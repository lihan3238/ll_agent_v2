# src/phases/phase_05_coder.py
import os
import json
import re
import yaml
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.tools.conda_env import CondaManager
from src.core.schema import CodeExecutionLog, ExperimentResults, ExecutionStatus, CoderOutput
from src.utils.logger import sys_logger
from src.agents.coder_aider import CoderAgentAider

class CoderPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="coder")

    def check_completion(self, state: ProjectState) -> bool:
        return state.coder is not None and state.coder.results and state.coder.results.status == ExecutionStatus.SUCCESS

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        if not state.architecture: raise ValueError("❌ Missing Architecture.")
        if not state.paper: raise ValueError("❌ Missing Paper Draft.") 
        
        config = state_manager._load_config()
        
        # --- [新增] 读取配置参数 ---
        workflow_conf = config.get("workflow", {})
        max_retries = workflow_conf.get("coder_retries", 3)
        # 默认 3 次环境重试
        env_retries = workflow_conf.get("env_setup_retries", 3) 
        # 默认 3 次 Lint 反思
        lint_retries = workflow_conf.get("lint_retries", 3)
        
        # 0. 获取模型配置
        coder_config = config.get("agents", {}).get("coder", {})
        default_model = config.get("llm", {}).get("default_model", "gpt-4o")
        target_model = coder_config.get("model", default_model)
        
        global_max = config.get("llm", {}).get("default_max_tokens", 16384)
        target_max_tokens = coder_config.get("max_tokens", global_max)
        
        # 1. Setup Environment Manager
        conda = CondaManager(state.project_name)
        
        # 2. Setup Aider [修改] 传入 lint_retries
        aider_agent = CoderAgentAider(
            project_path=conda.code_dir, 
            model_name=target_model,
            max_tokens=target_max_tokens,
            lint_retries=lint_retries # <--- 传入自定义的反思次数
        )

        # ====================================================
        # Step 1: Environment Setup (Template)
        # ====================================================
        sys_logger.info(f">>> [Phase 5.1] Setting up Conda Environment: {conda.env_name}")
        template_path = os.path.join("assets", "templates", "env", "base_environment.yaml")
        
        if not os.path.exists(template_path):
             os.makedirs(os.path.dirname(template_path), exist_ok=True)
             with open(template_path, "w") as f:
                 f.write("name: placeholder\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - pip\n  - numpy\n  - pip:\n    - torch\n")

        with open(template_path, "r", encoding="utf-8") as f:
            env_data = yaml.safe_load(f)
        
        env_data["name"] = conda.env_name
        target_env_path = os.path.join(conda.code_dir, "environment.yaml")
        
        if not os.path.exists(conda.code_dir):
            os.makedirs(conda.code_dir)
            
        with open(target_env_path, "w", encoding="utf-8") as f:
            yaml.dump(env_data, f, sort_keys=False)
            
        # [修改] 使用 Config 中的 env_retries
        env_success = False
        for k in range(env_retries):
            with open(target_env_path, 'r', encoding='utf-8') as f:
                current_env_content = f.read()
                
            success, msg = conda.create_env(current_env_content)
            if success:
                sys_logger.info("✅ Conda environment created successfully.")
                env_success = True
                break
            else:
                sys_logger.warning(f"⚠️ Env failed ({k+1}/{env_retries}). Asking Aider to fix...")
                clean_msg = re.sub(r'https?://\S+', '', msg)
                hint = "\n\n[HINT]: If packages are missing in Conda, move them to the `pip:` section."
                aider_agent.fix_error("conda env update", clean_msg + hint)

        if not env_success:
            sys_logger.error(f"⛔ Base env creation failed after {env_retries} attempts.")
            raise RuntimeError("Failed to create base conda environment.")
        
        # ====================================================
        # Step 2: Aider Implementation
        # ====================================================
        missing_source_files = []
        for file_spec in state.architecture.file_structure:
            clean_name = file_spec.filename.replace("\\", "/")
            fpath = os.path.join(conda.code_dir, clean_name)
            if not os.path.exists(fpath):
                missing_source_files.append(fpath)

        if not os.path.exists(os.path.join(conda.code_dir, "main.py")) or len(missing_source_files) > len(state.architecture.file_structure) * 0.5:
            sys_logger.info(">>> [Phase 5.2] Aider Implementation (Paper-Driven)...")
            aider_agent.implement_design(state.architecture)
        else:
            sys_logger.info(">>> Code exists. Skipping full implementation...")
            if missing_source_files:
                sys_logger.info(f"⚠️ Found {len(missing_source_files)} missing files. Creating placeholders...")
                for fpath in missing_source_files:
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(f'"""\nPlaceholder for missing file: {os.path.basename(fpath)}\n"""\n')

        # ====================================================
        # Step 3: Execution & Artifact Verification Loop
        # ====================================================
        sys_logger.info(">>> [Phase 5.3] Execution & Verification Loop...")
        run_command = "python main.py"
        logs = []
        final_results = None
        
        required_artifacts = []
        required_artifacts.append({"path": "results.json", "desc": "Numerical Metrics"})
        if state.architecture.experiments_plan:
            for exp in state.architecture.experiments_plan:
                required_artifacts.append({"path": exp.filename, "desc": exp.description})

        for i in range(max_retries):
            sys_logger.info(f"   -> Run Attempt {i+1}/{max_retries}...")
            
            ret, stdout, stderr = conda.run_code(run_command)
            
            logs.append(CodeExecutionLog(
                command=run_command, return_code=ret,
                stdout=stdout[-2000:], stderr=stderr[-2000:]
            ))
            
            needs_fix = False
            fix_message = ""
            
            if ret == 0:
                sys_logger.info("✅ Execution Successful (Exit 0). Verifying Artifacts...")
                
                missing_artifacts = []
                for artifact in required_artifacts:
                    file_path = os.path.join(conda.code_dir, artifact["path"])
                    if not os.path.exists(file_path):
                        missing_artifacts.append(f"- Artifact: {artifact['path']} ({artifact['desc']})")
                
                missing_source = []
                for file_spec in state.architecture.file_structure:
                    clean_name = file_spec.filename.replace("\\", "/")
                    fpath = os.path.join(conda.code_dir, clean_name)
                    if not os.path.exists(fpath) or os.path.getsize(fpath) < 50:
                        missing_source.append(f"- Source: {clean_name}")

                if not missing_artifacts and not missing_source:
                    sys_logger.info("🏆 All required artifacts and source files generated!")
                    
                    try:
                        with open(os.path.join(conda.code_dir, "results.json"), "r") as f:
                            metrics = json.load(f)
                        
                        figures_found = [
                            f["path"] for f in required_artifacts 
                            if f["path"].endswith(('.png', '.pdf', '.jpg'))
                        ]
                        
                        final_results = ExperimentResults(
                            metrics=metrics, figures=figures_found, status=ExecutionStatus.SUCCESS
                        )
                        break
                    except Exception as e:
                        needs_fix = True
                        fix_message = f"Execution success, but 'results.json' is invalid: {e}"
                else:
                    needs_fix = True
                    missing_str = "\n".join(missing_artifacts + missing_source)
                    fix_message = f"""
                    [SYSTEM ERROR] The code ran successfully (exit code 0), BUT failed to generate MANDATORY files or implement required source files:
                    
                    {missing_str}
                    
                    **ACTION REQUIRED**:
                    1. If 'Source' is missing: Implement the missing python file logic based on the architecture. OVERWRITE the file with full code.
                    2. If 'Artifact' is missing: Implement the plotting/saving logic in `main.py` or relevant files.
                    3. Ensure all files are saved to the correct paths.
                    """
                    sys_logger.warning(f"⚠️ Missing {len(missing_artifacts) + len(missing_source)} items. Triggering fix...")
            else:
                needs_fix = True
                fix_message = stderr if stderr.strip() else stdout[-1000:]
                sys_logger.warning(f"❌ Run failed with code {ret}.")

            if needs_fix and i < max_retries - 1:
                clean_msg = re.sub(r'https?://\S+', '', fix_message)
                aider_agent.fix_error(run_command, clean_msg)
                
                if "ModuleNotFoundError" in fix_message:
                    with open(target_env_path, "r") as f:
                        conda.create_env(f.read())

        state.coder = CoderOutput(
            environment_yaml="Managed",
            execution_log=logs,
            results=final_results
        )
        
        if final_results and final_results.status == ExecutionStatus.SUCCESS:
            return state
        else:
            sys_logger.error("Coder Phase finished without full success.")
            return state