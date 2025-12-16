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
        if not state.paper: raise ValueError("❌ Missing Paper Draft.") # 确保已规划
        
        config = state_manager._load_config()
        max_retries = config.get("workflow", {}).get("coder_retries", 3)
        
        # 0. 获取模型配置
        coder_config = config.get("agents", {}).get("coder", {})
        default_model = config.get("llm", {}).get("default_model", "gpt-4o")
        target_model = coder_config.get("model", default_model)
        # [修改] 获取 max_tokens
        # 优先级: Agent配置 > 全局默认 > 16384
        global_max = config.get("llm", {}).get("default_max_tokens", 16384)
        target_max_tokens = coder_config.get("max_tokens", global_max)
        # 1. Setup Environment Manager
        conda = CondaManager(state.project_name)
        
        # 2. Setup Aider [修改] 传入 max_tokens
        aider_agent = CoderAgentAider(
            project_path=conda.code_dir, 
            model_name=target_model,
            max_tokens=target_max_tokens # <--- 传进去
        )
        # ====================================================
        # Step 1: Environment Setup (Template)
        # ====================================================
        sys_logger.info(f">>> [Phase 5.1] Setting up Conda Environment: {conda.env_name}")
        template_path = os.path.join("assets", "templates", "env", "base_environment.yaml")
        
        if not os.path.exists(template_path):
             # 兜底：如果没有模板，创建一个默认的
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
            
        success, msg = conda.create_env(yaml.dump(env_data))
        if not success:
            sys_logger.error(f"⛔ Base env creation failed: {msg}")
            raise RuntimeError("Failed to create base conda environment.")
        
        # ====================================================
        # Step 2: Aider Implementation
        # ====================================================
        if not os.path.exists(os.path.join(conda.code_dir, "main.py")):
            sys_logger.info(">>> [Phase 5.2] Aider Implementation (Paper-Driven)...")
            aider_agent.implement_design(state.architecture)
        else:
            sys_logger.info(">>> Code exists. Skipping implementation...")

        # ====================================================
        # Step 3: Execution & Artifact Verification Loop
        # ====================================================
        sys_logger.info(">>> [Phase 5.3] Execution & Verification Loop...")
        run_command = "python main.py"
        logs = []
        final_results = None
        
        # 获取必须生成的产物清单
        required_artifacts = []
        # 1. 基础数据
        required_artifacts.append({"path": "results.json", "desc": "Numerical Metrics"})
        # 2. 论文图表
        if state.architecture.experiments_plan:
            for exp in state.architecture.experiments_plan:
                required_artifacts.append({"path": exp.filename, "desc": exp.description})

        for i in range(max_retries):
            sys_logger.info(f"   -> Run Attempt {i+1}/{max_retries}...")
            
            # 运行代码
            ret, stdout, stderr = conda.run_code(run_command)
            
            logs.append(CodeExecutionLog(
                command=run_command, return_code=ret,
                stdout=stdout[-2000:], stderr=stderr[-2000:]
            ))
            
            needs_fix = False
            fix_message = ""
            
            if ret == 0:
                sys_logger.info("✅ Execution Successful (Exit 0). Verifying Artifacts...")
                
                # --- 核心验证逻辑 ---
                missing_files = []
                for artifact in required_artifacts:
                    file_path = os.path.join(conda.code_dir, artifact["path"])
                    if not os.path.exists(file_path):
                        missing_files.append(f"- {artifact['path']} ({artifact['desc']})")
                
                if not missing_files:
                    # 全部通过
                    sys_logger.info("🏆 All required artifacts generated!")
                    
                    # 读取结果
                    try:
                        with open(os.path.join(conda.code_dir, "results.json"), "r") as f:
                            metrics = json.load(f)
                        
                        # 收集实际存在的图片路径 (用于展示)
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
                    # 运行成功，但缺文件
                    needs_fix = True
                    missing_str = "\n".join(missing_files)
                    fix_message = f"""
                    [SYSTEM ERROR] The code ran successfully (exit code 0), BUT failed to generate these MANDATORY files required for the paper:
                    
                    {missing_str}
                    
                    **ACTION REQUIRED**:
                    1. You MUST implement the plotting/saving logic for these files.
                    2. Ensure they are saved to the correct paths (e.g. `figures/`).
                    3. Update `main.py` to call these functions.
                    """
                    sys_logger.warning(f"⚠️ Missing {len(missing_files)} artifacts. Triggering fix...")
            else:
                # 运行报错
                needs_fix = True
                fix_message = stderr if stderr.strip() else stdout[-1000:]
                sys_logger.warning(f"❌ Run failed with code {ret}.")

            # 调用修复
            if needs_fix and i < max_retries - 1:
                # 简单的 URL 清洗，防止 Aider 爬取
                clean_msg = re.sub(r'https?://\S+', '', fix_message)
                aider_agent.fix_error(run_command, clean_msg)
                
                # 如果是环境问题，尝试同步
                if "ModuleNotFoundError" in fix_message:
                    with open(target_env_path, "r") as f:
                        conda.create_env(f.read())

        # Finalize
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