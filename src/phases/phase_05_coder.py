# src/phases/phase_05_coder.py
import os
import json
from typing import Dict
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.agents.coder import CoderAgent, Codebase, CodeFile
from src.tools.conda_env import CondaManager
from src.tools.code_utils import code_utils
from src.core.schema import CodeExecutionLog, ExperimentResults, ExecutionStatus
from src.utils.logger import sys_logger

class CoderPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="coder")

    def check_completion(self, state: ProjectState) -> bool:
        return state.coder is not None and state.coder.results and state.coder.results.status == ExecutionStatus.SUCCESS

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        if not state.architecture: raise ValueError("❌ Missing Architecture.")
        if not state.paper: raise ValueError("❌ Missing Paper Draft.")
        
        # 1. Setup
        config = state_manager._load_config()
        env_config = config.get("execution_env", {})
        max_retries = config.get("workflow", {}).get("coder_retries", 5)
        
        conda = CondaManager(state.project_name)
        coder = CoderAgent()
        
        # ====================================================
        # Step 1: Smart Scaffolding (LLM 生成带注释骨架)
        # ====================================================
        sys_logger.info(">>> [Phase 5.1] Smart Scaffolding (LLM-driven)...")
        
        initial_files = []
        
        # 遍历 Architect 定义的所有文件
        for file_spec in state.architecture.file_structure:
            # 1.1 使用 Agent 生成智能骨架
            # 传入 Research Idea 和 Architect Design 上下文
            skeleton_file = coder.write_smart_skeleton(
                file_spec=file_spec,
                design=state.architecture,
                research=state.research,
                env_config=env_config
            )
            initial_files.append(skeleton_file)
            
            # 1.2 为了保险，我们可以把 Agent 生成的文件名强制纠正回 Spec 里的文件名
            # 防止 LLM 只有内容对，文件名写错了
            skeleton_file.filename = file_spec.filename
        
        # 写入骨架
        self._write_files(conda.code_dir, initial_files)
        
        # 1.3 生成环境配置
        sys_logger.info("Generating environment config...")
        env_codebase = coder.generate_env_yaml(state.architecture, env_config)
        self._write_files(conda.code_dir, env_codebase.files)
        
        # ====================================================
        # Step 2: Environment Setup (带自动修复循环)
        # ====================================================
        sys_logger.info(">>> [Phase 5.2] Setting up Conda Environment (Auto-Fix Enabled)...")
        
        env_yaml_file = next((f for f in env_codebase.files if "environment" in f.filename), None)
        
        if env_yaml_file:
            env_retries = 3
            current_env_content = env_yaml_file.content
            
            for k in range(env_retries):
                sys_logger.info(f"   -> Env Setup Attempt {k+1}/{env_retries}")
                
                # 尝试创建环境
                success, stderr = conda.create_env(current_env_content)
                
                if success:
                    sys_logger.info("✅ Conda environment created successfully.")
                    break
                else:
                    sys_logger.warning(f"⚠️ Env creation failed. Triggering Agent to fix dependency issues...")
                    
                    if k == env_retries - 1:
                        raise RuntimeError(f"Failed to create Conda environment after {env_retries} attempts.\nLast Error: {stderr[:500]}")

                    # --- 修复环境文件 ---
                    # 构造错误上下文
                    error_context = f"Conda environment creation failed.\nError Log:\n{stderr}\n\nTask: Fix `environment.yaml`. If a package is not found in channels, move it to the `pip` section."
                    
                    # 使用 Coder 的 fix_code 能力，只传入 environment.yaml
                    fixed_codebase = coder.fix_code(
                        command="conda env update",
                        error_log=error_context,
                        files={"environment.yaml": current_env_content},
                        env_config=env_config
                    )
                    
                    # 获取修复后的内容
                    new_env_file = next((f for f in fixed_codebase.files if "environment" in f.filename), None)
                    if new_env_file:
                        # [关键] 修复后，再次注入 Config 中的强制依赖，防止 LLM 改乱了基础配置
                        # 我们临时构造一个 Codebase 对象来复用 _inject_requirements
                        coder._inject_requirements(fixed_codebase, env_config)
                        
                        # 更新当前内容并写入硬盘
                        updated_env_file = next((f for f in fixed_codebase.files if "environment" in f.filename), None)
                        current_env_content = updated_env_file.content
                        self._write_files(conda.code_dir, [updated_env_file])
                        sys_logger.info("🔧 Rewrote environment.yaml with fixes.")
                    else:
                        sys_logger.error("Agent failed to return a fixed environment.yaml.")
                        break
        else:
            sys_logger.warning("No environment.yaml found! Skipping environment setup.")

        # ====================================================
        # Step 2.5: Skeleton Verification (Smoke Test)
        # ====================================================
        sys_logger.info(">>> [Phase 5.2.5] Verifying Skeleton Structure...")
        
        # [核心修复] 必须加上 python 前缀
        run_command = "python main.py" 

        skeleton_retries = 3
        for i in range(skeleton_retries):
            ret, stdout, stderr = conda.run_code(run_command)
            
            if ret == 0:
                sys_logger.info("✅ Skeleton verification passed! Structure is valid.")
                break
            
            sys_logger.warning(f"⚠️ Skeleton failed (Attempt {i+1}/{skeleton_retries}). Fixing imports/structure...")
            
            # 触发修复
            error_msg = stderr if stderr.strip() else stdout[-1000:]
            current_files = self._read_all_files(conda.code_dir)
            
            error_context = f"SKELETON VERIFICATION FAILED. Do not implement logic yet. Fix imports/syntax/structure only.\nError:\n{error_msg}"
            
            fixed_codebase = coder.fix_code(
                command=run_command, # 使用带 python 的命令
                error_log=error_context,
                files=current_files,
                env_config=env_config
            )
            self._write_files(conda.code_dir, fixed_codebase.files)
            
            # 检查环境是否更新
            if any("environment.yaml" in f.filename for f in fixed_codebase.files):
                new_env = next(f.content for f in fixed_codebase.files if "environment.yaml" in f.filename)
                conda.create_env(new_env)
        
        # ====================================================
        # Step 3: Incremental Implementation (分步填肉)
        # ====================================================
        sys_logger.info(">>> [Phase 5.3] Implementing Logic File-by-File...")
        
        def sort_key(spec):
            name = spec.filename.lower()
            if "utils" in name or "config" in name: return 0
            if "data" in name: return 1
            if "model" in name: return 2
            if "train" in name: return 3
            if "main" in name: return 4
            return 5
        
        # 使用 state.architecture.file_structure 遍历
        sorted_specs = sorted(state.architecture.file_structure, key=sort_key)
        
        for file_spec in sorted_specs:
            if not file_spec.filename.endswith(".py"): continue
            
            sys_logger.info(f"   -> Working on: {file_spec.filename}")
            
            disk_path = os.path.join(conda.code_dir, file_spec.filename)
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    skeleton = f.read()
            except:
                skeleton = code_utils.generate_skeleton_from_design(file_spec)

            context_str = ""
            for root, _, filenames in os.walk(conda.code_dir):
                for fname in filenames:
                    if fname.endswith(".py") and fname != os.path.basename(file_spec.filename):
                        fpath = os.path.join(root, fname)
                        rel_path = os.path.relpath(fpath, conda.code_dir)
                        signature = code_utils.extract_ast_skeleton(fpath)
                        context_str += f"--- FILE: {rel_path} (Signatures Only) ---\n{signature}\n\n"
            
            impl_file = coder.implement_single_file(
                file_spec=file_spec,
                current_skeleton=skeleton,
                project_context=context_str,
                env_config=env_config
            )
            
            self._write_files(conda.code_dir, [impl_file])

        # ====================================================
        # Step 4: Execution Loop (Final Run)
        # ====================================================
        logs = []
        final_results = None
        
        # [核心修复] 不要尝试从 Codebase 对象读取 run_command，直接使用定义好的命令
        run_command = "python main.py"

        for i in range(max_retries + 1):
            sys_logger.info(f"\n>>> [Phase 5.4] Final Execution Attempt {i+1}/{max_retries+1}...")
            
            ret, stdout, stderr = conda.run_code(run_command)
            
            log = CodeExecutionLog(
                command=run_command, return_code=ret,
                stdout=stdout[-5000:], stderr=stderr[-5000:]
            )
            logs.append(log)
            
            if ret == 0:
                res_path = os.path.join(conda.code_dir, "results.json")
                if os.path.exists(res_path):
                    try:
                        with open(res_path, "r") as f: metrics = json.load(f)
                        final_results = ExperimentResults(
                            metrics=metrics, 
                            figures=[], 
                            status=ExecutionStatus.SUCCESS
                        )
                        # 扫描图表
                        figures_path = os.path.join(conda.code_dir, "figures")
                        if os.path.exists(figures_path):
                            for fig in os.listdir(figures_path):
                                if fig.endswith(('.png', '.pdf')):
                                    final_results.figures.append(os.path.join("code", "figures", fig))
                        
                        sys_logger.info(f"🏆 SUCCESS! Metrics: {metrics}")
                        break
                    except Exception as e:
                        stderr = f"Results JSON parse error: {e}"
                else:
                    stderr = "Execution success but 'results.json' missing."
            
            sys_logger.error("❌ Execution Failed. Analyzing...")
            if i == max_retries: break
            
            current_files = self._read_all_files(conda.code_dir)
            error_context = stderr if stderr.strip() else stdout[-2000:]
            
            fixed_codebase = coder.fix_code(
                command=run_command,
                error_log=error_context,
                files=current_files,
                env_config=env_config
            )
            
            self._write_files(conda.code_dir, fixed_codebase.files)
            sys_logger.info(f"🔧 Applied fixes to {len(fixed_codebase.files)} files.")
            
            if any("environment.yaml" in f.filename for f in fixed_codebase.files):
                sys_logger.info("♻️ Environment changed. Updating...")
                new_env = next(f.content for f in fixed_codebase.files if "environment.yaml" in f.filename)
                conda.create_env(new_env)

        state.coder = CoderOutput(
            environment_yaml="", 
            execution_log=logs,
            results=final_results
        )
        return state

    def _write_files(self, base_dir, files):
        for file in files:
            normalized_name = file.filename.replace("\\", "/")
            safe_filename = normalized_name.replace("..", "").lstrip("/")
            path = os.path.join(base_dir, safe_filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(file.content)
            sys_logger.info(f"Wrote {safe_filename}")

    def _read_all_files(self, base_dir) -> Dict[str, str]:
        files = {}
        for root, _, filenames in os.walk(base_dir):
            for name in filenames:
                if any(x in root for x in ["__pycache__", ".git", ".vscode", "figures"]): continue
                if name.endswith((".py", ".yaml", ".yml", ".json", ".md")):
                    rel = os.path.relpath(os.path.join(root, name), base_dir)
                    try:
                        with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                            files[rel] = f.read()
                    except: pass
        return files