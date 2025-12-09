# src/phases/phase_05_coder.py
import os
import json
from typing import Dict
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.agents.coder import CoderAgent
from src.tools.conda_env import CondaManager
from src.core.schema import CoderOutput, CodeExecutionLog, ExperimentResults, ExecutionStatus
from src.utils.logger import sys_logger

class CoderPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="coder")

    def check_completion(self, state: ProjectState) -> bool:
        # 如果有成功的运行结果，视为完成
        return state.coder is not None and state.coder.results and state.coder.results.status == ExecutionStatus.SUCCESS

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        # 前置依赖检查
        if not state.architecture:
            raise ValueError("❌ Missing Architecture Design.")
        if not state.paper:
            raise ValueError("❌ Missing Paper Draft. Please complete Paper Phase first.")

        # 1. 准备配置与工具
        config = state_manager._load_config()
        env_config = config.get("execution_env", {})
        # 从 config 读取重试次数，默认 5 次
        max_retries = config.get("workflow", {}).get("coder_retries", 5)
        
        conda = CondaManager(state.project_name)
        coder = CoderAgent()
        
        # 2. 生成初始代码
        sys_logger.info(">>> Step 1: Generating Codebase...")
        codebase = coder.generate_code(state.architecture, env_config)
        
        # 写入硬盘
        self._write_files(conda.code_dir, codebase.files)
        
        # 3. 创建/更新初始 Conda 环境
        env_yaml_file = next((f for f in codebase.files if "environment.yaml" in f.filename or "environment.yml" in f.filename), None)
        
        if env_yaml_file:
            success = conda.create_env(env_yaml_file.content)
            if not success:
                sys_logger.error("Failed to create initial Conda environment. Aborting.")
                raise RuntimeError("Conda environment creation failed.")
        else:
            sys_logger.warning("No environment.yaml found! Code generation might be incomplete.")

        # 4. 运行 & 调试循环
        logs = []
        final_results = None
        
        # 循环次数 = 初始运行(1) + 重试次数(max_retries)
        for i in range(max_retries + 1):
            sys_logger.info(f"\n>>> Step 2: Execution Attempt {i+1}/{max_retries+1}...")
            
            # --- A. 运行代码 ---
            ret, stdout, stderr = conda.run_code("main.py")
            
            # 记录日志
            log = CodeExecutionLog(
                command="python main.py",
                return_code=ret,
                stdout=stdout[-5000:], # 防止日志过大，截取最后部分
                stderr=stderr[-5000:]
            )
            logs.append(log)
            
            # --- B. 成功判定 ---
            if ret == 0:
                sys_logger.info("✅ Code executed successfully (Exit Code 0).")
                # 检查 results.json
                results_path = os.path.join(conda.code_dir, "results.json")
                if os.path.exists(results_path):
                    try:
                        with open(results_path, "r") as f:
                            metrics = json.load(f)
                        final_results = ExperimentResults(
                            metrics=metrics,
                            figures=[], # 后续可扩展：扫描 figures 目录
                            status=ExecutionStatus.SUCCESS
                        )
                        sys_logger.info(f"🏆 Metrics captured: {metrics}")
                        break # 成功退出循环
                    except Exception as e:
                        sys_logger.error(f"Failed to read results.json: {e}")
                        stderr = f"Code ran successfully but results.json parse failed: {e}"
                else:
                    sys_logger.warning("Code ran but results.json not found.")
                    stderr = "Code execution finished (exit code 0), but 'results.json' was not found. Did you save the metrics?"
            
            # --- C. 失败处理 & 退出条件 ---
            sys_logger.error(f"❌ Execution Issue detected.")
            
            if i == max_retries:
                sys_logger.error("Max retries reached. Coding phase failed to produce valid results.")
                break
            
            # --- D. 自我修复 (Self-Healing) ---
            # 读取当前所有代码作为 Context
            current_files = self._read_all_files(conda.code_dir)
            
            # 构造错误信息 (优先 stderr, 其次 stdout 后几行)
            error_msg = stderr if stderr.strip() else stdout[-1000:]
            if "ModuleNotFoundError" in error_msg:
                error_msg += "\n\nHINT: Missing library. Please update `environment.yaml`."

            # 调用 Agent 修复
            fixed_codebase = coder.fix_code(error_msg, current_files)
            
            # 覆盖写入修复后的文件
            self._write_files(conda.code_dir, fixed_codebase.files)
            sys_logger.info(f"🔧 Applied fixes to {len(fixed_codebase.files)} files.")

            # --- E. 环境自动修复 (Environment Auto-Fix) ---
            # 检查是否有 environment.yaml 的更新
            updated_env_file = next((f for f in fixed_codebase.files if "environment.yaml" in f.filename), None)
            
            if updated_env_file:
                sys_logger.info("♻️ Detected environment definition change. Updating Conda env...")
                # 再次调用注入逻辑，确保 config 中的 base_requirements 依然存在
                # 注意：这里我们假设 coder.fix_code 返回的内容是纯 LLM 生成的，
                # 为了保险，最好再次注入一次 base_requirements。
                # 但由于 coder.fix_code 内部逻辑比较独立，这里为了保持简单，
                # 我们假设 LLM 在修复时保留了原有的结构。
                # 更严谨的做法是调用 coder._inject_requirements，但那个方法是私有的且设计用于 generate 阶段。
                # 鉴于 fix 阶段 LLM 是基于原文修改，通常不会丢掉 pip 依赖。
                
                env_success = conda.create_env(updated_env_file.content)
                if not env_success:
                    sys_logger.error("Environment update failed during fix loop. Subsequent run might fail.")

        # 5. 保存结果到 State
        state.coder = CoderOutput(
            environment_yaml=env_yaml_file.content if env_yaml_file else "",
            execution_log=logs,
            results=final_results
        )
        
        return state

    def _write_files(self, base_dir, files):
        for file in files:
            # 1. 统一路径分隔符：将 Windows 的 \ 替换为 /
            normalized_name = file.filename.replace("\\", "/")
            
            # 2. 防止路径穿越
            safe_filename = normalized_name.replace("..", "").lstrip("/")
            
            path = os.path.join(base_dir, safe_filename)
            
            # 3. 确保父目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(file.content)
            sys_logger.info(f"Wrote {safe_filename}")

    def _read_all_files(self, base_dir) -> Dict[str, str]:
        files = {}
        for root, _, filenames in os.walk(base_dir):
            for name in filenames:
                # 排除 pycache, git, vscode 等目录
                if any(x in root for x in ["__pycache__", ".git", ".vscode"]):
                    continue
                    
                if name.endswith(".py") or name.endswith(".yaml") or name.endswith(".yml") or name.endswith(".sh"):
                    rel_path = os.path.relpath(os.path.join(root, name), base_dir)
                    try:
                        with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                            files[rel_path] = f.read()
                    except Exception:
                        pass 
        return files