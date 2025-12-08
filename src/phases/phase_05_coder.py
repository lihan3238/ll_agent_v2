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
        # [修改] 增加 Paper 检查，确保流程顺序
        # if not state.architecture:
        #     raise ValueError("❌ Missing Architecture Design.")
        if not state.paper:
            raise ValueError("❌ Missing Paper Draft. Please complete Paper Phase first.")
        # 1. 准备环境
        config = state_manager._load_config()
        env_config = config.get("execution_env", {})
        
        conda = CondaManager(state.project_name)
        coder = CoderAgent()
        
        # 2. 生成初始代码
        sys_logger.info(">>> Step 1: Generating Codebase...")
        codebase = coder.generate_code(state.architecture, env_config)
        
        # 写入硬盘
        self._write_files(conda.code_dir, codebase.files)
        
        # 3. 创建/更新 Conda 环境
        env_yaml = next((f.content for f in codebase.files if "environment" in f.filename), None)
        if env_yaml:
            success = conda.create_env(env_yaml)
            if not success:
                sys_logger.error("Failed to create Conda environment. Aborting Coder Phase.")
                raise RuntimeError("Conda environment creation failed. Check logs for details.")
        else:
            sys_logger.warning("No environment.yaml found! Code generation might be incomplete.")

        # 4. 运行 & 调试循环
        # 注意：这里我们设定一个固定的重试次数，比如 5 次
        max_retries = 5 
        logs = []
        final_results = None
        
        for i in range(max_retries + 1):
            sys_logger.info(f"\n>>> Step 2: Execution Attempt {i+1}/{max_retries+1}...")
            
            # 运行 main.py
            ret, stdout, stderr = conda.run_code("main.py")
            
            log = CodeExecutionLog(
                command="python main.py",
                return_code=ret,
                stdout=stdout[-2000:], # 只存最后一部分 log，防止 state.json 爆炸
                stderr=stderr[-2000:]
            )
            logs.append(log)
            
            if ret == 0:
                sys_logger.info("✅ Code executed successfully!")
                # 检查 results.json
                results_path = os.path.join(conda.code_dir, "results.json")
                if os.path.exists(results_path):
                    try:
                        with open(results_path, "r") as f:
                            metrics = json.load(f)
                        final_results = ExperimentResults(
                            metrics=metrics,
                            figures=[], # 可以在这里 scan figures 目录
                            status=ExecutionStatus.SUCCESS
                        )
                        break # 成功退出
                    except Exception as e:
                        sys_logger.error(f"Failed to read results.json: {e}")
                        # 这是一个特殊的错误，代码跑通了但没生成结果，也需要 fix
                        stderr = f"Code ran successfully but results.json could not be read: {e}"
                else:
                    sys_logger.warning("Code ran but results.json not found.")
                    stderr = "Code execution finished (exit code 0), but 'results.json' was not found. Did you save the metrics?"
            
            # 如果失败（ret!=0）或者 没生成 results.json
            sys_logger.error(f"❌ Execution/Result Issue. Triggering Auto-Fix...")
            
            if i == max_retries:
                sys_logger.error("Max retries reached. Coding failed.")
                break
            
            # 触发自我修复
            current_files = self._read_all_files(conda.code_dir)
            
            # 把报错信息喂给 Coder
            # 注意：如果 stderr 为空但 ret!=0 (极少见)，用 stdout 的最后部分
            error_msg = stderr if stderr.strip() else stdout[-1000:]
            
            fixed_codebase = coder.fix_code(error_msg, current_files)
            
            # 覆盖写入 (只写入修改过的文件)
            self._write_files(conda.code_dir, fixed_codebase.files)
            sys_logger.info(f"🔧 Applied fixes to {len(fixed_codebase.files)} files.")

        # 5. 保存结果到 State
        state.coder = CoderOutput(
            environment_yaml=env_yaml or "",
            execution_log=logs,
            results=final_results
        )
        
        return state

    def _write_files(self, base_dir, files):
        for file in files:
            # 防止路径穿越
            safe_filename = file.filename.replace("..", "").lstrip("/\\")
            path = os.path.join(base_dir, safe_filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(file.content)
            sys_logger.info(f"Wrote {safe_filename}")

    def _read_all_files(self, base_dir) -> Dict[str, str]:
        files = {}
        for root, _, filenames in os.walk(base_dir):
            for name in filenames:
                if name.endswith(".py") or name.endswith(".yaml") or name.endswith(".sh"):
                    rel_path = os.path.relpath(os.path.join(root, name), base_dir)
                    try:
                        with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                            files[rel_path] = f.read()
                    except:
                        pass # 忽略二进制文件等
        return files