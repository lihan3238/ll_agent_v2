# src/tools/conda_env.py
import os
import subprocess
import shlex # [新增] 用于正确拆分命令字符串
from src.utils.logger import sys_logger

class CondaManager:
    def __init__(self, project_name: str):
        self.env_name = f"pf_{project_name}"
        self.workspace_root = os.path.join("workspace", project_name)
        self.code_dir = os.path.join(self.workspace_root, "code")
        if not os.path.exists(self.code_dir):
            os.makedirs(self.code_dir)

    def create_env(self, env_yaml_content: str) -> tuple[bool, str]:
        """
        根据 yaml 内容创建/更新环境
        Returns: (success, error_message)
        """
        yaml_path = os.path.join(self.code_dir, "environment.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(env_yaml_content)
            
        abs_yaml_path = os.path.abspath(yaml_path)
        
        # 这里的命令不需要改，保持原样
        cmd = ["conda", "env", "update", "-f", abs_yaml_path, "-n", self.env_name, "--prune"]
        
        sys_logger.info(f"[CMD] {' '.join(cmd)}") 
        
        return_code, _, stderr = self._run_subprocess(cmd, capture_output=True)
        
        if return_code == 0:
            return True, ""
        else:
            return False, stderr

    def run_code(self, command: str) -> tuple[int, str, str]:
        """
        在环境中运行任意命令
        :param command: e.g. "python main.py" or "pytest"
        """
        sys_logger.info(f"🏃 Running command: '{command}' in env {self.env_name}...")
        
        # [核心修复] 使用 shlex.split 正确拆分输入的命令字符串
        # 之前是硬编码 ["python", script_name]，导致 python python main.py
        cmd_parts = shlex.split(command)
        
        # 构造最终命令：conda run -n env_name --no-capture-output [parts...]
        cmd = ["conda", "run", "-n", self.env_name, "--no-capture-output"] + cmd_parts
        
        sys_logger.info(f"[CMD] {' '.join(cmd)}") 
        return self._run_subprocess(cmd, capture_output=True)

    def _run_subprocess(self, cmd: list, capture_output=False) -> tuple:
        try:
            sys_logger.debug(f"[CWD] {self.code_dir}")
            
            result = subprocess.run(
                cmd,
                cwd=self.code_dir,
                text=True,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            
            # 日志过滤逻辑
            ignore_keywords = ["FutureWarning", "Terms of Service", "remote_definition"]
            clean_stderr = "\n".join([line for line in stderr.splitlines() if not any(k in line for k in ignore_keywords)])
            
            if result.returncode != 0:
                sys_logger.error(f"[EXEC FAIL] Code: {result.returncode}")
                if clean_stderr:
                    sys_logger.error(f"[STDERR Sample]:\n{clean_stderr[:500]}...") 
            elif clean_stderr:
                sys_logger.warning(f"[STDERR (Warning)]: {clean_stderr[:200]}...")

            return result.returncode, stdout, clean_stderr
            
        except Exception as e:
            sys_logger.error(f"[EXEC ERROR] Exception: {e}")
            return -1, "", str(e)