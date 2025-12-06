# src/tools/conda_env.py
import os
import subprocess
import yaml
from src.utils.logger import sys_logger

class CondaManager:
    def __init__(self, project_name: str):
        # 环境名加前缀，防止污染 base
        self.env_name = f"pf_{project_name}"
        self.workspace_root = os.path.join("workspace", project_name)
        self.code_dir = os.path.join(self.workspace_root, "code")
        
        if not os.path.exists(self.code_dir):
            os.makedirs(self.code_dir)

    def create_env(self, env_yaml_content: str) -> bool:
        """根据 yaml 内容创建/更新环境"""
        yaml_path = os.path.join(self.code_dir, "environment.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(env_yaml_content)
            
        sys_logger.info(f"🐍 Creating/Updating Conda env: {self.env_name}...")
        
        # 使用 conda env update --prune 确保环境干净
        cmd = ["conda", "env", "update", "-f", yaml_path, "-n", self.env_name, "--prune"]
        
        return self._run_subprocess(cmd)

    def run_code(self, script_name: str = "main.py") -> tuple[int, str, str]:
        """
        在环境中运行 Python 脚本
        Returns: (return_code, stdout, stderr)
        """
        sys_logger.info(f"🏃 Running {script_name} in env {self.env_name}...")
        
        # 使用 conda run -n {env} python {script}
        # 这是最稳健的方式，不需要激活环境
        cmd = ["conda", "run", "-n", self.env_name, "--no-capture-output", "python", script_name]
        
        return self._run_subprocess(cmd, capture_output=True)

    def _run_subprocess(self, cmd: list, capture_output=False) -> bool | tuple:
        try:
            # 统一在 code 目录下运行，这样相对路径 (如 data/) 才是对的
            result = subprocess.run(
                cmd,
                cwd=self.code_dir,
                text=True, # 自动解码为字符串
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                encoding='utf-8', 
                errors='replace' # 防止编码错误 crash
            )
            
            if capture_output:
                return result.returncode, result.stdout or "", result.stderr or ""
            
            if result.returncode != 0:
                sys_logger.error(f"Command failed with code {result.returncode}")
                return False
            return True
            
        except Exception as e:
            sys_logger.error(f"Conda command exception: {e}")
            if capture_output:
                return -1, "", str(e)
            return False