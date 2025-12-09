# src/tools/conda_env.py
import os
import subprocess
import yaml
from src.utils.logger import sys_logger

class CondaManager:
    def __init__(self, project_name: str):
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
        
        # 使用绝对路径防止路径拼接错误
        abs_yaml_path = os.path.abspath(yaml_path)
        
        # 增加 --quiet 减少不必要的日志输出 (Terms of Service 等)
        # 增加 --yes 虽然 update 不需要，但加上更保险
        cmd = ["conda", "env", "update", "-f", abs_yaml_path, "-n", self.env_name, "--prune", "--quiet"]
        
        # 注意：create_env 不返回 output，只返回是否成功
        success, _, _ = self._run_subprocess(cmd)
        return success

    def run_code(self, script_name: str = "main.py") -> tuple[int, str, str]:
        """
        在环境中运行 Python 脚本
        """
        sys_logger.info(f"🏃 Running {script_name} in env {self.env_name}...")
        
        cmd = ["conda", "run", "-n", self.env_name, "--no-capture-output", "python", script_name]
        
        return self._run_subprocess(cmd, capture_output=True)

    def _run_subprocess(self, cmd: list, capture_output=False) -> tuple[bool, str, str] | tuple[int, str, str]:
        """
        统一的子进程执行器，带智能日志降噪
        """
        try:
            # 统一在 code 目录下运行
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
            
            # --- 智能日志处理 ---
            
            # 1. 过滤掉 Conda 的已知良性 Warning
            ignore_keywords = [
                "FutureWarning", 
                "Terms of Service", 
                "remote_definition", 
                "subparser"
            ]
            
            # 如果 stderr 只有这些噪音，就视为空，或者只作为 Debug 信息
            is_real_error = False
            if stderr.strip():
                clean_stderr = []
                for line in stderr.splitlines():
                    if not any(k in line for k in ignore_keywords):
                        clean_stderr.append(line)
                    else:
                        # 记录一下噪音，但在 debug 级别
                        # sys_logger.debug(f"Ignored Conda Noise: {line}")
                        pass
                
                # 如果过滤后还有内容，且 returncode != 0，那才是真报错
                if clean_stderr:
                    # 重新组装真正有用的报错信息
                    stderr = "\n".join(clean_stderr)
                    is_real_error = True

            # 2. 判断最终结果
            if result.returncode != 0:
                sys_logger.error(f"Command failed (Code {result.returncode})")
                if is_real_error:
                    sys_logger.error(f"Error Details:\n{stderr}")
                
                if capture_output:
                    return result.returncode, stdout, stderr
                return False, stdout, stderr
            
            # 3. 成功时的处理
            # 即使成功了，stderr 里也可能有 warning，我们只用 warning 级别打印
            if is_real_error and capture_output:
                # 这是一个 Warning
                sys_logger.warning(f"Command succeeded with warnings:\n{stderr}")

            if capture_output:
                return result.returncode, stdout, stderr
            
            return True, stdout, stderr
            
        except Exception as e:
            sys_logger.error(f"Conda command exception: {e}")
            if capture_output:
                return -1, "", str(e)
            return False, "", str(e)