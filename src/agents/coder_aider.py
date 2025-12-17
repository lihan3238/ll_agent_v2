# src/agents/coder_aider.py
import os
import glob
import subprocess
from typing import List, Optional
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from src.core.schema import DesignDocument
from src.utils.logger import sys_logger

class CoderAgentAider:
    def __init__(self, project_path: str, model_name: str = "gpt-4o", max_tokens: int = None, lint_retries: int = 3):
        """
        初始化 Aider 代理
        :param project_path: 代码根目录
        :param model_name: 模型名称 (e.g. openai/deepseek-chat)
        :param max_tokens: 最大输出 token 数限制
        :param lint_retries: 语法检查自动修复的最大次数
        """
        self.project_path = os.path.abspath(project_path)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.lint_retries = lint_retries # [新增] 保存重试次数
        self._init_git()
        
    def _init_git(self):
        """Aider 需要在 git 仓库中运行以进行版本控制和回滚"""
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
            
        # 1. 创建 .gitignore 防止 Aider 读取日志和环境文件
        gitignore_path = os.path.join(self.project_path, ".gitignore")
        ignore_content = [
            "aider_chat_history.md",  # 忽略日志
            ".aider*",                # 忽略 aider 内部文件
            "__pycache__/",
            "*.pyc",
            "results.json",           # 结果文件
            "figures/",               # 图片
            ".DS_Store",
            "*.log"
        ]
        
        # 只有文件不存在时才创建
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("\n".join(ignore_content))
                
        # 2. 初始化 Git
        git_dir = os.path.join(self.project_path, ".git")
        if not os.path.exists(git_dir):
            try:
                # 忽略 git init 的输出
                subprocess.run(["git", "init"], cwd=self.project_path, check=False, capture_output=True)
                subprocess.run(["git", "config", "user.email", "ai@coder.com"], cwd=self.project_path, check=False)
                subprocess.run(["git", "config", "user.name", "AI Coder"], cwd=self.project_path, check=False)
                
                # 立即提交 gitignore
                subprocess.run(["git", "add", ".gitignore"], cwd=self.project_path, check=False, capture_output=True)
                subprocess.run(["git", "commit", "-m", "chore: add gitignore"], cwd=self.project_path, check=False, capture_output=True)
            except Exception:
                pass

    def _create_aider(self, fnames: List[str] = None, auto_commit=True) -> Coder:
        """
        创建一个 Aider Coder 实例
        """
        chat_history_path = os.path.join(self.project_path, "aider_chat_history.md")
        
        io = InputOutput(
            pretty=False,
            yes=True,
            input_history_file=None,
            chat_history_file=chat_history_path
        )
        
        model = Model(self.model_name)
        
        if self.max_tokens:
            model.max_output_tokens = self.max_tokens
            sys_logger.info(f"🔧 Forced Aider max_output_tokens to {self.max_tokens}")
        
        edit_format = None
        if "deepseek" in self.model_name.lower() or "claude" in self.model_name.lower():
            edit_format = "whole" 
            sys_logger.info(f"🤖 Detected non-GPT4 model ({self.model_name}), enforcing 'whole' edit format.")
        
        lint_cmd = "python -m py_compile"

        # [修复] 移除 max_reflections 参数
        coder = Coder.create(
            main_model=model, 
            io=io, 
            fnames=fnames, 
            auto_commits=auto_commit, 
            dirty_commits=False,
            edit_format=edit_format, 
            
            use_git=True,
            lint_cmds={
                "python": lint_cmd
            },
            auto_lint=True
        )

        # [修复] 手动设置 max_reflections
        if hasattr(coder, 'max_reflections'):
            coder.max_reflections = self.lint_retries
            
        return coder

    def implement_design(self, design: DesignDocument):
        """
        Phase 1: 基于设计文档，从零构建项目 (迭代式生成)
        """
        sys_logger.info(f"🤖 Aider Coder started in {self.project_path}")

        # 1. Scaffolding
        all_files = []
        for file_spec in design.file_structure:
            clean_filename = file_spec.filename.replace("\\", "/")
            full_path = os.path.join(self.project_path, clean_filename)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if not os.path.exists(full_path):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(f'"""\n{file_spec.description}\n"""\n')
            all_files.append(full_path)

        # 2. 启动 Aider
        coder = self._create_aider(fnames=all_files)

        # 3. Implement Logic (Iterative)
        sys_logger.info("Aider: Implementing Core Logic (Iterative Mode)...")
        
        experiments_instruction = "\n=== MANDATORY OUTPUTS ===\n"
        experiments_instruction += "1. `main.py` MUST save numerical metrics to `results.json`.\n"
        
        if hasattr(design, 'experiments_plan') and design.experiments_plan:
            for exp in design.experiments_plan:
                experiments_instruction += f"- Artifact: {exp.filename} ({exp.description})\n"
                if exp.metrics_source:
                    experiments_instruction += f"  - Data Source: {exp.metrics_source}\n"
        else:
            experiments_instruction += "  - Save generic metrics to `results.json`."

        global_context = f"""
        **Project**: {design.project_name}
        **Style**: {design.architecture_style}
        **Data Flow**: {design.data_flow_diagram}
        **Global Goal**: Produce valid `results.json` and figures.
        """

        def sort_priority(f):
            name = f.filename.lower()
            if "config" in name: return 0
            if "util" in name: return 1
            if "data" in name or "loader" in name: return 2
            if "model" in name or "net" in name: return 3
            if "train" in name or "eval" in name: return 4
            if "main" in name: return 10 
            return 5
            
        sorted_files = sorted(design.file_structure, key=sort_priority)

        for i, f in enumerate(sorted_files):
            sys_logger.info(f"  > Generating {i+1}/{len(sorted_files)}: {f.filename}")
            
            file_spec_str = f"--- Target: {f.filename} ---\nDesc: {f.description}\n"
            if f.classes:
                for c in f.classes:
                    file_spec_str += f"Class `{c.name}`: {c.description}\n"
                    for m in c.methods:
                        file_spec_str += f"  - Method `{m.name}`: {m.docstring}\n"
                        if m.core_logic_steps:
                            file_spec_str += f"    Logic: {'; '.join(m.core_logic_steps)}\n"
            if f.functions:
                for func in f.functions:
                    file_spec_str += f"Function `{func.name}`: {func.docstring}\n"

            current_context = global_context
            if "main.py" in f.filename or "plot" in f.filename or "vis" in f.filename or "util" in f.filename:
                current_context += experiments_instruction + "\n**INSTRUCTION**: Implement logic to generate/save these artifacts."

            file_prompt = f"""
            You are the Lead Engineer.
            
            {current_context}
            
            **CURRENT TASK**: Implement `{f.filename}`.
            
            **SPECIFICATION**:
            {file_spec_str}
            
            **CRITICAL RULES**:
            1. **OVERWRITE**: Output the **ENTIRE** content of the file. Do not use diffs.
            2. **FORMAT**: Start with the filename on its own line.
               Example:
               src/utils.py
               ```python
               ...
               ```
            3. **NO CHATTER**: Do not say "Here is the code". Just the file block.
            4. **IMPORTS**: Use absolute imports (e.g., `from src.models import ...`).
            5. **COMPLETENESS**: No `pass`. Implement full logic.
            """
            
            coder.run(file_prompt)

        sys_logger.info("✅ Aider finished implementation.")

    def fix_error(self, run_command: str, error_log: str):
        sys_logger.info(f"🚑 Aider Fixing Error for: {run_command}")
        
        py_files = glob.glob(os.path.join(self.project_path, "**", "*.py"), recursive=True)
        yaml_files = glob.glob(os.path.join(self.project_path, "**", "*.yaml"), recursive=True)
        all_context_files = py_files + yaml_files
        
        coder = self._create_aider(fnames=all_context_files)
        
        task_instruction = "Fix the code to resolve the error."
        
        if "MANDATORY files" in error_log or "FileNotFound" in error_log:
             task_instruction += "\n**CRITICAL**: You are missing logic to save specific files. Do NOT just catch the error. You MUST write the code to generate/save these files."
        
        if "SyntaxError" in error_log or "unmatched" in error_log:
             task_instruction += "\n**CRITICAL**: Check for mismatched parentheses `()` or braces `{}`."
             
        if "ModuleNotFoundError" in error_log:
            task_instruction += "\n**CRITICAL**: If a library is missing, add it to `environment.yaml` (prefer pip section)."

        fix_prompt = f"""
        Command `{run_command}` failed OR produced incomplete results.
        
        **Error / Issue**:
        ```
        {error_log}
        ```
        
        **TASK**:
        {task_instruction}
        
        **FORMAT**:
        filename.ext
        ```language
        ... full content ...
        ```
        """
        
        coder.run(fix_prompt)