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
    def __init__(self, project_path: str, model_name: str = "gpt-4o", max_tokens: int = None):
        """
        初始化 Aider 代理
        :param project_path: 代码根目录
        :param model_name: 模型名称 (e.g. openai/deepseek-chat)
        :param max_tokens: 最大输出 token 数限制
        """
        self.project_path = os.path.abspath(project_path)
        self.model_name = model_name
        self.max_tokens = max_tokens
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
        
        # 只有文件不存在时才创建，避免覆盖用户设置
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("\n".join(ignore_content))
                
        # 2. 初始化 Git
        git_dir = os.path.join(self.project_path, ".git")
        if not os.path.exists(git_dir):
            try:
                # 忽略 git init 的输出，防止污染日志
                subprocess.run(["git", "init"], cwd=self.project_path, check=False, capture_output=True)
                # 配置临时的 git user，防止 commit 报错
                subprocess.run(["git", "config", "user.email", "ai@coder.com"], cwd=self.project_path, check=False)
                subprocess.run(["git", "config", "user.name", "AI Coder"], cwd=self.project_path, check=False)
                
                # 立即提交 gitignore，使其生效
                subprocess.run(["git", "add", ".gitignore"], cwd=self.project_path, check=False, capture_output=True)
                subprocess.run(["git", "commit", "-m", "chore: add gitignore"], cwd=self.project_path, check=False, capture_output=True)
            except Exception:
                pass

    def _create_aider(self, fnames: List[str] = None, auto_commit=True) -> Coder:
        """
        创建一个 Aider Coder 实例
        """
        # 设置日志路径
        chat_history_path = os.path.join(self.project_path, "aider_chat_history.md")
        
        io = InputOutput(
            pretty=False,
            yes=True,
            input_history_file=None,
            chat_history_file=chat_history_path
        )
        
        model = Model(self.model_name)
        
        # [关键] 强制覆盖 Aider 的最大输出限制，防止长代码截断
        if self.max_tokens:
            model.max_output_tokens = self.max_tokens
            sys_logger.info(f"🔧 Forced Aider max_output_tokens to {self.max_tokens}")
        
        # [关键] 针对 DeepSeek/非GPT4模型，强制使用 'whole' 模式
        # 这会让模型输出整个文件内容，而不是 Diff，解决"只生成注释"或"Diff匹配失败"的问题
        edit_format = None
        if "deepseek" in self.model_name.lower() or "claude" in self.model_name.lower():
            edit_format = "whole" 
            sys_logger.info(f"🤖 Detected non-GPT4 model ({self.model_name}), enforcing 'whole' edit format.")
        
        return Coder.create(
            main_model=model, 
            io=io, 
            fnames=fnames, # 传入初始文件列表
            auto_commits=auto_commit, 
            dirty_commits=False,
            edit_format=edit_format # 显式传入编辑格式
        )

    def implement_design(self, design: DesignDocument):
        """
        Phase 1: 基于设计文档，从零构建项目
        """
        sys_logger.info(f"🤖 Aider Coder started in {self.project_path}")

        # 1. Scaffolding: 创建空文件，给 Aider 明确的“靶子”
        all_files = []
        for file_spec in design.file_structure:
            clean_filename = file_spec.filename.replace("\\", "/") # 规范化路径
            full_path = os.path.join(self.project_path, clean_filename)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if not os.path.exists(full_path):
                with open(full_path, 'w', encoding='utf-8') as f:
                    # 写入 docstring 帮助 Aider 理解文件用途
                    f.write(f'"""\n{file_spec.description}\n"""\n')
            all_files.append(full_path)

        # 2. 启动 Aider
        coder = self._create_aider(fnames=all_files)

        # 3. Implement Logic
        sys_logger.info("Aider: Implementing Core Logic...")
        
        # 构建文件结构说明
        files_instruction = ""
        for f in design.file_structure:
            files_instruction += f"\n--- File: {f.filename} ---\n"
            if f.classes:
                for c in f.classes:
                    files_instruction += f"Class {c.name}: {c.description}\n"
                    for m in c.methods:
                        files_instruction += f"  - Method {m.name}: {m.docstring}\n"
                        if m.core_logic_steps:
                            # 传入伪代码逻辑
                            files_instruction += f"    Logic: {'; '.join(m.core_logic_steps)}\n"

        # 构建实验产物指令 (Results.json & Figures)
        experiments_instruction = "\n\n=== MANDATORY OUTPUTS ===\n"
        experiments_instruction += "1. `main.py` MUST save numerical metrics to `results.json`.\n"
        if hasattr(design, 'experiments_plan') and design.experiments_plan:
            for exp in design.experiments_plan:
                experiments_instruction += f"- Generate Artifact: {exp.filename} ({exp.description})\n"

        # [Prompt 强化] 明确告诉模型输出完整代码，并规定输出格式以防止垃圾文件
        master_prompt = f"""
        You are the Lead Research Engineer.
        
        **Objective**: Implement the complete codebase based on the specs below.
        
        {experiments_instruction}

        **Architecture Overview**:
        {files_instruction}
        
        **Execution Flow**:
        {design.main_execution_flow}
        
        **CRITICAL INSTRUCTIONS (READ CAREFULLY)**:
        1. **OVERWRITE MODE**: The current files contain only skeletons/placeholders. **IGNORE** the existing content. **OVERWRITE** them with the full, working implementation.
        2. **WRITE FULL CODE**: Output the **entire content** of each file you edit. Do not use diffs or search/replace blocks.
        3. **FILE FORMAT**: Start each file with the filename on its own line, followed by the code block.
           Example:
           src/main.py
           ```python
           import os
           ...
           ```
        4. **NO CHATTER**: Do not output conversational text like "Here is the code". Just the file paths and code.
        5. **Imports**: Use absolute imports (e.g. `from src.models import ...`).
        6. **Completeness**: Write working code. **REMOVE** all `raise NotImplementedError` and `pass`.
        """
        
        coder.run(master_prompt)
        sys_logger.info("✅ Aider finished implementation.")

    def fix_error(self, run_command: str, error_log: str):
        """
        Phase 2: 自动修复模式
        """
        sys_logger.info(f"🚑 Aider Fixing Error for: {run_command}")
        
        # 1. 自动发现项目中的所有 py 文件和 yaml 文件
        py_files = glob.glob(os.path.join(self.project_path, "**", "*.py"), recursive=True)
        yaml_files = glob.glob(os.path.join(self.project_path, "**", "*.yaml"), recursive=True)
        all_context_files = py_files + yaml_files
        
        coder = self._create_aider(fnames=all_context_files)
        
        # 2. 构造修复 Prompt
        fix_prompt = f"""
        Command `{run_command}` failed OR produced incomplete results.
        
        **Error / Issue**:
        ```
        {error_log}
        ```
        
        **TASK**:
        1. Analyze the error.
        2. Fix the code. **Output the FULL content** of the fixed file(s).
        3. If "Missing File": Implement the missing logic to save that file.
        4. If "ModuleNotFoundError": Update `environment.yaml`.
        
        **FORMAT**:
        filename.ext
        ```language
        ... content ...
        ```
        """
        
        coder.run(fix_prompt)