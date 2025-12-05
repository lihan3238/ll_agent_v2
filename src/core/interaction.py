import os
import yaml
from jinja2 import Environment, FileSystemLoader
from src.utils.logger import sys_logger
from src.agents.translator import TranslatorAgent
from src.agents.reviewer import ReviewerAgent # 确保导入了 Reviewer
from src.core.schema import UserFeedback, ActionType

class InteractionManager:
    def __init__(self):
        self.config = self._load_project_config()
        
        project_conf = self.config.get("project", {})
        self.project_name = project_conf.get("name", "default_project")
        self.mode = project_conf.get("mode", "interactive")
        
        # Workspace
        self.workspace = os.path.join("workspace", self.project_name, "reviews")
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
            
        # Agents
        self.translator = TranslatorAgent()
        self.reviewer = ReviewerAgent()

        # Jinja2
        template_dir = os.path.join("assets", "templates", "reviews")
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _load_project_config(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def start_review(self, phase_name: str, template_name: str, context_data: dict, iteration_idx: int = 0) -> UserFeedback:
        """
        统一流程：
        1. 渲染 Markdown
        2. 保存文件 (留档)
        3. 分支：
           - Autonomous: 调用 Reviewer Agent
           - Interactive: 等待用户输入 -> Translator
        """
        
        # --- Step 1: Render Template (无论何种模式都执行) ---
        try:
            template = self.jinja_env.get_template(template_name)
            render_vars = {"phase_name": phase_name, **context_data}
            content = template.render(**render_vars)
        except Exception as e:
            sys_logger.error(f"Template rendering failed: {e}")
            # 降级处理：如果没有模板，转为字符串
            content = f"# Review: {phase_name}\n\nData:\n{str(context_data)}"

        # --- Step 2: Save File (留档) ---
        file_path = os.path.join(self.workspace, f"{phase_name}_review.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        sys_logger.info(f"📄 Review file generated: {file_path}")

        # --- Step 3: Branch Logic ---
        
        # A. 无人监管模式 (Autonomous) -> Reviewer Agent 介入
        if self.mode == "autonomous":
            sys_logger.info(f"[{phase_name}] Autonomous Mode: Delegating to Reviewer Agent.")
            
            # [修正] 智能提取要审查的核心对象
            # 我们约定 context_data 里通常有一个主键，比如 'report' 或 'framework'
            # 如果能找到，就传对象；找不到就传 context_data 字典
            data_to_review = context_data
            if 'report' in context_data:
                data_to_review = context_data['report']
            elif 'framework' in context_data:
                data_to_review = context_data['framework']
            
            # 传入对象给 Reviewer，而不是字符串
            return self.reviewer.review(phase_name, data_to_review, iteration_idx=iteration_idx)

        # B. 人机交互模式 (Interactive) -> 阻塞等待
        sys_logger.info(f"🛑 ACTION REQUIRED: Check {file_path}")
        print(f"\n{'='*60}")
        print(f"  ⏸️  SYSTEM PAUSED: {phase_name}")
        print(f"  📂 Review File: {file_path}")
        print(f"  📝 Please edit the file and Save.")
        print(f"{'='*60}")
        
        input(">>> Press ENTER after saving...")

        # 读取用户修改后的文件
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except FileNotFoundError:
            return UserFeedback(action=ActionType.APPROVE, feedback_en="", comments="File missing")

        # 翻译并返回
        return self.translator.process_feedback(raw_content)

interactor = InteractionManager()