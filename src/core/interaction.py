import os
import yaml
from jinja2 import Environment, FileSystemLoader
from src.utils.logger import sys_logger
from src.agents.translator import TranslatorAgent
from src.agents.reviewer import ReviewerAgent
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
        统一流程：渲染 -> 保存 -> 评审 -> [新增]追加评审结果到文件 -> 返回
        """
        
        # --- Step 1: Render Template ---
        try:
            template = self.jinja_env.get_template(template_name)
            render_vars = {"phase_name": phase_name, **context_data}
            content = template.render(**render_vars)
        except Exception as e:
            sys_logger.error(f"Template rendering failed: {e}")
            content = f"# Review: {phase_name}\n\nData:\n{str(context_data)}"

        # --- Step 2: Save Initial File (Agent's Output) ---
        file_path = os.path.join(self.workspace, f"{phase_name}_review.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        sys_logger.info(f"📄 Review file generated: {file_path}")

        # --- Step 3: Branch Logic ---
        
        # A. 无人监管模式 (Autonomous)
        if self.mode == "autonomous":
            sys_logger.info(f"[{phase_name}] Autonomous Mode: Delegating to Reviewer Agent.")
            
            # 提取对象
            data_to_review = context_data.get('report') or context_data.get('framework') or context_data.get('design') or context_data
            
            # 1. 调用 Reviewer
            feedback = self.reviewer.review(phase_name, data_to_review, iteration_idx=iteration_idx)
            
            # 2. [核心修改] 将评审结果追加写入 Markdown 文件
            self._append_review_to_file(file_path, feedback)
            
            return feedback

        # B. 人机交互模式 (Interactive)
        sys_logger.info(f"🛑 ACTION REQUIRED: Check {file_path}")
        print(f"\n{'='*60}")
        print(f"  ⏸️  SYSTEM PAUSED: {phase_name}")
        print(f"  📂 Review File: {file_path}")
        print(f"  📝 Please edit the file and Save.")
        print(f"{'='*60}")
        
        input(">>> Press ENTER after saving...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except FileNotFoundError:
            return UserFeedback(action=ActionType.APPROVE, feedback_en="", comments="File missing")

        return self.translator.process_feedback(raw_content)

    def _append_review_to_file(self, file_path: str, feedback: UserFeedback):
        """
        辅助方法：将 Reviewer 的意见追加到 MD 文件末尾，形成完整的记录。
        """
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n\n---\n")
                f.write("# 🤖 Reviewer Report (Automated)\n\n")
                
                # 状态图标
                icon = "✅" if feedback.action == ActionType.APPROVE else "❌"
                f.write(f"**Decision**: {icon} **{feedback.action}**\n\n")
                
                # 提取分数和简评 (通常存储在 comments 里)
                f.write(f"**Evaluation**: {feedback.comments}\n\n")
                
                # 如果有具体的修改建议
                if feedback.feedback_en and feedback.action == ActionType.REVISE:
                    f.write("### 🛠️ Required Revisions\n")
                    f.write(f"> {feedback.feedback_en}\n")
                    
            sys_logger.info(f"📝 Review results appended to {file_path}")
            
        except Exception as e:
            sys_logger.error(f"Failed to append review to file: {e}")

interactor = InteractionManager()