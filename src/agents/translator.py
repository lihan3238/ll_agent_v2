# src/agents/translator.py
from src.agents.base import BaseAgent
from src.core.schema import UserFeedback
from src.utils.logger import sys_logger

class TranslatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="translator")

    def process_feedback(self, raw_content: str) -> UserFeedback:
        """
        输入：Markdown 文件的全部内容（含中文）
        输出：结构化的 UserFeedback 对象
        """
        sys_logger.info("Translator: Processing user input...")
        
        # 1. 拼接完整 Prompt
        # 我们把 system 和 user_template 拼成一个大的模板
        full_prompt_template = self.prompts["system"] + "\n\n" + self.prompts["user_template"]
        
        # 2. 调用通用能力
        # BaseAgent 会自动处理 {json_schema} 的注入
        # 我们只需要传入 {raw_content}
        result = self.call_llm_with_struct(
            prompt_template=full_prompt_template,
            schema=UserFeedback,
            raw_content=raw_content
        )
        
        sys_logger.info(f"Translator Result: Action={result.action}, Feedback='{result.feedback_en[:50]}...'")
        return result