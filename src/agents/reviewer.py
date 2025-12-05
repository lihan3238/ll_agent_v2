# src/agents/reviewer.py
import json
from src.agents.base import BaseAgent
from src.core.schema import ReviewReport, ReviewDecision, UserFeedback, ActionType
from src.utils.logger import sys_logger

class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="reviewer")

    def review(self, phase_name: str, data_object: object, iteration_idx: int = 0) -> UserFeedback:
        sys_logger.info(f"🧐 Reviewer #2 evaluating: {phase_name} (Iter: {iteration_idx})...")
        
        if hasattr(data_object, "model_dump_json"):
            content_json = data_object.model_dump_json(indent=2)
        else:
            content_json = str(data_object)

        # [修改] 模板选择逻辑
        phase_lower = phase_name.lower()
        if "theory" in phase_lower:
            template_key = "theory_template"
        elif "architect" in phase_lower or "design" in phase_lower:
            template_key = "architect_template"
        else:
            template_key = "research_template"

        full_prompt = self.prompts["system"] + "\n\n" + self.prompts[template_key]

        # 首轮严厉模式
        if iteration_idx == 0:
            full_prompt += """
            **SPECIAL INSTRUCTION FOR ROUND 0**:
            This is the FIRST DRAFT. Output 'decision': 'REVISE'. Max Score: 6.
            Be extremely critical about structure and completeness.
            """

        report = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=ReviewReport,
            content_json=content_json
        )
        
        sys_logger.info(f"Review Result: {report.decision} (Score: {report.score}/10)")
        
        action = ActionType.APPROVE if report.decision == ReviewDecision.ACCEPT else ActionType.REVISE
        
        # 强制首轮打回
        if iteration_idx == 0 and action == ActionType.APPROVE:
             sys_logger.warning("🤖 LLM tried to approve first draft. System override -> REVISE.")
             action = ActionType.REVISE
             report.specific_instructions = "System Override: First draft requires iteration. Check for hardcoded values or missing dependencies."

        feedback_text = report.specific_instructions if action == ActionType.REVISE else "Quality check passed."
        
        return UserFeedback(
            action=action,
            feedback_en=feedback_text,
            comments=f"Score: {report.score}. Critique: {report.critique}"
        )