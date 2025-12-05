# src/agents/theorist.py
from src.agents.base import BaseAgent
from src.core.schema import TheoreticalFramework, ResearchReport
from src.utils.logger import sys_logger

class TheoristAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="theorist")

    def run(self, report: ResearchReport, full_paper_context: str = "", feedback_instruction: str = "", previous_theory: TheoreticalFramework = None) -> TheoreticalFramework:
        sys_logger.info("Task Started: Theorizing...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["user_template"]
        
        # 构造 Feedback 上下文
        feedback_context = ""
        
        # [核心修改] 显式区分 "Reviewer Feedback" 和 "Self-Correction"
        if feedback_instruction:
            feedback_context += f"""
            ### 🛠️ REVISION GOAL
            The objective for this iteration is:
            "{feedback_instruction}"
            """
        
        if previous_theory:
            # 提取上一轮的核心内容作为参考
            feedback_context += f"""
            
            ### 📜 PREVIOUS DRAFT (For Reference)
            **Title**: {previous_theory.research_field}
            **Existing Math**: {previous_theory.problem_formulation[:500]}...
            
            **INSTRUCTION**: You are iterating on this draft. IMPROVE IT based on the Revision Goal. 
            Do NOT output the exact same content. Add details, fix errors, or expand proofs.
            """
        else:
            feedback_context += "\n(No previous draft. Start fresh.)"
        
        # [Context Injection]
        # 将 PDF 全文拼接到 Related Work 中，这是给 LLM 的一种 Trick
        enhanced_related_work = report.related_work_summary
        if full_paper_context:
            enhanced_related_work += f"\n\n=== DEEP DIVE: FULL TEXT CONTENT ===\n{full_paper_context}\n======================================"

        # 格式化 Gap Analysis
        gaps_str = ""
        for g in report.gap_analysis:
            gaps_str += f"- **Target**: {g.existing_method}\n  **Flaw**: {g.limitation_description}\n  **Root Cause**: {g.mathematical_root_cause}\n\n"

        # 调用 LLM
        framework = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=TheoreticalFramework,
            refined_idea=report.refined_idea,
            related_work=enhanced_related_work, # 包含全文
            gap_analysis_context=gaps_str,       # 包含痛点
            feedback_context=feedback_context
        )
        
        sys_logger.info(f"Theory Developed. Field: {framework.research_field}")
        return framework