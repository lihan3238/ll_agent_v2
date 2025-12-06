# src/agents/paper_writer.py
from typing import List
from src.agents.base import BaseAgent
from src.core.schema import PaperDraft, PaperOutline, SectionContent, ResearchReport, TheoreticalFramework, DesignDocument
from src.utils.logger import sys_logger

class PaperWriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="paper_writer")

    def plan_outline(self, research: ResearchReport, theory: TheoreticalFramework, architect: DesignDocument) -> PaperOutline:
        sys_logger.info("Task Started: Planning paper outline...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["outline_template"]
        
        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=PaperOutline,
            refined_idea=research.refined_idea,
            theory_summary=theory.proposed_methodology[:1000], # 摘要一下防止爆
            arch_summary=architect.main_execution_flow
        )

    def write_section(self, section_name: str, 
                      research: ResearchReport, 
                      theory: TheoreticalFramework, 
                      architect: DesignDocument,
                      previous_content: str = "") -> SectionContent:
        
        sys_logger.info(f"Writing Section: {section_name}...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["section_template"]
        
        # 格式化 Gap Analysis
        gaps_str = "\n".join([f"- {g.existing_method}: {g.limitation_description}" for g in research.gap_analysis])
        
        # 格式化前文 (简单截取最后 2000 字符作为上下文，或者传递上一节的摘要)
        # 这里为了连贯性，传入上一节的完整内容（如果 token 允许）
        context_window = previous_content[-3000:] if len(previous_content) > 3000 else previous_content
        
        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=SectionContent,
            title=research.refined_idea, # 暂用 Idea 当标题上下文
            section_name=section_name,
            refined_idea=research.refined_idea,
            gaps=gaps_str,
            methodology=theory.proposed_methodology,
            experiments=architect.hyperparameters,
            previous_content=context_window
        )