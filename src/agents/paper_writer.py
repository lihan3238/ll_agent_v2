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
                      previous_content: str = "",
                      references_context: str = "",
                      existing_text: str = "", 
                      feedback: str = "") -> SectionContent:
        
        sys_logger.info(f"Writing Section: {section_name}...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["section_template"]
        
        # 1. 格式化 Gap Analysis
        gaps_str = "\n".join([f"- {g.existing_method}: {g.limitation_description}" for g in research.gap_analysis])
        
        # 2. [核心修复] 格式化实验计划 (Experiments Plan)
        # 告诉 Writer 必须用到哪些图表
        exp_context = "### Hyperparameters:\n" + str(architect.hyperparameters) + "\n\n"
        
        if architect.experiments_plan:
            exp_context += "### REQUIRED EXPERIMENT ARTIFACTS (MUST INSERT PLACEHOLDERS):\n"
            exp_context += "You MUST include the following figures/tables in the 'Experiments' section using standard LaTeX:\n"
            for exp in architect.experiments_plan:
                exp_context += f"- Type: {exp.type}\n"
                exp_context += f"  Filename: {{{exp.filename}}}  <-- USE EXACTLY THIS PATH\n"
                exp_context += f"  Caption: {exp.description}\n"
                exp_context += f"  Data Source: {exp.metrics_source}\n\n"
        else:
            exp_context += "(No specific figures designed. Use standard text description.)"

        # 3. 构造 Revision Context
        revision_context = ""
        if feedback and existing_text:
            revision_context = f"""
            === REVISION MODE ===
            **ORIGINAL DRAFT**:
            {existing_text}
            
            **REVIEWER FEEDBACK**:
            {feedback}
            
            **TASK**: Rewrite the draft. Fix issues but KEEP the valid citations and figure placeholders.
            """
        else:
            revision_context = "(Writing new content from scratch)"

        # 4. 调用 LLM
        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=SectionContent,
            title=research.refined_idea,
            section_name=section_name,
            refined_idea=research.refined_idea,
            gaps=gaps_str,
            methodology=theory.proposed_methodology,
            experiments_context=exp_context, # [修改] 传入完整的实验上下文
            previous_content=previous_content[-3000:] if len(previous_content) > 3000 else previous_content,
            references_context=references_context,
            revision_context=revision_context
        )