# src/agents/architect.py
from src.agents.base import BaseAgent
from src.core.schema import DesignDocument, TheoreticalFramework
from src.utils.logger import sys_logger

class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="architect")

    def run(self, theory: TheoreticalFramework, 
            feedback_instruction: str = "",
            previous_design: DesignDocument = None) -> DesignDocument:
        
        sys_logger.info(f"Task Started: Architecting solution for {theory.research_field}...")
        
        # 拼接 Prompt
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["user_template"]
        
        # 构造 Feedback 上下文
        feedback_context = ""
        if feedback_instruction:
            feedback_context = f"""
            !!! REVISION REQUEST (CRITICAL) !!!
            The previous design was reviewed.
            Feedback: "{feedback_instruction}"
            
            Action: Fix the issues in the Previous Draft. 
            **CRITICAL INSTRUCTION**: You must regenerate the ENTIRE JSON object. Do not output only the changed parts. Copy the unchanged parts from the Previous Draft if necessary.
            """
        else:
            feedback_context = "No previous feedback. Initial design."

        # 注入上一轮的设计草稿
        if previous_design:
            # exclude={'main_execution_flow'} 可能会导致 LLM 忘记生成这个字段，
            # 既然我们要全量输出，最好把全量参考给它，或者只 exclude 极少部分
            # 这里改为全量 dump，让 LLM 方便抄作业
            prev_json = previous_design.model_dump_json(indent=2)
            
            feedback_context += f"""
            
            === PREVIOUS DRAFT (REFERENCE) ===
            {prev_json}
            ==================================
            """

        design = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=DesignDocument,
            field=theory.research_field,
            methodology=theory.proposed_methodology,
            gaps=theory.theoretical_analysis, 
            feedback_context=feedback_context
        )
        
        sys_logger.info(f"Blueprint generated. Planned {len(design.file_structure)} files.")
        return design