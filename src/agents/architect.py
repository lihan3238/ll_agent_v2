# src/agents/architect.py
from src.agents.base import BaseAgent
from src.core.schema import DesignDocument, TheoreticalFramework
from src.utils.logger import sys_logger
import json

class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="architect")

    def run(self, theory: TheoreticalFramework, 
            feedback_instruction: str = "",
            previous_design: DesignDocument = None) -> DesignDocument:
        
        sys_logger.info(f"🏗️ Architect: Designing system for '{theory.research_field}'...")
        
        # 1. 准备 Prompt
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["user_template"]
        
        # 2. 处理 Feedback 上下文
        feedback_context = "No previous feedback. Start from scratch."
        if feedback_instruction:
            feedback_context = f"""
            !!! REVISION REQUIRED !!!
            **Reviewer Feedback**: "{feedback_instruction}"
            
            **Action**:
            - Modify the Previous Draft to address the feedback.
            - Keep the parts that work, fix the parts that don't.
            - Ensure the JSON structure remains valid.
            """

        # 3. 注入上一轮设计 (如果存在)
        # 这是一个 Trick: 把上一轮的 JSON 放在 Prompt 里，让 LLM "修改" 而不是 "凭空想象"
        if previous_design:
            try:
                # 只取前 3000 字符防止 Token 爆炸，或者完整放进去（取决于模型窗口）
                # 既然用 GPT-4o/DeepSeek，通常可以放完整的
                prev_json = previous_design.model_dump_json(indent=2)
                feedback_context += f"\n\n=== PREVIOUS DRAFT ===\n{prev_json}\n======================"
            except Exception as e:
                sys_logger.warning(f"Failed to serialize previous design: {e}")

        # 4. 调用 LLM
        design = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=DesignDocument,
            field=theory.research_field,
            methodology=theory.proposed_methodology,
            gaps=theory.theoretical_analysis, 
            feedback_context=feedback_context
        )
        
        # 5. 后处理/校验
        self._post_process_check(design)
        
        sys_logger.info(f"✅ Design ready: {design.project_name} ({len(design.file_structure)} files)")
        return design

    def _post_process_check(self, design: DesignDocument):
        """简单校验，防止低级错误"""
        files = [f.filename for f in design.file_structure]
        
        # 强制检查 main.py
        if "main.py" not in files:
            sys_logger.warning("Architect forgot main.py! Injecting a placeholder.")
            from src.core.schema import FileSpec
            design.file_structure.append(FileSpec(
                filename="main.py",
                description="Entry point for training and evaluation.",
                imports=["src.train"],
                classes=[],
                functions=[],
                # 这里的逻辑描述会传给 Aider
                core_logic_steps=[
                    "Initialize config",
                    "Run training loop",
                    "Evaluate model",
                    "Save metrics to results.json (MANDATORY)"
                ]
            ))
            
        # 强制检查 __init__.py
        dirs = set()
        for f in files:
            if "/" in f:
                d = f.rsplit("/", 1)[0]
                dirs.add(d)
        
        for d in dirs:
            init_file = f"{d}/__init__.py"
            if init_file not in files:
                # 可以在这里自动补全，或者只是由 Coder 处理（Aider 通常懂这个，但显式更好）
                pass