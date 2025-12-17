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
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["user_template"]
        
        # 构造初始 Feedback
        current_feedback = feedback_instruction
        if previous_design:
            # 只有在人工反馈时才带上旧设计，防止Token过长
            try:
                prev_json = previous_design.model_dump_json(indent=2)
                # 截取一部分以防爆炸，或者只保留 key information
                if not current_feedback:
                    current_feedback = "Refine the previous design."
                current_feedback += f"\n\n[Reference Previous Draft]:\n{prev_json[:4000]}..." 
            except: pass

        # --- 自我修正循环 (Self-Correction Loop) ---
        max_retries = 3
        best_design = None
        
        for i in range(max_retries):
            # 1. 构造 Feedback Context
            feedback_context = ""
            if current_feedback:
                feedback_context = f"### REVISION REQUEST:\n{current_feedback}"
            else:
                feedback_context = "(Initial Design Task)"

            # 2. 调用 LLM
            try:
                design = self.call_llm_with_struct(
                    prompt_template=full_prompt,
                    schema=DesignDocument,
                    field=theory.research_field,
                    methodology=theory.proposed_methodology,
                    gaps=theory.theoretical_analysis, 
                    feedback_context=feedback_context
                )
            except Exception as e:
                sys_logger.error(f"Architect LLM Error: {e}")
                continue

            # 3. 完整性校验
            is_valid, critique = self._validate_design(design)
            
            if is_valid:
                sys_logger.info(f"✅ Architect Design Passed Validation (Iter {i+1}).")
                # 最后的修补
                self._post_process_check(design)
                return design
            else:
                sys_logger.warning(f"⚠️ Architect Design Incomplete (Iter {i+1}): {critique}")
                # 将批评意见加入下一次的 Prompt
                current_feedback = f"""
                Your previous output was REJECTED because:
                {critique}
                
                **INSTRUCTION**: 
                1. Keep the `experiments_plan` (it was good).
                2. BUT YOU MUST FILL IN THE MISSING PARTS (`requirements` and `file_structure`).
                3. Do not be lazy. Design the full file tree.
                """
                best_design = design # 暂存，如果最后都失败了就用这个

        sys_logger.error("❌ Architect failed to produce complete design after retries.")
        if best_design:
            self._post_process_check(best_design)
            return best_design
        
        # 兜底返回一个空对象防止 Crash
        return DesignDocument(project_name="Fallback_Project", data_flow_diagram="Error", main_execution_flow="Error")

    def _validate_design(self, design: DesignDocument) -> tuple[bool, str]:
        """检查设计是否偷懒"""
        errors = []
        
        # 1. 检查依赖
        if not design.requirements or len(design.requirements) < 2:
            errors.append("- `requirements` list is empty or too short.")
            
        # 2. 检查文件数量
        if not design.file_structure or len(design.file_structure) < 3:
            errors.append(f"- `file_structure` only has {len(design.file_structure)} files. A real project needs more (data, model, utils, main).")
            
        # 3. 检查是否有 main.py
        has_main = any("main.py" in f.filename for f in design.file_structure)
        if not has_main:
            errors.append("- Missing `main.py` entry point.")
            
        # 4. 检查是否有绘图代码 (对应 experiments_plan)
        if design.experiments_plan:
            has_plotter = any("plot" in f.filename.lower() or "vis" in f.filename.lower() or "utils" in f.filename.lower() for f in design.file_structure)
            if not has_plotter:
                errors.append("- Defined experiments but no `utils/plotter.py` or similar file to generate figures.")

        if errors:
            return False, "\n".join(errors)
        return True, ""

    def _post_process_check(self, design: DesignDocument):
        """最后的兜底修补"""
        # 确保有 main.py
        files = [f.filename for f in design.file_structure]
        if "main.py" not in files:
            from src.core.schema import FileSpec
            design.file_structure.append(FileSpec(
                filename="main.py",
                description="Entry point.",
                imports=["src.utils"],
                core_logic_steps=["Run experiments", "Save results.json"]
            ))
            
        # 确保 requirements 不为空
        if not design.requirements:
            design.requirements = ["numpy", "pandas", "matplotlib", "torch"]