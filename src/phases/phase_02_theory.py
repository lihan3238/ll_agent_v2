# src/phases/phase_02_theory.py
import os
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.schema import ActionType, ResearchReport
from src.core.interaction import interactor
from src.core.state_manager import state_manager
from src.agents.theorist import TheoristAgent
from src.tools.pdf_reader import pdf_tool
from src.utils.logger import sys_logger

class TheoryPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="theory")

    def check_completion(self, state: ProjectState) -> bool:
        return state.theory is not None

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        if not state.research:
            raise ValueError("❌ Missing Research Report.")

        full_paper_context = self._handle_paper_ingestion(state.project_name, state.research)
        
        theorist = TheoristAgent()
        current_framework = None
        current_feedback = ""
        
        config = state_manager._load_config()
        rounds = config.get("workflow", {}).get("theory_rounds", 3)
        internal_loops = config.get("workflow", {}).get("internal_loops", 2)
        
        for r in range(rounds):
            sys_logger.info(f"\n>>> 🛡️ Theory Review Cycle {r+1}/{rounds} <<<")
            
            # 标记本轮是否更新成功
            updated_in_this_round = False
            
            for k in range(internal_loops):
                try:
                    # 动态指令
                    if k == 0:
                        instruction = current_feedback if current_feedback else "Draft the initial theoretical framework."
                    else:
                        instruction = "CRITICAL SELF-REFLECTION: Identify one weak mathematical definition and expand it. Do NOT output the same content."

                    sys_logger.info(f"   --- Internal Step {k+1} ---")

                    new_framework = theorist.run(
                        report=state.research,
                        full_paper_context=full_paper_context,
                        feedback_instruction=instruction,
                        previous_theory=current_framework
                    )
                    
                    # [Check] 检查是否真的更新了
                    if current_framework and new_framework.proposed_methodology == current_framework.proposed_methodology:
                        sys_logger.warning("⚠️ Theorist output is identical to previous draft. Agent might be stuck.")
                    
                    current_framework = new_framework
                    updated_in_this_round = True
                    
                except Exception as e:
                    sys_logger.error(f"Theorist internal loop error: {e}")
                    # 不要 continue 跳过 Review，而是尝试重试或者中断
                    continue
            
            # --- Interaction / Review ---
            # 只有当 current_framework 存在时才 Review
            if current_framework:
                # 如果本轮全是报错（updated_in_this_round=False），那这是上一轮的旧货
                if not updated_in_this_round:
                    sys_logger.warning("⛔ Skipping review because Theorist failed to generate new content this round.")
                    continue 

                user_feedback = interactor.start_review(
                    phase_name=f"02_Theory_Round_{r+1}",
                    template_name="theory_review.md.j2",
                    context_data={"framework": current_framework},
                    iteration_idx=r
                )
                
                if user_feedback.action == ActionType.APPROVE:
                    sys_logger.info("✅ Theory Phase Approved.")
                    state.theory = current_framework
                    return state
                elif user_feedback.action == ActionType.REVISE:
                    current_feedback = user_feedback.feedback_en
        
        if current_framework:
             state.theory = current_framework
             
        return state

    def _handle_paper_ingestion(self, project_name: str, report: ResearchReport) -> str:
        """
        处理 PDF 下载和读取的子流程
        """
        sys_logger.info("\n=== 📖 Starting Paper Ingestion Workflow ===")
        
        # 1. 目录准备
        papers_dir = os.path.join("workspace", project_name, "papers")
        if not os.path.exists(papers_dir):
            os.makedirs(papers_dir)
            
        # 2. 筛选 Top 3 论文
        papers_to_read = report.top_papers[:3]
        if not papers_to_read:
            sys_logger.warning("No papers to read.")
            return ""

        # 3. 生成 Markdown 指令单
        readme_path = os.path.join(papers_dir, "DOWNLOAD_INSTRUCTIONS.md")
        content = f"# 📥 论文下载清单\n\n请下载以下 PDF 并重命名为指定文件名，放入当前文件夹：`{os.path.abspath(papers_dir)}`\n\n"
        content += "| ID | Title | Link | Target Filename |\n|---|---|---|---|\n"
        
        mapping = {}
        for idx, p in enumerate(papers_to_read):
            file_id = f"paper_{idx+1}.pdf"
            mapping[file_id] = p
            content += f"| {idx+1} | {p.title} | [Link]({p.url}) | `{file_id}` |\n"
            
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # 4. 阻塞交互
        print(f"\n{'='*60}")
        print(f"  🛑 ACTION REQUIRED: DOWNLOAD PAPERS")
        print(f"  📂 Folder: {papers_dir}")
        print(f"  📄 Please download {len(papers_to_read)} PDFs according to 'DOWNLOAD_INSTRUCTIONS.md'.")
        print(f"{'='*60}")
        
        while True:
            # 这里为了自动化测试方便，可以加一个 check
            # 如果是 autonomous 且文件已存在，自动跳过
            # 但首次运行必须暂停
            user_input = input(">>> Type 'ok' when ready: ")
            if user_input.lower().strip() == 'ok':
                missing = [f for f in mapping.keys() if not os.path.exists(os.path.join(papers_dir, f))]
                if not missing:
                    break
                print(f"❌ Missing: {missing}")
            else:
                print("Type 'ok' to continue.")

        # 5. 读取
        sys_logger.info("Ingesting PDFs...")
        full_text = ""
        for fname, info in mapping.items():
            path = os.path.join(papers_dir, fname)
            text = pdf_tool.read_pdf(path)
            full_text += f"\n\n=== PAPER: {info.title} ===\n{text}"
            
        return full_text