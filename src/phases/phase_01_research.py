# src/phases/phase_01_research.py
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.schema import ActionType
from src.core.interaction import interactor
from src.core.state_manager import state_manager
from src.agents.researcher import ResearcherAgent
from src.utils.logger import sys_logger

class ResearchPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="research")

    def check_completion(self, state: ProjectState) -> bool:
        # 只要 research 字段有值，就算完成（这里可以加更复杂的判断）
        return state.research is not None

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        # 1. 初始化 Agent，注入全局记忆
        sys_logger.info("Initializing Researcher with existing knowledge base...")
        researcher = ResearcherAgent(
            existing_papers=state.paper_library,
            existing_gaps=state.known_gaps
        )
        
        # 确定输入 Idea
        initial_idea = state.user_initial_idea or "Please provide an initial idea."
        current_input = state.refined_idea or initial_idea
        last_report = None
        
        # 读取 Config (轮次配置)
        config = state_manager._load_config()
        review_rounds = config.get("workflow", {}).get("research_rounds", 3)
        internal_loops = config.get("workflow", {}).get("internal_loops", 2)
        
        # --- Main Loop ---
        for r in range(review_rounds):
            sys_logger.info(f"\n>>> 🛡️ Research Review Cycle {r+1}/{review_rounds} <<<")
            
            # --- Internal Self-Correction Loop ---
            for k in range(internal_loops):
                sys_logger.info(f"   --- Internal Step {k+1}/{internal_loops} ---")
                try:
                    last_report = researcher.run(user_idea=current_input)
                    # 自我迭代：把最新 refined idea 作为下一次输入
                    current_input = last_report.refined_idea 
                except Exception as e:
                    sys_logger.error(f"Researcher internal loop error: {e}")
                    import traceback
                    sys_logger.error(traceback.format_exc())
                    continue
            
            # --- Interaction / Review ---
            if last_report:
                # 调用 InteractionManager (自动触发 Reviewer Agent 或 人类)
                user_feedback = interactor.start_review(
                    phase_name=f"01_Research_Round_{r+1}",
                    template_name="research_review.md.j2",
                    context_data={"report": last_report},
                    iteration_idx=r # 传入轮次，触发 Reviewer 的严格模式
                )
                
                if user_feedback.action == ActionType.APPROVE:
                    sys_logger.info("✅ Research Phase Approved.")
                    
                    # [核心] Merge back to State
                    state.research = last_report
                    state.refined_idea = last_report.refined_idea
                    state.merge_papers(last_report.top_papers)
                    state.merge_gaps(last_report.gap_analysis)
                    
                    return state
                
                elif user_feedback.action == ActionType.REVISE:
                    sys_logger.info(f"🔄 Revision Requested: {user_feedback.feedback_en}")
                    current_input = f"{last_report.refined_idea}\n\n[FEEDBACK]: {user_feedback.feedback_en}"
        
        # 如果跑完次数还没过，保存最后一次结果
        if last_report:
             sys_logger.warning("⚠️ Max loops reached. Saving latest draft as final.")
             state.research = last_report
             state.refined_idea = last_report.refined_idea
             state.merge_papers(last_report.top_papers)
             state.merge_gaps(last_report.gap_analysis)

        return state