# src/phases/phase_03_architect.py
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.schema import ActionType
from src.core.interaction import interactor
from src.core.state_manager import state_manager
from src.agents.architect import ArchitectAgent
from src.utils.logger import sys_logger

class ArchitectPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="architect")

    def check_completion(self, state: ProjectState) -> bool:
        return state.architecture is not None

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        if not state.theory:
            raise ValueError("❌ Missing Theoretical Framework.")

        architect = ArchitectAgent()
        current_design = None
        current_feedback = ""
        
        config = state_manager._load_config()
        rounds = config.get("workflow", {}).get("architect_rounds", 2)
        internal_loops = 1 
        
        for r in range(rounds):
            sys_logger.info(f"\n>>> 🛡️ Architect Review Cycle {r+1}/{rounds} <<<")
            
            updated_in_this_round = False

            for k in range(internal_loops):
                try:
                    instruction = current_feedback if (k==0 and current_feedback) else "Optimize modularity and clarify logic steps."
                    
                    new_design = architect.run(
                        theory=state.theory,
                        feedback_instruction=instruction,
                        previous_design=current_design
                    )
                    
                    current_design = new_design
                    updated_in_this_round = True
                    
                except Exception as e:
                    sys_logger.error(f"Architect internal loop error: {e}")
                    continue
            
            if current_design:
                if not updated_in_this_round:
                    sys_logger.warning("⛔ Skipping review because Architect failed to update.")
                    continue

                user_feedback = interactor.start_review(
                    phase_name=f"03_Architect_Round_{r+1}",
                    template_name="design_review.md.j2",
                    context_data={"design": current_design},
                    iteration_idx=r
                )
                
                if user_feedback.action == ActionType.APPROVE:
                    sys_logger.info("✅ Architect Phase Approved.")
                    state.architecture = current_design
                    return state
                elif user_feedback.action == ActionType.REVISE:
                    current_feedback = user_feedback.feedback_en
        
        if current_design:
             state.architecture = current_design
             
        return state