# main.py
import sys
import os

# 确保 Python 路径包含 src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.state_manager import state_manager
from src.phases.phase_01_research import ResearchPhase
from src.phases.phase_02_theory import TheoryPhase
from src.phases.phase_03_architect import ArchitectPhase
# from src.phases.phase_04_paper import PaperPhase (未来)
# from src.phases.phase_05_coder import CoderPhase (未来)

def run_pipeline(initial_idea: str):
    print("🚀 Starting PaperForge Pipeline...")
    
    # 1. 初始化
    state = state_manager.load_state()
    if not state.user_initial_idea:
        state.user_initial_idea = initial_idea
        state_manager.save_state(state)

    # 2. 定义流水线
    pipeline = [
        ResearchPhase(),
        TheoryPhase(),
        ArchitectPhase()
    ]
    
    # 3. 执行
    for phase in pipeline:
        try:
            # 自动处理 Load -> Check Breakpoint -> Run -> Save
            state = phase.execute()
            
            # 额外检查产出是否生成
            if phase.phase_name == "research" and not state.research:
                print("❌ Critical Error: Research failed. Aborting.")
                return
            if phase.phase_name == "theory" and not state.theory:
                print("❌ Critical Error: Theory failed. Aborting.")
                return
                
        except Exception as e:
            print(f"❌ Pipeline Failed at {phase.phase_name}: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*50)
    print("🎉 ALL PHASES COMPLETED")
    print("="*50)
    print(f"Final State saved to: {state_manager.state_file}")
    
    if state.architecture:
        print(f"Blueprint Classes: {sum(len(f.classes) for f in state.architecture.file_structure)}")

if __name__ == "__main__":
    my_idea = "I want to use Mamba state space models for time series forecasting on weather data, comparing it with Transformer."
    run_pipeline(my_idea)