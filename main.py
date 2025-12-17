import sys
import os

# 确保 Python 路径包含 src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.state_manager import state_manager
# 显式导入 ProjectState 以便类型提示（虽然运行时由 state_manager 返回实例）
from src.core.state import ProjectState 
from src.phases.phase_01_research import ResearchPhase
from src.phases.phase_02_theory import TheoryPhase
from src.phases.phase_03_architect import ArchitectPhase
from src.phases.phase_04_paper import PaperPhase
from src.phases.phase_05_coder import CoderPhase
from src.phases.phase_06_refine import RefinePhase # [新增]

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
        ArchitectPhase(),
        PaperPhase(),
        CoderPhase(),
        RefinePhase() # [新增] Refine 是最后一步
    ]
    
    # 3. 执行循环
    for phase in pipeline:
        try:
            state = phase.execute()
            
            # 状态检查
            if phase.phase_name == "research" and not state.research:
                print("❌ Critical Error: Research failed. Aborting.")
                return
            if phase.phase_name == "theory" and not state.theory:
                print("❌ Critical Error: Theory failed. Aborting.")
                return
            if phase.phase_name == "architect" and not state.architecture:
                print("❌ Critical Error: Architect failed. Aborting.")
                return
            if phase.phase_name == "paper_draft" and not state.paper:
                print("❌ Critical Error: Paper writing failed. Aborting.")
                return
            if phase.phase_name == "coder":
                if not state.coder:
                    print("❌ Critical Error: Coder failed to produce output.")
                    return
                if not state.coder.results:
                    print("⚠️ Warning: Coder ran but produced no results. Refiner will run in 'Layout Only' mode.")
            if phase.phase_name == "refine" and not state.refiner:
                print("❌ Critical Error: Refine failed. Aborting.")
                return
                
        except Exception as e:
            print(f"❌ Pipeline Failed at {phase.phase_name}: {e}")
            import traceback
            traceback.print_exc()
            return

    # 4. 结束汇总
    print("\n" + "="*50)
    print("🎉 ALL PHASES COMPLETED (Research -> Theory -> Architect -> Paper -> Coder -> Refine)")
    print("="*50)
    print(f"Final State saved to: {state_manager.state_file}")
    
    if state.architecture:
        print(f"Blueprint Files: {len(state.architecture.file_structure)}")
    
    # [修改] 直接输出 Refiner 的最终成果
    if state.refiner and state.refiner.final_pdf_path:
        print(f"\n📄 Final PDF Generated Successfully!")
        print(f"   📍 Path: {state.refiner.final_pdf_path}")
        print(f"   📂 Latex Source: {state.refiner.latex_source_path}")
        if state.refiner.injected_data:
            print(f"   💉 Data Injected: {len(state.refiner.injected_data)} metric(s) updated in text.")
    else:
        print("\n⚠️ Pipeline finished but no final PDF was found in Refiner output.")

if __name__ == "__main__":
    my_idea = "A verifiable multi-keyword dynamic searchable encryption scheme that integrates blockchain with a Counting Bloom Filter (CBF). By designing a multi-chain radial index structure, the scheme enables two-layer parallel search under multi-keyword queries and achieves forward and backward security with minimal client storage. The blockchain stores accumulated CBF proofs, allowing any participant to publicly verify the correctness and completeness of search results without requiring any additional trusted party. The scheme supports both conjunctive (AND) and disjunctive (OR) multi-keyword queries, significantly improving dynamic update and search efficiency while maintaining strong security. This results in a fully functional DSSE system with low storage overhead and public verifiability."
    run_pipeline(my_idea)