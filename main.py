import sys
import os

# 确保 Python 路径包含 src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.state_manager import state_manager
from src.phases.phase_01_research import ResearchPhase
from src.phases.phase_02_theory import TheoryPhase
from src.phases.phase_03_architect import ArchitectPhase
from src.phases.phase_04_paper import PaperPhase
# [新增] 引入编译工具
from src.tools.latex_compiler import latex_compiler 

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
        PaperPhase()
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
                
        except Exception as e:
            print(f"❌ Pipeline Failed at {phase.phase_name}: {e}")
            import traceback
            traceback.print_exc()
            return

    # 4. 结束汇总 & 编译 PDF
    print("\n" + "="*50)
    print("🎉 ALL PHASES COMPLETED (Research -> Theory -> Architect -> Paper)")
    print("="*50)
    print(f"Final State saved to: {state_manager.state_file}")
    
    if state.architecture:
        print(f"Blueprint Files: {len(state.architecture.file_structure)}")
    
    if state.paper:
        latex_dir = os.path.join("workspace", state.project_name, "latex")
        print(f"\n📄 Paper Draft Generated at: {latex_dir}")
        
        # [新增] 自动编译逻辑
        print(f"🔨 Compiling PDF...")
        try:
            # 尝试编译 main.tex
            success = latex_compiler.compile(latex_dir, "main.tex")
            
            if success:
                pdf_path = os.path.join(latex_dir, "main.pdf")
                print(f"✅ PDF Generated Successfully: {pdf_path}")
            else:
                print("⚠️ PDF Compilation Failed.")
                print(f"   Debug Hint: Check 'logs/system.log' or run 'pdflatex main.tex' manually in {latex_dir}")
                
        except Exception as e:
            print(f"❌ Compiler Error: {e}")

if __name__ == "__main__":
    my_idea = "A verifiable multi-keyword dynamic searchable encryption scheme that integrates blockchain with a Counting Bloom Filter (CBF). By designing a multi-chain radial index structure, the scheme enables two-layer parallel search under multi-keyword queries and achieves forward and backward security with minimal client storage. The blockchain stores accumulated CBF proofs, allowing any participant to publicly verify the correctness and completeness of search results without requiring any additional trusted party. The scheme supports both conjunctive (AND) and disjunctive (OR) multi-keyword queries, significantly improving dynamic update and search efficiency while maintaining strong security. This results in a fully functional DSSE system with low storage overhead and public verifiability."
    run_pipeline(my_idea)