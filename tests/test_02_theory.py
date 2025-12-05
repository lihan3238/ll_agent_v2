# tests/test_02_theory.py
import os
import sys

# 路径 Hack，确保能找到 src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.state_manager import state_manager
from src.phases.phase_02_theory import TheoryPhase

def test_theory():
    print("🧪 Testing Phase 2: Theory...")
    
    # 1. 加载 State
    state = state_manager.load_state()
    
    # 2. 检查前置条件
    if not state.research:
        print("❌ Error: 'research' data missing in state.")
        print("   Please run 'tests/test_01_research.py' first!")
        return

    print(f"-> Pre-requisite met. Idea: {state.research.refined_idea[:50]}...")

    # 3. 运行 Phase
    phase = TheoryPhase()
    
    try:
        final_state = phase.execute()
        print(f"\n✅ Theory Phase Finished.")
        if final_state.theory:
            print(f"Field: {final_state.theory.research_field}")
            print(f"Innovations: {len(final_state.theory.key_innovations)}")
            
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_theory()