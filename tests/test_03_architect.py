# tests/test_03_architect.py
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.state_manager import state_manager
from src.phases.phase_03_architect import ArchitectPhase

def test_architect():
    print("🧪 Testing Phase 3: Architect...")
    
    # 1. 加载 State
    state = state_manager.load_state()
    
    # 2. 检查前置条件
    if not state.theory:
        print("❌ Error: 'theory' data missing in state.")
        print("   Please run 'tests/test_02_theory.py' first!")
        return

    print(f"-> Pre-requisite met. Theory Field: {state.theory.research_field}")

    # 3. 运行 Phase
    phase = ArchitectPhase()
    
    try:
        final_state = phase.execute()
        print(f"\n✅ Architect Phase Finished.")
        if final_state.architecture:
            print(f"Project Name: {final_state.architecture.project_name}")
            print(f"File Count: {len(final_state.architecture.file_structure)}")
            # 打印第一个文件的核心逻辑，检查 Deep Architect 是否生效
            first_file = final_state.architecture.file_structure[0]
            if first_file.classes:
                first_method = first_file.classes[0].methods[0]
                print(f"Sample Logic ({first_method.name}): {first_method.core_logic_steps[:2]}...")
            
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_architect()