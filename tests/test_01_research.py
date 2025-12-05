# tests/test_01_research.py
import os
import sys

# 确保能找到 src 目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.state_manager import state_manager
from src.phases.phase_01_research import ResearchPhase

def test_research():
    print("🧪 Testing Phase 1: Research...")
    
    # 1. 加载状态
    state = state_manager.load_state()
    
    # 2. 注入测试 Idea (如果为空)
    if not state.user_initial_idea:
        print("Injecting initial idea...")
        state.user_initial_idea = "I want to use Mamba state space models for time series forecasting on weather data, comparing it with Transformer."
        state_manager.save_state(state)

    # 3. 实例化 Phase
    phase = ResearchPhase()
    
    # 4. 执行
    try:
        final_state = phase.execute()
        
        print(f"\n✅ Research Phase Test Finished.")
        if final_state.research:
            print(f"Outcome: {len(final_state.paper_library)} papers, {len(final_state.known_gaps)} gaps.")
            print(f"Refined Idea: {final_state.refined_idea[:50]}...")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_research()