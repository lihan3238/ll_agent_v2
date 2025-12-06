# tests/test_05_coder.py
import os
import sys
import json

# 1. 路径设置：确保能找到 src 目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.state_manager import state_manager
from src.phases.phase_05_coder import CoderPhase
from src.core.schema import ExecutionStatus

def test_coder():
    print("🧪 Testing Phase 5: AI Coder & Experiment Runner...")
    
    # 2. 加载状态
    state = state_manager.load_state()
    
    # 2. 前置条件检查 (更新版)
    missing = []
    #if not state.architecture: missing.append("Architect (Design)")
    if not state.paper: missing.append("Paper (Draft)") # [新增检查]
    if missing:
        print(f"❌ Error: Missing pre-requisites: {', '.join(missing)}")
        print("   Please run previous phases first.")
        return

    print(f"-> Pre-requisite met.")
    print(f"   Project: {state.project_name}")
    print(f"   Architecture Style: {state.architecture.architecture_style}")
    print(f"   Planned Files: {len(state.architecture.file_structure)}")

    # 4. 实例化 Phase
    phase = CoderPhase()
    
    # 5. 执行
    try:
        # 注意：这会触发 Conda 环境创建和代码运行，可能需要几分钟
        print("\n⏳ Starting Coder Phase (This may take time due to Conda setup)...")
        final_state = phase.execute()
        
        print(f"\n✅ Coder Phase Finished.")
        
        if final_state.coder:
            # 打印环境信息
            print(f"\n🌍 Environment: pf_{state.project_name}")
            
            # 打印执行日志摘要
            logs = final_state.coder.execution_log
            print(f"📝 Execution Attempts: {len(logs)}")
            if logs:
                last_log = logs[-1]
                print(f"   Last Command: {last_log.command}")
                print(f"   Return Code: {last_log.return_code}")
                if last_log.return_code != 0:
                    print(f"   ⚠️ Error Tail:\n{last_log.stderr[-300:]}")

            # 打印最终结果
            if final_state.coder.results:
                print("\n🏆 EXPERIMENT RESULTS:")
                print(json.dumps(final_state.coder.results.metrics, indent=2))
                
                if final_state.coder.results.status == ExecutionStatus.SUCCESS:
                    print("\n🎉 SUCCESS! The code runs and produced metrics.")
                else:
                    print("\n⚠️ Code ran but status is marked as FAILED.")
            else:
                print("\n❌ No results generated. Auto-debugging might have failed max retries.")
            
            # 提示产物位置
            code_dir = os.path.join("workspace", state.project_name, "code")
            print(f"\n📂 Codebase Location: {code_dir}")
            
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coder()