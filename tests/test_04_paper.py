# tests/test_04_paper.py
import os
import sys

# 1. 路径设置：确保能找到 src 目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.tools.latex_compiler import latex_compiler # [新增]
from src.core.state_manager import state_manager
from src.phases.phase_04_paper import PaperPhase

def test_paper_generation():
    print("🧪 Testing Phase 4: Paper Draft Generation...")
    
    # 2. 加载状态
    state = state_manager.load_state()
    
    # 3. 前置条件检查
    # 论文写作必须依赖前三个阶段的产出
    missing_modules = []
    if not state.research: missing_modules.append("Research")
    if not state.theory: missing_modules.append("Theory")
    if not state.architecture: missing_modules.append("Architect")
    
    if missing_modules:
        print(f"❌ Error: Missing pre-requisites: {', '.join(missing_modules)}")
        print("   Please run tests for previous phases or 'main.py' first.")
        return

    print(f"-> Pre-requisites met.")
    print(f"   Idea: {state.research.refined_idea[:50]}...")
    print(f"   Architecture: {len(state.architecture.file_structure)} files planned.")

    # 4. 实例化 Phase
    phase = PaperPhase()
    
    # 5. 执行
    try:
        # execute() 包含: Load -> Check -> Run (Plan -> Write -> Save) -> Save State
        final_state = phase.execute()
        
    # [新增] 尝试编译
        if final_state.paper:
            project_dir = os.path.join("workspace", final_state.project_name, "latex")
            print(f"\n🔨 Attempting to compile PDF in: {project_dir}")
        
            success = latex_compiler.compile(project_dir, "main.tex")
        
            if success:
                print("🎉 PDF Generated Successfully!")
            # Windows 下自动打开 PDF (可选)
            # os.startfile(os.path.join(project_dir, "main.pdf"))
            else:
                print("⚠️ PDF Compilation Failed. Check logs or try manual compilation.")
            
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paper_generation()