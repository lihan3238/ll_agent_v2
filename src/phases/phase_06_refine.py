# src/phases/phase_06_refine.py
import os
import shutil
import json
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.core.schema import RefinerOutput, SectionContent
from src.agents.refiner import RefinerAgent
from src.tools.latex_compiler import latex_compiler
from src.utils.logger import sys_logger

class RefinePhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="refine")

    def check_completion(self, state: ProjectState) -> bool:
        return state.refiner is not None and os.path.exists(state.refiner.final_pdf_path)

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        if not state.coder or not state.coder.results:
            sys_logger.warning("⚠️ Coder results missing. Refiner will run in 'Layout Only' mode.")
        
        refiner = RefinerAgent()
        
        # 1. 准备目录
        workspace = os.path.join("workspace", state.project_name)
        code_dir = os.path.join(workspace, "code")
        latex_dir = os.path.join(workspace, "latex")
        figures_dir = os.path.join(latex_dir, "figures")
        
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        # ==========================================
        # Step 1: 资产同步 (Asset Sync)
        # ==========================================
        sys_logger.info(">>> [Phase 6.1] Synchronizing Artifacts...")
        self._sync_artifacts(code_dir, figures_dir)

        # ==========================================
        # Step 2: 数据注入 (Data Injection)
        # ==========================================
        sys_logger.info(">>> [Phase 6.2] Injecting Experimental Data...")
        metrics = state.coder.results.metrics if state.coder and state.coder.results else {}
        
        # 我们只更新 Abstract, Conclusion, Experiments 三个部分
        # 其他部分通常是理论，不需要动
        sections_to_update = ["Abstract", "Experiments", "Conclusion", "Introduction"]
        
        # 更新 Abstract (单独处理，因为它不在 sections 列表里)
        new_abstract = refiner.inject_data(state.paper.abstract, "Abstract", metrics)
        state.paper.abstract = new_abstract
        
        # 更新正文 Sections
        new_sections = []
        for sec in state.paper.sections:
            if any(k in sec.section_name for k in sections_to_update):
                updated_text = refiner.inject_data(sec.latex_content, sec.section_name, metrics)
                # 更新对象
                sec.latex_content = updated_text
            new_sections.append(sec)
        state.paper.sections = new_sections
        
        # 重新生成 main.tex (因为内容变了)
        # 这里的逻辑其实有点 tricky，我们需要调用 PaperPhase 的私有方法，或者在这里重写
        # 简单起见，我们手动重写 main.tex
        self._rewrite_latex_files(latex_dir, state.paper)

        # ==========================================
        # Step 3: 编译与修复 (Compile & Fix)
        # ==========================================
        sys_logger.info(">>> [Phase 6.3] Compiling PDF...")
        
        max_compile_retries = 3
        pdf_path = ""
        compile_log = ""
        
        for i in range(max_compile_retries):
            success = latex_compiler.compile(latex_dir, "main.tex")
            
            if success:
                pdf_path = os.path.join(latex_dir, "main.pdf")
                sys_logger.info(f"✅ PDF Generated: {pdf_path}")
                break
            else:
                sys_logger.warning(f"⚠️ Compilation failed (Attempt {i+1}). Analyzing log...")
                # 读取 log 文件
                log_file = os.path.join(latex_dir, "main.log")
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="latin-1") as f:
                        compile_log = f.read()
                        
                    # 这里的修复逻辑比较复杂，通常是修复 main.tex 或 body.tex
                    # 简单起见，我们只尝试修复 body.tex (大部分内容在这里)
                    body_path = os.path.join(latex_dir, "body.tex")
                    with open(body_path, "r", encoding="utf-8") as f:
                        body_content = f.read()
                        
                    fixed_body = refiner.fix_latex("body.tex", body_content, compile_log)
                    
                    with open(body_path, "w", encoding="utf-8") as f:
                        f.write(fixed_body)
                else:
                    sys_logger.error("No log file found.")

        # Finalize
        state.refiner = RefinerOutput(
            final_pdf_path=pdf_path,
            latex_source_path=latex_dir,
            compilation_log=compile_log[-2000:],
            injected_data=metrics
        )
        return state

    def _sync_artifacts(self, src_dir, dst_dir):
        """递归查找并移动图片"""
        count = 0
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.csv', '.tex')):
                    # 排除 LaTeX 源码本身，只找 Coder 生成的
                    if "main.tex" in file or "body.tex" in file: continue
                    
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dir, file) # 扁平化放入 figures/
                    
                    shutil.copy2(src_path, dst_path)
                    sys_logger.info(f"   -> Copied artifact: {file}")
                    count += 1
        if count == 0:
            sys_logger.warning("No artifacts found in code directory!")

    def _rewrite_latex_files(self, target_dir, draft):
        """重写 LaTeX 文件 (复用 PaperPhase 的逻辑，这里简化实现)"""
        # Rewrite body.tex
        body_path = os.path.join(target_dir, "body.tex")
        with open(body_path, "w", encoding="utf-8") as f:
            for sec in draft.sections:
                f.write(f"\\section{{{sec.section_name}}}\n")
                f.write(sec.latex_content)
                f.write("\n\n")
        
        # Rewrite main.tex (Header/Abstract might change)
        # 这里假设 main.tex 模板比较固定，主要变动在 Abstract
        # 为了稳健，我们可以读取旧 main.tex，用正则替换 Abstract
        # 或者直接覆盖（如果 PaperWriter 生成了完整的 main 结构）
        
        # 简单策略：重新读取模板并填空
        # (这里为了省事，假设 main.tex 主体不变，只改了 body.tex)
        # 如果 Abstract 变了，我们需要更新 main.tex
        main_path = os.path.join(target_dir, "main.tex")
        if os.path.exists(main_path):
            with open(main_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 极其简陋的替换，实际建议用 Jinja2 模板重绘
            import re
            # 替换 Abstract
            pattern = r"\\begin\{abstract\}(.*?)\\end\{abstract\}"
            replacement = f"\\\\begin{{abstract}}\n{draft.abstract}\n\\\\end{{abstract}}"
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(new_content)