# src/phases/phase_06_refine.py
import os
import shutil
import json
import re
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.core.schema import RefinerOutput
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
        
        sections_to_update = ["Abstract", "Experiments", "Conclusion", "Introduction"]
        
        # 更新 Abstract
        new_abstract = refiner.inject_data(state.paper.abstract, "Abstract", metrics)
        state.paper.abstract = new_abstract
        
        # 更新正文 Sections
        new_sections = []
        for sec in state.paper.sections:
            if any(k.lower() in sec.section_name.lower() for k in sections_to_update):
                updated_text = refiner.inject_data(sec.latex_content, sec.section_name, metrics)
                sec.latex_content = updated_text
            new_sections.append(sec)
        state.paper.sections = new_sections
        
        # 写入文件
        self._rewrite_latex_files(latex_dir, state.paper)

        # ==========================================
        # Step 3: 编译与智能修复 (Smart Compile & Fix)
        # ==========================================
        sys_logger.info(">>> [Phase 6.3] Compiling PDF...")
        
        max_compile_retries = 5 # 给多几次机会修复语法错误
        pdf_path = ""
        compile_log = ""
        
        for i in range(max_compile_retries):
            success = latex_compiler.compile(latex_dir, "main.tex")
            
            if success:
                pdf_path = os.path.join(latex_dir, "main.pdf")
                sys_logger.info(f"✅ PDF Generated: {pdf_path}")
                break
            
            # --- 修复逻辑 ---
            sys_logger.warning(f"⚠️ Compilation failed (Attempt {i+1}/{max_compile_retries}). Analyzing log...")
            
            log_file = os.path.join(latex_dir, "main.log")
            if not os.path.exists(log_file):
                sys_logger.error("No log file found.")
                break
                
            with open(log_file, "r", encoding="latin-1", errors="ignore") as f:
                compile_log = f.read()
            
            # [关键] 智能判断出错文件
            # LaTeX log 通常会有 "./body.tex:12: ..." 这样的提示
            target_file = "body.tex" # 默认修 body
            if "main.tex" in compile_log[-2000:] and "body.tex" not in compile_log[-2000:]:
                target_file = "main.tex"
            
            # 某些特定的错误可能需要检查 main.tex (如 \begin{document} 缺失)
            if "! LaTeX Error: Missing \\begin{document}" in compile_log:
                target_file = "main.tex"

            sys_logger.info(f"🔍 Detected error likely in: {target_file}")
            
            # 读取目标文件
            file_path = os.path.join(latex_dir, target_file)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 调用 Agent 修复
                fixed_content = refiner.fix_latex(target_file, content, compile_log)
                
                # 写回文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                
                sys_logger.info(f"🔧 Applied fix to {target_file}")
            else:
                sys_logger.error(f"Target file {target_file} not found!")

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
        if not os.path.exists(src_dir):
            sys_logger.warning(f"Source code dir {src_dir} does not exist.")
            return

        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.csv', '.tex')):
                    # 排除 LaTeX 源码本身，只找 Coder 生成的
                    if "main.tex" in file or "body.tex" in file: continue
                    
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dir, file)
                    
                    shutil.copy2(src_path, dst_path)
                    sys_logger.info(f"   -> Copied artifact: {file}")
                    count += 1
        if count == 0:
            sys_logger.warning("No artifacts found in code directory!")

    def _rewrite_latex_files(self, target_dir, draft):
        """重写 LaTeX 文件"""
        # Rewrite body.tex
        body_path = os.path.join(target_dir, "body.tex")
        with open(body_path, "w", encoding="utf-8") as f:
            for sec in draft.sections:
                # 简单清洗：防止 LLM 重复写 \section
                content = sec.latex_content
                if content.strip().startswith(f"\\section{{{sec.section_name}}}"):
                    content = content.replace(f"\\section{{{sec.section_name}}}", "", 1)
                
                f.write(f"\\section{{{sec.section_name}}}\n")
                f.write(content)
                f.write("\n\n")
        
        # Rewrite main.tex (Abstract Update)
        main_path = os.path.join(target_dir, "main.tex")
        if os.path.exists(main_path):
            with open(main_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 使用正则替换 Abstract
            # 匹配 \begin{abstract} ... \end{abstract} 之间的内容
            pattern = r"(\\begin\{abstract\})(.*?)(\\end\{abstract\})"
            replacement = f"\\1\n{draft.abstract}\n\\3"
            
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(new_content)