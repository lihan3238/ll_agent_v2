# src/tools/latex_compiler.py
import os
import subprocess
from src.utils.logger import sys_logger

class LatexCompiler:
    def compile(self, project_dir: str, main_filename: str = "main.tex"):
        """
        运行 pdflatex -> bibtex -> pdflatex -> pdflatex 完整流程
        """
        if not os.path.exists(os.path.join(project_dir, main_filename)):
            sys_logger.error(f"Main tex file not found: {main_filename}")
            return False

        sys_logger.info(f"Compiling PDF in {project_dir}...")
        
        # 基础命令
        cmd_pdf = ["pdflatex", "-interaction=nonstopmode", main_filename]
        cmd_bib = ["bibtex", main_filename.replace(".tex", "")]
        
        try:
            # Round 1: PDF
            self._run_command(cmd_pdf, project_dir)
            
            # Round 2: BibTeX
            self._run_command(cmd_bib, project_dir)
            
            # Round 3: PDF (Link)
            self._run_command(cmd_pdf, project_dir)
            
            # Round 4: PDF (Final)
            self._run_command(cmd_pdf, project_dir)
            
            pdf_path = os.path.join(project_dir, main_filename.replace(".tex", ".pdf"))
            if os.path.exists(pdf_path):
                sys_logger.info(f"✅ PDF Compilation Successful: {pdf_path}")
                return True
            else:
                sys_logger.error("❌ PDF file not generated.")
                return False
                
        except Exception as e:
            sys_logger.error(f"Compilation failed: {e}")
            return False

    def _run_command(self, cmd, cwd):
        """Helper to run subprocess"""
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60 # 防止卡死
        )
        if result.returncode != 0:
            # 只在出错时打印 Log，防止刷屏
            # sys_logger.warning(f"LaTeX Warning/Error: {result.stdout.decode('latin-1')[-500:]}")
            pass

latex_compiler = LatexCompiler()