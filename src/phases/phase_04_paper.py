# src/phases/phase_04_paper.py

import os
import yaml
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.agents.paper_writer import PaperWriterAgent
from src.core.schema import PaperDraft, SectionContent
from src.utils.logger import sys_logger

class PaperPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="paper_draft")

    def check_completion(self, state: ProjectState) -> bool:
        return state.paper is not None and state.paper.is_complete

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        # 前置检查
        if not (state.research and state.theory and state.architecture):
            raise ValueError("❌ Missing pre-requisites (Research/Theory/Architect).")

        writer = PaperWriterAgent()
        
        # 1. 生成大纲
        sys_logger.info(">>> Step 1: Generating Outline...")
        outline = writer.plan_outline(state.research, state.theory, state.architecture)
        sys_logger.info(f"Outline generated: {outline.section_names}")
        
        # 2. 逐章写作 (Sequential Writing)
        completed_sections = []
        accumulated_text = "" 
        
        for sec_name in outline.section_names:
            sys_logger.info(f">>> Step 2: Writing Section '{sec_name}'...")
            
            section_content = writer.write_section(
                section_name=sec_name,
                research=state.research,
                theory=state.theory,
                architect=state.architecture,
                previous_content=accumulated_text
            )
            
            completed_sections.append(section_content)
            accumulated_text += f"\n\n\\section{{{sec_name}}}\n{section_content.latex_content}"

        # 3. 生成 BibTeX
        bib_content = ""
        for p in state.paper_library.values():
            # 生成 Citation Key: AuthorYear (e.g., Liu2024)
            # 简单处理：取标题第一个词 + 年份，防止空格
            key = p.title.split()[0].strip() + str(p.year)
            # 清洗 key 中的非字母字符
            key = "".join([c for c in key if c.isalnum()])
            
            bib_content += f"@article{{{key},\n  title={{{p.title}}},\n  year={{{p.year}}},\n  url={{{p.url}}}\n}}\n\n"

        # 4. 组装 PaperDraft
        draft = PaperDraft(
            title=outline.title,
            abstract=outline.abstract,
            sections=completed_sections,
            bibliography_content=bib_content,
            is_complete=True
        )
        
        # 5. 保存文件
        self._save_latex_files(state.project_name, draft)
        
        # 6. 更新 State
        state.paper = draft
        return state

    def _save_latex_files(self, project_name: str, draft: PaperDraft):
        config = state_manager._load_config()
        template_name = config.get("project", {}).get("latex_template", "blank_icml_latex")
        
        src_template_dir = os.path.join("assets", "templates", "paper", template_name)
        target_dir = os.path.join("workspace", project_name, "latex")
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        import shutil
        for item in os.listdir(src_template_dir):
            s = os.path.join(src_template_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.isfile(s) and not item.endswith(".tex"):
                shutil.copy2(s, d)
        
        sys_logger.info(f"Copied style files from {template_name}")

        # 生成 body.tex
        body_path = os.path.join(target_dir, "body.tex")
        with open(body_path, "w", encoding="utf-8") as f:
            for sec in draft.sections:
                f.write(f"\\section{{{sec.section_name}}}\n")
                f.write(sec.latex_content)
                f.write("\n\n")
        
        # 生成 main.tex
        main_tex_content = self._generate_main_tex_content(draft)
        main_tex_path = os.path.join(target_dir, "main.tex")
        with open(main_tex_path, "w", encoding="utf-8") as f:
            f.write(main_tex_content)

        # 生成 references.bib
        bib_path = os.path.join(target_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(draft.bibliography_content)
            
        sys_logger.info(f"✅ LaTeX project ready at: {target_dir}")

    def _generate_main_tex_content(self, draft: PaperDraft) -> str:
        """
        基于 ICML 模板结构生成 main.tex。
        【关键修复】：所有 LaTeX 命令的大括号都变成了双大括号 {{ }}，
        只有 draft.title 等 Python 变量使用单大括号 { }。
        """
        return fr"""%%%%%%%% ICML 2025 SUBMISSION %%%%%%%%%%%%%%%%%

\documentclass{{article}}

% --- Packages ---
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{subfigure}}
\usepackage{{booktabs}} 
\usepackage{{hyperref}}
\newcommand{{\theHalgorithm}}{{\arabic{{algorithm}}}}
\usepackage{{icml2025}}

% --- Math & Theorems ---
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{mathtools}}
\usepackage{{amsthm}}
\usepackage{{multirow}}
\usepackage{{color}}
\usepackage{{colortbl}}
\usepackage[capitalize,noabbrev]{{cleveref}}

% --- Custom Definitions ---
\theoremstyle{{plain}}
\newtheorem{{theorem}}{{Theorem}}[section]
\newtheorem{{proposition}}[theorem]{{Proposition}}
\newtheorem{{lemma}}[theorem]{{Lemma}}
\newtheorem{{corollary}}[theorem]{{Corollary}}
\theoremstyle{{definition}}
\newtheorem{{definition}}[theorem]{{Definition}}
\newtheorem{{assumption}}[theorem]{{Assumption}}
\theoremstyle{{remark}}
\newtheorem{{remark}}[theorem]{{Remark}}

% --- Title & Author ---
\icmltitlerunning{{{draft.title[:50]}...}}

\begin{{document}}

\twocolumn[
\icmltitle{{{draft.title}}}

\begin{{icmlauthorlist}}
\icmlauthor{{Anonymous Authors}}{{inst1}}
\end{{icmlauthorlist}}

\icmlaffiliation{{inst1}}{{Institution Name, Location}}
\icmlcorrespondingauthor{{Anonymous}}{{email@domain.com}}
\icmlkeywords{{Machine Learning, ICML}}

\vskip 0.3in
]

\printAffiliationsAndNotice{{}}

% --- Abstract ---
\begin{{abstract}}
{draft.abstract}
\end{{abstract}}

% --- Body ---
\input{{body}}

% --- Bibliography ---
\bibliography{{references}}
\bibliographystyle{{icml2025}}

% --- Appendix ---
\newpage
\appendix
\onecolumn
\section{{Appendix}}
Additional proofs and details...

\end{{document}}
"""