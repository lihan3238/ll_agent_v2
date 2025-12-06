# src/phases/phase_04_paper.py

import os
from src.core.lifecycle import BasePhase
from src.core.state import ProjectState
from src.core.state_manager import state_manager
from src.core.interaction import interactor
from src.core.schema import ActionType, PaperDraft, SectionContent
from src.agents.paper_writer import PaperWriterAgent
from src.utils.logger import sys_logger
from src.tools.latex_compiler import latex_compiler # [新增] 用于每轮编译

class PaperPhase(BasePhase):
    def __init__(self):
        super().__init__(phase_name="paper_draft")

    def check_completion(self, state: ProjectState) -> bool:
        # 如果已经有 paper 且 is_complete，视为完成
        # 但如果是断点续传，我们可能想允许它是 draft 状态。这里暂定严格检查。
        return state.paper is not None and state.paper.is_complete

    def run_phase_logic(self, state: ProjectState) -> ProjectState:
        if not (state.research and state.theory and state.architecture):
            raise ValueError("❌ Missing pre-requisites.")

        writer = PaperWriterAgent()
        
        # 读取配置
        config = state_manager._load_config()
        rounds = config.get("workflow", {}).get("paper_rounds", 2)
        
        # Step 0: Pre-calculate Keys
        sys_logger.info(">>> Step 0: Pre-calculating Citation Keys...")
        bib_entries = []
        citation_map_str = "Available Papers for Citation:\n"
        for p in state.paper_library.values():
            first_word = "".join(filter(str.isalpha, p.title.split()[0]))
            key = f"{first_word}{p.year}"
            entry = f"@article{{{key},\n  title={{{p.title}}},\n  author={{{' and '.join(p.title.split()[:2])}}},\n  year={{{p.year}}},\n  url={{{p.url}}}\n}}"
            bib_entries.append(entry)
            citation_map_str += f"- Key: \\cite{{{key}}} | Title: {p.title} ({p.year})\n"
        full_bib_content = "\n\n".join(bib_entries)

        # 状态变量
        current_draft = state.paper # 如果有旧草稿，加载
        current_feedback = ""

        # 记录大纲，如果是断点恢复，从 draft 中取
        current_outline = current_draft.outline if current_draft else None
        
        for r in range(rounds):
            sys_logger.info(f"\n>>> 🛡️ Paper Writing Cycle {r+1}/{rounds} <<<")
            
            # A. 生成/修改内容
            if r == 0 and not current_draft:
                sys_logger.info("Drafting from scratch...")
                
                # 1. 生成大纲
                current_outline = writer.plan_outline(state.research, state.theory, state.architecture)
                
                new_sections = []
                accumulated_text = ""
                for sec_name in current_outline.section_names:
                    sec_content = writer.write_section(
                        section_name=sec_name,
                        research=state.research,
                        theory=state.theory,
                        architect=state.architecture,
                        previous_content=accumulated_text,
                        references_context=citation_map_str
                    )
                    new_sections.append(sec_content)
                    accumulated_text += f"\n\n{sec_content.latex_content}"
                
                current_draft = PaperDraft(
                    outline=current_outline, # [核心新增] 保存大纲
                    title=current_outline.title,
                    abstract=current_outline.abstract,
                    sections=new_sections,
                    bibliography_content=full_bib_content
                )
            
            else:
                # 后续轮次：基于 Feedback 修改
                sys_logger.info(f"Refining draft based on feedback: {current_feedback[:50]}...")
                updated_sections = []
                accumulated_text = ""
                
                for old_sec in current_draft.sections:
                    # 只有当 feedback 明确提到某个部分，或者我们可以让 LLM 自行判断是否需要修改
                    # 简单起见，我们把 feedback 传给每一章，让 LLM 决定是否重写
                    # (或者你可以设计更复杂的逻辑，只重写特定章节)
                    
                    new_sec = writer.write_section(
                        section_name=old_sec.section_name,
                        research=state.research,
                        theory=state.theory,
                        architect=state.architecture,
                        previous_content=accumulated_text,
                        references_context=citation_map_str,
                        existing_text=old_sec.latex_content, # 传入旧文本
                        feedback=current_feedback            # 传入反馈
                    )
                    updated_sections.append(new_sec)
                    accumulated_text += f"\n\n{new_sec.latex_content}"
                
                # 更新 Draft 对象
                current_draft.sections = updated_sections

            # B. 保存并编译
            self._save_latex_files(state.project_name, current_draft)
            
            # [放弃] 尝试编译以供 Reviewer 检查 (可选，Reviewer 主要看 MD/Text)
            # 但编译能暴露 LaTeX 语法错误
            # latex_dir = os.path.join("workspace", state.project_name, "latex")
            # compile_success = latex_compiler.compile(latex_dir, "main.tex")
            
            # C. 评审 (Interaction)
            # 我们把 Draft 转为文本给 Reviewer 看，或者只给 Abstract + Intro + Method
            # 这里简单处理：把全文章节拼接给 Reviewer
            full_text_for_review = f"Title: {current_draft.title}\nAbstract: {current_draft.abstract}\n\n"
            for sec in current_draft.sections:
                full_text_for_review += f"## {sec.section_name}\n{sec.latex_content}\n\n"
            
            # # 如果编译失败，把错误信息也喂给 Reviewer
            # if not compile_success:
            #     full_text_for_review += "\n\n[SYSTEM WARNING]: The LaTeX failed to compile. Please check for syntax errors."

            user_feedback = interactor.start_review(
                phase_name=f"04_Paper_Round_{r+1}",
                template_name="paper_review.md.j2", # 需要新建这个模板
                context_data={
                    "outline": current_outline, 
                    "draft_text": full_text_for_review,
                    "draft_obj": current_draft # 传对象给 Reviewer Agent 备用
                },
                iteration_idx=r
            )
            
            if user_feedback.action == ActionType.APPROVE:
                sys_logger.info("✅ Paper Draft Approved.")
                current_draft.is_complete = True
                state.paper = current_draft
                return state
                
            elif user_feedback.action == ActionType.REVISE:
                sys_logger.info(f"🔄 Revision Requested: {user_feedback.feedback_en}")
                current_feedback = user_feedback.feedback_en

        # End Loop
        if current_draft:
             sys_logger.warning("⚠️ Max paper rounds reached. Saving latest draft.")
             state.paper = current_draft
             
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