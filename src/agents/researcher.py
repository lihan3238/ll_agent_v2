# src/agents/researcher.py
from typing import List, Dict, Optional
from src.agents.base import BaseAgent
from src.core.schema import ResearchReport, SearchQueries, PaperInfo, TechnicalGap
from src.tools.scholarly import s2_tool
from src.utils.logger import sys_logger

class ResearcherAgent(BaseAgent):
    def __init__(self, existing_papers: Dict[str, PaperInfo] = None, existing_gaps: List[TechnicalGap] = None):
        """
        :param existing_papers: 从 ProjectState 传入的已知论文
        :param existing_gaps: 从 ProjectState 传入的已知痛点
        """
        super().__init__(role_name="researcher")
        
        # 配置
        self.query_count = self.config.get("query_count", 3)
        self.search_limit = self.config.get("search_limit_per_query", 5)
        self.max_papers_context = self.config.get("max_context_papers", 10) 
        
        # 初始化记忆 (继承自全局状态)
        self.paper_memory = existing_papers.copy() if existing_papers else {}
        self.gap_memory = existing_gaps.copy() if existing_gaps else []
        self.current_refined_idea = ""

    def run(self, user_idea: str) -> ResearchReport:
        # ... (这里的逻辑保持不变，复制你之前发给我的 run 方法即可) ...
        # ... 唯一区别是它现在操作的是 self.paper_memory，
        # ... 这个 memory 初始包含了历史所有论文。
        sys_logger.info(f"Task Started. Knowledge Base size: {len(self.paper_memory)} papers.")
        
        if not self.current_refined_idea:
            self.current_refined_idea = user_idea

        # Step 1: Expand
        query_result = self.call_llm_with_struct(
            prompt_template=self.prompts["query_gen_prompt"],
            schema=SearchQueries,
            user_idea=self.current_refined_idea,
            query_count=self.query_count
        )
        sys_logger.info(f"🔍 Queries: {query_result.queries}")

        new_count = 0
        for q in query_result.queries:
            sys_logger.info(f"API -> '{q}'")
            found_dicts = s2_tool.search(q, limit=self.search_limit)
            for p_data in found_dicts:
                p_obj = PaperInfo(
                    title=p_data['title'], year=str(p_data['year']), citations=p_data['citations'],
                    summary=p_data['abstract'], url=p_data['url']
                )
                if p_obj.title not in self.paper_memory:
                    self.paper_memory[p_obj.title] = p_obj
                    new_count += 1
        
        sys_logger.info(f"📚 Indexed {new_count} new papers. Total: {len(self.paper_memory)}")

        # Step 2: Select
        all_papers = list(self.paper_memory.values())
        top_context_papers = sorted(all_papers, key=lambda x: x.citations, reverse=True)[:self.max_papers_context]
        
        context_str = ""
        for i, p in enumerate(top_context_papers):
            context_str += f"[{i+1}] {p.title} ({p.year}) | Citations: {p.citations}\nAbstract: {p.summary[:300]}...\n\n"

        # Step 3: Synthesize
        llm_report = self.call_llm_with_struct(
            prompt_template=self.prompts["report_gen_prompt"],
            schema=ResearchReport,
            user_idea=user_idea,
            search_results_context=context_str
        )

        # Step 4: Update Internal State
        self.current_refined_idea = llm_report.refined_idea
        
        for new_gap in llm_report.gap_analysis:
            # 简单去重
            is_dup = any(g.existing_method == new_gap.existing_method and g.limitation_description == new_gap.limitation_description for g in self.gap_memory)
            if not is_dup:
                self.gap_memory.append(new_gap)

        return ResearchReport(
            refined_idea=self.current_refined_idea,
            keywords=llm_report.keywords,
            gap_analysis=self.gap_memory, # 返回全量痛点
            related_work_summary=llm_report.related_work_summary,
            top_papers=top_context_papers,
            implementation_suggestions=llm_report.implementation_suggestions
        )