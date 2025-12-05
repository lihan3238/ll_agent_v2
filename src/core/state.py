# src/core/state.py
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.core.schema import (
    ResearchReport, TheoreticalFramework, DesignDocument,
    PaperInfo, TechnicalGap
)

class ProjectState(BaseModel):
    project_name: str
    current_phase: str = "research" 
    
    user_initial_idea: str = Field(default="")
    refined_idea: str = Field(default="")
    
    paper_library: Dict[str, PaperInfo] = Field(default_factory=dict)
    known_gaps: List[TechnicalGap] = Field(default_factory=list)
    
    research: Optional[ResearchReport] = None
    theory: Optional[TheoreticalFramework] = None
    architecture: Optional[DesignDocument] = None
    
    def merge_papers(self, new_papers: List[PaperInfo]):
        for p in new_papers:
            # 归一化 Title 做 Key，防止大小写差异
            key = p.title.lower().strip()
            # 这里我们还是存原始 title 方便展示，但判断用归一化 key
            # 为了简单，直接用 title 覆盖（新来的可能引用数更新了）
            self.paper_library[p.title] = p

    def merge_gaps(self, new_gaps: List[TechnicalGap]):
        """
        增强去重逻辑：
        防止 Research 阶段每一轮都把 'Transformer fails long range' 加一遍。
        """
        # 生成现有指纹集合
        # 指纹 = "方法名(小写)|原因前20字符(小写)"
        existing_sigs = set()
        for g in self.known_gaps:
            sig = f"{g.existing_method.lower().strip()}|{g.mathematical_root_cause.lower().strip()[:20]}"
            existing_sigs.add(sig)
        
        for g in new_gaps:
            sig = f"{g.existing_method.lower().strip()}|{g.mathematical_root_cause.lower().strip()[:20]}"
            
            if sig not in existing_sigs:
                self.known_gaps.append(g)
                existing_sigs.add(sig)