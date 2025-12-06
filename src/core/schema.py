# src/core/schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

# --- 基础枚举 ---
class ActionType(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"

# --- Phase 1: Researcher Outputs ---
class PaperInfo(BaseModel):
    title: str
    year: str
    citations: int
    summary: str = Field(..., description="One sentence summary.")
    url: str

# [新增] 技术痛点结构
class TechnicalGap(BaseModel):
    existing_method: str = Field(..., description="The name of the existing SOTA method or protocol.")
    limitation_description: str = Field(..., description="Specifically what this method fails to do.")
    mathematical_root_cause: str = Field(..., description="The theoretical reason for this failure (e.g., 'Softmax bottleneck', 'Gradient vanishing').")

class ResearchReport(BaseModel):
    refined_idea: str
    keywords: List[str]
    # [新增] 必须包含 Gap Analysis
    gap_analysis: List[TechnicalGap] = Field(..., description="Detailed analysis of why current methods fail.")
    
    related_work_summary: str
    top_papers: List[PaperInfo]
    implementation_suggestions: str

class SearchQueries(BaseModel):
    queries: List[str]

# --- Phase 2: Theorist Outputs ---
class TheoreticalFramework(BaseModel):
    research_field: str
    problem_formulation: str
    proposed_methodology: str
    theoretical_analysis: str
    key_innovations: List[str]

# --- User Interaction ---
class UserFeedback(BaseModel):
    action: ActionType
    feedback_en: str
    comments: str

# --- [新增] Phase 3: Architect Outputs ---
class MethodSpec(BaseModel):
    name: str = Field(..., description="Method name, e.g., 'forward'")
    args: List[str] = Field(..., description="Arguments with type hints, e.g., ['x: torch.Tensor', 'state: Optional[Tensor] = None']")
    return_type: str = Field(..., description="Return type, e.g., 'torch.Tensor'")
    docstring: str = Field(..., description="Brief explanation of what this method does.")
    # [核心新增]：核心逻辑描述，指导 Coder 写代码体
    core_logic_steps: List[str] = Field(..., description="Step-by-step logic or pseudo-code. CRITICAL for complex math.")

class ClassSpec(BaseModel):
    name: str = Field(..., description="Class name, e.g., 'MambaBlock'")
    inherits_from: str = Field(default="nn.Module", description="Parent class.")
    description: str = Field(..., description="High-level purpose.")
    attributes: List[str] = Field(..., description="Class attributes/fields to initialize in __init__.")
    methods: List[MethodSpec] = Field(..., description="List of methods.")

class FileSpec(BaseModel):
    filename: str = Field(..., description="Path/Name, e.g., 'src/models/mamba.py'")
    description: str = Field(..., description="Purpose of this file.")
    imports: List[str] = Field(..., description="Key libraries to import.")
    classes: List[ClassSpec] = Field(default=[], description="Classes to define.")
    functions: List[MethodSpec] = Field(default=[], description="Global functions.")

class DesignDocument(BaseModel):
    project_name: str
    architecture_style: str
    requirements: List[str]
    file_structure: List[FileSpec]
    
    # [新增] 数据流与张量形状约定
    data_flow_diagram: str = Field(..., description="Text-based diagram describing how data moves (e.g., Data -> [B, L, D] -> Model).")
    
    hyperparameters: Dict[str, str]
    main_execution_flow: str

# --- Phase 4: Paper Writer Outputs ---

class SectionContent(BaseModel):
    section_name: str = Field(..., description="e.g., 'Introduction', 'Methodology'")
    latex_content: str = Field(..., description="Raw LaTeX content for this section. NO preamble, NO \\begin{document}.")
    word_count: int = Field(..., description="Approximate word count.")

class PaperOutline(BaseModel):
    """先规划，后写作"""
    title: str
    abstract: str
    section_names: List[str] = Field(..., description="Ordered list of section titles to be written.")

class PaperDraft(BaseModel):
    # [新增] 显式保存大纲结构
    outline: PaperOutline 
    
    title: str
    abstract: str
    sections: List[SectionContent] = Field(..., description="Ordered list of completed sections.")
    bibliography_content: str = Field(..., description="Content of the .bib file.")
    
    # 状态标记，方便后续 Refine    
    is_complete: bool = False

# reviewer

class ReviewDecision(str, Enum):
    ACCEPT = "APPROVE" # 对应 ActionType.APPROVE
    REJECT = "REVISE"  # 对应 ActionType.REVISE

class ReviewReport(BaseModel):
    decision: ReviewDecision = Field(..., description="The final decision: APPROVE for next phase, or REVISE for improvements.")
    score: int = Field(..., description="Quality score from 1-10.")
    critique: str = Field(..., description="Detailed critique of the work, highlighting weaknesses.")
    specific_instructions: str = Field(..., description="If REJECT, provide specific actionable instructions for the next iteration.")