# src/core/schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
from enum import Enum

# --- 基础枚举 ---
class ActionType(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"

# --- Phase 1: Researcher Outputs ---
class PaperInfo(BaseModel):
    title: str = Field(default="Unknown Title")
    year: str = Field(default="N/A")
    citations: int = Field(default=0)
    summary: str = Field(default="No summary provided.", description="One sentence summary.")
    url: str = Field(default="")

class TechnicalGap(BaseModel):
    existing_method: str = Field(default="Unknown Method", description="The name of the existing SOTA method or protocol.")
    limitation_description: str = Field(default="Limitation details missing.", description="Specifically what this method fails to do.")
    mathematical_root_cause: str = Field(default="Root cause not specified.", description="The theoretical reason for this failure.")

class ResearchReport(BaseModel):
    refined_idea: str = Field(default="")
    keywords: List[str] = Field(default_factory=list)
    gap_analysis: List[TechnicalGap] = Field(default_factory=list, description="Detailed analysis of why current methods fail.")
    related_work_summary: str = Field(default="")
    top_papers: List[PaperInfo] = Field(default_factory=list)
    implementation_suggestions: str = Field(default="")

class SearchQueries(BaseModel):
    queries: List[str] = Field(default_factory=list)

# --- Phase 2: Theorist Outputs ---
class TheoreticalFramework(BaseModel):
    research_field: str = Field(default="Unspecified Field")
    problem_formulation: str = Field(default="Pending formulation...")
    proposed_methodology: str = Field(default="Pending methodology...")
    theoretical_analysis: str = Field(default="Pending analysis...")
    key_innovations: List[str] = Field(default_factory=list)

# --- User Interaction ---
class UserFeedback(BaseModel):
    action: ActionType
    feedback_en: str = Field(default="")
    comments: str = Field(default="")

# --- Phase 3: Architect Outputs (关键修改点) ---
class MethodSpec(BaseModel):
    name: str = Field(default="unknown_method", description="Method name, e.g., 'forward'")
    args: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="Arguments with type hints.")
    return_type: str = Field(default="Any", description="Return type.")
    docstring: str = Field(default="", description="Brief explanation.")
    # [修复] 增加默认值，防止 LLM 漏写或截断
    core_logic_steps: List[str] = Field(
        default_factory=list, 
        description="High-level logic steps. E.g., '1. Load data from X. 2. Init model Y. 3. Train loop...'"
    )

class ClassSpec(BaseModel):
    name: str = Field(default="UnknownClass", description="Class name.")
    inherits_from: str = Field(default="object", description="Parent class.")
    description: str = Field(default="", description="High-level purpose.")
    attributes: List[str] = Field(default_factory=list, description="Class attributes.")
    methods: List[MethodSpec] = Field(default_factory=list, description="List of methods.")

class FileSpec(BaseModel):
    filename: str = Field(..., description="Path/Name, e.g., 'src/models/mamba.py'") # filename 最好必填，否则没法写文件
    description: str = Field(default="", description="Purpose of this file.")
    imports: List[str] = Field(default_factory=list, description="Key libraries to import.")
    classes: List[ClassSpec] = Field(default_factory=list, description="Classes to define.")
    functions: List[MethodSpec] = Field(default_factory=list, description="Global functions.")

# [新增] 实验产物定义
class ArtifactType(str, Enum):
    FIGURE = "figure"
    TABLE = "table"
    OTHER = "other" # 兜底

class ExperimentArtifact(BaseModel):
    artifact_id: str = Field(default="artifact_unknown")
    type: ArtifactType = Field(default=ArtifactType.OTHER)
    description: str = Field(default="")
    filename: str = Field(default="plot.png")
    metrics_source: List[str] = Field(default_factory=list)

class DesignDocument(BaseModel):
    project_name: str = Field(default="Project")
    architecture_style: str = Field(default="Standard")
    requirements: List[str] = Field(default_factory=list)
    file_structure: List[FileSpec] = Field(default_factory=list)
    
    # [修复] 增加默认值，防止 JSON 截断导致报错
    experiments_plan: List[ExperimentArtifact] = Field(default_factory=list, description="List of figures/tables needed.")
    data_flow_diagram: str = Field(default="See code.", description="Text-based diagram.")
    hyperparameters: Dict[str, str] = Field(default_factory=dict)
    main_execution_flow: str = Field(default="Run main.py", description="Steps to run.")

# --- Phase 4: Paper Writer Outputs ---
class SectionContent(BaseModel):
    section_name: str = Field(default="Section")
    latex_content: str = Field(default="TODO")
    word_count: int = Field(default=0)

class PaperOutline(BaseModel):
    title: str = Field(default="Draft Title")
    abstract: str = Field(default="Pending abstract...")
    section_names: List[str] = Field(default_factory=list)

class PaperDraft(BaseModel):
    outline: PaperOutline = Field(default_factory=PaperOutline)
    title: str = Field(default="")
    abstract: str = Field(default="")
    sections: List[SectionContent] = Field(default_factory=list)
    bibliography_content: str = Field(default="")
    is_complete: bool = False

# --- Phase 5: Coder Outputs ---
class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

class CodeExecutionLog(BaseModel):
    command: str
    return_code: int
    stdout: str
    stderr: str
    error_analysis: Optional[str] = None

class ExperimentResults(BaseModel):
    metrics: Dict[str, float] = Field(default_factory=dict)
    figures: List[str] = Field(default_factory=list)
    status: ExecutionStatus = Field(default=ExecutionStatus.FAILED)

class CoderOutput(BaseModel):
    environment_yaml: str = Field(default="")
    execution_log: List[CodeExecutionLog] = Field(default_factory=list)
    results: Optional[ExperimentResults] = None

class EnvRequirements(BaseModel):
    additional_conda_packages: List[str] = Field(default_factory=list)
    additional_pip_packages: List[str] = Field(default_factory=list)

# --- Reviewer ---
class ReviewDecision(str, Enum):
    ACCEPT = "APPROVE"
    REJECT = "REVISE"

class ReviewReport(BaseModel):
    decision: ReviewDecision
    score: int
    critique: str
    specific_instructions: str