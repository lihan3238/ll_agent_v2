# src/agents/coder.py
from typing import List, Dict
from pydantic import BaseModel, Field
from src.agents.base import BaseAgent
from src.core.schema import DesignDocument
from src.utils.logger import sys_logger

# Coder 的输出结构
class CodeFile(BaseModel):
    filename: str
    content: str

class Codebase(BaseModel):
    files: List[CodeFile]

class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="coder")

    def generate_code(self, design: DesignDocument, env_config: dict) -> Codebase:
        sys_logger.info("Coder: Generating initial codebase...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["gen_code_template"]
        
        # 序列化 Design Document 供 LLM 阅读
        design_str = design.model_dump_json(indent=2)
        
        # 提取关键环境信息
        base_reqs = env_config.get("base_requirements", [])
        hw_ctx = env_config.get("hardware_context", "CPU")
        
        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            design_doc=design_str,
            env_config=str(env_config),
            base_requirements=str(base_reqs),
            hardware_context=hw_ctx
        )

    def fix_code(self, error_log: str, files: Dict[str, str]) -> Codebase:
        sys_logger.info("Coder: Fixing bugs based on error log...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["fix_bug_template"]
        
        # 构造上下文：把当前代码喂给 LLM
        code_context = ""
        for name, content in files.items():
            # 简单截断，防止文件过大
            content_preview = content if len(content) < 5000 else content[:2000] + "\n...[truncated]..." + content[-2000:]
            code_context += f"--- {name} ---\n{content_preview}\n\n"
            
        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            error_log=error_log[-4000:], # 取最后 4000 字符的报错信息
            file_content=code_context
        )