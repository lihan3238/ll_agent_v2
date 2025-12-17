# src/agents/refiner.py
import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field  # [关键修复] 统一在头部导入
from src.agents.base import BaseAgent
from src.core.schema import SectionContent
from src.utils.logger import sys_logger

class RefinerAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="refiner")

    def inject_data(self, section_text: str, section_name: str, metrics: Dict[str, Any]) -> str:
        """
        将实验数据注入到文本中
        """
        # 如果没有数据，直接返回原文本
        if not metrics:
            return section_text
            
        sys_logger.info(f"💉 Injecting data into {section_name}...")
        
        # 定义一个简单的 Schema 用于接收返回结果
        class InjectionResult(BaseModel):
            updated_content: str = Field(..., description="The full latex text with data injected.")
            changes_made: List[str] = Field(default_factory=list, description="List of changes made.")

        try:
            result = self.call_llm_with_struct(
                prompt_template=self.prompts["system"] + "\n\n" + self.prompts["inject_template"],
                schema=InjectionResult,
                metrics_json=json.dumps(metrics, indent=2),
                section_name=section_name,
                latex_content=section_text
            )
            
            if result.changes_made:
                sys_logger.info(f"   -> Changes: {result.changes_made}")
            return result.updated_content
            
        except Exception as e:
            sys_logger.warning(f"Data injection failed for {section_name}: {e}")
            return section_text

    def fix_latex(self, filename: str, content: str, error_log: str) -> str:
        """
        修复 LaTeX 语法错误
        """
        sys_logger.info(f"🔧 Fixing LaTeX error in {filename}...")
        
        # 简单提取行号（假设 Log 格式包含 "line X"）
        line_match = re.search(r"line (\d+)", error_log)
        line_num = line_match.group(1) if line_match else "unknown"
        
        # [修复] 现在 BaseModel 已经在文件头部导入，不会报错了
        class FixResult(BaseModel):
            fixed_content: str = Field(..., description="The full fixed latex content.")
        
        try:
            result = self.call_llm_with_struct(
                prompt_template=self.prompts["system"] + "\n\n" + self.prompts["fix_template"],
                schema=FixResult,
                error_log=error_log[-3000:], # 稍微多取一点 log
                filename=filename,
                context_lines=content, 
                line_num=line_num
            )
            return result.fixed_content
        except Exception as e:
            sys_logger.error(f"Latex fix failed: {e}")
            return content