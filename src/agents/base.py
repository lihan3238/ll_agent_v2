# src/agents/base.py
import yaml
import os
import json
from abc import ABC
from typing import TypeVar, Generic, Type, Dict, Any, Optional
from pydantic import BaseModel
from json_repair import repair_json

from src.core.llm import call_llm
from src.utils.logger import sys_logger

T = TypeVar('T', bound=BaseModel)

class BaseAgent(ABC, Generic[T]):
    def __init__(self, role_name: str):
        self.role_name = role_name
        self.config = self._load_agent_config()
        self.prompts = self._load_prompt_file()

    def _load_agent_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            
            global_llm = full_config.get("llm", {})
            agent_specific = full_config.get("agents", {}).get(self.role_name, {})
            
            merged = {
                "model": agent_specific.get("model", global_llm.get("default_model")),
                "base_url": agent_specific.get("base_url", global_llm.get("default_base_url")),
                "temperature": agent_specific.get("temperature", 0.7),
                "max_tokens": agent_specific.get("max_tokens", global_llm.get("default_max_tokens", 16384)),
                **{k:v for k,v in agent_specific.items() if k not in ["model", "base_url", "temperature", "max_tokens"]}
            }
            return merged
        except Exception as e:
            sys_logger.error(f"Config load failed for {self.role_name}: {e}")
            raise

    def _load_prompt_file(self) -> Dict[str, str]:
        path = os.path.join("prompts", f"{self.role_name}.yaml")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            sys_logger.error(f"Prompt file missing: {path}")
            raise

    def call_llm_with_struct(self, prompt_template: str, schema: Type[T], **kwargs) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        kwargs["json_schema"] = schema_json
        
        try:
            prompt = prompt_template.format(**kwargs)
        except KeyError as e:
            sys_logger.error(f"Prompt formatting error. Missing key: {e}")
            raise ValueError(f"Prompt template missing variable: {e}")

        max_retries = self.config.get("max_retries", 3)
        for i in range(max_retries):
            try:
                raw_resp = call_llm(
                    prompt=prompt,
                    model=self.config["model"],
                    base_url=self.config["base_url"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    agent_name=self.role_name
                )
                return self._parse_output(raw_resp, schema)
            
            except Exception as e:
                if i == max_retries - 1:
                    sys_logger.error(f"[{self.role_name}] Struct parse failed after {max_retries} retries: {e}")
                    raise RuntimeError(f"Agent {self.role_name} failed struct parsing.")
                sys_logger.warning(f"[{self.role_name}] Parse failed (Attempt {i+1}). Retrying...")

    def _parse_output(self, raw_text: str, schema: Type[T]) -> T:
        """强化版解析器：处理 Markdown、无效字符、以及列表包装"""
        clean_text = raw_text.strip()
        
        # 1. 移除 Markdown 块
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]
        
        clean_text = clean_text.strip()

        # 2. 尝试使用 json_repair 转换为标准 JSON 字符串
        try:
            repaired_json_str = repair_json(clean_text)
            # 加载为 Python 对象
            obj = json.loads(repaired_json_str)
        except Exception as e:
            sys_logger.error(f"[{self.role_name}] JSON Repair failed to produce valid JSON: {e}")
            raise ValueError(f"Invalid JSON format: {clean_text[:200]}...")

        # 3. [关键修复逻辑]：处理 List 包装
        # 如果模型返回了 [{...}] 但我们期望的是 {...}
        if isinstance(obj, list) and len(obj) > 0:
            # 如果目标 Schema 并没有定义为 List 类型（即它是一个普通的 BaseModel）
            # 我们取列表中的第一个有效对象
            sys_logger.info(f"[{self.role_name}] Detected JSON list wrapper, unwrapping first element.")
            obj = obj[0]

        # 4. 使用 Pydantic 验证 Python 对象
        try:
            return schema.model_validate(obj)
        except Exception as e:
            sys_logger.error(f"[{self.role_name}] Pydantic validation failed: {e}")
            sys_logger.debug(f"Object causing error: {obj}")
            raise e