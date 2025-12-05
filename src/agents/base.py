# src/agents/base.py
import yaml
import os
import json
from abc import ABC
from typing import TypeVar, Generic, Type, Dict, Any, Optional
from pydantic import BaseModel
from json_repair import repair_json # [新增] 引入修复库

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
            
            # 合并配置
            merged = {
                "model": agent_specific.get("model", global_llm.get("default_model")),
                "base_url": agent_specific.get("base_url", global_llm.get("default_base_url")),
                "temperature": agent_specific.get("temperature", 0.7),
                # 传入其他自定义参数
                **{k:v for k,v in agent_specific.items() if k not in ["model", "base_url", "temperature"]}
            }
            return merged
        except Exception as e:
            sys_logger.error(f"Config load failed for {self.role_name}: {e}")
            raise

    def _load_prompt_file(self) -> Dict[str, str]:
        """加载 Prompt 字典"""
        path = os.path.join("prompts", f"{self.role_name}.yaml")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            sys_logger.error(f"Prompt file missing: {path}")
            raise

    def call_llm_with_struct(self, prompt_template: str, schema: Type[T], **kwargs) -> T:
        """
        原子能力：渲染 Prompt -> 调用 LLM -> 强制解析为 Schema T
        """
        # 1. 自动注入 JSON Schema
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        kwargs["json_schema"] = schema_json
        
        # 2. 渲染文本 [这里就是刚才报错的地方]
        try:
            prompt = prompt_template.format(**kwargs)
        except KeyError as e:
            # 如果 YAML 里没写双大括号，这里就会报错
            sys_logger.error(f"Prompt formatting error. Did you forget to escape {{}} in your YAML? Error key: {e}")
            raise ValueError(f"Prompt template missing variable: {e}")

        # 3. 重试循环
        max_retries = self.config.get("max_retries", 3)
        for i in range(max_retries):
            try:
                raw_resp = call_llm(
                    prompt=prompt,
                    model=self.config["model"],
                    base_url=self.config["base_url"],
                    temperature=self.config["temperature"]
                )
                
                return self._parse_output(raw_resp, schema)
            
            except Exception as e:
                if i == max_retries - 1:
                    sys_logger.error(f"[{self.role_name}] Struct parse failed finally: {e}")
                    raise RuntimeError(f"Agent {self.role_name} failed struct parsing.")

    def _parse_output(self, raw_text: str, schema: Type[T]) -> T:
        """纯净解析 + json_repair 救急"""
        clean_text = raw_text.strip()
        
        # 去除 Markdown 标记
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]
        
        try:
            # 方案 A: 尝试标准解析 (Pydantic V2)
            return schema.model_validate_json(clean_text)
        except Exception:
            # 方案 B: 如果失败，使用 json_repair 进行强力修复
            try:
                sys_logger.warning(f"[{self.role_name}] Standard JSON parse failed. Attempting json_repair...")
                
                # json_repair 返回的是修复后的 JSON 字符串
                repaired_str = repair_json(clean_text)
                
                # 再次尝试解析
                return schema.model_validate_json(repaired_str)
            except Exception as e:
                # 彻底没救了
                sys_logger.error(f"[{self.role_name}] Fatal JSON Error even after repair: {e}")
                sys_logger.debug(f"Bad Output: {clean_text}")
                raise ValueError(f"JSON Parse Error: {e}")