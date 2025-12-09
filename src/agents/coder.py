# src/agents/coder.py
import yaml
import json # 确保导入
from typing import List, Dict
from pydantic import BaseModel
from src.agents.base import BaseAgent
from src.core.schema import DesignDocument
from src.utils.logger import sys_logger

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
        design_str = design.model_dump_json(indent=2)
        
        # 将结构化的 config 转为 JSON 字符串展示给 LLM，让它知道有哪些包
        env_str = json.dumps(env_config, indent=2)
        
        codebase = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            design_doc=design_str,
            env_config=env_str
        )

        # 后处理：注入 Config 中的依赖
        self._inject_requirements(codebase, env_config)
        
        return codebase

    def fix_code(self, error_log: str, files: Dict[str, str]) -> Codebase:
        sys_logger.info("Coder: Fixing bugs based on error log...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["fix_bug_template"]
        
        code_context = ""
        for name, content in files.items():
            content_preview = content if len(content) < 5000 else content[:2000] + "\n...[truncated]..." + content[-2000:]
            code_context += f"--- {name} ---\n{content_preview}\n\n"
            
        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            error_log=error_log[-4000:],
            file_content=code_context
        )

    def _inject_requirements(self, codebase: Codebase, env_config: dict):
        """
        [简化版] 解析 config 结构化依赖并注入 environment.yaml
        """
        base_reqs = env_config.get("base_requirements", {})
        python_ver = env_config.get("python_version", "3.10")
        
        # 1. 找到或创建 environment.yaml
        yaml_file = next((f for f in codebase.files if "environment.yaml" in f.filename or "environment.yml" in f.filename), None)
        
        if not yaml_file:
            yaml_file = CodeFile(filename="environment.yaml", content="name: project_env\nchannels:\n  - defaults\ndependencies:\n")
            codebase.files.append(yaml_file)

        try:
            # 2. 解析 YAML
            env_data = yaml.safe_load(yaml_file.content) or {}
            
            if "dependencies" not in env_data:
                env_data["dependencies"] = []
            
            deps = env_data["dependencies"]
            
            # 3. 注入 Python 版本
            has_python = any(d.startswith("python=") if isinstance(d, str) else False for d in deps)
            if not has_python:
                deps.insert(0, f"python={python_ver}")
            
            # 确保有 pip 工具
            if "pip" not in deps:
                deps.append("pip")

            # 4. 提取 Config 数据
            # 兼容旧格式（如果是列表）和新格式（如果是字典）
            if isinstance(base_reqs, list):
                # 如果用户还在用旧格式，记录警告并跳过或尝试简单处理
                sys_logger.warning("Config base_requirements is a list. Please update to dict format (conda/pip) for better handling.")
                config_conda = []
                config_pip = []
            else:
                config_conda = base_reqs.get("conda", [])
                config_pip = base_reqs.get("pip", [])

            # 5. 注入 Conda 包
            for pkg in config_conda:
                if pkg not in deps:
                    deps.append(pkg)

            # 6. 注入 Pip 包
            if config_pip:
                # 寻找现有的 pip 字典块
                pip_section = next((d for d in deps if isinstance(d, dict) and "pip" in d), None)
                
                if not pip_section:
                    pip_section = {"pip": []}
                    deps.append(pip_section)
                
                if not pip_section['pip']:
                    pip_section['pip'] = []
                
                # 追加 Pip 包
                for item in config_pip:
                    # 简单去重：如果该项已存在则跳过 (参数类的 --index-url 除外，为了安全可以不去重直接追加)
                    if not str(item).startswith("-") and item in pip_section['pip']:
                        continue
                    pip_section['pip'].append(item)

            # 7. 写回
            yaml_file.content = yaml.dump(env_data, sort_keys=False, default_flow_style=False)
            sys_logger.info("✅ Successfully injected structured base requirements.")

        except Exception as e:
            sys_logger.error(f"Failed to inject requirements: {e}")