# src/agents/coder.py
import json
import yaml
import re
from typing import List, Dict
from pydantic import BaseModel
from src.agents.base import BaseAgent
from src.core.schema import ResearchReport,DesignDocument, FileSpec
from src.utils.logger import sys_logger

class CodeFile(BaseModel):
    filename: str
    content: str

class Codebase(BaseModel):
    files: List[CodeFile]

class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(role_name="coder")

    def _get_env_vars(self, env_config: dict) -> dict:
        return {
            "os_platform": env_config.get("os_platform", "linux"),
            "hardware_context": env_config.get("experience_context", env_config.get("hardware_context", "CPU"))
        }

    # [新增] 智能骨架生成方法
    def write_smart_skeleton(self, 
                             file_spec: FileSpec, 
                             design: DesignDocument, 
                             research: ResearchReport, 
                             env_config: dict) -> CodeFile:
        
        sys_logger.info(f"🧱 Smart Scaffolding: {file_spec.filename}...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["smart_skeleton_template"]
        
        # 准备上下文
        spec_json = file_spec.model_dump_json(indent=2)
        # 提取 Architect 的关键信息 (防止 token 爆炸，不传整个 design)
        design_summary = f"Style: {design.architecture_style}\nFlow: {design.main_execution_flow}"
        
        # 调用 LLM 生成单个文件的骨架代码
        # 注意：这里我们让 LLM 直接返回 CodeFile 结构
        # 或者为了简单，如果模板返回的是纯代码，我们需要包装一下
        # 为了复用 call_llm_with_struct，我们让它返回 CodeFile
        
        # 这里需要稍微 trick 一下，因为 smart_skeleton_template 期望返回代码
        # 但我们的 base agent 期望 JSON。
        # 建议：修改 prompt 让它返回 JSON 包含 {filename, content}
        # 或者：我们这里手动构造 CodeFile
        
        # 让我们复用 call_llm_with_struct, 让 prompt 指示返回 JSON
        # 此时 prompt 里的 Target Schema 会生效
        
        result = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase, # 复用 Codebase 结构 (虽然只返回一个文件)
            filename=file_spec.filename,
            file_spec_json=spec_json,
            idea=research.refined_idea,
            design_context=design_summary,
            **self._get_env_vars(env_config)
        )
        
        # 提取结果
        for f in result.files:
            # 模糊匹配文件名，防止 LLM 改名
            if file_spec.filename in f.filename or f.filename in file_spec.filename:
                return f
        
        # 兜底
        return result.files[0] if result.files else CodeFile(filename=file_spec.filename, content="# Generation Failed")

    def generate_env_yaml(self, design: DesignDocument, env_config: dict) -> Codebase:
        """只生成 environment.yaml"""
        sys_logger.info("Coder: Generating environment configuration...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["env_gen_template"]
        
        # 准备 Requirements 字符串
        reqs_str = "\n".join(design.requirements)
        
        # 序列化 Design Doc (防御性措施：即使模板里误写了 {design_doc}，传进去也不会报错)
        design_str = design.model_dump_json(indent=2)
        
        codebase = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            requirements=reqs_str,
            design_doc=design_str, # [新增] 传入此变量以防模板需要
            **self._get_env_vars(env_config)
        )
        
        # 注入依赖 [之前报错就是因为下面这个方法没定义]
        self._inject_requirements(codebase, env_config)
        return codebase

    def implement_single_file(self, 
                              file_spec: FileSpec, 
                              current_skeleton: str, 
                              project_context: str, 
                              env_config: dict) -> CodeFile:
        
        sys_logger.info(f"✍️ Coder: Implementing {file_spec.filename}...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["implement_template"]
        
        spec_json = file_spec.model_dump_json(indent=2)
        
        result = self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            filename=file_spec.filename,
            file_spec_json=spec_json,
            current_skeleton=current_skeleton,
            project_context=project_context,
            **self._get_env_vars(env_config)
        )
        
        for f in result.files:
            if f.filename == filename or filename in f.filename: # 简单模糊匹配
                return f
        return result.files[0] if result.files else CodeFile(filename=file_spec.filename, content=current_skeleton)

    def fix_code(self, command: str, error_log: str, files: Dict[str, str], env_config: dict) -> Codebase:
        sys_logger.info("🚑 Coder: Analyzing error and fixing code...")
        
        full_prompt = self.prompts["system"] + "\n\n" + self.prompts["fix_bug_template"]
        
        code_context = ""
        for name, content in files.items():
            content_trunc = content if len(content) < 3000 else content[:1500] + "\n...[truncated]...\n" + content[-1500:]
            code_context += f"--- FILE: {name} ---\n{content_trunc}\n\n"

        return self.call_llm_with_struct(
            prompt_template=full_prompt,
            schema=Codebase,
            command=command,
            error_log=error_log[-5000:],
            file_content=code_context,
            **self._get_env_vars(env_config)
        )

    def _inject_requirements(self, codebase: Codebase, env_config: dict):
        """
        [补全的方法] 解析 config 结构化依赖并注入 environment.yaml
        """
        base_reqs = env_config.get("base_requirements", {})
        python_ver = env_config.get("python_version", "3.11")
        
        config_conda_pkgs = base_reqs.get("conda", [])
        config_pip_pkgs = base_reqs.get("pip", [])

        # 1. 建立 Pip 黑名单
        pip_blacklist = set()
        for item in config_pip_pkgs:
            item_str = str(item).strip()
            if item_str.startswith("-"): continue
            pkg_name = re.split(r'[<>=!]', item_str)[0].strip()
            pip_blacklist.add(pkg_name)
            if pkg_name == "torch":
                pip_blacklist.add("pytorch")
                pip_blacklist.add("pytorch-cuda")

        # 2. 找到/创建 environment.yaml
        yaml_file = next((f for f in codebase.files if "environment.yaml" in f.filename or "environment.yml" in f.filename), None)
        if not yaml_file:
            # [修改] 默认加入 conda-forge
            yaml_file = CodeFile(filename="environment.yaml", content="name: project_env\nchannels:\n  - conda-forge\n  - defaults\ndependencies:\n")
            codebase.files.append(yaml_file)

        try:
            env_data = yaml.safe_load(yaml_file.content) or {}
            
            # [新增] 强制确保 conda-forge 存在且优先级最高
            if "channels" not in env_data:
                env_data["channels"] = ["conda-forge", "defaults"]
            else:
                if "conda-forge" not in env_data["channels"]:
                    env_data["channels"].insert(0, "conda-forge")
            
            if "dependencies" not in env_data:
                env_data["dependencies"] = []
            
            original_deps = env_data["dependencies"]
            
            # --- 构建新的 dependencies 列表 ---
            new_deps = []
            
            # A. 强制 Python 版本
            new_deps.append(f"python={python_ver}")
            new_deps.append("pip")

            # B. 注入 Config 中的 Conda 包
            for pkg in config_conda_pkgs:
                if pkg not in new_deps:
                    new_deps.append(pkg)
            
            # C. 筛选 LLM 生成的 Conda 包
            for item in original_deps:
                if isinstance(item, str):
                    if item.startswith("python=") or item == "pip":
                        continue
                    
                    llm_pkg_name = re.split(r'[<>=!]', item)[0].strip()
                    
                    if llm_pkg_name in pip_blacklist:
                        sys_logger.warning(f"🚫 Removing '{item}' from Conda list because it is defined in Pip config.")
                        continue
                        
                    if item not in new_deps:
                        new_deps.append(item)
            
            # D. 处理 Pip 包
            llm_pip_list = []
            for item in original_deps:
                if isinstance(item, dict) and "pip" in item:
                    llm_pip_list.extend(item["pip"])
            
            final_pip_list = []
            index_url_line = None
            
            # D1. Config Pip
            for pkg in config_pip_pkgs:
                pkg_str = str(pkg).strip()
                if "--index-url" in pkg_str:
                    index_url_line = pkg_str
                else:
                    if pkg_str not in final_pip_list:
                        final_pip_list.append(pkg_str)
            
            # D2. LLM Pip (去重)
            for pkg in llm_pip_list:
                pkg_str = str(pkg).strip()
                if "--index-url" in pkg_str: continue 
                
                pkg_name = re.split(r'[<>=!]', pkg_str)[0].strip()
                is_duplicate = False
                for existing in final_pip_list:
                    existing_name = re.split(r'[<>=!]', existing)[0].strip()
                    if pkg_name == existing_name:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    final_pip_list.append(pkg_str)

            # E. 组装
            if final_pip_list or index_url_line:
                pip_block = []
                if index_url_line:
                    pip_block.append(index_url_line)
                pip_block.extend(final_pip_list)
                new_deps.append({"pip": pip_block})

            # F. 写回
            env_data["dependencies"] = new_deps
            yaml_file.content = yaml.dump(env_data, sort_keys=False, default_flow_style=False)
            sys_logger.info("✅ Successfully injected and sanitized requirements.")

        except Exception as e:
            sys_logger.error(f"Failed to inject requirements: {e}")