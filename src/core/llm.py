# src/core/llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import llm_logger

load_dotenv()

def call_llm(
    prompt: str, 
    model: str, 
    base_url: str = None, 
    temperature: float = 0.7, 
    max_tokens: int = 16384,
    agent_name: str = "unknown" # [新增] Agent 身份标识
) -> str:
    """
    统一 LLM 接口，支持多厂商自动切换 Key，并记录 Token 消耗
    """
    # 1. 确定 Base URL 和 API Key
    api_key = None
    final_base_url = base_url
    
    if "deepseek" in model.lower():
        if not final_base_url: final_base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not final_base_url: final_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key: api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key: raise ValueError(f"Missing API Key for model {model}")

    # ================= LOGGING START =================
    llm_logger.info(f"======== [REQUEST] Agent: {agent_name} | Model: {model} ========")
    llm_logger.info(f"URL: {final_base_url}")
    llm_logger.info(f"PROMPT HEAD:\n{prompt[:200]}...") 
    llm_logger.info("-" * 50)
    # =================================================

    client = OpenAI(base_url=final_base_url, api_key=api_key)

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        # DeepSeek-reasoner (R1) 兼容性处理
        if "reasoner" not in model:
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        
        # [新增] 提取 Token 消耗信息
        usage = response.usage
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            usage_info = f"Tokens: Input={prompt_tokens} | Output={completion_tokens} | Total={total_tokens}"
        else:
            usage_info = "Tokens: N/A (Usage info not returned)"

        # ================= LOGGING END =================
        llm_logger.info(f"======== [RESPONSE] Agent: {agent_name} ========")
        llm_logger.info(usage_info) # [新增] 记录 Token
        llm_logger.info(f"CONTENT HEAD:\n{content[:200]}...")
        llm_logger.info("=" * 60 + "\n") 

        return content

    except Exception as e:
        llm_logger.error(f"LLM Call Failed for {agent_name}: {str(e)}")
        raise RuntimeError(f"LLM API Call Failed: {str(e)}")