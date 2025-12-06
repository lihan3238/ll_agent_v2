# src/core/llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import llm_logger

load_dotenv()

def call_llm(prompt: str, model: str, base_url: str = None, temperature: float = 0.7, max_tokens: int = 8192) -> str:
    """
    统一 LLM 接口，支持多厂商自动切换 Key
    """
    # 1. 确定 Base URL 和 API Key
    # 策略：如果 Config 里没传 URL，先看是不是 DeepSeek 模型，是则优先用 DeepSeek 环境配置
    
    api_key = None
    final_base_url = base_url
    
    if "deepseek" in model.lower():
        if not final_base_url: final_base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not final_base_url: final_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key: api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key: raise ValueError(f"Missing API Key for model {model}")

    # ================= LOGGING START =================
    llm_logger.info(f"======== [REQUEST] Model: {model} ========")
    llm_logger.info(f"URL: {final_base_url}")
    llm_logger.info(f"PROMPT HEAD:\n{prompt[:200]}...") # 只打头200字防止刷屏，完整在文件里
    llm_logger.info("-" * 50)
    # =================================================

    client = OpenAI(base_url=final_base_url, api_key=api_key)

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens # [新增] 传入限制
        }
        
        # DeepSeek-reasoner (R1) 建议不要传 temperature 或者设为默认，
        # 但 deepseek-chat (V3) 支持。为了安全，如果不是 reasoner 才传 temp
        if "reasoner" not in model:
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        # ================= LOGGING END =================
        llm_logger.info(f"======== [RESPONSE] ========")
        llm_logger.info(f"CONTENT HEAD:\n{content[:200]}...")
        llm_logger.info("=" * 60 + "\n") 

        return content

    except Exception as e:
        llm_logger.error(f"LLM Call Failed: {str(e)}")
        raise RuntimeError(f"LLM API Call Failed: {str(e)}")