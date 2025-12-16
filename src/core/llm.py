# src/core/llm.py
import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import llm_logger
import sys

load_dotenv()

def call_llm(
    prompt: str, 
    model: str, 
    base_url: str = None, 
    temperature: float = 0.7, 
    max_tokens: int = 16384,
    agent_name: str = "unknown"
) -> str:
    """
    统一 LLM 接口 (Streaming + 实时控制台输出 + 鲁棒性增强)
    """
    # 1. 鉴权逻辑
    api_key = None
    final_base_url = base_url
    
    if "deepseek" in model.lower():
        if not final_base_url: final_base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not final_base_url: final_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key: api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key: 
        api_key = os.getenv("API_KEY") # 最后的尝试
        if not api_key:
            raise ValueError(f"Missing API Key for model {model}")

    # ================= LOGGING (FILE) =================
    llm_logger.info(f"======== [REQUEST] Agent: {agent_name} | Model: {model} ========")
    llm_logger.info(f"URL: {final_base_url}")
    llm_logger.info(f"PROMPT HEAD (First 500 chars):\n{prompt[:500]}...") 
    llm_logger.info("-" * 50)
    # =================================================

    # 超时设置：Architect 生成长文本需要很长时间
    # read=600 意味着如果服务器 600秒 不吐字才算超时
    timeout = httpx.Timeout(connect=15.0, read=600.0, write=15.0, pool=15.0)

    client = OpenAI(
        base_url=final_base_url, 
        api_key=api_key,
        timeout=timeout
    )

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True # 强制流式
        }
        
        # O1/Reasoning 模型兼容性
        if "o1" not in model.lower() and "reasoner" not in model.lower():
            kwargs["temperature"] = temperature

        response_stream = client.chat.completions.create(**kwargs)
        
        collected_content = []
        
        print(f"\n🤖 [{agent_name}] Generating:", end="\n", flush=True)
        print("-" * 40) 
        
        chunk_count = 0
        
        for chunk in response_stream:
            chunk_count += 1
            delta = chunk.choices[0].delta
            
            # [兼容性修复] 优先获取 content，如果没有，尝试获取 reasoning_content (针对 DeepSeek R1)
            # 注意：标准 OpenAI 库可能没有 reasoning_content 属性，需用 getattr 安全获取
            content = delta.content
            reasoning = getattr(delta, 'reasoning_content', None)
            
            # 优先使用 content；如果是 R1 且 content 为空但有 reasoning，也可以暂时打印出来看看
            # 但最终我们只需要 content。如果模型只返回 reasoning，说明 Prompt 没引导它输出结论。
            
            part = content if content is not None else ""
            
            if part:
                collected_content.append(part)
                print(part, end="", flush=True)
            
            # 如果是 DeepSeek R1 的思考过程，选择性打印（可选）
            # if reasoning:
            #     print(f"[Think: {reasoning}]", end="", flush=True)

        print("\n" + "-" * 40)
        
        full_content = "".join(collected_content)
        
        # [核心修复] 空响应检查
        if not full_content.strip():
            err_msg = f"LLM returned empty response! (Chunks received: {chunk_count})"
            llm_logger.error(err_msg)
            # 打印 Prompt 尾部以供调试
            llm_logger.error(f"PROMPT TAIL:\n...{prompt[-500:]}")
            raise RuntimeError(err_msg)

        print(f"✅ [{agent_name}] Generation Complete. Length: {len(full_content)}")
        
        # ================= LOGGING (FILE) =================
        llm_logger.info(f"======== [RESPONSE] Agent: {agent_name} ========")
        llm_logger.info(f"Total Length: {len(full_content)} chars")
        llm_logger.info(f"CONTENT HEAD:\n{full_content[:500]}...") 
        llm_logger.info("=" * 60 + "\n") 

        return full_content

    except Exception as e:
        llm_logger.error(f"LLM Call Failed for {agent_name}: {str(e)}")
        print(f"\n❌ LLM Error: {str(e)}")
        raise RuntimeError(f"LLM API Call Failed: {str(e)}")