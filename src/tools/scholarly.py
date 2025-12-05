# src/tools/scholarly.py
import requests
import os
import time
from typing import List, Dict
from src.utils.logger import sys_logger

class SemanticScholarTool:
    def __init__(self):
        self.api_key = os.getenv("S2_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        根据 query 搜索论文，返回清洗后的列表
        """
        sys_logger.info(f"Checking Semantic Scholar for: '{query}'")
        time.sleep(3)

        params = {
            "query": query,
            "limit": limit,
            # 指定我们需要返回的字段，节省带宽和Token
            "fields": "title,abstract,year,citationCount,authors,url"
        }

        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                sys_logger.info(f"🌐 S2 API Request (Attempt {attempt+1}/{max_retries}): '{query}'")
                
                response = requests.get(
                    self.base_url, 
                    params=params, 
                    headers=self.headers, 
                    timeout=15 # 稍微增加超时时间
                )
                
                # Case A: 成功
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("data", [])
                    if not papers:
                        sys_logger.warning(f"No papers found for query: {query}")
                        return []
                    
                    # 清洗数据
                    cleaned_papers = []
                    for p in papers:
                        if not p.get("abstract"):
                            continue
                        cleaned_papers.append({
                            "title": p.get("title", "Unknown Title"),
                            "year": p.get("year", "N/A"),
                            "citations": p.get("citationCount", 0),
                            "abstract": p.get("abstract", "").replace("\n", " "),
                            "url": p.get("url", ""),
                            "authors": ", ".join([a["name"] for a in p.get("authors", [])][:3])
                        })
                    return cleaned_papers
                
                # Case B: 需要重试的错误 (429 Too Many Requests, 5xx Server Error)
                elif response.status_code == 429 or response.status_code >= 500:
                    sys_logger.warning(f"⚠️ API Status {response.status_code}. Retrying in 5s...")
                    time.sleep(3)
                    continue # 进入下一次循环
                
                # Case C: 客户端错误 (400 Bad Request 等)，通常是因为 Query 格式不对，重试没用
                else:
                    sys_logger.error(f"❌ S2 API Error {response.status_code}: {response.text}")
                    return []

            except requests.exceptions.RequestException as e:
                # 网络层面报错 (断网、DNS解析失败等)
                sys_logger.warning(f"⚠️ Network Error: {e}. Retrying in 5s...")
                time.sleep(3)
                continue

        # 如果循环结束还没返回，说明失败了
        sys_logger.error(f"❌ Failed to fetch papers for '{query}' after {max_retries} attempts.")
        return []

# 单例实例
s2_tool = SemanticScholarTool()