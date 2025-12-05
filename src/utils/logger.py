# src/utils/logger.py
import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 格式定义
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
CONSOLE_FORMAT = ">> %(message)s" 
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def _create_file_handler(name: str):
    """辅助：创建按天滚动的文件Handler"""
    today = datetime.now().strftime("%Y-%m-%d")
    # llm trace 独立文件，其他系统日志一个文件
    filename = f"{name}_{today}.log" if "llm" in name.lower() else f"session_{today}.log"
    path = os.path.join(LOG_DIR, filename)
    
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    return handler

def get_console_logger(name: str):
    """系统主Logger：控制台输出简略信息，文件记录详细信息"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 1. Console Handler (简略)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        c_handler.setLevel(logging.INFO)
        logger.addHandler(c_handler)

        # 2. File Handler (详细)
        f_handler = _create_file_handler(name)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)
        
    return logger

def get_file_only_logger(name: str):
    """LLM Trace Logger：只写文件，不输出到控制台"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 关键：防止冒泡到 root logger

    if not logger.handlers:
        f_handler = _create_file_handler(name)
        logger.addHandler(f_handler)
        
    return logger

# 对外暴露
sys_logger = get_console_logger("SYSTEM")
llm_logger = get_file_only_logger("LLM_TRACE")