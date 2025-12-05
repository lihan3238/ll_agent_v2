# src/tools/pdf_reader.py
import fitz  # PyMuPDF
import os
import re
from src.utils.logger import sys_logger

class PDFReaderTool:
    def read_pdf(self, file_path: str, max_pages: int = 12) -> str:
        """
        读取 PDF 文本，包含智能清洗和截断逻辑。
        :param max_pages: 限制读取页数（CVPR/NeurIPS 正文通常 < 10页，加上附录给12页够了）
        """
        if not os.path.exists(file_path):
            sys_logger.warning(f"PDF file not found: {file_path}")
            return ""

        try:
            doc = fitz.open(file_path)
            text_content = []
            
            # 限制页数
            read_limit = min(len(doc), max_pages)
            
            sys_logger.info(f"📄 Reading {os.path.basename(file_path)} (Pages 1-{read_limit})...")

            for i in range(read_limit):
                page = doc.load_page(i)
                raw_text = page.get_text()
                
                # --- 清洗逻辑 ---
                cleaned_text = self._clean_page_text(raw_text)
                
                # --- 智能截断逻辑 ---
                # 如果发现这一页全是参考文献，不仅这页不要，后面也不要了
                if self._is_reference_page(cleaned_text):
                    sys_logger.info(f"   -> Detected References at page {i+1}. Stopping early.")
                    break
                
                text_content.append(f"\n--- Page {i+1} ---\n{cleaned_text}")
                
            doc.close()
            final_text = "".join(text_content)
            sys_logger.info(f"   -> Extraction finished. Length: {len(final_text)} chars.")
            return final_text
            
        except Exception as e:
            sys_logger.error(f"Failed to read PDF {file_path}: {e}")
            return ""

    def _clean_page_text(self, text: str) -> str:
        """简单的文本清洗"""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            s_line = line.strip()
            # 1. 去掉太短的行（可能是页码或页眉）
            if len(s_line) < 4 and s_line.isdigit(): 
                continue 
            cleaned_lines.append(s_line)
        return "\n".join(cleaned_lines)

    def _is_reference_page(self, text: str) -> bool:
        """判断是否进入了参考文献区域"""
        # 简单规则：包含大写的 "REFERENCES" 且位于行首附近
        if re.search(r'^\s*(REFERENCES|References|Bibliography)\s*$', text, re.MULTILINE):
            return True
        return False

pdf_tool = PDFReaderTool()