#!/bin/bash

# 若任一步失败则退出
set -e

TEX_FILE="workspace\mamba_weather_v3\latex\template.tex"
BASE="template"

echo "▶ 第一遍编译正文..."
pdflatex "$TEX_FILE"

echo "▶ 生成参考文献索引..."
bibtex "$BASE"

echo "▶ 第二遍编译（插入参考文献）..."
pdflatex "$TEX_FILE"

echo "▶ 第三遍编译（修复交叉引用）..."
pdflatex "$TEX_FILE"

echo "✓ 完成！输出文件：${BASE}.pdf"
