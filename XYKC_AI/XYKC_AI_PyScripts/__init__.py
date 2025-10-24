# XYKC_AI_PyScripts package initialization file
# This file makes the directory a Python package

# 导出模块，使它们可以被外部导入
from .qwen3_vl_4b_instruct import process_with_model as process_with_qwen3_vl_4b_instruct
from .deepseek_ocr import process_with_model as process_with_deepseek_ocr

__all__ = [
    "process_with_qwen3_vl_4b_instruct",
    "process_with_deepseek_ocr"
]