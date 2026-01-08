#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen Image Extension - 主脚本文件（重构版）
该文件现在只是一个入口点，实际功能已迁移到主目录下的相应模块
"""

# 从新的模块化结构导入函数
from main import (
    run_preprocess_control_image_wrapper as run_preprocess_control_image,
    run_text_to_image_wrapper as run_text_to_image,
    run_image_editing_wrapper as run_image_editing,
    dispatch_command
)

# 保持原有API接口不变，以确保向后兼容
__all__ = [
    'run_preprocess_control_image',
    'run_text_to_image', 
    'run_image_editing',
    'dispatch_command'
]

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen Image Scripts - 模块化重构版本")
    parser.add_argument("command", choices=["preprocess", "text_to_image", "image_editing"],
                       help="要执行的命令")
    parser.add_argument("args_file", help="参数文件路径")
    
    args = parser.parse_args()
    
    try:
        result = dispatch_command(args.command, args.args_file)
        print(f"命令执行完成: {result}")
    except Exception as e:
        print(f"执行命令时出错: {e}")
        sys.exit(1)