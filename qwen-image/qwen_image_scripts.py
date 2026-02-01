"""
Qwen Image Extension - 主脚本文件（重构版）
该文件现在只是一个入口点，实际功能已迁移到主目录下的相应模块
"""

try:
    # 尝试相对导入（当作为包的一部分被导入时）
    try:
        from .main import (
            run_preprocess_control_image_wrapper as run_preprocess_control_image,
            run_text_to_image_wrapper as run_text_to_image,
            run_image_editing_wrapper as run_image_editing,
            dispatch_command
        )
    except ImportError:
        # 如果相对导入失败，尝试绝对导入
        import sys
        import os
        from pathlib import Path
        
        # 获取当前脚本所在的目录
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
            
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
except ImportError as e:
    print(f"qwen_image_scripts.py 导入错误: {e}")
    import traceback
    traceback.print_exc()

    # 定义空函数作为备用
    def run_preprocess_control_image(args_file):
        print(f"无法导入预处理函数: {args_file}")
        return None

    def run_text_to_image(args_file):
        print(f"无法导入文本到图像函数: {args_file}")
        return None

    def run_image_editing(args_file):
        print(f"无法导入图像编辑函数: {args_file}")
        return None

    def dispatch_command(command, args):
        print(f"无法导入命令分发函数: {command}, {args}")
        return None