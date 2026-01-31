
"""
Qwen Image Extension - 主入口模块
整合所有功能模块
"""

# 确保路径正确设置
import sys
import os
from pathlib import Path

# 添加必要的路径到sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
webui_root = parent_dir.parent.parent

# 添加路径
paths_to_add = [
    str(current_dir),  # qwen-image目录
    str(webui_root),  # 主目录
    str(webui_root / "extensions-builtin"),
    str(webui_root / "extensions-builtin" / "forge_legacy_preprocessors")
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 现在导入模块
try:
    from preprocessor import run_preprocess_control_image
    from text_to_image import run_text_to_image
    from image_editing import run_image_editing
except ImportError as e:
    print(f"main.py 模块导入错误: {e}")
    import traceback
    traceback.print_exc()
    
    # 定义备用函数
    def run_preprocess_control_image(args_file):
        print(f"无法导入预处理控制图像函数: {args_file}")
        return None

    def run_text_to_image(args_file):
        print(f"无法导入文本到图像函数: {args_file}")
        return None

    def run_image_editing(args_file):
        print(f"无法导入图像编辑函数: {args_file}")
        return None

def run_preprocess_control_image_wrapper(args_file):
    """包装函数，用于预处理控制图像"""
    try:
        return run_preprocess_control_image(args_file)
    except Exception as e:
        print(f"run_preprocess_control_image_wrapper 执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_text_to_image_wrapper(args_file):
    """包装函数，用于文生图功能"""
    try:
        return run_text_to_image(args_file)
    except Exception as e:
        print(f"run_text_to_image_wrapper 执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_image_editing_wrapper(args_file):
    """包装函数，用于图像编辑功能"""
    try:
        return run_image_editing(args_file)
    except Exception as e:
        print(f"run_image_editing_wrapper 执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def dispatch_command(command, args_file):
    """分发命令到相应功能"""
    try:
        if command == "preprocess":
            return run_preprocess_control_image_wrapper(args_file)
        elif command == "text_to_image":
            return run_text_to_image_wrapper(args_file)
        elif command == "image_editing":
            return run_image_editing_wrapper(args_file)
        else:
            print(f"未知命令: {command}")
            return None
    except Exception as e:
        print(f"dispatch_command 执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None