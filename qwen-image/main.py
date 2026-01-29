
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
from preprocessor import run_preprocess_control_image
from text_to_image import run_text_to_image
from image_editing import run_image_editing


def run_preprocess_control_image_wrapper(args_file):
    """包装函数，用于预处理控制图像"""
    return run_preprocess_control_image(args_file)


def run_text_to_image_wrapper(args_file):
    """包装函数，用于文生图功能"""
    return run_text_to_image(args_file)


def run_image_editing_wrapper(args_file):
    """包装函数，用于图像编辑功能"""
    return run_image_editing(args_file)


def dispatch_command(command, args):
    """分发命令到相应功能"""
    if command == "preprocess":
        return run_preprocess_control_image_wrapper(args)
    elif command == "text_to_image":
        return run_text_to_image_wrapper(args)
    elif command == "image_editing":
        return run_image_editing_wrapper(args)
    else:
        print(f"未知命令: {command}")
        return None