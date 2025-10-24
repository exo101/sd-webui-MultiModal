"""
Qwen3-VL-8B-Instruct 模型支持模块
用于在 sd-webui-MultiModal 插件中添加对 Qwen3-VL-8B-Instruct 模型的支持
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# 添加当前目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入Qwen3-VL-8B-Instruct模型处理模块
try:
    from .qwen3_vl_8b_instruct import (
        process_with_model as process_with_qwen3_vl_8b_instruct_model,
        is_model_available as is_qwen3_vl_8b_instruct_available,
        get_model_info as get_qwen3_vl_8b_instruct_info,
        download_model as download_qwen3_vl_8b_instruct_model,
        is_model_downloaded as is_qwen3_vl_8b_instruct_downloaded,
        load_model as load_qwen3_vl_8b_instruct_model
    )
    QWEN3_VL_AVAILABLE = True
except ImportError as e:
    QWEN3_VL_AVAILABLE = False
    print(f"警告: 无法导入 Qwen3-VL-8B-Instruct 模型支持模块: {e}")

def get_qwen3_vl_8b_instruct_model():
    """
    返回 Qwen3-VL-8B-Instruct 模型的配置信息
    """
    if QWEN3_VL_AVAILABLE:
        return get_qwen3_vl_8b_instruct_info()
    else:
        return {
            "model_name": "Qwen3-VL-8B-Instruct",
            "model_id": "qwen3-vl-8b-instruct",
            "description": "阿里通义千问推出的80亿参数视觉语言模型，具备强大的多模态理解能力",
            "capabilities": [
                "视觉问答",
                "图像描述生成",
                "OCR文本识别",
                "视觉推理",
                "GUI操作理解",
                "长文本理解"
            ],
            "parameters": {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

def is_qwen3_vl_8b_instruct_available():
    """
    检查 Qwen3-VL-8B-Instruct 模型是否可用
    """
    if QWEN3_VL_AVAILABLE:
        return is_qwen3_vl_8b_instruct_available()
    else:
        # 如果模块导入失败，检查基本依赖
        try:
            import transformers
            import torch
            import modelscope
            return True
        except ImportError:
            return False

def download_qwen3_vl_8b_instruct_if_needed():
    """
    如果需要，下载 Qwen3-VL-8B-Instruct 模型
    """
    if QWEN3_VL_AVAILABLE:
        try:
            # 只有当模型未下载时才尝试下载
            if not is_qwen3_vl_8b_instruct_downloaded():
                return download_qwen3_vl_8b_instruct_model()
            return True
        except Exception as e:
            print(f"模型下载检查失败: {e}")
            return False
    return False

def process_with_qwen3_vl_8b_instruct(image_path=None, prompt="", **kwargs):
    """
    使用 Qwen3-VL-8B-Instruct 模型处理输入
    
    Args:
        image_path (str): 图像文件路径（可选）
        prompt (str): 输入提示词
        **kwargs: 其他参数
    
    Returns:
        str: 模型处理结果
    """
    # 确保模型已下载
    download_qwen3_vl_8b_instruct_if_needed()
    
    if QWEN3_VL_AVAILABLE:
        # 调用从qwen3_vl_8b_instruct模块导入的实际处理函数
        return process_with_qwen3_vl_8b_instruct_model(image_path, prompt, **kwargs)
    else:
        try:
            # 这里应该是实际调用模型的代码
            # 由于 Qwen3-VL-8B-Instruct 是一个大型模型，通常通过API或专门的推理框架调用
            # 此处仅为示例实现
            
            if image_path and os.path.exists(image_path):
                # 处理图像和文本输入
                result = f"使用 Qwen3-VL-8B-Instruct 模型处理图像 {os.path.basename(image_path)} 和提示: {prompt}"
            else:
                # 仅处理文本输入
                result = f"使用 Qwen3-VL-8B-Instruct 模型处理文本: {prompt}"
                
            # 模拟模型处理时间
            import time
            time.sleep(1)
            
            return result
        except Exception as e:
            return f"处理过程中出现错误: {str(e)}"

# 定义支持的视觉模型列表
vision_model_names = [
    "qwen3-vl-8b-instruct",  # 新增的 Qwen3-VL-8B-Instruct 模型
    "qwen2.5vl:latest",
    "qwen2.5vl:3b",    
    "llama3.2-vision:latest",
]

def register_model():
    """
    注册模型到系统中
    """
    model_info = get_qwen3_vl_8b_instruct_model()
    print(f"已注册模型: {model_info['model_name']}")
    return model_info

if __name__ == "__main__":
    # 测试模型注册
    register_model()
"""
Qwen3-VL-8B-Instruct 模型支持模块
用于在 sd-webui-MultiModal 插件中添加对 Qwen3-VL-8B-Instruct 模型的支持
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# 添加当前目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入Qwen3-VL-8B-Instruct模型处理模块
try:
    from .qwen3_vl_8b_instruct import (
        process_with_model as process_with_qwen3_vl_8b_instruct_model,
        is_model_available as is_qwen3_vl_8b_instruct_available,
        get_model_info as get_qwen3_vl_8b_instruct_info,
        download_model as download_qwen3_vl_8b_instruct_model,
        is_model_downloaded as is_qwen3_vl_8b_instruct_downloaded,
        load_model as load_qwen3_vl_8b_instruct_model
    )
    QWEN3_VL_AVAILABLE = True
except ImportError as e:
    QWEN3_VL_AVAILABLE = False
    print(f"警告: 无法导入 Qwen3-VL-8B-Instruct 模型支持模块: {e}")

def get_qwen3_vl_8b_instruct_model():
    """
    返回 Qwen3-VL-8B-Instruct 模型的配置信息
    """
    if QWEN3_VL_AVAILABLE:
        return get_qwen3_vl_8b_instruct_info()
    else:
        return {
            "model_name": "Qwen3-VL-8B-Instruct",
            "model_id": "qwen3-vl-8b-instruct",
            "description": "阿里通义千问推出的80亿参数视觉语言模型，具备强大的多模态理解能力",
            "capabilities": [
                "视觉问答",
                "图像描述生成",
                "OCR文本识别",
                "视觉推理",
                "GUI操作理解",
                "长文本理解"
            ],
            "parameters": {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

def is_qwen3_vl_8b_instruct_available():
    """
    检查 Qwen3-VL-8B-Instruct 模型是否可用
    """
    if QWEN3_VL_AVAILABLE:
        return is_qwen3_vl_8b_instruct_available()
    else:
        # 如果模块导入失败，检查基本依赖
        try:
            import transformers
            import torch
            import modelscope
            return True
        except ImportError:
            return False

def download_qwen3_vl_8b_instruct_if_needed():
    """
    如果需要，下载 Qwen3-VL-8B-Instruct 模型
    """
    if QWEN3_VL_AVAILABLE:
        try:
            # 只有当模型未下载时才尝试下载
            if not is_qwen3_vl_8b_instruct_downloaded():
                return download_qwen3_vl_8b_instruct_model()
            return True
        except Exception as e:
            print(f"模型下载检查失败: {e}")
            return False
    return False

def process_with_qwen3_vl_8b_instruct(image_path=None, prompt="", **kwargs):
    """
    使用 Qwen3-VL-8B-Instruct 模型处理输入
    
    Args:
        image_path (str): 图像文件路径（可选）
        prompt (str): 输入提示词
        **kwargs: 其他参数
    
    Returns:
        str: 模型处理结果
    """
    # 确保模型已下载
    download_qwen3_vl_8b_instruct_if_needed()
    
    if QWEN3_VL_AVAILABLE:
        # 调用从qwen3_vl_8b_instruct模块导入的实际处理函数
        return process_with_qwen3_vl_8b_instruct_model(image_path, prompt, **kwargs)
    else:
        try:
            # 这里应该是实际调用模型的代码
            # 由于 Qwen3-VL-8B-Instruct 是一个大型模型，通常通过API或专门的推理框架调用
            # 此处仅为示例实现
            
            if image_path and os.path.exists(image_path):
                # 处理图像和文本输入
                result = f"使用 Qwen3-VL-8B-Instruct 模型处理图像 {os.path.basename(image_path)} 和提示: {prompt}"
            else:
                # 仅处理文本输入
                result = f"使用 Qwen3-VL-8B-Instruct 模型处理文本: {prompt}"
                
            # 模拟模型处理时间
            import time
            time.sleep(1)
            
            return result
        except Exception as e:
            return f"处理过程中出现错误: {str(e)}"

# 定义支持的视觉模型列表
vision_model_names = [
    "qwen3-vl-8b-instruct",  # 新增的 Qwen3-VL-8B-Instruct 模型
    "qwen2.5vl:latest",
    "qwen2.5vl:3b",    
    "llama3.2-vision:latest",
]

def register_model():
    """
    注册模型到系统中
    """
    model_info = get_qwen3_vl_8b_instruct_model()
    print(f"已注册模型: {model_info['model_name']}")
    return model_info

if __name__ == "__main__":
    # 测试模型注册
    register_model()