#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen Image Extension - 预处理器模块
用于处理图像预处理相关功能
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json
import time

# 添加项目路径到sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
webui_root = parent_dir.parent.parent.parent

# 添加必要的路径
paths_to_add = [
    str(parent_dir),  # qwen-image目录
    str(webui_root),  # 主目录
    str(webui_root / "extensions-builtin"),
    str(webui_root / "extensions-builtin" / "forge_legacy_preprocessors")
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入qwen_image_controlnet模块以使用其中的预处理功能
from qwen_image_controlnet import preprocess_for_qwen_image_controlnet
from config import CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL


def preprocess_control_image(image_input, preprocessor_display_name, mask_path=None):
    """预处理控制图像"""
    try:
        # 将UI显示名称转换为内部标识符
        mapped_preprocessor_type = CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_display_name, "none")
        
        # 直接使用qwen_image_controlnet.py中的预处理函数
        processed_image = preprocess_for_qwen_image_controlnet(image_input, mapped_preprocessor_type, mask_path)
        
        return processed_image

    except Exception as e:
        print(f"预处理控制图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return image_input


def run_preprocess_control_image(args_file):
    """运行预处理控制图像的主函数"""
    try:
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        print(f"开始执行控制图像预处理功能，参数文件: {args_file}")
        
        # 获取参数
        image_path = args.get('image_path')
        preprocessor_type = args.get('preprocessor_type')
        mask_path = args.get('mask_path', None)  # 从参数中获取mask_path
        
        # 使用qwen_image_controlnet.py中的预处理函数
        result = preprocess_for_qwen_image_controlnet(image_path, preprocessor_type, mask_path)
        
        # 修复：正确判断预处理结果是否有效
        if result is not None:
            # 检查numpy数组是否有效
            if isinstance(result, np.ndarray):
                # 检查数组是否非空
                if result.size > 0:
                    # 如果是全零数组，可能表示处理失败
                    if not np.all(result == 0):
                        print("检测到有效的预处理结果（非全零数组）")
                    else:
                        print("警告：预处理结果为全零数组，但仍视为有效结果")
                    
                    # 保存并返回结果
                    outputs_dir = Path(__file__).parent / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    
                    # 使用已导入的time模块，避免重复导入导致的变量遮蔽问题
                    timestamp = int(time.time() * 1000)
                    output_path = outputs_dir / f"preprocess_preview_{timestamp}.png"
                    
                    # 将numpy数组转换为PIL图像并保存
                    if isinstance(result, np.ndarray):
                        # 确保数值在正确范围内
                        if result.dtype != np.uint8:
                            # 归一化到0-255范围
                            result_min = result.min()
                            result_max = result.max()
                            if result_max > result_min:  # 避免除零错误
                                result = ((result - result_min) / (result_max - result_min) * 255).astype(np.uint8)
                            else:
                                result = np.zeros_like(result, dtype=np.uint8)
                        
                        # 转换为PIL图像并保存
                        result_image = Image.fromarray(result)
                        result_image.save(output_path)
                        print(f"SUCCESS:{output_path}")
                        return str(output_path)
                else:
                    print("预处理结果为空数组")
                    return None
            # 如果返回的是PIL图像对象，保存它并输出路径
            elif isinstance(result, Image.Image):
                outputs_dir = Path(__file__).parent / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                
                # 使用已导入的time模块，避免重复导入导致的变量遮蔽问题
                timestamp = int(time.time() * 1000)
                output_path = outputs_dir / f"preprocess_preview_{timestamp}.png"
                result.save(output_path)
                print(f"SUCCESS:{output_path}")
                return str(output_path)
            else:
                # 如果返回的是路径字符串
                print(f"SUCCESS:{result}")
                return result
        else:
            print("预处理失败，返回None")
            return None
            
    except Exception as e:
        print(f"运行预处理控制图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_attention_optimizations(pipe, is_quantized_model=False):
    """应用注意力优化到模型"""
    try:
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 如果是量化模型，跳过优化以避免冲突
            if is_quantized_model:
                return
            # 应用SageAttention或Flash Attention优化
            if SAGE_ATTENTION_AVAILABLE:
                replace_transformer_attention_with_sage(pipe.transformer)
            elif FLASH_ATTENTION_AVAILABLE:
                replace_transformer_attention_with_flash(pipe.transformer)
        else:
            pass  # 不输出日志
    except Exception as e:
        pass  # 不输出日志
