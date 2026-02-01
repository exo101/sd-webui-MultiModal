import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time
import cv2
from diffusers import QwenImagePipeline
import json
from diffusers.utils import load_image
import logging
import warnings
from typing import Union, List, Optional, Dict, Any
from enum import Enum

# 尝试导入SageAttention和Flash Attention
try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# Flash Attention检测
FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass

from backend import attention


def apply_attention_optimizations(pipe):
    """应用注意力优化到模型"""
    try:
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 应用SageAttention或Flash Attention优化
            if SAGE_ATTENTION_AVAILABLE:
                print(f"[INFO] 为模型应用SageAttention优化...")
                replace_transformer_attention_with_sage(pipe.transformer)
            elif FLASH_ATTENTION_AVAILABLE:
                print(f"[INFO] 为模型应用Flash Attention优化...")
                replace_transformer_attention_with_flash(pipe.transformer)
        else:
            print(f"[WARNING] 无法找到pipe.transformer组件，跳过注意力优化")
    except Exception as e:
        print(f"[ERROR] 应用注意力优化失败: {str(e)}")


def replace_transformer_attention_with_sage(transformer):
    """将transformer中的注意力机制替换为SageAttention"""
    try:
        for name, module in transformer.named_modules():
            if 'attn' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # 替换forward方法以使用SageAttention
                original_forward = module.forward
                
                def make_new_forward(orig_forward):
                    def sage_forward(hidden_states, *args, **kwargs):
                        # 原始的query/key/value投影
                        query = orig_forward.__self__.to_q(hidden_states)
                        key = orig_forward.__self__.to_k(hidden_states)
                        value = orig_forward.__self__.to_v(hidden_states)

                        # 确保维度正确
                        batch_size, seq_len, dim = query.shape
                        head_dim = dim // orig_forward.__self__.heads
                        heads = orig_forward.__self__.heads

                        # 重塑为多头形式
                        query = query.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        key = key.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

                        # 使用SageAttention进行计算
                        out = sageattn(query, key, value, 
                                     scale=head_dim**(-0.5), 
                                     attention_dropout=0.0, 
                                     causal=False)
                        
                        # 重塑回原始格式
                        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
                        
                        # 通过输出投影
                        if hasattr(orig_forward.__self__, 'to_out'):
                            if not isinstance(orig_forward.__self__.to_out, (list, tuple)):
                                out = orig_forward.__self__.to_out(out)
                            else:
                                for layer in orig_forward.__self__.to_out:
                                    out = layer(out)
                        
                        return out
                    return sage_forward
                
                module.forward = make_new_forward(original_forward).__get__(module, type(module))
        print("[INFO] SageAttention优化应用成功")
    except Exception as e:
        print(f"[ERROR] 应用SageAttention优化失败: {str(e)}")


def replace_transformer_attention_with_flash(transformer):
    """将transformer中的注意力机制替换为FlashAttention"""
    try:
        import torch.nn.functional as F
        
        for name, module in transformer.named_modules():
            if 'attn' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # 替换forward方法以使用Flash Attention
                original_forward = module.forward
                
                def make_new_forward(orig_forward):
                    def flash_forward(hidden_states, *args, **kwargs):
                        # 原始的query/key/value投影
                        query = orig_forward.__self__.to_q(hidden_states)
                        key = orig_forward.__self__.to_k(hidden_states)
                        value = orig_forward.__self__.to_v(hidden_states)

                        # 确保维度正确
                        batch_size, seq_len, dim = query.shape
                        head_dim = dim // orig_forward.__self__.heads
                        heads = orig_forward.__self__.heads

                        # 重塑为多头形式
                        query = query.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        key = key.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

                        # 尝试使用Flash Attention
                        try:
                            # Flash Attention 2 implementation
                            from flash_attn import flash_attn_func
                            out = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
                        except Exception:
                            # 回退到PyTorch的scaled_dot_product_attention
                            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

                        # 重塑回原始格式
                        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
                        
                        # 通过输出投影
                        if hasattr(orig_forward.__self__, 'to_out'):
                            if not isinstance(orig_forward.__self__.to_out, (list, tuple)):
                                out = orig_forward.__self__.to_out(out)
                            else:
                                for layer in orig_forward.__self__.to_out:
                                    out = layer(out)
                                    
                        return out
                    return flash_forward
                
                module.forward = make_new_forward(original_forward).__get__(module, type(module))
        print("[INFO] Flash Attention优化应用成功")
    except Exception as e:
        print(f"[ERROR] 应用Flash Attention优化失败: {str(e)}")


def preprocess_for_qwen_image_controlnet(image_path, preprocessor_type, mask_path=None):
    """为Qwen-Image-ControlNet预处理图像"""
    try:
        # 检查image_path是否为字符串并且有效
        if image_path is None or (isinstance(image_path, str) and (not image_path or not os.path.exists(image_path))) or (hasattr(image_path, '__len__') and len(image_path) == 0):
            print(f"预处理图像路径无效: {image_path}")
            return None
        
        # 如果image_path是numpy数组或其他非字符串类型，需要特殊处理
        if not isinstance(image_path, str):
            # 如果是numpy数组，说明已经是图像数据，直接处理
            if isinstance(image_path, np.ndarray):
                image = Image.fromarray(image_path.astype('uint8'), 'RGB')
            else:
                print(f"预处理图像路径无效: {image_path}，不是有效的路径或图像数据")
                return None
        else:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
        
        # 使用WebUI的预处理器管理系统
        try:
            # 添加WebUI根目录到系统路径
            webui_root = Path(__file__).parent.parent.parent.parent
            extensions_builtin = webui_root / "extensions-builtin"
            
            paths_to_add = [
                str(webui_root),
                str(extensions_builtin),
                str(extensions_builtin / "forge_preprocessor_inpaint")
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.append(path)
            
            # 导入WebUI的预处理器管理模块
            from modules_forge.shared import supported_preprocessors
            from modules_forge.initialization import initialize_forge
            
            # 初始化Forge系统
            initialize_forge()
            
            # 手动导入inpaint预处理器以确保预处理器被正确加载
            try:
                import forge_preprocessor_inpaint.scripts.preprocessor_inpaint
            except Exception as e:
                # 即使导入失败，也要确保预处理器在supported_preprocessors中
                try:
                    # 尝试直接导入并注册inpaint预处理器
                    from forge_preprocessor_inpaint.scripts.preprocessor_inpaint import PreprocessorInpaintOnly, PreprocessorInpaint, PreprocessorInpaintLama
                    from modules_forge.shared import add_supported_preprocessor
                    
                    # 检查预处理器是否已经注册
                    inpaint_only_registered = False
                    inpaint_global_harmonious_registered = False
                    inpaint_lama_registered = False
                    
                    for name, preprocessor in supported_preprocessors.items():
                        if hasattr(preprocessor, 'name'):
                            if preprocessor.name == 'inpaint_only':
                                inpaint_only_registered = True
                            elif preprocessor.name == 'inpaint_global_harmonious':
                                inpaint_global_harmonious_registered = True
                            elif preprocessor.name == 'inpaint_lama':
                                inpaint_lama_registered = True
                    
                    # 只有在未注册时才添加
                    if not inpaint_only_registered:
                        inpaint_only_preprocessor = PreprocessorInpaintOnly()
                        add_supported_preprocessor(inpaint_only_preprocessor)
                    
                    if not inpaint_global_harmonious_registered:
                        inpaint_preprocessor = PreprocessorInpaint()
                        add_supported_preprocessor(inpaint_preprocessor)
                    
                    if not inpaint_lama_registered:
                        inpaint_lama_preprocessor = PreprocessorInpaintLama()
                        add_supported_preprocessor(inpaint_lama_preprocessor)
                        
                except Exception as manual_register_error:
                    pass
            
            # 手动导入legacy_preprocessors以确保预处理器被正确加载
            try:
                import forge_legacy_preprocessors.scripts.legacy_preprocessors
            except Exception as e:
                pass
            
            # 特殊处理"none"预处理器 - 直接返回原始图像
            if preprocessor_type.lower() in ["none", "无", "none (default)"]:
                if isinstance(image, np.ndarray):
                    return np.array(image)
                else:
                    return np.array(image)
            
            # 先尝试使用预处理器映射表来转换UI显示名称到内部标识符
            # 从同目录下的config模块导入映射表
            internal_preprocessor_type = preprocessor_type  # 默认使用原始类型
            try:
                # 使用已导入的sys，不再重复导入
                current_dir = os.path.dirname(__file__)
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                
                # 尝试从config模块导入映射表
                from config import CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL
                internal_preprocessor_type = CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_type, preprocessor_type)
            except ImportError:
                # 如果找不到映射，则直接使用原名称
                internal_preprocessor_type = preprocessor_type
            
            # 获取预处理器对象
            preprocessor = supported_preprocessors.get(internal_preprocessor_type)
            if preprocessor is None:
                # 处理带前缀的预处理器名称，如"[Pose] dw_openpose_full"
                clean_preprocessor_type = preprocessor_type
                if isinstance(clean_preprocessor_type, str) and "]" in clean_preprocessor_type:
                    # 去除"[Pose] "这样的前缀
                    clean_preprocessor_type = clean_preprocessor_type.split("]", 1)[1].strip()
                
                # 尝试不同的命名变体
                variants = [
                    clean_preprocessor_type,
                    clean_preprocessor_type.lower(),
                    clean_preprocessor_type.lower().replace(" ", "_"),
                    clean_preprocessor_type.lower().replace("-", "_"),
                    clean_preprocessor_type.replace("-", "_"),
                    clean_preprocessor_type.replace(" ", "_")
                ]
                
                # 添加更多常见的预处理器名称变体
                common_variants = {
                    "dw_openpose_full": ["openpose_full", "dw openpose full", "openpose_full", "dw-openpose-full"],
                    "openpose_full": ["dw_openpose_full", "dw openpose full", "dw-openpose-full"],
                    "depth_midas": ["midas", "depth_midas", "depth-midas"],
                    "depth_anything_v2": ["depth_anything", "depth anything v2", "depth-anything-v2"],
                    "softedge_hed": ["hed", "softedge_hed", "softedge-hed"],
                    "lineart_standard": ["lineart", "lineart_standard", "lineart-standard"],
                    "lineart_realistic": ["lineart_realistic", "lineart-realistic"],
                    "lineart_anime_denoise": ["lineart_anime", "lineart-anime-denoise", "lineart_anime_denoise"],
                    "canny": ["canny"]
                }
                
                # 如果当前预处理器类型在常见变体映射中，添加这些变体
                if clean_preprocessor_type in common_variants:
                    variants.extend(common_variants[clean_preprocessor_type])
                
                for variant in variants:
                    if variant in supported_preprocessors:
                        preprocessor = supported_preprocessors[variant]
                        print(f"找到预处理器 '{variant}' 替代 '{preprocessor_type}'")
                        break
            
            # 特殊处理"inpaint_only"预处理器名称
            # 在某些情况下，用户可能使用"Inpaint Only"而不是"inpaint_only"
            clean_preprocessor_type = preprocessor_type
            if isinstance(clean_preprocessor_type, str) and "]" in clean_preprocessor_type:
                # 去除"[Pose] "这样的前缀
                clean_preprocessor_type = clean_preprocessor_type.split("]", 1)[1].strip()
            
            if preprocessor is None and clean_preprocessor_type.lower().replace(" ", "_") in ["inpaint_only", "inpaintonly"]:
                # 尝试查找"inpaint_only"
                if "inpaint_only" in supported_preprocessors:
                    preprocessor = supported_preprocessors["inpaint_only"]
                    print(f"找到预处理器 'inpaint_only' 替代 '{preprocessor_type}'")
            
            # 如果还是找不到，直接报错
            if preprocessor is None:
                print(f"错误：未找到预处理器 {preprocessor_type}")
                raise ValueError(f"未找到预处理器: {preprocessor_type}，请检查预处理器名称是否正确")
            
            # 使用预处理器处理图像
            # 注意：WebUI预处理器通常接受RGB格式的numpy数组，值范围为0-255
            if isinstance(image, np.ndarray):
                # 如果图像是numpy数组格式
                image_array = image
            else:
                # 如果图像是PIL Image格式
                image_array = np.array(image)
            
            # 确保图像是RGB格式
            if len(image_array.shape) == 2:
                # 灰度图转RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                # RGBA转RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] == 3:
                # 已经是RGB格式
                pass
            else:
                # 其他情况，假设是BGR格式转RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # 调用预处理器处理图像
            # 注意：不同的预处理器可能有不同的参数要求
            try:
                # 尝试以不同方式调用预处理器
                if hasattr(preprocessor, '__call__'):
                    # 大多数预处理器是可调用对象
                    # 检查预处理器需要的参数并提供默认值
                    import inspect
                    sig = inspect.signature(preprocessor.__call__)
                    kwargs = {}
                    
                    # 为常见参数提供默认值
                    if 'resolution' in sig.parameters:
                        kwargs['resolution'] = min(image_array.shape[0], image_array.shape[1])
                    if 'slider_1' in sig.parameters:
                        # 对于Canny预处理器，slider_1是低阈值
                        kwargs['slider_1'] = 100 if preprocessor.name == 'canny' else None
                    if 'slider_2' in sig.parameters:
                        # 对于Canny预处理器，slider_2是高阈值
                        kwargs['slider_2'] = 200 if preprocessor.name == 'canny' else None
                    if 'slider_3' in sig.parameters:
                        kwargs['slider_3'] = None
                    
                    # 特殊处理inpaint_only预处理器，它需要input_mask参数
                    if preprocessor.name == "inpaint_only":
                        # 对于inpaint_only，我们需要提供蒙版图像
                        if mask_path and os.path.exists(mask_path):
                            # 如果提供了蒙版路径，直接使用
                            mask_image = Image.open(mask_path).convert("RGB")
                            mask_array = np.array(mask_image)
                            # 确保蒙版是单通道的
                            if len(mask_array.shape) == 3:
                                # 将RGB蒙版转换为灰度图
                                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                            # 添加input_mask参数
                            kwargs['input_mask'] = mask_array
                            print(f"为inpaint_only预处理器提供了蒙版图像: {mask_path}")
                        else:
                            # 检查是否是ForgeCanvas格式的图像（包含背景和前景）
                            # 如果image_path是包含蒙版信息的图像，则从中提取蒙版
                            try:
                                # 尝试从图像的alpha通道获取蒙版
                                pil_image = Image.open(image_path)
                                if pil_image.mode == 'RGBA':
                                    # 从alpha通道提取蒙版
                                    alpha_channel = np.array(pil_image)[:, :, 3]
                                    kwargs['input_mask'] = alpha_channel
                                    print("从RGBA图像的alpha通道提取蒙版用于inpaint_only预处理器")
                                else:
                                    # 检查是否是灰度图作为蒙版
                                    if len(np.array(pil_image).shape) == 2:
                                        kwargs['input_mask'] = np.array(pil_image)
                                        print("使用灰度图作为蒙版用于inpaint_only预处理器")
                                    else:
                                        # 尝试从图像路径中提取蒙版信息（假设是ControlNet绘制的蒙版）
                                        # 对于inpaint_only预处理器，直接返回原始图像，因为真正的处理在扩散过程中进行
                                        print("inpaint_only预处理器直接返回原始图像，真正的处理将在扩散过程中进行")
                                        return image_array
                            except Exception as e:
                                print(f"尝试从图像提取蒙版时出错: {e}")
                                # 对于inpaint_only预处理器，如果没有蒙版，直接返回原始图像
                                print("inpaint_only预处理器未提供有效蒙版，直接返回原始图像")
                                return image_array
                    
                    # 确保所有数字参数都不是None
                    if 'slider_1' in kwargs and kwargs['slider_1'] is None:
                        # 检查预处理器是否需要特定的默认值
                        if hasattr(preprocessor, 'slider_1') and preprocessor.slider_1 is not None:
                            if hasattr(preprocessor.slider_1, 'gradio_update_kwargs'):
                                kwargs['slider_1'] = preprocessor.slider_1.gradio_update_kwargs.get('value', 0)
                        else:
                            kwargs['slider_1'] = 0
                    
                    if 'slider_2' in kwargs and kwargs['slider_2'] is None:
                        # 检查预处理器是否需要特定的默认值
                        if hasattr(preprocessor, 'slider_2') and preprocessor.slider_2 is not None:
                            if hasattr(preprocessor.slider_2, 'gradio_update_kwargs'):
                                kwargs['slider_2'] = preprocessor.slider_2.gradio_update_kwargs.get('value', 0)
                        else:
                            kwargs['slider_2'] = 0
                    
                    processed_image_array = preprocessor(image_array, **kwargs)
                else:
                    # 一些预处理器可能需要特殊的调用方式
                    processed_image_array = preprocessor(image_array)
                
                # 确保输出是正确的格式
                if isinstance(processed_image_array, tuple):
                    # 有些预处理器返回元组，第一个元素是图像
                    processed_image_array = processed_image_array[0]
                
                # 确保输出是numpy数组
                if not isinstance(processed_image_array, np.ndarray):
                    raise ValueError(f"预处理器返回了意外的类型: {type(processed_image_array)}")
                
                # 检查输出是否为空
                if processed_image_array.size == 0:
                    raise ValueError("预处理器返回了空数组")
                
                # 确保输出是3通道RGB图像
                if len(processed_image_array.shape) == 2:
                    # 灰度图转RGB
                    processed_image = cv2.cvtColor(processed_image_array, cv2.COLOR_GRAY2RGB)
                elif processed_image_array.shape[2] == 1:
                    # 单通道转RGB
                    processed_image = cv2.cvtColor(processed_image_array.squeeze(), cv2.COLOR_GRAY2RGB)
                elif processed_image_array.shape[2] == 3:
                    # 已经是RGB格式
                    processed_image = processed_image_array
                elif processed_image_array.shape[2] == 4:
                    # RGBA转RGB
                    processed_image = cv2.cvtColor(processed_image_array, cv2.COLOR_RGBA2RGB)
                else:
                    # 其他情况，默认使用原始输出
                    processed_image = processed_image_array
                
                # 确保输出数组是非空的
                if processed_image is not None and processed_image.size > 0:
                    return processed_image
                else:
                    print("预处理器返回了空结果")
                    return None
                
            except Exception as process_error:
                print(f"使用WebUI预处理器时出错: {process_error}")
                import traceback
                traceback.print_exc()
                # 出错时不再回退，直接抛出异常
                raise
            
        except Exception as e:
            print(f"使用WebUI预处理器时出错: {e}")
            import traceback
            traceback.print_exc()
            # 不再回退到默认处理，直接抛出异常
            raise
        
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
        # 不再返回None，直接抛出异常
        raise
