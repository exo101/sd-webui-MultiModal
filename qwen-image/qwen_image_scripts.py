#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
import warnings

# 保存原始的标准输出和错误流
original_stdout = sys.stdout
original_stderr = sys.stderr
original_showwarning = warnings.showwarning

# 创建StringIO对象来捕获输出
captured_stdout = io.StringIO()
captured_stderr = io.StringIO()

# 重定向标准输出和错误流
sys.stdout = captured_stdout
sys.stderr = captured_stderr

# 重定向警告
def ignore_warnings(*args, **kwargs):
    pass

warnings.showwarning = ignore_warnings

# 设置日志级别以减少输出
import logging
logging.getLogger().setLevel(logging.CRITICAL)

# ==================== 导入模块 ====================
import json
import os
import copy
from pathlib import Path
import torch
import math
from safetensors.torch import load_file as load_state_dict_in_safetensors
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import FlowMatchHeunDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
import time
import psutil
import gc
import cv2
import numpy as np
from PIL import Image

# ==================== 预处理器导入和可用性检查 ====================
# 尝试导入WebUI的ControlNet预处理器
PREPROCESSORS_AVAILABLE = False
try:
    # 添加WebUI根目录和相关路径到系统路径中
    webui_root = Path(__file__).parent.parent.parent.parent
    extensions_builtin = webui_root / "extensions-builtin"
    forge_preprocessors = extensions_builtin / "forge_legacy_preprocessors"
    
    # 添加必要的路径
    paths_to_add = [
        str(webui_root),
        str(extensions_builtin),
        str(forge_preprocessors)
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
    
    # 尝试导入ControlNet预处理器
    try:
        from annotator.hed import apply_hed as HEDdetectorImported
        from annotator.midas import apply_midas as MidasDetectorImported
        from annotator.openpose import OpenposeDetector as OpenposeDetectorImported
        from annotator.canny import apply_canny as CannyDetectorImported
        from annotator.depth_anything_v2 import DepthAnythingV2Detector as DepthAnythingV2DetectorImported
        from annotator.lineart import LineartDetector as LineartDetectorImported
        from annotator.lineart_anime import LineartAnimeDetector as LineartAnimeDetectorImported
        PREPROCESSORS_AVAILABLE = True
    except ImportError:
        # 尝试从forge_legacy_preprocessors导入
        try:
            from forge_legacy_preprocessors.annotator.hed import apply_hed as HEDdetectorImported
            from forge_legacy_preprocessors.annotator.midas import apply_midas as MidasDetectorImported
            from forge_legacy_preprocessors.annotator.openpose import OpenposeDetector as OpenposeDetectorImported
            from forge_legacy_preprocessors.annotator.canny import apply_canny as CannyDetectorImported
            from forge_legacy_preprocessors.annotator.depth_anything_v2 import DepthAnythingV2Detector as DepthAnythingV2DetectorImported
            from forge_legacy_preprocessors.annotator.lineart import LineartDetector as LineartDetectorImported
            from forge_legacy_preprocessors.annotator.lineart_anime import LineartAnimeDetector as LineartAnimeDetectorImported
            PREPROCESSORS_AVAILABLE = True
        except ImportError:
            PREPROCESSORS_AVAILABLE = False

except Exception:
    PREPROCESSORS_AVAILABLE = False

# ==================== ControlNet 可用性检查 ====================
# 尝试导入ControlNet模型
CONTROLNET_AVAILABLE = False
try:
    from diffusers.models import QwenImageControlNetModel
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False


# ==================== 预处理器函数 ====================
def apply_canny(image, low_threshold=100, high_threshold=200):
    """应用Canny边缘检测"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        # 注意：canny预处理器是一个函数而不是类
        result = CannyDetectorImported(image, low_threshold, high_threshold)
        # 确保输出为3通道图像
        if len(result.shape) == 2:
            result = result[:, :, None]
            result = np.concatenate([result, result, result], axis=2)
        return result
    else:
        # 回退到简化版本
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        # 确保输出为3通道图像
        if len(edges.shape) == 2:
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
        return edges

def apply_depth(image):
    """应用深度估计"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        # 注意：MidasDetector是一个函数而不是类
        result, _ = MidasDetectorImported(image)
        return result
    else:
        # 回退到简化版本
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # 扩展为3通道
        if len(gray.shape) == 2:
            gray = gray[:, :, None]
            gray = np.concatenate([gray, gray, gray], axis=2)
        return gray

def apply_depth_anything_v2(image):
    """应用Depth Anything V2深度估计"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        # DepthAnythingV2Detector是一个类，需要实例化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = DepthAnythingV2DetectorImported(device)
        result = processor(image, colored=False)  # WebUI中使用colored=False
        return result
    else:
        # 回退到普通深度估计
        return apply_depth(image)

def apply_pose(image, include_body=True, include_hand=False, include_face=False, use_dw_pose=False):
    """应用姿态检测"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        processor = OpenposeDetectorImported()
        # DWPose是OpenPose的一个特殊模式，通过use_dw_pose参数控制
        result = processor(image, include_body=include_body, include_hand=include_hand, include_face=include_face, use_dw_pose=use_dw_pose)
        return result
    else:
        return image

def apply_openpose_full(image):
    """应用完整姿态检测（包含手部和面部）"""
    return apply_pose(image, include_body=True, include_hand=True, include_face=True)

def apply_openpose_hand(image):
    """应用手部姿态检测"""
    return apply_pose(image, include_body=False, include_hand=True, include_face=False)

def apply_openpose_face(image):
    """应用面部姿态检测"""
    return apply_pose(image, include_body=False, include_hand=False, include_face=True)

def apply_softedge(image):
    """应用软边缘检测"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        # 注意：HEDdetector是一个函数而不是类
        result = HEDdetectorImported(image)
        # 确保输出为3通道图像
        if len(result.shape) == 2:
            result = result[:, :, None]
            result = np.concatenate([result, result, result], axis=2)
        return result
    else:
        # 回退到简化版本
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        # 扩展为3通道
        if len(edges.shape) == 2:
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
        return edges

def apply_lineart_standard(image):
    """应用标准线稿检测"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        processor = LineartDetectorImported(LineartDetectorImported.model_default)
        result = 255 - processor(image)  # WebUI中使用反转
        return result
    else:
        # 回退到软边缘检测
        return apply_softedge(image)

def apply_lineart_realistic(image):
    """应用写实线稿检测"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        processor = LineartDetectorImported(LineartDetectorImported.model_realistic)
        result = 255 - processor(image)  # WebUI中使用反转
        return result
    else:
        # 回退到标准线稿检测
        return apply_lineart_standard(image)

def apply_lineart_anime_denoise(image):
    """应用动漫线稿去噪检测"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if PREPROCESSORS_AVAILABLE:
        processor = LineartAnimeDetectorImported()
        result = 255 - processor(image)  # WebUI中使用反转
        return result
    else:
        # 回退到标准线稿检测
        return apply_lineart_standard(image)

# ==================== 控制图像预处理函数 ====================
def preprocess_control_image(image_path, preprocessor_type, mask_path=None):
    """预处理控制图像，使用WebUI内置的ControlNet预处理器，支持蒙版"""
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 如果提供了蒙版路径，加载并应用蒙版
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")  # 转换为灰度图作为蒙版
            # 调整蒙版大小以匹配图像
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            # 将蒙版应用到图像上
            # 创建一个透明图层，然后将原图和蒙版合并
            image_array = np.array(image)
            mask_array = np.array(mask)
            
            # 使用蒙版调整图像的亮度或直接混合
            # 这里采用简单的混合方式：图像像素值乘以(蒙版值/255)
            masked_array = (image_array * (mask_array[:, :, np.newaxis] / 255.0)).astype(np.uint8)
            image = Image.fromarray(masked_array, mode="RGB")
        
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
            except Exception:
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
                        
                except Exception:
                    pass
            
            # 手动导入legacy_preprocessors以确保预处理器被正确加载
            try:
                import forge_legacy_preprocessors.scripts.legacy_preprocessors
            except Exception:
                pass
            
            # 特殊处理"none"预处理器 - 直接返回原始图像
            if preprocessor_type.lower() in ["none", "无", "none (default)"]:
                if isinstance(image, np.ndarray):
                    return image
                else:
                    return np.array(image)
            
            # 处理带前缀的预处理器名称，如"[Pose] openpose"
            clean_preprocessor_type = preprocessor_type
            if isinstance(clean_preprocessor_type, str) and "]" in clean_preprocessor_type:
                # 去除"[Pose] "这样的前缀
                clean_preprocessor_type = clean_preprocessor_type.split("]", 1)[1].strip()
            
            # 获取预处理器对象
            preprocessor = supported_preprocessors.get(clean_preprocessor_type)
            if preprocessor is None:
                # 尝试不同的命名变体
                variants = [
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
                    "openpose": ["openpose_full", "dw_openpose_full"],
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
                        break
            
            # 特殊处理"inpaint_only"预处理器名称
            # 在某些情况下，用户可能使用"Inpaint Only"而不是"inpaint_only"
            if preprocessor is None and clean_preprocessor_type.lower().replace(" ", "_") in ["inpaint_only", "inpaintonly"]:
                # 尝试查找"inpaint_only"
                if "inpaint_only" in supported_preprocessors:
                    preprocessor = supported_preprocessors["inpaint_only"]
            
            # 如果还是找不到，直接报错而不是回退到canny
            if preprocessor is None:
                raise ValueError(f"未找到预处理器: {preprocessor_type}，请检查预处理器名称是否正确")
            
            # 使用预处理器处理图像
            # 注意：WebUI预处理器通常接受RGB格式的numpy数组，值范围为0-255
            
            # 确保图像数据是正确的格式
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
                                else:
                                    # 检查是否是灰度图作为蒙版
                                    if len(np.array(pil_image).shape) == 2:
                                        kwargs['input_mask'] = np.array(pil_image)
                                    else:
                                        # 对于inpaint_only预处理器，直接返回原始图像，因为真正的处理在扩散过程中进行
                                        return image_array
                            except Exception:
                                # 对于inpaint_only预处理器，如果没有蒙版，直接返回原始图像
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
                
                # 确保数组包含有效数据
                if np.isnan(processed_image_array).any() or np.isinf(processed_image_array).any():
                    raise ValueError("预处理器返回的数组包含NaN或Inf值")
                
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
                    # 其他情况，假定已经是正确的RGB格式
                    processed_image = processed_image_array
                
                # 转换为PIL Image对象
                processed_image_pil = Image.fromarray(processed_image, 'RGB')
                
                # 修复：保持原始图像的宽高比，避免裁剪
                # 获取原始图像尺寸
                orig_width, orig_height = processed_image_pil.size
                
                # 不改变图像内容，只确保格式正确
                return processed_image
            except Exception:
                # 出错时不再回退，直接抛出异常
                raise
            
        except Exception:
            # 不再回退到默认处理，直接抛出异常
            raise
        
    except Exception:
        # 不再返回None，直接抛出异常
        raise

# ==================== 预处理控制图像主函数 ====================
def run_preprocess_control_image(args_file):
    """运行预处理控制图像的主函数"""
    try:
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 获取参数
        image_path = args.get('image_path')
        preprocessor_type = args.get('preprocessor_type')
        
        # 预处理图像
        result = preprocess_control_image(image_path, preprocessor_type)
        
        # 修复：正确判断预处理结果是否有效
        if result is not None:
            # 检查numpy数组是否有效
            if isinstance(result, np.ndarray):
                # 检查数组是否非空
                if result.size > 0:
                    # 保存并返回结果
                    outputs_dir = Path(__file__).parent / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    
                    import time
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
                        
                    # 准备成功信息
                    success_msg = f"SUCCESS:{output_path}"
                    # 在 finally 块之前，通过一个统一的位置输出所有消息
                    _print_captured_output([success_msg])
                    return str(output_path)
                else:
                    return None
            # 如果返回的是PIL图像对象，保存它并输出路径
            elif isinstance(result, Image.Image):
                outputs_dir = Path(__file__).parent / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                
                import time
                timestamp = int(time.time() * 1000)
                output_path = outputs_dir / f"preprocess_preview_{timestamp}.png"
                result.save(output_path)
                # 准备成功信息
                success_msg = f"SUCCESS:{output_path}"
                # 在 finally 块之前，通过一个统一的位置输出所有消息
                _print_captured_output([success_msg])
                return str(output_path)
            else:
                # 如果返回的是路径字符串
                # 准备成功信息
                success_msg = f"SUCCESS:{result}"
                # 在 finally 块之前，通过一个统一的位置输出所有消息
                _print_captured_output([success_msg])
                return result
        else:
            return None
            
    except Exception as e:
        # 将错误信息传递给统一的输出函数
        _print_captured_output([f"运行预处理控制图像时出错: {e}"])
        return None
    finally:
        # 恢复原始的标准输出和错误流
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        warnings.showwarning = original_showwarning

# ==================== 文生图功能 ====================
def _print_captured_output(messages):
    """
    统一处理被捕获的输出，仅在需要时恢复原始stdout进行打印。
    :param messages: 要输出的消息列表
    """
    # 临时恢复原始stdout以打印消息
    temp_stdout = sys.stdout
    sys.stdout = original_stdout
    try:
        for msg in messages:
            print(msg)
    finally:
        # 立刻将stdout重新定向回捕获对象
        sys.stdout = temp_stdout

def run_text_to_image(args_file):
    """运行文生图功能"""
    try:
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 获取参数
        prompt = args["prompt"]
        negative_prompt = args.get("negative_prompt", "")
        width = args["width"]
        height = args["height"]
        steps = args["steps"]
        cfg_scale = args["cfg_scale"]
        scheduler_type = args["scheduler"]
        
        # 导入必要的库
        from diffusers import QwenImagePipeline
        # 使用稳健的方式导入Transformer模型，优先使用支持LoRA的版本
        LightningTransformer = None
        try:
            # 首先尝试导入支持LoRA的nunchaku版本
            from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as LightningTransformer
        except (ImportError, ModuleNotFoundError):
            try:
                # 回退到diffusers标准版本
                from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as LightningTransformer
            except Exception:
                LightningTransformer = None
        
        if LightningTransformer is None:
            return
            
        from nunchaku.utils import get_gpu_memory, get_precision
        from PIL import Image
        
        # 获取用户选择的采样方法
        scheduler_type = args.get("scheduler", "euler")
        
        # Scheduler 配置
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        
        # 根据用户选择创建相应的调度器
        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "euler_ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "dpmpp_2m":
            # DPM++ 2M 调度器配置稍有不同
            dpm_config = scheduler_config.copy()
            dpm_config.update({
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
            })
            scheduler = DPMSolverMultistepScheduler.from_config(dpm_config)
        else:
            # 默认使用 Euler 调度器
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        
        # 获取模型路径
        # 修复：使用传递的model_dir参数而不是硬编码路径
        model_dir = args.get("model_dir")
        if model_dir:
            qwenimage_models_dir = Path(model_dir)
        else:
            # 回退到默认路径
            models_dir = Path(__file__).parent / "models"
            qwenimage_models_dir = models_dir / "qwenimage"
        steps = args["steps"]
        
        # 定义torch_dtype
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # 获取用户选择的模型文件
        model_file = args.get("model_file")
        if model_file:
            # 使用用户选择的模型文件
            model_path = qwenimage_models_dir / model_file
        else:
            # 默认使用第一个模型文件
            model_files = list(qwenimage_models_dir.glob("*.safetensors"))
            if model_files:
                model_path = model_files[0]
            else:
                model_path = None
        
        # 检查模型文件是否存在
        if not model_path or not model_path.exists():
            return
        
        # 添加ControlNet相关路径到系统路径
        controlnet_path = Path(__file__).parent / "ControlNet" / "Qwen-Image-ControlNet-Union"
        if str(controlnet_path) not in sys.path:
            sys.path.append(str(controlnet_path))
        
        # 获取ControlNet相关参数
        controlnet_model_selected = args.get("controlnet_model", "无")
        control_image_path = args.get("control_image")
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        controlnet_enable = (control_image_path is not None 
                           and control_image_path != "" 
                           and controlnet_model_selected != "无" 
                           and PREPROCESSORS_AVAILABLE)

        # 加载ControlNet模型
        controlnet = None
        if controlnet_enable:
            try:
                # 加载ControlNet模型
                if controlnet_model_selected and controlnet_model_selected != "无":
                    # 检查是否为本地路径
                    controlnet_base_path = Path(__file__).parent.parent.parent.parent / "models" / "ControlNet"
                    model_name = controlnet_model_selected.split('/')[-1] if '/' in controlnet_model_selected else controlnet_model_selected
                    controlnet_local_path = controlnet_base_path / model_name
                    
                    # 确保目录存在
                    controlnet_local_path.mkdir(parents=True, exist_ok=True)
                    
                    if controlnet_local_path and (controlnet_local_path / "config.json").exists():
                        controlnet = QwenImageControlNetModel.from_pretrained(
                            str(controlnet_local_path),
                            torch_dtype=torch_dtype
                        )
                    else:
                        controlnet = None
                        controlnet_enable = False
                        
                    if controlnet is not None:
                        pass
                else:
                    controlnet = None
                    controlnet_enable = False
            except Exception:
                controlnet = None
                controlnet_enable = False
        else:
            controlnet = None

        # 加载模型
        transformer = None
        pipe = None

        # 直接使用nunchaku的正确加载方式
        try:
            # 导入相应的类
            from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
            
            # 检查模型路径
            if model_path is None:
                raise ValueError("模型路径为None")
            
            # 检查模型文件是否存在且可读
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 检查是否为特殊的大模型文件 qwen_2.5_vl_7b_fp8_scaled.safetensors
            if model_path.name == "qwen_2.5_vl_7b_fp8_scaled.safetensors":
                # 对于这个特殊的大模型文件，我们使用不同的加载方式
                transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                    str(model_path),  # 直接使用文件路径
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    max_memory={0: "15GB"}  # 限制GPU内存使用
                )
            else:
                # 尝试加载transformer
                transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                    str(model_path),
                    torch_dtype=torch_dtype
                )

            # 添加ControlNet相关路径到系统路径
            controlnet_path = Path(__file__).parent / "ControlNet" / "Qwen-Image-ControlNet-Union"
            if str(controlnet_path) not in sys.path:
                sys.path.append(str(controlnet_path))
            
            # 使用模型根目录作为基础路径，而不是模型文件所在子目录
            # 模型根目录包含model_index.json和其他必要组件
            base_model_path = str(model_path.parent.parent)  # models/qwen-image
            
            # 检查text_encoder组件是否存在，如果不存在则尝试使用替代方案
            text_encoder_path = Path(base_model_path) / "text_encoder"
            text_encoder_missing = False
            if not text_encoder_path.exists():
                text_encoder_missing = True
            elif not (text_encoder_path / "model-00001-of-00004.safetensors").exists():
                text_encoder_missing = True
            
            # 如果text_encoder组件缺失，检查是否存在qwen_2.5_vl_7b_fp8_scaled.safetensors作为替代
            use_alternative_text_encoder = False
            alternative_text_encoder_model_path = None
            if text_encoder_missing:
                alternative_text_encoder = text_encoder_path / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
                if alternative_text_encoder.exists():
                    use_alternative_text_encoder = True
                    alternative_text_encoder_model_path = str(alternative_text_encoder)
            
            if controlnet_enable and controlnet is not None:
                try:
                    from diffusers import QwenImageControlNetPipeline
                    # 创建ControlNet Pipeline，使用模型根目录作为基础路径
                    # 注意：from_pretrained 方法可能不直接接受 torch_dtype 参数
                    pipe = QwenImageControlNetPipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        controlnet=controlnet,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
                    # 将整个pipeline移动到指定的数据类型
                    if torch_dtype != pipe.transformer.dtype:
                        pipe.to(torch_dtype)
                except Exception:
                    # 回退到标准QwenImagePipeline管道
                    from diffusers import QwenImagePipeline
                    
                    # 检查是否需要使用替代的text_encoder
                    if use_alternative_text_encoder and alternative_text_encoder_model_path:
                        # 手动加载各组件然后创建Pipeline
                        # 从完整模型路径加载其他组件，但排除text_encoder
                        pipe = QwenImagePipeline.from_pretrained(
                            base_model_path,
                            transformer=transformer,
                            scheduler=scheduler,
                            torch_dtype=torch_dtype,
                            text_encoder=None  # 不加载text_encoder
                        )
                        
                        # 手动加载替代的text_encoder
                        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
                        
                        # 根据项目规范，使用from_pretrained方法加载模型
                        # 并使用父目录作为路径，同时启用内存优化参数
                        text_encoder_alt = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            str(Path(alternative_text_encoder_model_path).parent),  # 使用父目录作为路径
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                            max_memory={0: "15GB"}
                        )
                        
                        # 加载tokenizer（从原始路径）
                        tokenizer = Qwen2Tokenizer.from_pretrained(
                            base_model_path,
                            subfolder="tokenizer"
                        )
                        
                        # 手动设置组件
                        pipe.text_encoder = text_encoder_alt
                        pipe.tokenizer = tokenizer
                        
                    else:
                        pipe = QwenImagePipeline.from_pretrained(
                            base_model_path,
                            transformer=transformer,
                            scheduler=scheduler,
                            torch_dtype=torch_dtype
                        )
                    
                    # 将整个pipeline移动到指定的数据类型
                    if torch_dtype != pipe.transformer.dtype:
                        pipe.to(torch_dtype)
                    controlnet_enable = False
            else:
                from diffusers import QwenImagePipeline
                
                # 检查是否需要使用替代的text_encoder
                if use_alternative_text_encoder and alternative_text_encoder_model_path:
                    # 手动加载各组件然后创建Pipeline
                    # 从完整模型路径加载其他组件，但排除text_encoder
                    pipe = QwenImagePipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype,
                        text_encoder=None  # 不加载text_encoder
                    )
                    
                    # 手动加载替代的text_encoder
                    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
                    
                    # 根据项目规范，使用from_pretrained方法加载模型
                    # 并使用父目录作为路径，同时启用内存优化参数
                    text_encoder_alt = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        str(Path(alternative_text_encoder_model_path).parent),  # 使用父目录作为路径
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        max_memory={0: "15GB"}
                    )
                    
                    # 加载tokenizer（从原始路径）
                    tokenizer = Qwen2Tokenizer.from_pretrained(
                        base_model_path,
                        subfolder="tokenizer"
                    )
                    
                    # 手动设置组件
                    pipe.text_encoder = text_encoder_alt
                    pipe.tokenizer = tokenizer
                    
                else:
                    pipe = QwenImagePipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
                
                # 将整个pipeline移动到指定的数据类型
                if torch_dtype != pipe.transformer.dtype:
                    pipe.to(torch_dtype)
                    
        except Exception:
            # 确保在下一次尝试前清理可能损坏的对象
            transformer = None
            pipe = None
            return
        
        # 设置模型卸载
        from nunchaku.utils import get_gpu_memory
        if get_gpu_memory() > 18:
            pipe.enable_model_cpu_offload()
        else:
            if transformer is not None:
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
        
        # 获取随机种子
        seed = args.get("seed", -1)
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # 创建生成器
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        
        # 处理ControlNet相关参数
        control_image_path = args.get("control_image")
        controlnet_conditioning_scale = args.get("controlnet_conditioning_scale", 1.0)
        controlnet_preprocessor = args.get("controlnet_preprocessor", "none")
        controlnet_start = args.get("controlnet_start", 0.0)
        controlnet_end = args.get("controlnet_end", 1.0)
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        controlnet_model_selected = args.get("controlnet_model", "无")
        controlnet_enable = (control_image_path is not None 
                           and control_image_path != "" 
                           and controlnet_model_selected != "无")
        
        if controlnet_enable and control_image_path:
            # 预处理控制图像
            processed_control_image = preprocess_control_image(control_image_path, controlnet_preprocessor)
            if processed_control_image is None:
                controlnet_enable = False
            else:
                # 再次确保图像是RGB模式
                # 检查是numpy数组还是PIL图像
                if isinstance(processed_control_image, np.ndarray):
                    # 如果是numpy数组，先转换为PIL图像
                    processed_control_image = Image.fromarray(processed_control_image)
                
                # 现在确保是PIL图像并转换为RGB模式
                if processed_control_image.mode != 'RGB':
                    processed_control_image = processed_control_image.convert('RGB')
        else:
            processed_control_image = None
            controlnet_enable = False

        # 准备生成参数
        generation_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg_scale,
            "generator": generator,
        }
        
        # 获取生成批次大小并添加到生成参数中
        batch_size = args.get("batch_size", 1)
        generation_params["num_images_per_prompt"] = batch_size
        
        # 如果启用了ControlNet，添加ControlNet相关参数
        if controlnet_enable and controlnet is not None and processed_control_image is not None:
            generation_params.update({
                "control_image": processed_control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": controlnet_start,
                "control_guidance_end": controlnet_end,
            })
        else:
            pass

        # 处理LoRA模型
        lora_model_1 = args.get("lora_model_1")
        lora_model_2 = args.get("lora_model_2")
        lora_weight_1 = args.get("lora_weight_1", 1.0)
        lora_weight_2 = args.get("lora_weight_2", 1.0)
        
        # 添加nunchaku目录到sys.path
        nunchaku_path = Path(__file__).parent.parent / "nunchaku-2_lora_concat"
        if str(nunchaku_path) not in sys.path:
            sys.path.insert(0, str(nunchaku_path))  # 插入到路径开头确保优先级
        
        # 尝试加载LoRA模型
        try:
            def load_lora_model(model_path, weight, model_name, pipeline):
                if not model_path or model_path == "":
                    return False
                
                try:
                    # 直接从已复制到diffusers库中的nunchaku导入LoRA模块
                    from nunchaku.lora.flux.v1.lora_flux_v2 import update_lora_params_v2, set_lora_strength_v2, reset_lora_v2
                    
                    # 加载LoRA权重
                    lora_state_dict = load_state_dict_in_safetensors(model_path)
                    if lora_state_dict:
                        try:
                            # 尝试应用LoRA权重
                            # 首先检查LoRA模型与当前模型的兼容性
                            update_lora_params_v2(pipeline.transformer, lora_state_dict, strength=weight, allow_expand=True)
                            return True
                        except RuntimeError as e:
                            if "size of tensor" in str(e) and "must match" in str(e):
                                # 尝试重置LoRA（如果可能）
                                try:
                                    reset_lora_v2(pipeline.transformer)
                                except:
                                    pass

                                return False
                            elif "unexpected" in str(e):
                                return False
                            else:
                                raise e
                        except ValueError as e:
                            if "mismatch" in str(e).lower() or "shape" in str(e).lower():
                                return False

                            else:
                                raise e
                        except AttributeError:
                            return False
                        except Exception:
                            return False
                    else:
                        return False
                except Exception:
                    return False
            
            # 确保pipeline已定义后再加载LoRA模型
            if 'pipe' in locals() and pipe is not None:
                # 加载两个LoRA模型
                if lora_model_1:
                    load_lora_model(lora_model_1, lora_weight_1, "1", pipe)
                    
                if lora_model_2:
                    load_lora_model(lora_model_2, lora_weight_2, "2", pipe)
                
                # 重新初始化CPU卸载管理器以适应LoRA加载后模型参数维度的变化
                if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'offload') and pipe.transformer.offload:
                    try:
                        # 保存当前的卸载设置
                        use_pin_memory = getattr(pipe.transformer.offload_manager, 'use_pin_memory', True)
                        num_blocks_on_gpu = getattr(pipe.transformer.offload_manager, 'num_blocks_on_gpu', 1)
                        on_gpu_modules = getattr(pipe.transformer.offload_manager, 'on_gpu_modules', [])
                        
                        # 重新设置卸载
                        pipe.transformer.set_offload(False)  # 先关闭
                        # 重新创建缓冲区块以适应新的维度
                        pipe.transformer.set_offload(
                            True, 
                            use_pin_memory=use_pin_memory, 
                            num_blocks_on_gpu=num_blocks_on_gpu,
                            on_gpu_modules=on_gpu_modules
                        )  # 再开启
                        # 重新创建缓冲区块
                        if hasattr(pipe.transformer, 'offload_manager') and pipe.transformer.offload_manager is not None:
                            # 强制重新创建缓冲区块
                            blocks = pipe.transformer.offload_manager.blocks
                            # 创建新的缓冲区块，确保与当前模型参数维度一致，并放在正确的设备上
                            if len(blocks) > 0:
                                device = pipe.transformer.device
                                pipe.transformer.offload_manager.buffer_blocks = [
                                    copy.deepcopy(blocks[0]).to(device), 
                                    copy.deepcopy(blocks[0]).to(device)
                                ]
                    except Exception:
                        pass
                elif hasattr(pipe, 'transformer'):
                    # 如果没有启用卸载，确保所有模型参数在正确的设备上
                    try:
                        device = pipe.transformer.device
                        # 将整个transformer模型移动到指定设备
                        pipe.transformer.to(device)
                    except Exception:
                        pass
                else:
                    pass
            else:
                pass
        except Exception:
            pass

        # 生成图像
        # 使用官方推荐的参数
        images = pipe(**generation_params).images
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_paths = []
        
        # 处理输出图像 - 严格按照官方示例方式处理
        for i, image in enumerate(images):
            # 确保图像是PIL Image对象
            if not isinstance(image, Image.Image):
                # 如果是numpy数组，转换为PIL Image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
            
            # 严格按照官方示例方式处理图像
            # 确保图像是RGB模式
            image = image.convert("RGB")
            
            # 直接保存图像，不做任何额外处理
            output_path = Path(args["output_dir"]) / f"qwen_image_edit_{timestamp}_{i}.png"
            image.save(output_path)
            output_paths.append(output_path)
        
        # 输出成功信息，输出所有图像路径
        success_messages = []
        for output_path in output_paths:
            success_messages.append(f"SUCCESS: {output_path}")
        
        # 在 finally 块之前，通过一个统一的位置输出所有消息
        _print_captured_output(success_messages)
        
    except Exception as e:
        # 将错误信息传递给统一的输出函数
        _print_captured_output([f"运行图像编辑功能时发生错误: {str(e)}"])
        
    finally:
        # 恢复原始的标准输出和错误流
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        warnings.showwarning = original_showwarning

def _print_captured_output(messages):
    """
    统一处理被捕获的输出，仅在需要时恢复原始stdout进行打印。
    :param messages: 要输出的消息列表
    """
    # 临时恢复原始stdout以打印消息
    temp_stdout = sys.stdout
    sys.stdout = original_stdout
    try:
        for msg in messages:
            print(msg)
    finally:
        # 立刻将stdout重新定向回捕获对象
        sys.stdout = temp_stdout
# ==================== 图像编辑功能 ====================
def run_image_editing(args_file):
    """运行图像编辑功能"""
    # 确保关键模块在作用域内可用
    import os
    import sys
    import json
    import time

    import torch
    import numpy as np
    from pathlib import Path
    from PIL import Image
    try:
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 获取参数
        prompt = args["prompt"]
        negative_prompt = args.get("negative_prompt", "")
        # 确保negative_prompt是字符串类型
        if not isinstance(negative_prompt, str):
            negative_prompt = str(negative_prompt)
        input_images = args["images"]  # 这是输入图像
        steps = args["steps"] if args["steps"] is not None else 8  # 设置默认步数为8
        cfg_scale = args["cfg_scale"]
        scheduler_type = args["scheduler"]
        
        # 查找第一个非空的图像路径作为输入图像，第二个非空路径作为控制图像（如果存在）
        input_image_path = None
        control_image_path = None
        
        # 第一个非空路径作为输入图像，第二个非空路径作为控制图像（如果存在）
        non_empty_paths = [img_path.strip() for img_path in input_images 
                          if img_path and isinstance(img_path, str) and img_path.strip()]
        
        if len(non_empty_paths) > 0:
            input_image_path = non_empty_paths[0]
        if len(non_empty_paths) > 1:
            control_image_path = non_empty_paths[1]

        # 加载输入图像以获取尺寸
        init_image = None
        width = 1024
        height = 1024
        if input_image_path and os.path.exists(input_image_path):
            try:
                init_image = Image.open(input_image_path)
                orig_width, orig_height = init_image.size
                width, height = orig_width, orig_height
            except Exception:
                pass
        else:
            pass
        
        # 处理ControlNet相关参数
        controlnet_conditioning_scale = args.get("controlnet_conditioning_scale", 1.0)
        controlnet_preprocessor = args.get("controlnet_preprocessor", "none")
        controlnet_start = args.get("controlnet_start", 0.0)
        controlnet_end = args.get("controlnet_end", 1.0)
        
        # 获取ControlNet相关参数
        controlnet_model_selected = args.get("controlnet_model", "无")
        
        # 获取蒙版图像参数
        mask_image_path = args.get("mask_image")
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        has_control_image = control_image_path is not None
        controlnet_enable = (has_control_image 
                           and controlnet_model_selected != "无" 
                           and PREPROCESSORS_AVAILABLE)
        
        # 验证输入图像参数
        if not isinstance(input_images, list):
            return
            
        if len(input_images) == 0:
            return
            

        # 导入必要的库
        from diffusers import QwenImageEditPlusPipeline
        # 使用稳健的方式导入Transformer模型，优先使用支持LoRA的版本
        EditTransformer = None
        try:
            # 首先尝试导入支持LoRA的nunchaku版本
            from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
        except (ImportError, ModuleNotFoundError):
            try:
                # 回退到diffusers标准版本
                from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as EditTransformer
            except Exception:
                EditTransformer = None
        
        if EditTransformer is None:
            return
            
        from nunchaku.utils import get_gpu_memory
        from diffusers.utils import load_image
        from PIL import Image
        
        # 获取用户选择的采样方法
        scheduler_type = args.get("scheduler", "euler")
        
        # Scheduler 配置
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        
        # 根据用户选择创建相应的调度器
        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "euler_ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "dpmpp_2m":
            # DPM++ 2M 调度器配置稍有不同
            dpm_config = scheduler_config.copy()
            dpm_config.update({
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
            })
            scheduler = DPMSolverMultistepScheduler.from_config(dpm_config)
        else:
            # 默认使用 Euler 调度器
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        
        # 获取模型路径
        # 修复：使用传递的model_dir参数而不是硬编码路径
        model_dir = args.get("model_dir")
        if model_dir:
            qwenimage_edit_models_dir = Path(model_dir)
        else:
            # 回退到默认路径
            models_dir = Path(__file__).parent / "models"
            qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
        steps = args["steps"]
        
        # 获取用户选择的模型文件
        model_file = args.get("model_file")
        if model_file:
            # 使用用户选择的模型文件
            model_path = qwenimage_edit_models_dir / model_file
        else:
            # 如果没有指定模型文件，则使用默认模型
            model_path = None
            # 查找默认模型文件
            for file_path in qwenimage_edit_models_dir.glob("*.safetensors"):
                model_path = file_path
                break
            
            if model_path is None:
                return
        
        # 检查模型文件是否存在
        if not model_path or not model_path.exists():
            return
        
        # 加载模型
        transformer = EditTransformer.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16
        )
        
        # 使用模型根目录作为基础路径，而不是模型文件所在子目录
        # 模型根目录包含model_index.json和其他必要组件
        base_model_path = model_path.parent.parent  # 获取models/qwen-image目录
        base_model_path = base_model_path.resolve()  # 获取绝对路径
        
        # 确保基础路径存在
        if not base_model_path.exists():
            return
            
        # 使用本地组件创建pipeline
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(base_model_path),
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        )
        
        # 设置模型卸载
        if get_gpu_memory() > 18:
            pipeline.enable_model_cpu_offload()
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()
        
        # 处理LoRA模型
        lora_model_1 = args.get("lora_model_1")
        lora_model_2 = args.get("lora_model_2")
        lora_weight_1 = args.get("lora_weight_1", 1.0)
        lora_weight_2 = args.get("lora_weight_2", 1.0)
        
        # 添加nunchaku目录到sys.path
        nunchaku_path = Path(__file__).parent.parent / "nunchaku-2_lora_concat"
        if str(nunchaku_path) not in sys.path:
            sys.path.insert(0, str(nunchaku_path))  # 插入到路径开头确保优先级
        
        # 尝试加载LoRA模型
        try:
            def load_lora_model(model_path, weight, model_name, pipeline):
                if not model_path or model_path == "":
                    return False
                    
                try:
                    # 使用importlib.util直接从文件导入，避免触发整个nunchaku包的加载
                    import importlib.util
                    import sys
                    from pathlib import Path
                    
                    # 构建LoRA模块文件路径
                    lora_module_path = Path(__file__).parent.parent / "nunchaku-2_lora_concat" / "nunchaku" / "lora" / "flux" / "v1" / "lora_flux_v2.py"
                    
                    # 检查模块文件是否存在
                    if lora_module_path.exists():
                        spec = importlib.util.spec_from_file_location("lora_flux_v2", str(lora_module_path))
                        lora_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(lora_module)
                        
                        # 获取所需函数
                        update_lora_params = getattr(lora_module, 'update_lora_params_v2', None)
                        set_lora_strength_v2 = getattr(lora_module, 'set_lora_strength_v2', None)
                        
                        if update_lora_params:
                            # 加载LoRA权重
                            lora_state_dict = load_state_dict_in_safetensors(model_path)
                            if lora_state_dict:
                                try:
                                    # 尝试应用LoRA权重
                                    update_lora_params(pipeline.transformer, lora_state_dict, strength=weight)
                                    
                                    # 重新初始化CPU卸载管理器以适应LoRA加载后模型参数维度的变化
                                    if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, 'offload') and pipeline.transformer.offload:
                                        try:
                                            # 保存当前的卸载设置
                                            use_pin_memory = getattr(pipeline.transformer.offload_manager, 'use_pin_memory', True)
                                            num_blocks_on_gpu = getattr(pipeline.transformer.offload_manager, 'num_blocks_on_gpu', 1)
                                            
                                            # 重新设置卸载
                                            pipeline.transformer.set_offload(False)  # 先关闭
                                            pipeline.transformer.set_offload(True, use_pin_memory=use_pin_memory, num_blocks_on_gpu=num_blocks_on_gpu)  # 再开启
                                        except Exception:
                                            pass
                                    
                                    return True
                                except Exception:
                                    return False
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                except Exception:
                    return False
            
            # 确保pipeline已定义后再加载LoRA模型
            if 'pipeline' in locals() and pipeline is not None:
                # 加载两个LoRA模型
                if lora_model_1:
                    load_lora_model(lora_model_1, lora_weight_1, "1", pipeline)
                    
                if lora_model_2:
                    load_lora_model(lora_model_2, lora_weight_2, "2", pipeline)
            else:
                pass
        except Exception:
            pass
        # 处理输入图像和控制图像
        init_image = None
        control_image_path = None
        
        # 获取从单独参数传递的控制图像（编辑模型UI使用这种方式）
        control_image_path = args.get("control_image")
        
        # 查找第一个非空的图像路径作为输入图像，第二个非空路径作为控制图像（如果存在）
        non_empty_paths = [img_path.strip() for img_path in input_images 
                          if img_path and isinstance(img_path, str) and img_path.strip()]
        
        if len(non_empty_paths) > 0:
            input_image_path = non_empty_paths[0]
        # 如果没有从单独参数获取到控制图像，再尝试从images列表中获取
        if not control_image_path and len(non_empty_paths) > 1:
            control_image_path = non_empty_paths[1]
        
        # 使用UI传递的controlnet_enable参数来控制ControlNet启用状态
        controlnet_enable = args.get("controlnet_enable", False)

        # 预处理控制图像（如果启用）
        processed_control_image = None
        if controlnet_enable and control_image_path:
            # 处理预处理器名称，去除可能的前缀（如"[Pose] "）
            clean_preprocessor_type = controlnet_preprocessor
            if isinstance(clean_preprocessor_type, str) and "]" in clean_preprocessor_type:
                # 去除"[Pose] "这样的前缀
                clean_preprocessor_type = clean_preprocessor_type.split("]", 1)[1].strip()
            
            # 特殊处理一些常见的预处理器名称映射
            preprocessor_mapping = {
                "dw_openpose_full": "dw_openpose_full",
                "openpose_full": "openpose_full",
                "canny": "canny",
                "depth_midas": "depth_midas",
                "depth_anything_v2": "depth_anything_v2",
                "softedge_hed": "softedge_hed",
                "lineart_standard": "lineart_standard",
                "lineart_realistic": "lineart_realistic",
                "lineart_anime_denoise": "lineart_anime_denoise"
            }
            
            # 如果clean_preprocessor_type在映射中，使用映射值
            if clean_preprocessor_type in preprocessor_mapping:
                clean_preprocessor_type = preprocessor_mapping[clean_preprocessor_type]
            
            # 特殊处理inpaint_only预处理器，需要蒙版图像
            mask_path = mask_image_path  # 使用从参数中获取的蒙版图像路径
            if clean_preprocessor_type == "inpaint_only":
                if mask_image_path:
                    pass
                else:
                    # 如果没有单独提供蒙版，则使用control_image_path本身（假设它包含蒙版信息）
                    mask_path = control_image_path
            
            processed_control_image = preprocess_control_image(control_image_path, clean_preprocessor_type, mask_path)
            if processed_control_image is None:
                has_control_image = False
            else:
                # 再次确保图像是RGB模式
                # 检查是numpy数组还是PIL图像
                if isinstance(processed_control_image, np.ndarray):
                    # 如果是numpy数组，先转换为PIL图像
                    processed_control_image = Image.fromarray(processed_control_image)
                
                if processed_control_image.mode != 'RGB':
                    processed_control_image = processed_control_image.convert('RGB')
                
        else:
            processed_control_image = None

        # 准备生成参数
        
        if not input_image_path:
            return
            
        try:
            init_image = load_image(input_image_path)
            if init_image is None:
                return
                
            # 严格按照官方示例方式处理图像
            # 确保图像是RGB模式
            init_image = init_image.convert("RGB")
            
            # 获取原始尺寸
            orig_width, orig_height = init_image.size
            width, height = orig_width, orig_height

        except Exception:
            return
                
        # 生成图像

        # 获取生成批次大小
        batch_size = args.get("batch_size", 1)
        
        # 创建生成器
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.get("seed", -1))
        
        # 准备生成参数 - 严格按照官方示例方式准备
        generation_params = {
            "image": init_image,
            "prompt": prompt,
            "true_cfg_scale": cfg_scale,
            "negative_prompt": negative_prompt if negative_prompt else " ",
            "num_inference_steps": steps,
            "generator": generator,
            "num_images_per_prompt": batch_size,
        }
        
        # 根据使用的Pipeline类型和是否启用ControlNet来处理图像输入
        if controlnet_enable and processed_control_image is not None:
            # 对于启用了ControlNet的情况，将参考图像和控制图像作为列表传递
            generation_params["image"] = [init_image, processed_control_image]
        else:
            # 对于普通编辑模式，只传递输入图像
            generation_params["image"] = init_image
            
        images = pipeline(**generation_params).images

        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_paths = []
        
        # 处理输出图像 - 严格按照官方示例方式处理
        for i, image in enumerate(images):
            # 确保图像是PIL Image对象
            if not isinstance(image, Image.Image):
                # 如果是numpy数组，转换为PIL Image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
            
            # 严格按照官方示例方式处理图像
            # 确保图像是RGB模式
            image = image.convert("RGB")
            
            # 直接保存图像，不做任何额外处理
            output_path = Path(args["output_dir"]) / f"qwen_image_edit_{timestamp}_{i}.png"
            image.save(output_path)
            output_paths.append(output_path)
        
        # 输出成功信息，输出所有图像路径
        # 恢复标准输出以显示成功信息
        sys.stdout = original_stdout
        for output_path in output_paths:
            print(f"SUCCESS: {output_path}")
        sys.stdout = captured_stdout
        
    except Exception as e:
        # 恢复标准输出以显示错误信息
        sys.stdout = original_stdout
        print(f"运行图像编辑功能时发生错误: {str(e)}")
        sys.stdout = captured_stdout
        return
    finally:
        # 恢复原始的标准输出和错误流
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        warnings.showwarning = original_showwarning

        return
