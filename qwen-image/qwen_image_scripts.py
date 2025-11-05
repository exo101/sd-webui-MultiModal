#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==================== 导入模块 ====================
import json
import sys
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
            print(f"已添加路径到sys.path: {path}")
    
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
        print("ControlNet预处理器导入成功")
    except ImportError as e:
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
            # print("ControlNet预处理器从forge_legacy_preprocessors导入成功")  # 注释掉调试信息
        except ImportError as e2:
            # print(f"ControlNet预处理器导入失败: {e}")  # 注释掉调试信息
            # print(f"尝试从forge_legacy_preprocessors导入也失败: {e2}")  # 注释掉调试信息
            PREPROCESSORS_AVAILABLE = False

except Exception as e:
    # print(f"导入预处理器时出现未预期的错误: {e}")  # 注释掉调试信息
    PREPROCESSORS_AVAILABLE = False

# ==================== ControlNet 可用性检查 ====================
# 尝试导入ControlNet模型
CONTROLNET_AVAILABLE = False
try:
    from diffusers.models import QwenImageControlNetModel
    CONTROLNET_AVAILABLE = True
    # print("ControlNet功能可用")  # 注释掉调试信息
except ImportError:
    CONTROLNET_AVAILABLE = False
    # print("ControlNet功能不可用: 无法导入QwenImageControlNetModel")  # 注释掉调试信息

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
            print(f"预处理图像路径无效: {image_path}")
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
        print(f"开始使用预处理器 {preprocessor_type} 处理图像: {image_path}")
        
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
                    # print(f"已添加路径到sys.path: {path}")  # 注释掉调试信息
            
            # 导入WebUI的预处理器管理模块
            from modules_forge.shared import supported_preprocessors
            from modules_forge.initialization import initialize_forge
            
            # 初始化Forge系统
            initialize_forge()
            
            # 手动导入inpaint预处理器以确保预处理器被正确加载
            try:
                import forge_preprocessor_inpaint.scripts.preprocessor_inpaint
                # print("成功加载forge_preprocessor_inpaint模块")  # 注释掉调试信息
            except Exception as e:
                # print(f"加载forge_preprocessor_inpaint模块时出错: {e}")  # 注释掉调试信息
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
                        # print("手动注册inpaint_only预处理器成功")
                    
                    if not inpaint_global_harmonious_registered:
                        inpaint_preprocessor = PreprocessorInpaint()
                        add_supported_preprocessor(inpaint_preprocessor)
                        # print("手动注册inpaint_global_harmonious预处理器成功")
                    
                    if not inpaint_lama_registered:
                        inpaint_lama_preprocessor = PreprocessorInpaintLama()
                        add_supported_preprocessor(inpaint_lama_preprocessor)
                        # print("手动注册inpaint_lama预处理器成功")
                        
                except Exception as manual_register_error:
                    # print(f"手动注册inpaint预处理器失败: {manual_register_error}")
                    pass
            
            # 手动导入legacy_preprocessors以确保预处理器被正确加载
            try:
                import forge_legacy_preprocessors.scripts.legacy_preprocessors
                # print("成功加载legacy_preprocessors模块")  # 注释掉调试信息
            except Exception as e:
                # print(f"加载legacy_preprocessors模块时出错: {e}")  # 注释掉调试信息
                pass
            
            # 直接使用预处理器类型名称获取预处理器对象
            # 根据WebUI源码，预处理器的名称就是其在supported_preprocessors中的键
            # print(f"尝试查找预处理器: {preprocessor_type}")  # 注释掉调试信息
            
            # 特殊处理"none"预处理器 - 直接返回原始图像
            if preprocessor_type.lower() in ["none", "无", "none (default)"]:
                if isinstance(image, np.ndarray):
                    return image
                else:
                    return np.array(image)
            
            # 获取预处理器对象
            preprocessor = supported_preprocessors.get(preprocessor_type)
            if preprocessor is None:
                # 如果找不到对应预处理器，尝试转换命名格式查找
                internal_preprocessor_name = preprocessor_type.lower().replace(" ", "_")
                # print(f"未找到预处理器 {preprocessor_type}，尝试查找: {internal_preprocessor_name}")  # 注释掉调试信息
                preprocessor = supported_preprocessors.get(internal_preprocessor_name)
            
            # 特殊处理"inpaint_only"预处理器名称
            # 在某些情况下，用户可能使用"Inpaint Only"而不是"inpaint_only"
            if preprocessor is None and preprocessor_type.lower().replace(" ", "_") in ["inpaint_only", "inpaintonly"]:
                # 尝试查找"inpaint_only"
                if "inpaint_only" in supported_preprocessors:
                    preprocessor = supported_preprocessors["inpaint_only"]
                    # print("通过特殊处理找到预处理器: inpaint_only")
            
            # 如果还是找不到，直接报错而不是回退到canny
            if preprocessor is None:
                print(f"错误：未找到预处理器 {preprocessor_type}")
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
                
                # print("预处理器调用成功")  # 注释掉调试信息
                
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
                # 出错时不再回退，直接抛出异常
                raise
            
        except Exception as e:
            print(f"使用WebUI预处理器时出错: {e}")
            import traceback
            traceback.print_exc()
            # 不再回退到默认处理，直接抛出异常
            raise
        
    except Exception as e:
        print(f"预处理控制图像时出错: {e}")
        import traceback
        traceback.print_exc()
        # 不再返回None，直接抛出异常
        raise

# ==================== 预处理控制图像主函数 ====================
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
        
        # 预处理图像
        result = preprocess_control_image(image_path, preprocessor_type)
        
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
                        print(f"SUCCESS:{output_path}")
                        return str(output_path)
                else:
                    print("预处理结果为空数组")
                    return None
            # 如果返回的是PIL图像对象，保存它并输出路径
            elif isinstance(result, Image.Image):
                outputs_dir = Path(__file__).parent / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                
                import time
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

# ==================== 文生图功能 ====================
def run_text_to_image(args_file):
    """运行文生图功能"""
    try:
        print(f"开始执行文生图功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
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
            print("成功导入支持LoRA的NunchakuQwenImageTransformer2DModel")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"无法导入支持LoRA的nunchaku版本: {e}")
            try:
                # 回退到diffusers标准版本
                from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as LightningTransformer
                print("回退到diffusers标准版本的QwenImageTransformer2DModel")
            except Exception as e2:
                print(f"无法导入diffusers标准版本: {e2}")
                LightningTransformer = None
        
        if LightningTransformer is None:
            print("错误: 无法导入任何可用的Transformer模型")
            return
            
        from nunchaku.utils import get_gpu_memory, get_precision
        from PIL import Image
        
        print("依赖库导入成功")
        
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
        
        print(f"Scheduler配置完成: {scheduler_type}")
        
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
        print(f"用户选择的模型文件: {model_file}")
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
        
        print(f"用户选择步数: {steps}")
        if model_path:
            print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not model_path or not model_path.exists():
            print(f"模型文件不存在: {model_path}")
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
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        
        if controlnet_enable:
            # 加载ControlNet模型
            controlnet_model_path = controlnet_model_selected
            if controlnet_model_path and controlnet_model_path != "无":
                try:
                    # 检查是否为本地路径 (使用新的路径: D:\sd-webui-forge-aki-v4.0\models\ControlNet)
                    controlnet_base_path = Path(__file__).parent.parent.parent.parent / "models" / "ControlNet"
                    model_name = controlnet_model_path.split('/')[-1] if '/' in controlnet_model_path else controlnet_model_path
                    controlnet_local_path = controlnet_base_path / model_name
                    
                    # 确保目录存在
                    controlnet_local_path.mkdir(parents=True, exist_ok=True)
                    
                    # 只有在确实提供了控制图像的情况下才启用ControlNet
                    control_image = args.get("control_image")
                    if not control_image:
                        print("未提供控制图像，跳过ControlNet")
                        controlnet = None
                        controlnet_enable = False
                    elif controlnet_local_path and (controlnet_local_path / "config.json").exists():
                        print(f"从本地路径加载ControlNet模型: {controlnet_local_path}")
                        controlnet = QwenImageControlNetModel.from_pretrained(
                            str(controlnet_local_path),
                            # 这些参数在配置文件中存在但不被模型期望，会显示警告但不影响功能
                            # 保留torch_dtype以确保模型在正确的数据类型下运行
                            torch_dtype=torch_dtype
                        )
                    elif controlnet_model_available:
                        # 从HuggingFace下载
                        print(f"从HuggingFace下载ControlNet模型: {controlnet_model_path}")
                        # 只传递必要的参数，避免传递不支持的参数
                        controlnet = QwenImageControlNetModel.from_pretrained(
                            controlnet_model_path,
                            torch_dtype=torch_dtype
                        )
                        # 保存到本地以便下次使用
                        controlnet.save_pretrained(str(controlnet_local_path))
                    else:
                        controlnet = None
                        controlnet_enable = False
                        
                    if controlnet is not None:
                        print("ControlNet模型加载成功")
                        print(f"ControlNet模型类型: {type(controlnet)}")
                except Exception as e:
                    print(f"ControlNet模型加载失败: {e}")
                    import traceback
                    traceback.print_exc()
                    controlnet = None
                    controlnet_enable = False
            else:
                controlnet = None
                controlnet_enable = False
        else:
            controlnet = None

        # 加载模型
        print("开始加载模型...")
        transformer = None
        pipe = None
        
        # 直接使用nunchaku的正确加载方式
        try:
            print(f"尝试使用nunchaku加载模型...")
            # 导入相应的类
            from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
            
            # 检查模型路径
            print(f"正在从 {model_path} 加载transformer...")
            if model_path is None:
                raise ValueError("模型路径为None")
            
            # 检查模型文件是否存在且可读
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 尝试加载transformer
            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype
            )
            print("Transformer加载成功")

            # 添加ControlNet相关路径到系统路径
            controlnet_path = Path(__file__).parent / "ControlNet" / "Qwen-Image-ControlNet-Union"
            if str(controlnet_path) not in sys.path:
                sys.path.append(str(controlnet_path))
                print(f"已添加ControlNet路径到sys.path: {controlnet_path}")
            
            # 使用模型根目录作为基础路径，而不是模型文件所在子目录
            # 模型根目录包含model_index.json和其他必要组件
            base_model_path = str(model_path.parent.parent)  # models/qwen-image
            
            if controlnet_enable and controlnet is not None:
                print("尝试使用ControlNet管道")
                try:
                    from diffusers import QwenImageControlNetPipeline
                    print(f"ControlNet类类型: {type(controlnet)}")
                    print(f"ControlNet设备: {next(controlnet.parameters()).device if hasattr(controlnet, 'parameters') else 'unknown'}")
                    
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
                    print("ControlNet管道创建成功")
                except Exception as e:
                    print(f"ControlNet管道创建失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("回退到标准QwenImagePipeline管道")
                    from diffusers import QwenImagePipeline
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
                print("使用标准QwenImagePipeline管道")
                from diffusers import QwenImagePipeline
                pipe = QwenImagePipeline.from_pretrained(
                    base_model_path,
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=torch_dtype
                )
                # 将整个pipeline移动到指定的数据类型
                if torch_dtype != pipe.transformer.dtype:
                    pipe.to(torch_dtype)
            print("Pipeline已构建")
            print("模型加载完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            # 确保在下一次尝试前清理可能损坏的对象
            transformer = None
            pipe = None
            return
        
        # 设置模型卸载
        if get_gpu_memory() > 18:
            pipe.enable_model_cpu_offload()
            print("启用CPU卸载")
        else:
            if transformer is not None:
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
            print("启用顺序CPU卸载")
        
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
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        
        if controlnet_enable and control_image_path:
            # 预处理控制图像
            processed_control_image = preprocess_control_image(control_image_path, controlnet_preprocessor)
            if processed_control_image is None:
                print("控制图像处理失败")
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
        
        # 如果启用了ControlNet，添加ControlNet相关参数
        if controlnet_enable and controlnet is not None and processed_control_image is not None:
            generation_params.update({
                "control_image": processed_control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": controlnet_start,
                "control_guidance_end": controlnet_end,
            })
            print(f"ControlNet已启用，参数: 强度={controlnet_conditioning_scale}, 开始={controlnet_start}, 结束={controlnet_end}")
        else:
            print("ControlNet未启用或条件不满足")

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
                
                print(f"加载LoRA模型 {model_name}: {model_path} (强度: {weight})")
                try:
                    # 直接从已复制到diffusers库中的nunchaku导入LoRA模块
                    from nunchaku.lora.flux.v1.lora_flux_v2 import update_lora_params_v2, set_lora_strength_v2, reset_lora_v2
                    
                    # 加载LoRA权重
                    lora_state_dict = load_state_dict_in_safetensors(model_path)
                    if lora_state_dict:
                        try:
                            # 尝试应用LoRA权重
                            # 首先检查LoRA模型与当前模型的兼容性
                            print(f"正在检查LoRA模型 {model_name} 与当前模型的兼容性...")
                            update_lora_params_v2(pipeline.transformer, lora_state_dict, strength=weight, allow_expand=True)
                            print(f"LoRA模型 {model_name} 加载成功")
                            return True
                        except RuntimeError as e:
                            if "size of tensor" in str(e) and "must match" in str(e):
                                print(f"LoRA模型 {model_name} 与当前模型不兼容: 尺寸不匹配")
                                print(f"错误详情: {e}")
                                print("这可能是因为LoRA模型是为不同版本或架构的主模型设计的")
                                print("请确保LoRA模型与主模型架构匹配，或者尝试使用其他LoRA模型")
                                # 尝试重置LoRA（如果可能）
                                try:
                                    reset_lora_v2(pipeline.transformer)
                                except:
                                    pass
                                return False
                            elif "unexpected" in str(e):
                                print(f"LoRA模型 {model_name} 包含不支持的层: {e}")
                                print("请使用与当前模型架构兼容的LoRA模型")
                                return False
                            else:
                                raise e
                        except ValueError as e:
                            if "mismatch" in str(e).lower() or "shape" in str(e).lower():
                                print(f"LoRA模型 {model_name} 与当前模型结构不匹配: {e}")
                                print("请确保使用与主模型相同架构训练的LoRA模型")
                                return False
                            else:
                                raise e
                        except AttributeError as e:
                            print(f"LoRA模块缺少必要的函数: {e}")
                            print("请确保lora_flux_v2.py文件包含update_lora_params_v2和set_lora_strength_v2函数")
                            import traceback
                            traceback.print_exc()
                            return False
                        except Exception as e:
                            print(f"应用LoRA模型时发生未知错误: {e}")
                            import traceback
                            traceback.print_exc()
                            return False
                    else:
                        print(f"LoRA模型 {model_name} 加载失败: 无法加载权重")
                        return False
                except Exception as e:
                    print(f"LoRA模型 {model_name} 加载过程中出现错误: {e}")
                    import traceback
                    traceback.print_exc()
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
                    print("重新初始化CPU卸载管理器以适应LoRA模型参数")
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
                        print("CPU卸载管理器重新初始化完成")
                    except Exception as e:
                        print(f"重新初始化CPU卸载管理器时出错: {e}")
                        import traceback
                        traceback.print_exc()
                elif hasattr(pipeline, 'transformer'):
                    # 如果没有启用卸载，确保所有模型参数在正确的设备上
                    try:
                        device = pipeline.transformer.device
                        # 将整个transformer模型移动到指定设备
                        pipeline.transformer.to(device)
                        print(f"确保模型在设备 {device} 上")
                    except Exception as e:
                        print(f"移动模型到设备时出错: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("警告: pipe变量未定义，跳过LoRA模型加载")
        except Exception as e:
            print(f"LoRA加载过程中出现错误: {e}")
        # 生成图像
        print("开始生成图像...")
        # 使用官方推荐的参数
        images = pipe(**generation_params).images
        
        print("图像生成完成")
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_paths = []
        
        # 获取生成批次大小
        batch_size = args.get("batch_size", 1)
        
        for i, image in enumerate(images):
            output_path = Path(args["output_dir"]) / f"qwen_image_{timestamp}_{i}.png"
            image.save(output_path)
            output_paths.append(output_path)
            print(f"图像保存完成: {output_path}")
        
        # 输出成功信息，只输出图像路径
        # 如果有多张图像，只输出第一张图像的路径
        print(f"SUCCESS: {output_paths[0] if output_paths else ''}")
        
    except Exception as e:
        print(f"运行图像编辑功能时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return
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
    try:
        print(f"开始执行图像编辑功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 检查参数文件是否存在
        print(f"检查参数文件是否存在: {args_file}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"参数文件绝对路径: {os.path.abspath(args_file)}")
        
        if not os.path.exists(args_file):
            print(f"错误: 参数文件不存在: {args_file}")
            # 列出当前目录下的文件
            current_dir = os.path.dirname(args_file) if os.path.dirname(args_file) else "."
            if os.path.exists(current_dir):
                print(f"目录 {current_dir} 中的文件:")
                for file in os.listdir(current_dir):
                    print(f"  {file}")
            return
        
        if not os.path.isfile(args_file):
            print(f"错误: 参数文件不是一个有效的文件: {args_file}")
            return
            
        # 检查文件是否可读
        if not os.access(args_file, os.R_OK):
            print(f"错误: 参数文件不可读: {args_file}")
            return
            
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 添加详细的参数检查和日志
        print(f"接收到的完整参数: {args}")
        
        # 检查必需的参数是否存在
        required_args = ["prompt", "images", "steps", "cfg_scale", "scheduler"]
        for arg in required_args:
            if arg not in args:
                print(f"错误: 缺少必需参数 '{arg}'")
                return
        
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
        
        # 添加参数详细信息日志
        print(f"输入图像路径: {input_images}")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"推理步数: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"调度器类型: {scheduler_type}")
        
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
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像存在: {has_control_image}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        print(f"用户选择步数: {args['steps']}")
        
        # 验证输入图像参数
        if not isinstance(input_images, list):
            print(f"错误: images参数应该是一个列表，但实际类型是 {type(input_images)}")
            return
            
        if len(input_images) == 0:
            print("错误: images参数是空列表，未提供任何图像路径")
            return
            

        # 导入必要的库
        from diffusers import QwenImageEditPlusPipeline
        # 使用稳健的方式导入Transformer模型，优先使用支持LoRA的版本
        EditTransformer = None
        try:
            # 首先尝试导入支持LoRA的nunchaku版本
            from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
            print("成功导入支持LoRA的NunchakuQwenImageTransformer2DModel (编辑版)")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"无法导入支持LoRA的nunchaku版本 (编辑版): {e}")
            try:
                # 回退到diffusers标准版本
                from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as EditTransformer
                print("回退到diffusers标准版本的QwenImageTransformer2DModel (编辑版)")
            except Exception as e2:
                print(f"无法导入diffusers标准版本 (编辑版): {e2}")
                EditTransformer = None
        
        if EditTransformer is None:
            print("错误: 无法导入任何可用的Transformer模型 (编辑版)")
            return
            
        from nunchaku.utils import get_gpu_memory
        from diffusers.utils import load_image
        from PIL import Image
        
        print("依赖库导入成功")
        
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
                print("未找到任何编辑模型文件")
                return
        
        print(f"用户选择步数: {steps}")
        print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not model_path or not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
        
        # 加载模型
        print("开始加载模型...")
        transformer = EditTransformer.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16
        )
        
        # 使用模型根目录作为基础路径，而不是模型文件所在子目录
        # 模型根目录包含model_index.json和其他必要组件
        base_model_path = model_path.parent.parent  # 获取models/qwen-image目录
        base_model_path = base_model_path.resolve()  # 获取绝对路径
        
        print(f"模型根目录: {base_model_path}")
        
        # 确保基础路径存在
        if not base_model_path.exists():
            print(f"模型根目录不存在: {base_model_path}")
            return
            
        # 使用本地组件创建pipeline
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(base_model_path),
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        )
        
        print("模型加载完成")
        
        # 设置模型卸载
        if get_gpu_memory() > 18:
            pipeline.enable_model_cpu_offload()
            print("启用CPU卸载")
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()
            print("启用顺序CPU卸载")
        
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
                    
                print(f"加载LoRA模型 {model_name}: {model_path} (强度: {weight})")
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
                                    print(f"LoRA模型 {model_name} 加载成功")
                                    
                                    # 重新初始化CPU卸载管理器以适应LoRA加载后模型参数维度的变化
                                    if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, 'offload') and pipeline.transformer.offload:
                                        print("重新初始化CPU卸载管理器以适应LoRA模型参数")
                                        try:
                                            # 保存当前的卸载设置
                                            use_pin_memory = getattr(pipeline.transformer.offload_manager, 'use_pin_memory', True)
                                            num_blocks_on_gpu = getattr(pipeline.transformer.offload_manager, 'num_blocks_on_gpu', 1)
                                            
                                            # 重新设置卸载
                                            pipeline.transformer.set_offload(False)  # 先关闭
                                            pipeline.transformer.set_offload(True, use_pin_memory=use_pin_memory, num_blocks_on_gpu=num_blocks_on_gpu)  # 再开启
                                        except Exception as e:
                                            print(f"重新初始化CPU卸载管理器时出错: {e}")
                                    
                                    return True
                                except Exception as e:
                                    print(f"应用LoRA模型时发生错误: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    return False
                            else:
                                print(f"LoRA模型 {model_name} 加载失败: 无法加载权重")
                                return False
                        else:
                            print(f"LoRA模块缺少必要的函数: update_lora_params_v2")
                            return False
                    else:
                        print(f"LoRA模块文件不存在: {lora_module_path}")
                        return False
                except Exception as e:
                    print(f"LoRA模型 {model_name} 加载过程中出现错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # 确保pipeline已定义后再加载LoRA模型
            if 'pipeline' in locals() and pipeline is not None:
                # 加载两个LoRA模型
                if lora_model_1:
                    load_lora_model(lora_model_1, lora_weight_1, "1", pipeline)
                    
                if lora_model_2:
                    load_lora_model(lora_model_2, lora_weight_2, "2", pipeline)
            else:
                print("警告: pipeline变量未定义，跳过LoRA模型加载")
        except Exception as e:
            print(f"LoRA加载过程中出现错误: {e}")
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
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        print(f"用户选择步数: {args['steps']}")
        
        # 预处理控制图像（如果启用）
        processed_control_image = None
        if controlnet_enable and control_image_path:
            # 处理预处理器名称，去除可能的前缀（如"[Pose] "）
            clean_preprocessor_type = controlnet_preprocessor
            if isinstance(clean_preprocessor_type, str) and "]" in clean_preprocessor_type:
                # 去除"[Pose] "这样的前缀
                clean_preprocessor_type = clean_preprocessor_type.split("]", 1)[1].strip()
                print(f"预处理器名称已清理: '{controlnet_preprocessor}' -> '{clean_preprocessor_type}'")
            
            # 特殊处理inpaint_only预处理器，需要蒙版图像
            mask_path = mask_image_path  # 使用从参数中获取的蒙版图像路径
            if clean_preprocessor_type == "inpaint_only":
                if mask_image_path:
                    print(f"为inpaint_only预处理器提供蒙版图像: {mask_path}")
                else:
                    # 如果没有单独提供蒙版，则使用control_image_path本身（假设它包含蒙版信息）
                    mask_path = control_image_path
                    print("使用control_image本身作为inpaint_only预处理器的蒙版")
            
            processed_control_image = preprocess_control_image(control_image_path, clean_preprocessor_type, mask_path)
            if processed_control_image is None:
                print("控制图像处理失败")
                has_control_image = False
            else:
                # 再次确保图像是RGB模式
                # 检查是numpy数组还是PIL图像
                if isinstance(processed_control_image, np.ndarray):
                    # 如果是numpy数组，先转换为PIL图像
                    processed_control_image = Image.fromarray(processed_control_image)
                
                if processed_control_image.mode != 'RGB':
                    processed_control_image = processed_control_image.convert('RGB')
                
                print(f"控制图像预处理完成，尺寸: {processed_control_image.size}")
        else:
            print("未启用ControlNet或控制图像不存在")
            processed_control_image = None

        # 准备生成参数
        print(f"开始处理输入图像，图像路径列表: {input_images}")
        
        if not input_image_path:
            print("错误: 未提供输入图像")
            print(f"input_images变量详情: {input_images}")
            print(f"input_images长度: {len(input_images) if input_images else 'None'}")
            for i, img_path in enumerate(input_images or []):
                print(f"input_images[{i}]的值: '{img_path}' 类型: {type(img_path)}")
            return
            
        try:
            print(f"尝试加载图像: {input_image_path}")
            init_image = load_image(input_image_path)
            print(f"图像加载结果: {init_image}")
            if init_image is None:
                print("错误: 无法加载输入图像")
                return
                
            # 确保图像是RGB模式
            if init_image.mode != 'RGB':
                init_image = init_image.convert('RGB')
            print(f"输入图像加载成功，尺寸: {init_image.size}")
        except Exception as e:
            print(f"加载输入图像失败: {e}")
            import traceback
            traceback.print_exc()
            return
                
        # 生成图像
        print("开始生成图像...")
        # 使用官方推荐的参数
        generation_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg_scale,
            "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.get("seed", -1))
        }
        
        # 根据使用的Pipeline类型和是否启用ControlNet来处理图像输入
        if controlnet_enable and processed_control_image is not None:
            # 对于启用了ControlNet的情况，将参考图像和控制图像进行串联作为输入
            # Qwen-Image-Edit-2509支持通过图像串联实现多图像编辑
            from PIL import Image
            import numpy as np
            
            try:
                print(f"开始处理参考图像和控制图像的串联，参考图像类型: {type(init_image)}, 控制图像类型: {type(processed_control_image)}")
                
                # 确保两个图像都是PIL图像
                if not isinstance(init_image, Image.Image):
                    if isinstance(init_image, np.ndarray):
                        init_image = Image.fromarray(init_image)
                    else:
                        raise ValueError(f"参考图像格式不支持: {type(init_image)}")
                
                if not isinstance(processed_control_image, Image.Image):
                    if isinstance(processed_control_image, np.ndarray):
                        processed_control_image = Image.fromarray(processed_control_image)
                    else:
                        raise ValueError(f"控制图像格式不支持: {type(processed_control_image)}")
                
                # 确保图像是RGB模式
                if init_image.mode != 'RGB':
                    init_image = init_image.convert('RGB')
                if processed_control_image.mode != 'RGB':
                    processed_control_image = processed_control_image.convert('RGB')
                
                print(f"图像转换完成，参考图像尺寸: {init_image.size}，控制图像尺寸: {processed_control_image.size}")
                
                # 根据使用的Pipeline类型和是否启用ControlNet来处理图像输入
                if controlnet_enable and processed_control_image is not None:
                    # 对于启用了ControlNet的情况，将参考图像和控制图像作为列表传递
                    print(f"使用ControlNet功能，传递参考图像和控制图像")
                    print(f"参考图像尺寸: {init_image.size if hasattr(init_image, 'size') else 'N/A'}")
                    print(f"控制图像尺寸: {processed_control_image.size if hasattr(processed_control_image, 'size') else 'N/A'}")
                    # QwenImageEditPlusPipeline支持同时传递参考图像和控制图像作为列表
                    generation_params["image"] = [init_image, processed_control_image]
                else:
                    # 对于普通编辑模式，只传递输入图像
                    print(f"使用普通编辑模式，图像类型: {type(init_image)}")
                    generation_params["image"] = init_image
                    if hasattr(init_image, 'size'):
                        print(f"输入图像尺寸: {init_image.size}")
            except Exception as e:
                print(f"处理ControlNet图像时出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果无法处理为组合图像，则回退到仅使用参考图像
                print("回退到仅使用参考图像")
                generation_params["image"] = init_image
        else:
            # 对于普通编辑模式，只传递输入图像
            print(f"使用普通编辑模式，图像类型: {type(init_image)}")
            generation_params["image"] = init_image
            if hasattr(init_image, 'size'):
                print(f"输入图像尺寸: {init_image.size}")
            
        images = pipeline(**generation_params).images
        
        print("图像生成完成")
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_paths = []
        
        # 处理输出图像
        for i, image in enumerate(images):
            # 确保图像是PIL Image对象
            if not isinstance(image, Image.Image):
                # 如果是numpy数组，转换为PIL Image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
            
            # 确保图像是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 根据项目规范，如果输出图像宽度约为高度的两倍，则为拼接图像，需要提取右半部分
            try:
                if image.width >= image.height * 1.8 and image.width <= image.height * 2.2:
                    # 这是一个拼接图像，提取右半部分作为最终编辑结果
                    width = image.width
                    height = image.height
                    # 计算右半部分的边界框
                    right_image = image.crop((width // 2, 0, width, height))
                    image = right_image
                    print(f"检测到拼接图像，已提取右半部分作为编辑结果，尺寸: {image.size}")
            except Exception as e:
                print(f"处理输出图像时出错: {e}")
                # 继续使用原图像
                
            output_path = Path(args["output_dir"]) / f"qwen_image_edit_{timestamp}_{i}.png"
            image.save(output_path)
            output_paths.append(output_path)
            print(f"图像保存完成: {output_path}")
        
        # 输出成功信息，只输出图像路径
        # 如果有多张图像，只输出第一张图像的路径
        print(f"SUCCESS: {output_paths[0] if output_paths else ''}")
        
    except Exception as e:
        print(f"运行图像编辑功能时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return

        return