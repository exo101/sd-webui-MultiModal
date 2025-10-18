#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import os
from pathlib import Path
import torch
import math
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
            print("ControlNet预处理器从forge_legacy_preprocessors导入成功")
        except ImportError as e2:
            print(f"ControlNet预处理器导入失败: {e}")
            print(f"尝试从forge_legacy_preprocessors导入也失败: {e2}")
            PREPROCESSORS_AVAILABLE = False

except Exception as e:
    print(f"导入预处理器时出现未预期的错误: {e}")
    PREPROCESSORS_AVAILABLE = False

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
                str(extensions_builtin)
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.append(path)
                    print(f"已添加路径到sys.path: {path}")
            
            # 导入WebUI的预处理器管理模块
            from modules_forge.shared import supported_preprocessors
            from modules_forge.initialization import initialize_forge
            
            # 初始化Forge系统
            initialize_forge()
            
            # 手动导入legacy_preprocessors以确保预处理器被正确加载
            try:
                import forge_legacy_preprocessors.scripts.legacy_preprocessors
                print("成功加载legacy_preprocessors模块")
            except Exception as e:
                print(f"加载legacy_preprocessors模块时出错: {e}")
            
            # 打印所有可用的预处理器名称，用于调试
            print("WebUI中所有可用的预处理器:")
            for name, preprocessor in supported_preprocessors.items():
                print(f"  - {name}: {preprocessor.name}")
            
            # 直接使用预处理器类型名称获取预处理器对象
            # 根据WebUI源码，预处理器的名称就是其在supported_preprocessors中的键
            print(f"尝试查找预处理器: {preprocessor_type}")
            
            # 获取预处理器对象
            preprocessor = supported_preprocessors.get(preprocessor_type)
            if preprocessor is None:
                # 如果找不到对应预处理器，尝试转换命名格式查找
                internal_preprocessor_name = preprocessor_type.lower().replace(" ", "_")
                print(f"未找到预处理器 {preprocessor_type}，尝试查找: {internal_preprocessor_name}")
                preprocessor = supported_preprocessors.get(internal_preprocessor_name)
            
            # 如果还是找不到，使用默认的canny
            if preprocessor is None:
                print(f"未找到预处理器 {preprocessor_type}，使用默认的canny")
                preprocessor = supported_preprocessors.get("canny")
                if preprocessor is None:
                    # 最后的回退方案
                    print("无法获取任何预处理器，返回原始图像")
                    return image
            
            print(f"成功找到预处理器: {preprocessor.name}")
            
            # 使用预处理器处理图像
            # 注意：WebUI预处理器通常接受RGB格式的numpy数组，值范围为0-255
            print(f"使用预处理器 {preprocessor.name} 处理图像: {image_path}")
            
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
                        kwargs['resolution'] = 512
                    if 'slider_1' in sig.parameters:
                        # 对于Canny预处理器，slider_1是低阈值
                        kwargs['slider_1'] = 100 if preprocessor.name == 'canny' else None
                    if 'slider_2' in sig.parameters:
                        # 对于Canny预处理器，slider_2是高阈值
                        kwargs['slider_2'] = 200 if preprocessor.name == 'canny' else None
                    if 'slider_3' in sig.parameters:
                        kwargs['slider_3'] = None
                    
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
                
                print("预处理器调用成功")
                
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
                # 出错时回退到原始图像
                return image
            
        except Exception as e:
            print(f"使用WebUI预处理器时出错: {e}")
            import traceback
            traceback.print_exc()
            # 回退到默认处理
            return image
        
    except Exception as e:
        print(f"预处理控制图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_preprocess_control_image(args_file):
    """运行预处理控制图像的主函数"""
    try:
        # 读取参数文件
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        print(f"开始执行控制图像预处理功能，参数文件: {args_file}")
        print(f"接收到的参数: {args}")
        
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

def get_system_info():
    """获取系统配置信息"""
    try:
        # 获取GPU信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
        else:
            gpu_info = "CPU Only"
        
        # 获取系统内存信息
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024**3)  # GB
        available_memory = memory.available / (1024**3)  # GB
        
        return {
            "gpu": gpu_info,
            "system_memory": f"{total_memory:.1f}GB",
            "available_memory": f"{available_memory:.1f}GB"
        }
    except Exception as e:
        return {
            "gpu": "NVIDIA RTX 4070 Ti",
            "system_memory": "64GB",
            "available_memory": "Unknown"
        }

def run_text_to_image(args_file):
    """运行文生图功能"""
    try:
        print(f"开始执行文生图功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        print(f"接收到的参数: {args}")
        
        # 获取参数
        prompt = args["prompt"]
        negative_prompt = args.get("negative_prompt", "")
        width = args["width"]
        height = args["height"]
        steps = args["steps"]
        cfg_scale = args["cfg_scale"]
        scheduler_type = args["scheduler"]
        
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"图像尺寸: {width}x{height}")
        print(f"推理步数: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"采样方法: {scheduler_type}")
        
        # 导入必要的库
        from diffusers import QwenImagePipeline
        from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as LightningTransformer
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
        
        # 检查是否启用ControlNet
        controlnet_enable = args.get("controlnet_enable", False) and PREPROCESSORS_AVAILABLE
        
        if controlnet_enable:
            # 加载ControlNet模型
            controlnet_model_path = args.get("controlnet_model", "InstantX/Qwen-Image-ControlNet-Union")
            if controlnet_model_path:
                try:
                    # 检查是否为本地路径 (使用新的路径: D:\sd-webui-forge-aki-v4.0\models\ControlNet)
                    controlnet_base_path = Path(__file__).parent.parent.parent.parent / "models" / "ControlNet"
                    model_name = controlnet_model_path.split('/')[-1] if '/' in controlnet_model_path else controlnet_model_path
                    controlnet_local_path = controlnet_base_path / model_name
                    
                    # 确保目录存在
                    controlnet_local_path.mkdir(parents=True, exist_ok=True)
                    
                    # 尝试导入ControlNet模型
                    try:
                        from diffusers.models import QwenImageControlNetModel
                        controlnet_model_available = True
                    except ImportError:
                        controlnet_model_available = False
                        print("无法导入QwenImageControlNetModel")
                    
                    if controlnet_model_available and (controlnet_local_path / "config.json").exists():
                        print(f"从本地路径加载ControlNet模型: {controlnet_local_path}")
                        controlnet = QwenImageControlNetModel.from_pretrained(
                            str(controlnet_local_path), 
                            torch_dtype=torch_dtype,
                            local_files_only=True
                        )
                    elif controlnet_model_available:
                        # 从HuggingFace下载
                        print(f"从HuggingFace下载ControlNet模型: {controlnet_model_path}")
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
            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(str(model_path))
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
                    pipe = QwenImageControlNetPipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        controlnet=controlnet,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
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
        
        print(f"ControlNet参数: 强度={controlnet_conditioning_scale}, 预处理器={controlnet_preprocessor}, 开始={controlnet_start}, 结束={controlnet_end}")
        
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

        # 生成图像
        print("开始生成图像...")
        print(f"生成参数: {generation_params}")
        # 使用官方推荐的参数
        image = pipe(**generation_params).images[0]
        
        print("图像生成完成")
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_path = Path(args["output_dir"]) / f"qwen_image_{timestamp}.png"
        image.save(output_path)
        
        # 计算生成时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"图像保存完成: {output_path}")
        print(f"图像生成耗时: {generation_time:.2f}秒")
        
        # 准备生成信息
        generation_info = {
            "推理步数": steps,
            "提示词引导系数 (CFG Scale)": args["cfg_scale"],
            "宽度": args["width"],
            "高度": args["height"],
            "模型类型": "Qwen文生图模型",
            "模型文件": model_path.name if model_path else "未知",
            "采样方法": scheduler_type,
            "生成时间": f"{generation_time:.2f}秒",
            "GPU配置": system_info["gpu"],
            "系统内存": system_info["system_memory"]
        }
        
        # 如果启用了ControlNet，添加ControlNet相关信息
        if controlnet_enable:
            generation_info["ControlNet启用"] = True
            generation_info["ControlNet模型"] = args.get("controlnet_model", "InstantX/Qwen-Image-ControlNet-Union")
            generation_info["ControlNet强度"] = args.get("controlnet_conditioning_scale", 1.0)
            generation_info["ControlNet预处理器"] = args.get("controlnet_preprocessor", "none")
            generation_info["ControlNet开始时间步"] = args.get("controlnet_start", 0.0)
            generation_info["ControlNet结束时间步"] = args.get("controlnet_end", 1.0)
        else:
            generation_info["ControlNet启用"] = False
        
        # 将生成信息保存到文件，供UI读取
        info_file = Path(args["output_dir"]) / f"qwen_image_info_{timestamp}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, ensure_ascii=False, indent=2)
        
        # 输出成功信息，分别输出图像路径和信息文件路径
        print(f"SUCCESS: {output_path}")
        print(f"INFO_FILE: {info_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_image_editing(args_file):
    """运行图像编辑功能"""
    try:
        print(f"开始执行图像编辑功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        print(f"接收到的参数: {args}")
        
        # 获取参数
        prompt = args["prompt"]
        negative_prompt = args.get("negative_prompt", "")
        images = args["images"]
        steps = args["steps"]
        cfg_scale = args["cfg_scale"]
        scheduler_type = args["scheduler"]
        
        print(f"编辑指令: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"图像数量: {len(images)}")
        print(f"推理步数: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"采样方法: {scheduler_type}")
        
        # 导入必要的库
        from diffusers import QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
        from nunchaku.utils import get_gpu_memory, get_precision
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
        transformer = EditTransformer.from_pretrained(str(model_path))
        
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
        
        print("未启用LoRA功能")
        
        # 设置模型卸载
        if get_gpu_memory() > 18:
            pipeline.enable_model_cpu_offload()
            print("启用CPU卸载")
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()
            print("启用顺序CPU卸载")
        
        # 加载图像
        print("开始加载图像...")
        images = []
        for image_path in args["images"]:
            if image_path is not None:
                images.append(load_image(image_path).convert("RGB"))
        
        print("图像加载完成")
        
        # 准备输入
        inputs = {
            "image": images[0] if len(images) == 1 else images,  # 单张图像直接传递，多张图像传递列表
            "prompt": args["prompt"],
            "true_cfg_scale": args["cfg_scale"],
            "negative_prompt": args["negative_prompt"],
            "num_inference_steps": args["steps"],
            "generator": torch.manual_seed(0),  # 添加随机种子以确保结果可重现
            "guidance_scale": 1.0,  # 按照官方推荐设置
            "num_images_per_prompt": 1,
        }
        
        print("开始生成编辑后的图像...")
        # 生成图像
        output = pipeline(**inputs)
        output_image = output.images[0]
        
        print("图像生成完成")
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_path = Path(args["output_dir"]) / f"qwen_image_edit_{timestamp}.png"
        output_image.save(output_path)
        
        # 计算生成时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"图像保存完成: {output_path}")
        print(f"图像生成耗时: {generation_time:.2f}秒")
        
        # 准备生成信息
        generation_info = {
            "推理步数": steps,
            "提示词引导系数 (CFG Scale)": args["cfg_scale"],
            "模型类型": "Qwen图像编辑模型",
            "模型文件": model_path.name if model_path else "未知",
            "采样方法": scheduler_type,
            "生成时间": f"{generation_time:.2f}秒",
            "GPU配置": system_info["gpu"],
            "系统内存": system_info["system_memory"]
        }
        
        
        # 将生成信息保存到文件，供UI读取
        info_file = Path(args["output_dir"]) / f"qwen_image_edit_info_{timestamp}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, ensure_ascii=False, indent=2)
        
        # 输出成功信息，分别输出图像路径和信息文件路径
        print(f"SUCCESS: {output_path}")
        print(f"INFO_FILE: {info_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 用于测试
if __name__ == "__main__":
    if len(sys.argv) > 1:
        args_file = sys.argv[1]
        run_preprocess_control_image(args_file)
    if len(sys.argv) != 2:
        print("用法: python qwen_image_scripts.py <args_file>")
        sys.exit(1)
    
    args_file = sys.argv[1]
    
    # 检查参数文件是否存在
    if not os.path.exists(args_file):
        print(f"参数文件不存在: {args_file}")
        sys.exit(1)
    
    # 读取参数确定运行哪个功能
    with open(args_file, 'r', encoding='utf-8') as f:
        args = json.load(f)
    
    # 根据参数判断运行哪个功能
    if "images" in args:
        run_image_editing(args_file)
    else:
        run_text_to_image(args_file)
