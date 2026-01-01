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

# 导入qwen_image_controlnet模块以使用其中的预处理功能
sys.path.append(str(Path(__file__).parent))
from qwen_image_controlnet import preprocess_for_qwen_image_controlnet

# 预处理器类型映射（UI显示名称到内部标识符）
# 这些映射对应于qwen_image_edit.py中的定义
CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL = {
    # Canny 类别
    "[Canny] Canny": "canny",
    # Depth 类别
    "[Depth] Depth Midas": "depth_midas",
    "[Depth] Depth Leres": "depth_leres",
    "[Depth] Depth Leres++": "depth_leres++",
    "[Depth] Depth Anything": "depth_anything",
    "[Depth] Depth Anything V2": "depth_anything_v2",
    "[Depth] Depth Hand Refiner": "depth_hand_refiner",
    "[Depth] Depth Marigold": "depth_marigold",
    "[Depth] Depth Zoe": "depth_zoe",
    # Pose 类别
    "[Pose] Openpose Full": "openpose_full",
    "[Pose] Openpose": "openpose",
    "[Pose] Openpose Face": "openpose_face",
    "[Pose] Openpose Faceonly": "openpose_faceonly",
    "[Pose] Openpose Hand": "openpose_hand",
    "[Pose] DW Openpose Full": "dw_openpose_full",
    "[Pose] Animal Openpose": "animal_openpose",
    "[Pose] Densepose (purple bg & purple torso)": "densepose",
    "[Pose] Densepose Parula (black bg & blue torso)": "densepose_parula",
    # Lineart 类别
    "[Lineart] Lineart Standard (from white bg & black line)": "lineart_standard",
    "[Lineart] Lineart Realistic": "lineart_realistic",
    "[Lineart] Lineart Coarse": "lineart_coarse",
    "[Lineart] Lineart Anime": "lineart_anime",
    "[Lineart] Lineart Anime Denoise": "lineart_anime_denoise",
    # Softedge 类别
    "[Softedge] Scribble Pidinet": "scribble_pidinet",
    "[Softedge] Softedge Pidinet": "softedge_pidinet",
    "[Softedge] Softedge Pidinet Safe": "softedge_pidinet_safe",
    "[Softedge] Softedge Pidinstruct": "softedge_pidinstruct",
    "[Softedge] Softedge Hed": "softedge_hed",
    "[Softedge] Softedge Hedsafe": "softedge_hedsafe",
    # Inpaint 类别
    "[Inpaint] Inpaint Only": "inpaint_only",
    # 直接名称映射（为了兼容性）
    "canny": "canny",
    "depth_midas": "depth_midas",
    "depth_leres": "depth_leres",
    "depth_leres++": "depth_leres++",
    "depth_anything": "depth_anything",
    "depth_anything_v2": "depth_anything_v2",
    "depth_hand_refiner": "depth_hand_refiner",
    "depth_marigold": "depth_marigold",
    "depth_zoe": "depth_zoe",
    "openpose_full": "openpose_full",
    "openpose": "openpose",
    "openpose_face": "openpose_face",
    "openpose_faceonly": "openpose_faceonly",
    "openpose_hand": "openpose_hand",
    "dw_openpose_full": "dw_openpose_full",
    "animal_openpose": "animal_openpose",
    "densepose": "densepose",
    "densepose_parula": "densepose_parula",
    "lineart_standard": "lineart_standard",
    "lineart_realistic": "lineart_realistic",
    "lineart_coarse": "lineart_coarse",
    "lineart_anime": "lineart_anime",
    "lineart_anime_denoise": "lineart_anime_denoise",
    "scribble_pidinet": "scribble_pidinet",
    "softedge_pidinet": "softedge_pidinet",
    "softedge_pidinet_safe": "softedge_pidinet_safe",
    "softedge_pidinstruct": "softedge_pidinstruct",
    "softedge_hed": "softedge_hed",
    "softedge_hedsafe": "softedge_hedsafe",
    "inpaint_only": "inpaint_only",
    # 特殊值
    "None": "none",
    "none": "none",
    "": "none"  # 空字符串也视为"none"
}

# 定义preprocess_control_image函数，以便在subprocess环境中使用
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
        
        # 检查model_file是否有效
        if not model_file or model_file == "无" or model_file == "None" or model_file == "":
            print("错误: 未选择模型文件")
            return
        
        # 使用用户选择的模型文件
        model_path = qwenimage_models_dir / model_file
        
        print(f"用户选择步数: {steps}")
        print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not model_path.exists():
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
                        print(f"从本地路径加载ControlNet模型: {controlnet_local_path}")
                        controlnet = QwenImageControlNetModel.from_pretrained(
                            str(controlnet_local_path),
                            torch_dtype=torch_dtype
                        )
                    else:
                        print(f"ControlNet模型不存在: {controlnet_local_path}")
                        controlnet = None
                        controlnet_enable = False
                        
                    if controlnet is not None:
                        print("ControlNet模型加载成功")
                        print(f"ControlNet模型类型: {type(controlnet)}")
                else:
                    controlnet = None
                    controlnet_enable = False
            except Exception as e:
                print(f"ControlNet模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                controlnet = None
                controlnet_enable = False
        else:
            controlnet = None

        # 加载模型
        print("开始加载模型...")
        transformer = None
        pipeline = None  # 修复：定义pipeline变量
        
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
                    pipeline = QwenImageControlNetPipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        controlnet=controlnet,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
                    # 将整个pipeline移动到指定的数据类型
                    if torch_dtype != pipeline.transformer.dtype:
                        pipeline.to(torch_dtype)
                                
                    # 检查并修复Qwen2_5_VLConfig缺少的属性
                    if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'config'):
                        text_config = pipeline.text_encoder.config
                        # 确保必要的视觉token id存在
                        if not hasattr(text_config, 'vision_start_token_id'):
                            setattr(text_config, 'vision_start_token_id', 151652)
                        if not hasattr(text_config, 'vision_end_token_id'):
                            setattr(text_config, 'vision_end_token_id', 151653)
                        if not hasattr(text_config, 'vision_token_id'):
                            setattr(text_config, 'vision_token_id', 151654)
                        if not hasattr(text_config, 'image_token_id'):
                            setattr(text_config, 'image_token_id', 151655)
                        if not hasattr(text_config, 'video_token_id'):
                            setattr(text_config, 'video_token_id', 151656)
                                        
                    print("ControlNet管道创建成功")
                except Exception as e:
                    print(f"ControlNet管道创建失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("回退到标准QwenImagePipeline管道")
                    from diffusers import QwenImagePipeline
                    pipeline = QwenImagePipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
                    # 将整个pipeline移动到指定的数据类型
                    if torch_dtype != pipeline.transformer.dtype:
                        pipeline.to(torch_dtype)
                    controlnet_enable = False
            else:
                print("使用标准QwenImagePipeline管道")
                from diffusers import QwenImagePipeline
                pipeline = QwenImagePipeline.from_pretrained(
                    base_model_path,
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=torch_dtype
                )
                # 将整个pipeline移动到指定的数据类型
                if torch_dtype != pipeline.transformer.dtype:
                    pipeline.to(torch_dtype)
            print("Pipeline已构建")
            print("模型加载完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            # 确保在下一次尝试前清理可能损坏的对象
            transformer = None
            pipeline = None
            return
        
        # 设置模型卸载 - 与官方示例保持一致
        try:
            # 从nunchaku.utils导入get_gpu_memory函数
            from nunchaku.utils import get_gpu_memory
            
            if get_gpu_memory() > 18:
                pipeline.enable_model_cpu_offload()
            else:
                # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
                transformer.set_offload(
                    True, use_pin_memory=False, num_blocks_on_gpu=1
                )  # increase num_blocks_on_gpu if you have more VRAM
                pipeline._exclude_from_cpu_offload.append("transformer")
                pipeline.enable_sequential_cpu_offload()
            print("启用模型CPU卸载")
        except Exception as e:
            print(f"设置模型CPU卸载失败: {e}")
            import traceback
            traceback.print_exc()

        # 获取随机种子
        seed = args.get("seed", -1)
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # 创建生成器 - 修复：确保生成器与模型在相同设备上
        device = "cuda" if torch.cuda.is_available() and pipeline is not None and hasattr(pipeline, 'device') else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # 处理ControlNet相关参数
        control_image_path = args.get("control_image")
        controlnet_conditioning_scale = args.get("controlnet_conditioning_scale", 1.0)
        controlnet_preprocessor = args.get("controlnet_preprocessor", "none")
        controlnet_start = args.get("controlnet_start", 0.0)
        controlnet_end = args.get("controlnet_end", 1.0)
        mask_path = args.get("mask_path", None)  # 添加mask_path参数获取
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        controlnet_model_selected = args.get("controlnet_model", "无")
        controlnet_enable = (control_image_path is not None 
                           and control_image_path != "" 
                           and controlnet_model_selected != "无")
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        
        if controlnet_enable and control_image_path:
            # 使用qwen_image_controlnet.py中的预处理函数
            processed_control_image = preprocess_for_qwen_image_controlnet(control_image_path, controlnet_preprocessor, mask_path)
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
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg_scale,
            "generator": generator,
        }
        
        # 只有当true_cfg_scale > 1时才传递negative_prompt，因为这时才启用classifier-free guidance
        if cfg_scale > 1.0:
            generation_params["negative_prompt"] = negative_prompt
        
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
            def load_lora_model(model_path_str, weight, model_name, pipeline):
                # 检查model_path_str是否为"无"、"None"或空字符串
                if not model_path_str or model_path_str == "无" or model_path_str == "None" or model_path_str == "":
                    print(f"跳过LoRA模型加载（未选择模型或模型路径为空）: {model_name}")
                    return False  # 返回False表示未加载
                
                model_path = Path(model_path_str)
                
                print(f"尝试加载LoRA模型: {model_path}")
                
                if not model_path.exists():
                    print(f"LoRA模型文件不存在: {model_path}")
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
            try:
                if 'pipeline' in locals() and pipeline is not None:
                    # 加载两个LoRA模型
                    if lora_model_1 and lora_model_1 != "无" and lora_model_1 != "None":
                        load_lora_model(lora_model_1, lora_weight_1, "1", pipeline)
                        
                    if lora_model_2 and lora_model_2 != "无" and lora_model_2 != "None":
                        load_lora_model(lora_model_2, lora_weight_2, "2", pipeline)
                else:
                    print("警告: pipeline变量未定义，跳过LoRA模型加载")
            except Exception as e:
                print(f"LoRA加载过程中出现错误: {e}")
        except Exception as e:
            print(f"LoRA加载过程中出现错误: {e}")

        # 生成图像
        try:
            print("开始生成图像...")
            # 使用官方推荐的参数
            images = pipeline(**generation_params).images
            
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
            
            # 输出成功信息，输出所有图像路径
            for output_path in output_paths:
                print(f"SUCCESS: {output_path}")
        except Exception as e:
            print(f"图像生成过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
    except Exception as e:
        print(f"运行图像编辑功能时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return

def load_lora_model(model_path_str, weight, model_name, pipeline):
    """加载LoRA模型到pipeline"""
    try:
        # 检查model_path_str是否为"无"或空字符串
        if not model_path_str or model_path_str == "无" or model_path_str == "None" or model_path_str == "":
            print(f"跳过LoRA模型加载（未选择模型或模型路径为空）: {model_name}")
            return False  # 返回False表示未加载
        
        model_path = Path(model_path_str)
        
        print(f"尝试加载LoRA模型: {model_path}")
        
        if not model_path.exists():
            print(f"LoRA模型文件不存在: {model_path}")
            return False
        
        # 加载LoRA模型
        lora_state_dict = load_state_dict_in_safetensors(model_path)
        
        # 尝试将LoRA模型加载到pipeline
        if hasattr(pipeline, 'load_lora_weights'):
            pipeline.load_lora_weights(lora_state_dict)
        
            # 设置LoRA缩放
            if hasattr(pipeline, 'set_lora_tensor_split'):
                pipeline.set_lora_tensor_split([weight] * len(lora_state_dict.keys()))
            elif hasattr(pipeline, 'set_adapters'):
                # 尝试其他可能的设置方法
                pass  # 可能需要根据实际pipeline类型调整
        
        print(f"成功加载LoRA模型: {model_path} (权重: {weight})")
        return True  # 返回True表示已加载
    except Exception as e:
        print(f"加载LoRA模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        
        # 添加LoRA模型参数
        lora_model_1 = args.get("lora_model_1")
        lora_model_2 = args.get("lora_model_2")
        lora_weight_1 = args.get("lora_weight_1", 1.0)
        lora_weight_2 = args.get("lora_weight_2", 1.0)
        
        # 添加参数详细信息日志
        print(f"输入图像路径: {input_images}")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"推理步数: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"调度器类型: {scheduler_type}")
        print(f"LoRA模型1: {lora_model_1}, 权重: {lora_weight_1}")
        print(f"LoRA模型2: {lora_model_2}, 权重: {lora_weight_2}")
        
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
                print(f"从输入图像 {input_image_path} 获取尺寸: {width}x{height}")
            except Exception as e:
                print(f"无法加载输入图像以获取尺寸，使用默认尺寸 1024x1024: {e}")
        else:
            print(f"未提供有效输入图像，使用默认尺寸: {width}x{height}")
        
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
        
        # 检查是否使用SDNQ量化模型
        sdnq_enable = args.get("sdnq_enable", False)
        
        if sdnq_enable:
            # 自动检测SDNQ模型路径
            # 优先检查用户指定的路径
            sdnq_model_path = None
            
            # 首先尝试从参数中获取模型路径
            model_dir = args.get("model_dir")
            if model_dir:
                sdnq_model_path = Path(model_dir)
            else:
                # 尝试从标准模型目录查找
                from modules import shared
                models_dir = Path(shared.models_path) if hasattr(shared, 'models_path') else Path(__file__).parent.parent.parent.parent / "models"
                
                # 优先查找SDNQ模型
                sdnq_candidates = [
                    models_dir / "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
                    models_dir / "qwen-image" / "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
                    models_dir / "qwen-image-edit" / "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32"
                ]
                
                for candidate_path in sdnq_candidates:
                    if candidate_path.exists():
                        sdnq_model_path = candidate_path
                        break
            
            # 如果还是没找到，尝试扫描models目录下可能的SDNQ模型
            if sdnq_model_path is None or not sdnq_model_path.exists():
                models_dir = Path(shared.models_path) if hasattr(shared, 'models_path') else Path(__file__).parent.parent.parent.parent / "models"
                for item in models_dir.iterdir():
                    if item.is_dir() and "SDNQ" in item.name and "uint4" in item.name:
                        sdnq_model_path = models_dir / item.name
                        break
            
            # 如果仍然不存在，返回错误
            if sdnq_model_path is None or not sdnq_model_path.exists():
                print(f"SDNQ模型路径不存在: {sdnq_model_path}")
                return
            
            # 使用SDNQ量化模型
            model_path = sdnq_model_path
            print(f"使用SDNQ量化模型: {model_path}")
            
            # 检查目录是否包含必要的文件
            try:
                if not (model_path / "transformer").exists():
                    print(f"SDNQ模型缺少transformer目录: {model_path}")
                    return
                if not (model_path / "text_encoder").exists():
                    print(f"SDNQ模型缺少text_encoder目录: {model_path}")
                    return
                if not (model_path / "vae").exists():
                    print(f"SDNQ模型缺少vae目录: {model_path}")
                    return
            except Exception as e:
                print(f"检查SDNQ模型路径时出错: {model_path} - {str(e)}")
                return

            # 导入SDNQ相关模块
            try:
                import torch
                import diffusers
                from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
                from sdnq.common import use_torch_compile as triton_is_available
                from sdnq.loader import apply_sdnq_options_to_model
                print("成功导入SDNQ相关模块")
            except ImportError as e:
                print(f"无法导入SDNQ相关模块，错误: {e}")
                print("请确保已安装sdnq库")
                return
            
            # SDNQ模型的路径应该是目录，而不是单个文件
            base_model_path = model_path  # 对于SDNQ模型，路径参数已经是目录
            print(f"SDNQ模型根目录: {base_model_path}")
            
            # 使用from_pretrained加载完整pipeline - 按照官方示例方式
            pipeline = diffusers.QwenImageEditPlusPipeline.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.bfloat16
            )
            
            # 检查并修复Qwen2_5_VLConfig缺少的属性
            if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'config'):
                text_config = pipeline.text_encoder.config
                # 确保必要的视觉token id存在
                if not hasattr(text_config, 'vision_start_token_id'):
                    setattr(text_config, 'vision_start_token_id', 151652)
                if not hasattr(text_config, 'vision_end_token_id'):
                    setattr(text_config, 'vision_end_token_id', 151653)
                if not hasattr(text_config, 'vision_token_id'):
                    setattr(text_config, 'vision_token_id', 151654)
                if not hasattr(text_config, 'image_token_id'):
                    setattr(text_config, 'image_token_id', 151655)
                if not hasattr(text_config, 'video_token_id'):
                    setattr(text_config, 'video_token_id', 151656)
            
            # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
            if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
                try:
                    pipeline.transformer = apply_sdnq_options_to_model(pipeline.transformer, use_quantized_matmul=True)
                    if hasattr(pipeline, 'text_encoder'):
                        pipeline.text_encoder = apply_sdnq_options_to_model(pipeline.text_encoder, use_quantized_matmul=True)
                    print("已启用INT8 MatMul优化")
                except Exception as e:
                    print(f"SDNQ优化应用失败，回退到PyTorch Eager模式: {e}")
                    # 继续使用PyTorch Eager模式
            else:
                print("未启用INT8 MatMul优化（Triton不可用或无GPU）")
            
            # 设置SDNQ模型的内存管理 - 与官方示例保持一致
            try:
                pipeline.enable_model_cpu_offload()
                print("为SDNQ启用模型CPU卸载")
            except Exception as e:
                print(f"SDNQ CPU卸载设置失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("SDNQ模型加载完成")
        else:
            # 检查是否使用Nunchaku加速模型
            nunchaku_enable = args.get("nunchaku_enable", False)
            if nunchaku_enable:
                # 获取Nunchaku模型参数
                nunchaku_precision = args.get("nunchaku_precision", "fp4")
                nunchaku_rank = args.get("nunchaku_rank", 128)
                
                # 使用Nunchaku加速模型
                steps = args["steps"]
                
                # 获取模型路径 - 从参数中获取模型目录
                model_dir = args.get("model_dir")
                if model_dir:
                    qwenimage_edit_models_dir = Path(model_dir)
                else:
                    # 根据项目规范，使用正确的模型目录
                    # 模型应位于 shared.models_path / "qwen-image" / "qwen-image-edit" 目录下
                    from pathlib import Path
                    import sys
                    from modules import shared
                    models_dir = Path(shared.models_path) / "qwen-image"
                    qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
                
                # 获取用户选择的模型文件
                model_file = args.get("model_file")
                
                # 检查model_file是否有效
                if not model_file or model_file == "无" or model_file == "None" or model_file == "":
                    print("错误: 未选择模型文件")
                    return
                
                # 使用用户选择的模型文件
                model_path = qwenimage_edit_models_dir / model_file
                
                print(f"用户选择步数: {steps}")
                print(f"使用Nunchaku加速模型")
                print(f"模型路径: {model_path}")
                
                # 检查模型文件是否存在
                if not model_path.exists():
                    print(f"模型文件不存在: {model_path}")
                    return
                
                # 导入必要的库
                from diffusers import QwenImageEditPlusPipeline
                # 导入Transformer模型
                from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
                print("成功导入支持LoRA的NunchakuQwenImageTransformer2DModel (编辑版)")
                
                # 加载模型 - 按照官方示例方式
                print("开始加载模型...")
                transformer = EditTransformer.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16
                )
                
                # 使用基础模型路径，而不是模型文件所在子目录
                # 根据项目规范，文生图功能对应的Nunchaku模型必须存储在 models/qwen-image/qwenimage 子目录下
                # 图像编辑功能对应的Nunchaku模型必须存储在 models/qwen-image/qwen-image-edit 子目录下
                # 但基础模型路径应该是 models/qwen-image
                base_model_path = model_path.parent.parent  # 获取models/qwen-image目录
                base_model_path = base_model_path.resolve()  # 获取绝对路径
                
                print(f"模型根目录: {base_model_path}")
                
                # 确保基础路径存在
                if not base_model_path.exists():
                    print(f"模型根目录不存在: {base_model_path}")
                    return
                    
                # 使用本地组件创建pipeline - 按照官方示例方式
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    str(base_model_path),
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16
                )
                
                # 检查并修复Qwen2_5_VLConfig缺少的属性
                if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'config'):
                    text_config = pipeline.text_encoder.config
                    # 确保必要的视觉token id存在
                    if not hasattr(text_config, 'vision_start_token_id'):
                        setattr(text_config, 'vision_start_token_id', 151652)
                    if not hasattr(text_config, 'vision_end_token_id'):
                        setattr(text_config, 'vision_end_token_id', 151653)
                    if not hasattr(text_config, 'vision_token_id'):
                        setattr(text_config, 'vision_token_id', 151654)
                    if not hasattr(text_config, 'image_token_id'):
                        setattr(text_config, 'image_token_id', 151655)
                    if not hasattr(text_config, 'video_token_id'):
                        setattr(text_config, 'video_token_id', 151656)
                
                print("模型加载完成")
                
                # 设置Nunchaku模型的内存管理 - 与官方示例保持一致
                try:
                    # 从nunchaku.utils导入get_gpu_memory函数
                    from nunchaku.utils import get_gpu_memory
                    
                    if get_gpu_memory() > 18:
                        pipeline.enable_model_cpu_offload()
                    else:
                        # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
                        transformer.set_offload(
                            True, use_pin_memory=False, num_blocks_on_gpu=1
                        )  # increase num_blocks_on_gpu if you have more VRAM
                        pipeline._exclude_from_cpu_offload.append("transformer")
                        pipeline.enable_sequential_cpu_offload()
                    print("为Nunchaku启用模型CPU卸载")
                except Exception as e:
                    print(f"设置Nunchaku模型CPU卸载失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("错误: 未启用Nunchaku或SDNQ模型")
                return
            
            # 确保pipeline已定义后再加载LoRA模型
            try:
                if pipeline is not None:  # 修复：检查pipeline是否为None而不是使用'pipeline' in locals()
                    # 加载两个LoRA模型
                    if lora_model_1 and lora_model_1 != "无" and lora_model_1 != "None":
                        load_lora_model(lora_model_1, lora_weight_1, "1", pipeline)
                        
                    if lora_model_2 and lora_model_2 != "无" and lora_model_2 != "None":
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
            return
            
        try:
            print(f"尝试加载图像: {input_image_path}")
            init_image = load_image(input_image_path)
            print(f"图像加载结果: {init_image}")
            if init_image is None:
                print("错误: 无法加载输入图像")
                return
                
            # 严格按照官方示例方式处理图像
            # 确保图像是RGB模式
            init_image = init_image.convert("RGB")
            
            # 获取原始尺寸
            orig_width, orig_height = init_image.size
            width, height = orig_width, orig_height
            print(f"输入图像原始尺寸: {orig_width}x{orig_height}")
        except Exception as e:
            print(f"加载输入图像失败: {e}")
            import traceback
            traceback.print_exc()
            return
                
        # 生成图像
        print("开始生成图像...")
        
        # 获取生成批次大小
        batch_size = args.get("batch_size", 1)
        
        # 创建生成器 - 修复：确保生成器与模型在相同设备上
        device = "cuda" if torch.cuda.is_available() and 'pipeline' in locals() and pipeline is not None else "cpu"
        generator = torch.Generator(device=device).manual_seed(args.get("seed", -1))
        
        # 准备生成参数 - 严格按照官方示例方式准备
        generation_params = {
            "image": init_image,
            "prompt": prompt,
            "true_cfg_scale": cfg_scale,
            "num_inference_steps": steps,
            "generator": generator,
            "num_images_per_prompt": batch_size,
        }
        
        # 只有当true_cfg_scale > 1时才传递negative_prompt，因为这时才启用classifier-free guidance
        if cfg_scale > 1.0:
            generation_params["negative_prompt"] = negative_prompt if negative_prompt else " "
        else:
            # 当cfg_scale <= 1.0时，不传递negative_prompt参数
            pass
        
        # 根据使用的Pipeline类型和是否启用ControlNet来处理图像输入
        if controlnet_enable and processed_control_image is not None:
            # 对于启用了ControlNet的情况，将参考图像和控制图像作为列表传递
            generation_params["image"] = [init_image, processed_control_image]
        else:
            # 对于普通编辑模式，只传递输入图像
            generation_params["image"] = init_image

        # 为SDNQ模型添加官方推荐的参数
        if sdnq_enable:
            generation_params["guidance_scale"] = 1.0  # 官方示例中的参数
            # SDNQ模型需要negative_prompt参数，即使为空格
            generation_params["negative_prompt"] = negative_prompt if negative_prompt else " "

        # 生成图像 - 使用torch.inference_mode优化性能
        with torch.inference_mode():
            images = pipeline(**generation_params).images
        
        print("图像生成完成")
        
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
            print(f"图像保存完成: {output_path}")
        
        # 输出成功信息，输出所有图像路径
        for output_path in output_paths:
            print(f"SUCCESS: {output_path}")
        
    except Exception as e:
        print(f"运行图像编辑功能时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return

        return

