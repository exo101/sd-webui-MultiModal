#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 在文件开头就处理参数，避免与WebUI的argparse冲突
import sys
import os

# 保存原始参数并处理我们的参数
original_argv = sys.argv.copy()

# 提取我们的参数（第一个非选项参数应该就是我们的JSON文件）
our_args = [arg for arg in original_argv[1:] if not arg.startswith('-')]
args_file = None
if len(our_args) == 1:
    args_file = our_args[0]

# 清理sys.argv，只保留脚本名，避免与WebUI的argparse冲突
sys.argv = [original_argv[0]]

# 现在导入其他模块
import json
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
import argparse

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
    
    from modules_forge.shared import supported_preprocessors
    from modules_forge.initialization import initialize_forge
    from annotator.util import HWC3
    
    PREPROCESSORS_AVAILABLE = True
    print("成功导入WebUI预处理器系统")
    
except ImportError as e:
    print(f"警告: 无法导入WebUI预处理器系统: {e}")
    PREPROCESSORS_AVAILABLE = False

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 导入预处理函数
from qwen_image_controlnet import preprocess_for_qwen_image_controlnet

class QwenImageProcessor:
    """Qwen图像处理器类"""
    
    def __init__(self):
        self.temp_dir = Path(__file__).parent / "temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    def preprocess_control_image(self, image_path, preprocessor_type):
        """预处理控制图像"""
        try:
            print(f"开始预处理控制图像: {image_path} 使用预处理器: {preprocessor_type}")
            
            # 特殊处理"none"预处理器
            if preprocessor_type is None or preprocessor_type.lower() in ["none", "无", "none (default)"]:
                print("使用无预处理模式，直接返回图像路径")
                return image_path
            
            # 特殊处理reference预处理器
            if preprocessor_type in ["reference_only", "reference_adain", "reference_adain+attn"]:
                print(f"Reference预处理器 {preprocessor_type} 检测到，直接返回图像")
                return image_path
            
            # 使用WebUI的预处理器系统处理图像
            processed_image = preprocess_for_qwen_image_controlnet(image_path, preprocessor_type)
            
            if processed_image is not None:
                # 如果返回的是文件路径，直接返回
                if isinstance(processed_image, str) and os.path.exists(processed_image):
                    return processed_image
                
                # 如果返回的是numpy数组，转换为PIL图像并保存
                if isinstance(processed_image, np.ndarray):
                    # 确保输出目录存在
                    temp_dir = os.path.join(self.temp_dir, "preprocessed")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 转换为PIL图像
                    pil_image = Image.fromarray(processed_image).convert("RGB")
                    
                    # 生成临时文件名
                    temp_filename = f"preprocessed_{int(time.time() * 1000)}.png"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    
                    # 保存处理后的图像
                    pil_image.save(temp_path)
                    print(f"预处理完成，保存到: {temp_path}")
                    return temp_path
                else:
                    print(f"预处理器返回了意外的类型: {type(processed_image)}")
                    return image_path
            else:
                print("预处理器返回了空结果，使用原始图像")
                return image_path
                
        except Exception as e:
            print(f"预处理控制图像时出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回原始图像路径而不是抛出异常
            return image_path

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
        
        # 特殊处理"none"预处理器
        if preprocessor_type and preprocessor_type.lower() in ["none", "无", "none (default)"]:
            print(f"使用无预处理模式，直接返回原始图像路径: {image_path}")
            print(f"SUCCESS:{image_path}")
            return image_path
            
        # 创建QwenImageProcessor实例并处理图像
        processor = QwenImageProcessor()
        result = processor.preprocess_control_image(image_path, preprocessor_type)
        
        # 处理预处理器返回的结果
        if result is None:
            print("预处理失败，返回None")
            return None

        # 情况1: 结果已经是存在的文件路径
        if isinstance(result, str):
            if os.path.exists(result):
                print(f"SUCCESS:{result}")
                return result
            else:
                print(f"错误：预处理器返回的路径不存在: {result}")
                return None

        # 情况2: 结果是numpy数组
        if isinstance(result, np.ndarray):
            if result.size == 0:
                print("预处理结果为空数组")
                return None
            if np.all(result == 0):
                print("警告：预处理结果为全零数组")

            # 确保数据类型为uint8
            if result.dtype != np.uint8:
                result_min, result_max = result.min(), result.max()
                if result_max > result_min:
                    result = ((result - result_min) / (result_max - result_min) * 255).astype(np.uint8)
                else:
                    result = np.zeros_like(result, dtype=np.uint8)
            
            # 转换为PIL图像
            result = Image.fromarray(result)

        # 情况3: 结果是PIL Image对象
        if isinstance(result, Image.Image):
            # 确保输出目录存在
            output_dir = Path(image_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一文件名并保存
            output_path = output_dir / f"preprocessed_{int(time.time() * 1000)}.png"
            result.save(output_path)
            print(f"SUCCESS:{output_path}")
            return output_path
            
        print(f"预处理器返回了意外的类型: {type(result)}")
        return None
        
    except Exception as e:
        print(f"执行预处理控制图像时发生异常: {e}")
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
    else:
        controlnet = None

    # 加载模型
    print("开始加载模型...")
    transformer = None
    pipe = None
    vae = None  # 初始化vae变量
    
    # 直接使用nunchaku的正确加载方式
    print(f"尝试使用nunchaku加载模型...")
    # 导入相应的类
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    
    # 检查模型路径
    print(f"正在从 {model_path} 加载transformer...")
    if model_path is None:
        print("模型路径为None")
        return
    
    # 检查模型文件是否存在且可读
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        return
    
    # 尝试加载transformer
    try:
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
    
    # 获取生成批次数量
    batch_size = args.get("batch_size", 1)
    
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
        # 创建处理器实例
        processor = QwenImageProcessor()
        
        # 预处理控制图像（内部已处理尺寸调整）
        processed_control_image = processor.preprocess_control_image(control_image_path, controlnet_preprocessor)
        if processed_control_image is None:
            print("控制图像处理失败")
            controlnet_enable = False
        else:
            # 确保处理后的图像符合尺寸要求
            if isinstance(processed_control_image, str):
                # 如果是文件路径，加载图像
                processed_control_image = Image.open(processed_control_image)
            
            # 检查是numpy数组还是PIL图像
            if isinstance(processed_control_image, np.ndarray):
                # 如果是numpy数组，先转换为PIL图像
                processed_control_image = Image.fromarray(processed_control_image)
            
            # 确保是RGB模式
            if processed_control_image.mode != 'RGB':
                processed_control_image = processed_control_image.convert('RGB')
            
            # 已移除 resize_image_if_needed 函数调用
            processed_control_image = processed_control_image
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
        "generator": generator if batch_size == 1 else [generator] + [torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed + i + 1) for i in range(batch_size - 1)],  # 为每个批次创建不同的生成器
        "num_images_per_prompt": batch_size  # 设置生成的图像数量
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
    result = pipe(**generation_params)
    images = result.images
    
    print("图像生成完成")
    
    # 保存图像，使用时间戳确保文件名唯一
    timestamp = int(time.time() * 1000)  # 毫秒级时间戳
    saved_image_paths = []
    
    # 保存所有生成的图像
    for i, image in enumerate(images):
        output_path = Path(args["output_dir"]) / f"qwen_image_{timestamp}_{i}.png"
        image.save(output_path)
        saved_image_paths.append(str(output_path))
    
    # 如果只生成了一张图像，直接使用第一张图像的路径
    if len(saved_image_paths) == 1:
        final_output_path = saved_image_paths[0]
    else:
        # 如果生成了多张图像，保存所有图像路径到一个文本文件中
        paths_file = Path(args["output_dir"]) / f"qwen_image_paths_{timestamp}.txt"
        with open(paths_file, 'w', encoding='utf-8') as f:
            for path in saved_image_paths:
                f.write(path + '\n')
        # 使用第一张图像作为输出
        final_output_path = saved_image_paths[0]
    
    # 计算生成时间
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"图像保存完成: {final_output_path}")
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
        "系统内存": system_info["system_memory"],
        "随机种子": seed,
        "生成批次": batch_size
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
    
    # 输出成功信息
    print(f"SUCCESS:{final_output_path}")
    print(f"INFO_FILE:{info_file}")
    
    # 清理资源
    if pipe is not None:
        del pipe
    if transformer is not None:
        del transformer
    if controlnet is not None:
        del controlnet
    if vae is not None:
        del vae
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("资源清理完成")
    
    # 添加缺失的return语句
    return final_output_path


def run_image_editing(args_file):
    """运行图像编辑功能"""
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
            # 直接加载图像，不进行尺寸调整
            image = load_image(image_path).convert("RGB")
            images.append(image)
            # 打印图像尺寸信息用于调试
            print(f"加载图像: {image_path}, 尺寸: {image.size}")
    
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

# 用于测试
if __name__ == "__main__":
    # 检查是否提供了参数文件
    if args_file is None:
        print("用法: python qwen_image_scripts.py <args_file>")
        sys.exit(1)
    
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
    elif "prompt" in args:
        run_text_to_image(args_file)
    elif "image_path" in args and "preprocessor_type" in args:
        run_preprocess_control_image(args_file)
    else:
        print("未知的参数格式")
        sys.exit(1)