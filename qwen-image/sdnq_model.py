"""
Qwen Image Extension - SDNQ (Qwen-Image Quantized) 模型处理模块
用于加载和运行SDNQ量化模型
"""

import os
import sys
import json
import time
import gc
import torch
import traceback
from pathlib import Path
from diffusers import QwenImagePipeline

# 导入SDNQ相关模块
try:
    from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    SDNQ_AVAILABLE = True
except ImportError:
    print("Warning: SDNQ modules not found. Please install sdnq package.")
    SDNQ_AVAILABLE = False
    from diffusers import QwenImagePipeline


def apply_attention_optimizations(pipe):
    """应用注意力优化到模型 - 当前禁用以避免影响模型使用"""
    # 由于加速轮子可能影响模型加载和使用，暂时禁用此功能
    pass


def load_sdnq_model(model_path, torch_dtype):
    """
    加载SDNQ量化模型
    """
    try:
        # 确保模型路径存在
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"模型路径不存在: {model_path}")
            return None

        if not SDNQ_AVAILABLE:
            print("SDNQ模块不可用，无法加载SDNQ模型")
            return None

        # 根据模型名称判断使用哪个Pipeline类型
        model_name_lower = model_path_obj.name.lower()
        
        # 判断是否为编辑模型
        is_edit_model = "edit" in model_name_lower or "qwen-image-edit" in model_name_lower
        
        from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline
        
        if is_edit_model:
            pipeline_class = QwenImageEditPlusPipeline
            print(f"检测到编辑模型: {model_path_obj.name}，使用 QwenImageEditPlusPipeline")
        else:
            pipeline_class = QwenImagePipeline
            print(f"检测到文生图模型: {model_path_obj.name}，使用 QwenImagePipeline")

        # 尝试加载SDNQ模型
        try:
            # 使用官方推荐的加载方式
            model_path_str = str(model_path_obj)
            pipeline = pipeline_class.from_pretrained(
                model_path_str,
                torch_dtype=torch_dtype
            )
            
            # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
            if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
                pipeline.transformer = apply_sdnq_options_to_model(pipeline.transformer, use_quantized_matmul=True)
                pipeline.text_encoder = apply_sdnq_options_to_model(pipeline.text_encoder, use_quantized_matmul=True)
                
            # 由于加速轮子可能影响模型使用，暂不应用注意力优化
            # apply_attention_optimizations(pipeline)
            
            return pipeline

        except Exception as e:
            # 尝试使用更宽松的加载参数
            try:
                pipeline = pipeline_class.from_pretrained(
                    str(model_path_obj),
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    variant=None,
                    local_files_only=True,
                    # 添加这些参数来处理权重不匹配问题
                    low_cpu_mem_usage=False,
                    device_map=None
                )
                
                # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
                if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
                    pipeline.transformer = apply_sdnq_options_to_model(pipeline.transformer, use_quantized_matmul=True)
                    pipeline.text_encoder = apply_sdnq_options_to_model(pipeline.text_encoder, use_quantized_matmul=True)
                    
                # 由于加速轮子可能影响模型使用，暂不应用注意力优化
                # apply_attention_optimizations(pipeline)
                    
                return pipeline
            except Exception as e2:
                # 只有在真正失败时才打印错误
                print(f"SDNQ模型加载失败: {e2}")
                traceback.print_exc()
                return None

    except Exception as e:
        print(f"加载SDNQ模型时发生错误: {e}")
        traceback.print_exc()
        return None


def run_sdnq_generation(
    pipe,
    prompt,
    negative_prompt="",
    width=1024,
    height=1024,
    num_inference_steps=20,
    true_cfg_scale=4.0,
    seed=-1,
    num_images_per_prompt=1,
    control_image=None,
    controlnet_conditioning_scale=1.0,
    control_guidance_start=0.0,
    control_guidance_end=1.0
):
    """
    使用SDNQ模型执行图像生成
    """
    try:
        # 设置随机种子
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

        # 准备生成参数
        generation_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
            "num_images_per_prompt": num_images_per_prompt,
        }

        # 只有当true_cfg_scale > 1时才传递negative_prompt
        if true_cfg_scale > 1.0:
            generation_params["negative_prompt"] = negative_prompt

        # 如果提供了control_image，则添加ControlNet相关参数
        if control_image is not None:
            generation_params.update({
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            })

        # 执行生成
        output = pipe(**generation_params)
        images = output.images if hasattr(output, 'images') else [output]
        return images

    except Exception as e:
        error_msg = f"SDNQ图像生成失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None
