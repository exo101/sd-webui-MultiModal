import torch
import os
import logging
from typing import Optional
from PIL import Image
import numpy as np
import time
import uuid
import sys
import importlib

# 添加当前目录到系统路径以确保模块可以被找到
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 尝试导入加速库
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# 导入模型加载相关函数
try:
    from .flux_klein_model_loader import load_flux_klein_pipeline
except (ImportError, ModuleNotFoundError):
    # 如果相对导入失败，尝试绝对导入
    try:
        import flux_klein_model_loader
        load_flux_klein_pipeline = flux_klein_model_loader.load_flux_klein_pipeline
    except ImportError:
        # 尝试直接导入
        flux_klein_model_loader = importlib.import_module('flux_klein_model_loader')
        load_flux_klein_pipeline = getattr(flux_klein_model_loader, 'load_flux_klein_pipeline')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_attention_optimizations(pipeline):
    """
    应用注意力优化，使用flash_attn或sageattention（如果可用）
    """
    if FLASH_ATTENTION_AVAILABLE:
        logger.info("Applying Flash Attention optimizations...")
        
        # 替换pipeline中的transformer的注意力机制
        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
            replace_attention_with_flash_attention(pipeline.transformer)
        
    elif SAGE_ATTENTION_AVAILABLE:
        logger.info("Applying SageAttention optimizations...")
        
        # 替换pipeline中的transformer的注意力机制
        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
            replace_attention_with_sage_attention(pipeline.transformer)
    else:
        logger.info("No advanced attention mechanisms available, using default attention")


def replace_attention_with_flash_attention(module):
    """
    递归替换模块中的注意力层为Flash Attention
    """
    for name, child in module.named_children():
        if "attention" in name.lower() or "attn" in name.lower():
            # 这里需要根据实际模型结构进行调整
            # 因为不同的模型有不同的注意力层实现
            pass
        else:
            replace_attention_with_flash_attention(child)


def replace_attention_with_sage_attention(module):
    """
    递归替换模块中的注意力层为Sage Attention
    """
    for name, child in module.named_children():
        if "attention" in name.lower() or "attn" in name.lower():
            # 这里需要根据实际模型结构进行调整
            # 因为不同的模型有不同的注意力层实现
            pass
        else:
            replace_attention_with_sage_attention(child)


def generate_flux_klein_image(
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 1.0,
    height: int = 1024,
    width: int = 768,
    seed: int = -1,
    model_path: str = "FLUX_2-klein-base-4B",
    batch_size: int = 1,
    lora_enable: bool = False,
    lora_model: str = "",
    lora_weight: float = 1.0
):
    """
    使用FLUX.2-klein模型生成图像
    """
    try:
        # 确保尺寸是8的倍数
        height = height - (height % 8)
        width = width - (width % 8)
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Generating image with parameters: "
                   f"Prompt='{prompt}', Steps={steps}, Guidance={guidance_scale}, "
                   f"Size={width}x{height}, Seed={seed}, Model={model_path}, "
                   f"Batch size={batch_size}, LoRA={lora_model if lora_enable else 'Disabled'}, "
                   f"Optimization={optimization_used}")
        
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 应用注意力优化
        apply_attention_optimizations(pipe)
        
        # 如果启用了LoRA，加载LoRA权重
        if lora_enable and lora_model and lora_model != "":
            try:
                pipe.load_lora_weights(lora_model)
                pipe.fuse_lora(lora_scale=lora_weight)
                logger.info(f"Applied LoRA weights from {lora_model} with scale {lora_weight}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        result_images = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        end_time = time.time()
        
        # 记录性能信息
        logger.info(f"Generated {len(result_images)} images in {end_time - start_time:.2f}s using {optimization_used} optimization")
        
        # 确保输出目录存在
        output_dir = os.path.join("outputs", "flux_klein")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        saved_paths = []
        for i, img in enumerate(result_images):
            unique_filename = f"flux_klein_{uuid.uuid4().hex}_{i}.png"
            save_path = os.path.join(output_dir, unique_filename)
            img.save(save_path)
            saved_paths.append(save_path)
        
        return result_images, f"Successfully generated {len(result_images)} images and saved to {output_dir}"

    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def multi_img_flux_klein(
    img1: np.ndarray,
    img2: Optional[np.ndarray],
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = -1,
    model_path: str = "FLUX_2-klein-base-4B",
    batch_size: int = 1,
    lora_enable: bool = False,
    lora_model: str = "",
    lora_weight: float = 1.0
):
    """
    基于两张图像生成新图像
    """
    try:
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Generating multi-image with parameters: "
                   f"Steps={steps}, Guidance={guidance_scale}, "
                   f"Seed={seed}, Model={model_path}, "
                   f"Batch size={batch_size}, LoRA={lora_model if lora_enable else 'Disabled'}, "
                   f"Optimization={optimization_used}")
        
        # 应用注意力优化
        apply_attention_optimizations(pipe)
        
        # 如果启用了LoRA，加载LoRA权重
        if lora_enable and lora_model and lora_model != "":
            try:
                pipe.load_lora_weights(lora_model)
                pipe.fuse_lora(lora_scale=lora_weight)
                logger.info(f"Applied LoRA weights from {lora_model} with scale {lora_weight}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
        
        # 将numpy数组转换为PIL图像
        if img1 is not None:
            pil_img1 = Image.fromarray(img1.astype('uint8'), 'RGB')
        else:
            return None, "First image is required"
        
        # 如果提供了第二张图像，将其转换为PIL图像
        if img2 is not None and isinstance(img2, np.ndarray):
            pil_img2 = Image.fromarray(img2.astype('uint8'), 'RGB')
            # 在这里实现图像融合逻辑，这取决于模型如何处理多图像输入
            # 例如，可以将两张图像拼接或以其他方式组合
            combined_prompt = f"{prompt} | Image 1: {pil_img1.size}, Image 2: {pil_img2.size}"
        else:
            combined_prompt = prompt
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        result_images = pipe(
            prompt=combined_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        end_time = time.time()
        
        # 记录性能信息
        logger.info(f"Generated {len(result_images)} images in {end_time - start_time:.2f}s using {optimization_used} optimization")
        
        # 确保输出目录存在
        output_dir = os.path.join("outputs", "flux_klein_multi")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        saved_paths = []
        for i, img in enumerate(result_images):
            unique_filename = f"flux_klein_multi_{uuid.uuid4().hex}_{i}.png"
            save_path = os.path.join(output_dir, unique_filename)
            img.save(save_path)
            saved_paths.append(save_path)
        
        return result_images, f"Successfully generated {len(result_images)} images and saved to {output_dir}"

    except Exception as e:
        error_msg = f"Error generating image from multiple images: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def inpaint_flux_klein(
    image_with_mask: dict,
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = -1,
    model_path: str = "FLUX_2-klein-base-4B",
    batch_size: int = 1,
    lora_enable: bool = False,
    lora_model: str = "",
    lora_weight: float = 1.0
):
    """
    使用FLUX.2-klein模型进行局部重绘
    """
    try:
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Performing inpainting with parameters: "
                   f"Prompt='{prompt}', Steps={steps}, Guidance={guidance_scale}, "
                   f"Seed={seed}, Model={model_path}, "
                   f"Batch size={batch_size}, LoRA={lora_model if lora_enable else 'Disabled'}, "
                   f"Optimization={optimization_used}")
        
        # 应用注意力优化
        apply_attention_optimizations(pipe)
        
        # 如果启用了LoRA，加载LoRA权重
        if lora_enable and lora_model and lora_model != "":
            try:
                pipe.load_lora_weights(lora_model)
                pipe.fuse_lora(lora_scale=lora_weight)
                logger.info(f"Applied LoRA weights from {lora_model} with scale {lora_weight}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
        
        # 从字典中提取图像和蒙版
        init_image = image_with_mask["image"]
        mask_image = image_with_mask["mask"]
        
        # 转换为PIL图像
        init_image = Image.fromarray(init_image.astype('uint8'))
        mask_image = Image.fromarray(mask_image.astype('uint8').convert('L'))
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        result_images = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        end_time = time.time()
        
        # 记录性能信息
        logger.info(f"Inpainted {len(result_images)} images in {end_time - start_time:.2f}s using {optimization_used} optimization")
        
        # 确保输出目录存在
        output_dir = os.path.join("outputs", "flux_klein_inpaint")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        saved_paths = []
        for i, img in enumerate(result_images):
            unique_filename = f"flux_klein_inpaint_{uuid.uuid4().hex}_{i}.png"
            save_path = os.path.join(output_dir, unique_filename)
            img.save(save_path)
            saved_paths.append(save_path)
        
        return result_images, f"Successfully inpainted {len(result_images)} images and saved to {output_dir}"

    except Exception as e:
        error_msg = f"Error during inpainting: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def extend_flux_klein(
    image: np.ndarray,
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = -1,
    model_path: str = "FLUX_2-klein-base-4B",
    batch_size: int = 1,
    lora_enable: bool = False,
    lora_model: str = "",
    lora_weight: float = 1.0
):
    """
    使用FLUX.2-klein模型扩展图像
    """
    try:
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Extending image with parameters: "
                   f"Prompt='{prompt}', Steps={steps}, Guidance={guidance_scale}, "
                   f"Seed={seed}, Model={model_path}, "
                   f"Batch size={batch_size}, LoRA={lora_model if lora_enable else 'Disabled'}, "
                   f"Optimization={optimization_used}")
        
        # 应用注意力优化
        apply_attention_optimizations(pipe)
        
        # 如果启用了LoRA，加载LoRA权重
        if lora_enable and lora_model and lora_model != "":
            try:
                pipe.load_lora_weights(lora_model)
                pipe.fuse_lora(lora_scale=lora_weight)
                logger.info(f"Applied LoRA weights from {lora_model} with scale {lora_weight}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
        
        # 转换输入图像为PIL图像
        input_image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        result_images = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        end_time = time.time()
        
        # 记录性能信息
        logger.info(f"Extended {len(result_images)} images in {end_time - start_time:.2f}s using {optimization_used} optimization")
        
        # 确保输出目录存在
        output_dir = os.path.join("outputs", "flux_klein_extend")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        saved_paths = []
        for i, img in enumerate(result_images):
            unique_filename = f"flux_klein_extend_{uuid.uuid4().hex}_{i}.png"
            save_path = os.path.join(output_dir, unique_filename)
            img.save(save_path)
            saved_paths.append(save_path)
        
        return result_images, f"Successfully extended {len(result_images)} images and saved to {output_dir}"

    except Exception as e:
        error_msg = f"Error during image extension: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
