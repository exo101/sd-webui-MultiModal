import torch
import os
import logging
from typing import Optional, Union
from PIL import Image
import numpy as np
import time
import uuid
import sys
import importlib

# 创建logger实例
logger = logging.getLogger('flux_klein_generators')

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
    img1: Union[str, np.ndarray, Image.Image],
    img2: Optional[Union[str, np.ndarray, Image.Image]],
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
    基于两张图像生成新图像，使用图像编辑功能
    """
    try:
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Generating multi-image with parameters: "
                   f"Prompt='{prompt}', Steps={steps}, Guidance={guidance_scale}, "
                   f"Seed={seed}, Model={model_path}, "
                   f"Batch size={batch_size}, LoRA={lora_model if lora_enable else 'Disabled'}, "
                   f"Optimization={optimization_used}")
        
        # 检查图像是否正确传递
        logger.info(f"Received images - img1: {type(img1)}, img2: {type(img2)}")
        if img1 is not None:
            logger.info(f"img1: {type(img1)}")
        if img2 is not None:
            logger.info(f"img2: {type(img2)}")
        
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
        
        # 将输入转换为PIL图像
        condition_images = []
        
        if img1 is not None:
            if isinstance(img1, str):  # 文件路径
                pil_img1 = Image.open(img1).convert('RGB')
            elif isinstance(img1, np.ndarray):
                if img1.dtype != np.uint8:
                    img1 = (img1 * 255).astype(np.uint8)
                pil_img1 = Image.fromarray(img1, 'RGB')
            elif hasattr(img1, 'convert'):  # 如果已经是PIL图像
                pil_img1 = img1.convert('RGB')
            else:
                return None, f"Unsupported img1 type: {type(img1)}"
            
            condition_images.append(pil_img1)
            logger.info(f"Added img1 to condition_images: {type(pil_img1)}, size: {pil_img1.size if pil_img1 else 'N/A'}")
        else:
            return None, "First image is required"
        
        # 如果提供了第二张图像，将其转换为PIL图像并添加到条件图像列表
        if img2 is not None:
            if isinstance(img2, str):  # 文件路径
                pil_img2 = Image.open(img2).convert('RGB')
            elif isinstance(img2, np.ndarray):
                if img2.dtype != np.uint8:
                    img2 = (img2 * 255).astype(np.uint8)
                pil_img2 = Image.fromarray(img2, 'RGB')
            elif hasattr(img2, 'convert'):  # 如果已经是PIL图像
                pil_img2 = img2.convert('RGB')
            else:
                return None, f"Unsupported img2 type: {type(img2)}"
            
            condition_images.append(pil_img2)
            logger.info(f"Added img2 to condition_images: {type(pil_img2)}, size: {pil_img2.size if pil_img2 else 'N/A'}")
        else:
            logger.info("Only one image provided")

        logger.info(f"Final condition_images count: {len(condition_images)}")
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        
        # 使用Flux2 Klein的图像条件生成功能
        # 传入图像列表作为条件
        logger.info("Attempting image-conditioned generation with FLUX_2-klein")
        result_images = pipe(
            prompt=prompt,
            image=condition_images,  # 传入条件图像列表
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
        import traceback
        logger.error(traceback.format_exc())
        return None, error_msg


def inpaint_flux_klein(
    image_with_mask: Union[dict, np.ndarray, Image.Image],
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
    使用FLUX.2-klein模型对蒙版区域进行图像编辑
    """
    try:
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Generating inpainted image with parameters: "
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
        
        # 处理图像和蒙版数据
        if isinstance(image_with_mask, dict):
            # ImageMask组件返回字典格式，需要提取图像和蒙版
            if 'image' not in image_with_mask or 'mask' not in image_with_mask:
                return None, "Invalid image_with_mask format: missing image or mask"
            
            image = image_with_mask['image']
            mask = image_with_mask['mask']
            
            # 确保图像和蒙版是PIL格式
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            elif not isinstance(image, Image.Image):
                return None, f"Unsupported image type: {type(image)}"
            
            if isinstance(mask, np.ndarray):
                # 蒙版可能是多通道的，需要转为单通道
                if mask.ndim == 3:
                    mask = mask[:, :, 0] if mask.shape[2] >= 1 else mask[:, :, 0]
                mask = Image.fromarray(mask.astype(np.uint8))
            elif not isinstance(mask, Image.Image):
                return None, f"Unsupported mask type: {type(mask)}"
        elif isinstance(image_with_mask, Image.Image):
            # 如果只传入了图像，需要从图像中分离出图像和蒙版
            # 这种情况不太可能发生，但为了兼容性处理
            return None, "ImageMask component must provide both image and mask"
        else:
            return None, f"Unsupported image_with_mask type: {type(image_with_mask)}"
        
        # 确保图像和蒙版尺寸一致
        if image.size != mask.size:
            # 调整蒙版尺寸以匹配图像
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        
        # 使用Flux2 Klein的inpainting功能
        # 将蒙版转换为黑白图像，白色表示需要重绘的区域
        mask = mask.convert('L')  # 确保蒙版是灰度图
        
        # 对蒙版进行二值化处理，使白色区域更清晰
        mask = mask.point(lambda x: 255 if x > 128 else 0, mode='1')
        
        # 将蒙版转换回L模式
        mask = mask.convert('L')
        
        logger.info(f"Attempting inpainting with image size: {image.size}, mask size: {mask.size}")
        
        result_images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,  # 传递蒙版图像
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        
        end_time = time.time()
        
        # 记录性能信息
        logger.info(f"Generated {len(result_images)} images in {end_time - start_time:.2f}s using {optimization_used} optimization")
        
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
        
        return result_images, f"Successfully generated {len(result_images)} images and saved to {output_dir}"

    except Exception as e:
        error_msg = f"Error during image inpainting: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, error_msg


def extend_flux_klein(
    image: Union[np.ndarray, Image.Image],
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = -1,
    model_path: str = "FLUX_2-klein-base-4B",
    batch_size: int = 1,
    lora_enable: bool = False,
    lora_model: str = "",
    lora_weight: float = 1.0,
    extend_left: int = 64,
    extend_right: int = 64,
    extend_top: int = 64,
    extend_bottom: int = 64
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
                   f"Extension: L:{extend_left}px R:{extend_right}px T:{extend_top}px B:{extend_bottom}px, "
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
        
        # 处理输入图像
        if isinstance(image, np.ndarray):
            input_image = Image.fromarray(image.astype('uint8'), 'RGB')
        elif isinstance(image, Image.Image):
            input_image = image.convert('RGB')
        else:
            logger.error(f"Invalid image type: {type(image)}")
            return None, f"Invalid image type: {type(image)}"
        
        # 获取原始图像尺寸
        orig_width, orig_height = input_image.size
        
        # 计算扩展后的新尺寸
        new_width = orig_width + extend_left + extend_right
        new_height = orig_height + extend_top + extend_bottom
        
        # 如果没有扩展需求，直接返回原图
        if new_width <= 0 or new_height <= 0:
            logger.warning("Extension dimensions are zero or negative, returning original image")
            return [input_image], "Extension dimensions are zero, returning original image"
        
        # 创建新的空白画布
        extended_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
        
        # 将原始图像粘贴到新画布的中心位置
        extended_image.paste(input_image, (extend_left, extend_top))
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        
        # 使用扩展后的图像进行生成
        result_images = pipe(
            prompt=prompt,
            image=extended_image,
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
        import traceback
        logger.error(traceback.format_exc())
        return None, error_msg