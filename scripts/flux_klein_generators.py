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

# 增强日志配置 - 确保调试信息可见
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 强制输出到标准输出
        logging.FileHandler('flux_klein_debug.log', mode='w')  # 同时写入文件
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 确保记录DEBUG级别日志

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

# 尝试导入来自flux_klein_model_loader的优化函数
try:
    from .flux_klein_model_loader import apply_attention_optimizations
except (ImportError, ModuleNotFoundError):
    # 如果相对导入失败，尝试绝对导入
    try:
        import flux_klein_model_loader
        apply_attention_optimizations = flux_klein_model_loader.apply_attention_optimizations
    except ImportError:
        # 尝试直接导入
        flux_klein_model_loader = importlib.import_module('flux_klein_model_loader')
        apply_attention_optimizations = getattr(flux_klein_model_loader, 'apply_attention_optimizations')


def apply_attention_optimizations(pipeline, model_type='original'):
    """
    应用注意力优化，使用flash_attn或sageattention（如果可用）
    该函数已被移至flux_klein_model_loader.py中实现，此处仅为兼容性保留
    """
    # 导入来自model_loader的实现
    try:
        from .flux_klein_model_loader import apply_attention_optimizations as loader_apply_optimizations
        return loader_apply_optimizations(pipeline, model_type)
    except (ImportError, ModuleNotFoundError):
        try:
            import flux_klein_model_loader
            return flux_klein_model_loader.apply_attention_optimizations(pipeline, model_type)
        except ImportError:
            importlib = __import__('importlib')
            flux_klein_model_loader = importlib.import_module('flux_klein_model_loader')
            return flux_klein_model_loader.apply_attention_optimizations(pipeline, model_type)


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
        
        # 应用注意力优化（从model_loader模块导入）
        apply_attention_optimizations(pipe, model_path)
        
        # 如果启用了LoRA，加载LoRA权重
        if lora_enable and lora_model and lora_model != "":
            try:
                # 构建完整的LoRA文件路径
                lora_path = os.path.join("models", "Lora", lora_model)
                if os.path.exists(lora_path):
                    # 使用正确的参数加载LoRA权重
                    pipe.load_lora_weights(
                        lora_path, 
                        weight_name=os.path.basename(lora_path),
                        local_files_only=True
                    )
                    pipe.fuse_lora(lora_scale=lora_weight)
                    logger.info(f"Applied LoRA weights from {lora_path} with scale {lora_weight}")
                else:
                    logger.warning(f"LoRA file not found: {lora_path}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                # 继续执行而不中断生成过程
        
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
        
        # 应用注意力优化（从model_loader模块导入）
        apply_attention_optimizations(pipe, model_path)
        
        # 如果启用了LoRA，加载LoRA权重
        if lora_enable and lora_model and lora_model != "":
            try:
                # 构建完整的LoRA文件路径
                lora_path = os.path.join("models", "Lora", lora_model)
                if os.path.exists(lora_path):
                    # 使用正确的参数加载LoRA权重
                    pipe.load_lora_weights(
                        lora_path, 
                        weight_name=os.path.basename(lora_path),
                        local_files_only=True
                    )
                    pipe.fuse_lora(lora_scale=lora_weight)
                    logger.info(f"Applied LoRA weights from {lora_path} with scale {lora_weight}")
                else:
                    logger.warning(f"LoRA file not found: {lora_path}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                # 继续执行而不中断生成过程
        
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
                # 构建完整的LoRA文件路径
                lora_path = os.path.join("models", "Lora", lora_model)
                if os.path.exists(lora_path):
                    # 使用正确的参数加载LoRA权重
                    pipe.load_lora_weights(
                        lora_path, 
                        weight_name=os.path.basename(lora_path),
                        local_files_only=True
                    )
                    pipe.fuse_lora(lora_scale=lora_weight)
                    logger.info(f"Applied LoRA weights from {lora_path} with scale {lora_weight}")
                else:
                    logger.warning(f"LoRA file not found: {lora_path}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                # 继续执行而不中断生成过程
        
        # 处理图像和蒙版数据
        if isinstance(image_with_mask, dict):
            # ImageMask组件返回字典格式，需要提取图像和蒙版
            logger.info(f"ImageMask data keys: {list(image_with_mask.keys())}")
            logger.info(f"ImageMask data types: {[(k, type(v)) for k, v in image_with_mask.items()]}")
            
            # 根据项目规范，从'background'提取原始图像，从'layers'[0]提取蒙版
            if 'background' in image_with_mask and 'layers' in image_with_mask and len(image_with_mask['layers']) > 0:
                image = image_with_mask['background']
                mask = image_with_mask['layers'][0]
                logger.info("Extracted image from 'background' and mask from 'layers'[0]")
            elif 'image' in image_with_mask and 'mask' in image_with_mask:
                image = image_with_mask['image']
                mask = image_with_mask['mask']
                logger.info("Extracted image from 'image' and mask from 'mask'")
            else:
                available_keys = list(image_with_mask.keys())
                return None, f"Invalid image_with_mask format: missing required keys. Available keys: {available_keys}"
            
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
        
        # 重新设计蒙版处理：采用更直接的编辑区域标记
        # 确保图像为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode: {image.mode}")
            
        # 处理遮罩：确保为L模式（灰度）
        if mask.mode != 'L':
            mask = mask.convert('L')
            logger.info(f"Converted mask to L mode: {mask.mode}")
        
        # 创建编辑引导图像：在蒙版区域添加强烈的视觉标记
        image_for_pipeline = image.copy()
        
        logger.info(f"Image size: {image.size}, Mask size: {mask.size}")
        logger.info(f"Image mode: {image.mode}, Mask mode: {mask.mode}")
        
        # 统计蒙版信息 - 修正检测阈值
        width, height = image.size
        mask_pixels = mask.load()
        
        # 使用更低的阈值检测蒙版区域，提高敏感度
        white_pixel_count = 0
        gray_pixel_count = 0
        black_pixel_count = 0
        
        for x in range(width):
            for y in range(height):
                pixel_value = mask_pixels[x, y]
                if pixel_value > 200:  # 更宽松的白色检测阈值
                    white_pixel_count += 1
                elif pixel_value > 100:  # 灰色区域
                    gray_pixel_count += 1
                else:  # 黑色区域
                    black_pixel_count += 1
        
        total_pixels = width * height
        mask_coverage = (white_pixel_count + gray_pixel_count) / total_pixels  # 包含灰色区域
        
        logger.info(f"Mask pixel distribution:")
        logger.info(f"  White pixels (>200): {white_pixel_count}")
        logger.info(f"  Gray pixels (100-200): {gray_pixel_count}")  
        logger.info(f"  Black pixels (<100): {black_pixel_count}")
        logger.info(f"  Total coverage: {mask_coverage*100:.2f}%")
        
        # 核心改进：在蒙版区域添加强烈的视觉标记
        if mask_coverage > 0.001:  # 降低触发阈值到0.1%
            image_pixels = image_for_pipeline.load()
            
            # 使用纯白色强烈标记所有非黑色区域（用户绘制的蒙版区域）
            marked_pixels = 0
            for x in range(width):
                for y in range(height):
                    if mask_pixels[x, y] > 100:  # 标记所有灰色和白色区域
                        image_pixels[x, y] = (255, 255, 255)  # 纯白色标记
                        marked_pixels += 1
            
            logger.info(f"Applied white marking to {marked_pixels} pixels ({marked_pixels/total_pixels*100:.2f}%)")
            
            # 构建强调编辑意图的提示词
            if mask_coverage > 0.3:
                edit_intensity = "dramatically"
            elif mask_coverage > 0.1:
                edit_intensity = "significantly"
            else:
                edit_intensity = "carefully"
                
            enhanced_prompt = f"{prompt} [EDITOR: {edit_intensity} modify the white-marked region while maintaining realistic appearance]"
        else:
            # 没有蒙版时使用原始图像和提示词
            enhanced_prompt = prompt
            logger.warning("⚠️ No significant mask detected - using text-to-image mode")
            logger.info("蒙版检测建议：请确保使用黑色画笔绘制需要编辑的区域")
        
        logger.info(f"Final prompt for pipeline: {enhanced_prompt}")
        logger.info(f"Processing image with effective mask coverage: {mask_coverage*100:.3f}%")
        
        # 使用Flux2KleinPipeline进行编辑
        result_images = pipe(
            prompt=enhanced_prompt,
            image=image_for_pipeline,
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
                # 构建完整的LoRA文件路径
                lora_path = os.path.join("models", "Lora", lora_model)
                if os.path.exists(lora_path):
                    # 使用正确的参数加载LoRA权重
                    pipe.load_lora_weights(
                        lora_path, 
                        weight_name=os.path.basename(lora_path),
                        local_files_only=True
                    )
                    pipe.fuse_lora(lora_scale=lora_weight)
                    logger.info(f"Applied LoRA weights from {lora_path} with scale {lora_weight}")
                else:
                    logger.warning(f"LoRA file not found: {lora_path}")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                # 继续执行而不中断生成过程
        
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
