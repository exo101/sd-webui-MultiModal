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
import safetensors.torch
from collections import OrderedDict

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


def _convert_non_diffusers_flux2_lora_to_diffusers_manual(original_state_dict):
    """
    手动实现 FLUX.2 LoRA 格式转换（基于 diffusers 库的逻辑）
    将 non-diffusers 格式转换为 diffusers 格式
    """
    converted_state_dict = {}
    
    num_double_layers = 8
    num_single_layers = 48
    lora_keys = ("lora_A", "lora_B")
    
    try:
        # 转换 single transformer blocks
        for sl in range(num_single_layers):
            single_block_prefix = f"single_blocks.{sl}"
            attn_prefix = f"single_transformer_blocks.{sl}.attn"
            
            for lora_key in lora_keys:
                linear1_key = f"{single_block_prefix}.linear1.{lora_key}.weight"
                linear2_key = f"{single_block_prefix}.linear2.{lora_key}.weight"
                
                if linear1_key in original_state_dict:
                    converted_state_dict[f"{attn_prefix}.to_qkv_mlp_proj.{lora_key}.weight"] = \
                        original_state_dict.pop(linear1_key)
                
                if linear2_key in original_state_dict:
                    converted_state_dict[f"{attn_prefix}.to_out.{lora_key}.weight"] = \
                        original_state_dict.pop(linear2_key)
        
        # 转换 double transformer blocks
        for dl in range(num_double_layers):
            transformer_block_prefix = f"transformer_blocks.{dl}"
            
            for lora_key in lora_keys:
                # 处理 fused QKV
                for attn_type in ["img_attn", "txt_attn"]:
                    qkv_key = f"double_blocks.{dl}.{attn_type}.qkv.{lora_key}.weight"
                    
                    if qkv_key in original_state_dict:
                        fused_qkv_weight = original_state_dict.pop(qkv_key)
                        
                        if lora_key == "lora_A":
                            diff_attn_proj_keys = (
                                ["to_q", "to_k", "to_v"]
                                if attn_type == "img_attn"
                                else ["add_q_proj", "add_k_proj", "add_v_proj"]
                            )
                            for proj_key in diff_attn_proj_keys:
                                converted_state_dict[f"{transformer_block_prefix}.attn.{proj_key}.{lora_key}.weight"] = \
                                    torch.cat([fused_qkv_weight])
                        else:
                            sample_q, sample_k, sample_v = torch.chunk(fused_qkv_weight, 3, dim=0)
                            
                            if attn_type == "img_attn":
                                converted_state_dict[f"{transformer_block_prefix}.attn.to_q.{lora_key}.weight"] = \
                                    torch.cat([sample_q])
                                converted_state_dict[f"{transformer_block_prefix}.attn.to_k.{lora_key}.weight"] = \
                                    torch.cat([sample_k])
                                converted_state_dict[f"{transformer_block_prefix}.attn.to_v.{lora_key}.weight"] = \
                                    torch.cat([sample_v])
                            else:
                                converted_state_dict[f"{transformer_block_prefix}.attn.add_q_proj.{lora_key}.weight"] = \
                                    torch.cat([sample_q])
                                converted_state_dict[f"{transformer_block_prefix}.attn.add_k_proj.{lora_key}.weight"] = \
                                    torch.cat([sample_k])
                                converted_state_dict[f"{transformer_block_prefix}.attn.add_v_proj.{lora_key}.weight"] = \
                                    torch.cat([sample_v])
                
                # 处理投影层
                proj_mappings = [
                    (f"double_blocks.{dl}.img_attn.proj.{lora_key}.weight", 
                     f"{transformer_block_prefix}.attn.to_out.0.{lora_key}.weight"),
                    (f"double_blocks.{dl}.txt_attn.proj.{lora_key}.weight",
                     f"{transformer_block_prefix}.attn.to_add_out.{lora_key}.weight"),
                ]
                
                for orig_key, diff_key in proj_mappings:
                    if orig_key in original_state_dict:
                        converted_state_dict[diff_key] = original_state_dict.pop(orig_key)
                
                # 处理 MLP 层
                mlp_mappings = [
                    (f"double_blocks.{dl}.img_mlp.0.{lora_key}.weight", 
                     f"{transformer_block_prefix}.ff.linear_in.{lora_key}.weight"),
                    (f"double_blocks.{dl}.img_mlp.2.{lora_key}.weight",
                     f"{transformer_block_prefix}.ff.linear_out.{lora_key}.weight"),
                    (f"double_blocks.{dl}.txt_mlp.0.{lora_key}.weight",
                     f"{transformer_block_prefix}.ff_context.linear_in.{lora_key}.weight"),
                    (f"double_blocks.{dl}.txt_mlp.2.{lora_key}.weight",
                     f"{transformer_block_prefix}.ff_context.linear_out.{lora_key}.weight"),
                ]
                
                for orig_key, diff_key in mlp_mappings:
                    if orig_key in original_state_dict:
                        converted_state_dict[diff_key] = original_state_dict.pop(orig_key)
        
        # 添加 transformer. 前缀
        final_state_dict = {}
        for key, value in converted_state_dict.items():
            final_state_dict[f"transformer.{key}"] = value
        
        return final_state_dict
        
    except Exception as e:
        logger.error(f"Error during manual conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 如果转换失败，返回原始状态字典（至少移除了 diffusion_model 前缀）
        return {f"transformer.{k}": v for k, v in original_state_dict.items() if not k.startswith('diffusion_model.')}


def convert_lora_state_dict_for_flux(state_dict):
    """
    转换 LoRA 状态字典以匹配 FLUX.2-klein 模型的期望格式
    处理常见的键名不匹配问题
    """
    # 首先移除 diffusion_model 前缀
    original_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('diffusion_model.'):
            new_key = new_key.replace('diffusion_model.', '')
        original_state_dict[new_key] = value
    
    # 使用完整转换函数
    return _convert_non_diffusers_flux2_lora_to_diffusers_manual(original_state_dict)


def load_lora_with_conversion(pipe, lora_path, weight_name, lora_scale=1.0):
    """
    加载 LoRA 权重并进行必要的键名转换
    :param pipe: 模型管道
    :param lora_path: LoRA 文件路径（可以是相对路径或绝对路径）
    :param weight_name: 权重文件名
    :param lora_scale: LoRA 缩放因子
    :return: 是否成功加载
    """
    try:
        # 将相对路径转换为绝对路径
        if not os.path.isabs(lora_path):
            # 如果不是绝对路径，尝试多种方式查找文件
            found_path = None
            
            # 方法 1: 从当前工作目录开始向上搜索 Lora 目录
            current_search_dir = os.getcwd()
            while current_search_dir and os.path.dirname(current_search_dir) != current_search_dir:
                possible_lora_dirs = [
                    os.path.join(current_search_dir, 'models', 'Lora'),
                    os.path.join(current_search_dir, 'Lora'),
                ]
                
                for lora_dir in possible_lora_dirs:
                    if os.path.isdir(lora_dir):
                        logger.info(f"Searching for LoRA '{lora_path}' in directory: {lora_dir}")
                        
                        # 直接在 Lora 目录下查找
                        direct_path = os.path.join(lora_dir, lora_path)
                        if os.path.exists(direct_path):
                            found_path = direct_path
                            logger.info(f"Found LoRA at: {found_path}")
                            break
                        
                        # 递归搜索所有子目录
                        for root, dirs, files in os.walk(lora_dir):
                            if lora_path in files:
                                found_path = os.path.join(root, lora_path)
                                logger.info(f"Found LoRA in subdirectory: {found_path}")
                                break
                        
                        # 也尝试将 lora_path 作为子目录路径
                        subdir_path = os.path.join(lora_dir, lora_path)
                        if os.path.exists(subdir_path):
                            found_path = subdir_path
                            logger.info(f"Found LoRA with subdirectory path: {found_path}")
                            break
                
                if found_path:
                    break
                    
                # 向上一级目录继续搜索
                parent_dir = os.path.dirname(current_search_dir)
                if parent_dir == current_search_dir:
                    break
                current_search_dir = parent_dir
            
            # 方法 2: 如果还没找到，尝试相对于当前脚本目录
            if not found_path:
                possible_paths = [
                    os.path.join(current_dir, '..', '..', '..', 'models', 'Lora', lora_path),
                    os.path.join(current_dir, '..', '..', 'Lora', lora_path),
                    os.path.join(os.getcwd(), lora_path),
                ]
                
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    logger.info(f"Checking alternative path: {abs_path}")
                    if os.path.exists(abs_path):
                        found_path = abs_path
                        logger.info(f"Found LoRA at: {found_path}")
                        break
            
            if found_path:
                lora_path = found_path
            else:
                # 如果都没找到，记录所有尝试过的路径
                logger.error(f"LoRA file not found. Searched in directory trees from:")
                logger.error(f"  Base directory: {os.getcwd()}")
                logger.error(f"  Script directory: {current_dir}")
                logger.error(f"  Requested path: {lora_path}")
        
        # 最终检查文件是否存在
        if not os.path.exists(lora_path):
            logger.error(f"LoRA file not found: {lora_path}")
            return False
        
        logger.info(f"Attempting to load LoRA from: {lora_path}")
        
        # 直接加载状态字典并手动应用权重
        state_dict = safetensors.torch.load_file(lora_path)
        logger.info(f"Loaded LoRA state dict with {len(state_dict)} keys")
        
        # 检查并记录键名格式
        sample_keys = list(state_dict.keys())[:5]
        logger.info(f"Sample LoRA keys: {sample_keys}")
        
        # 先移除 diffusion_model 前缀
        original_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('diffusion_model.'):
                new_key = new_key.replace('diffusion_model.', '')
            original_state_dict[new_key] = value
        
        # 记录转换前的样本键名
        converted_sample_keys = list(original_state_dict.keys())[:5]
        logger.info(f"After removing prefix (sample): {converted_sample_keys}")
        
        # 使用完整转换函数
        logger.info("Converting to diffusers format...")
        converted_state_dict = _convert_non_diffusers_flux2_lora_to_diffusers_manual(original_state_dict.copy())
        
        # 记录转换后的样本键名
        final_sample_keys = list(converted_state_dict.keys())[:5]
        logger.info(f"Final converted keys (sample): {final_sample_keys}")
        
        # 尝试使用转换后的状态字典加载
        try:
            # 先卸载任何现有的 LoRA
            try:
                if hasattr(pipe, 'unload_lora_weights'):
                    pipe.unload_lora_weights()
            except Exception:
                pass
            
            # 临时保存转换后的状态字典
            import tempfile
            temp_path = None
            try:
                # safetensors.torch.save 返回 bytes，需要手动写入文件
                saved_bytes = safetensors.torch.save(converted_state_dict, metadata={"format": "pt"})
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as tmp_file:
                    temp_path = tmp_file.name
                    tmp_file.write(saved_bytes)
                
                logger.info(f"Saved converted state dict to temporary file: {temp_path}")
            except Exception as save_error:
                logger.error(f"Failed to save temporary file: {save_error}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                raise
            
            try:
                # 从临时文件加载 - 使用英文临时文件名避免编码问题
                pipe.load_lora_weights(
                    os.path.dirname(temp_path),
                    weight_name=os.path.basename(temp_path),
                    local_files_only=True
                )
                logger.info(f"Successfully loaded LoRA with converted state dict (scale: {lora_scale})")
                
                # 融合 LoRA 权重
                if hasattr(pipe, 'fuse_lora'):
                    pipe.fuse_lora(lora_scale=lora_scale)
                    logger.info(f"Fused LoRA weights with scale {lora_scale}")
                
                return True
            finally:
                # 清理临时文件
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                    
        except Exception as conversion_error:
            logger.error(f"Conversion attempt also failed: {conversion_error}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Failed to load LoRA weights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


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
    使用 FLUX.2-klein 模型生成图像
    """
    try:
        # 确保尺寸是 8 的倍数
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
        
        # 如果启用了 LoRA，加载 LoRA 权重
        if lora_enable and lora_model and lora_model != "":
            success = load_lora_with_conversion(pipe, lora_model, os.path.basename(lora_model), lora_weight)
            if not success:
                logger.warning("LoRA loading failed, continuing without LoRA")
        
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
        
        # 如果启用了 LoRA，加载 LoRA 权重
        if lora_enable and lora_model and lora_model != "":
            success = load_lora_with_conversion(pipe, lora_model, os.path.basename(lora_model), lora_weight)
            if not success:
                logger.warning("LoRA loading failed, continuing without LoRA")
        
        # 将输入转换为 PIL 图像
        condition_images = []
        
        if img1 is not None:
            if isinstance(img1, str):  # 文件路径
                pil_img1 = Image.open(img1).convert('RGB')
            elif isinstance(img1, np.ndarray):
                if img1.dtype != np.uint8:
                    img1 = (img1 * 255).astype(np.uint8)
                pil_img1 = Image.fromarray(img1, 'RGB')
            elif hasattr(img1, 'convert'):  # 如果已经是 PIL 图像
                pil_img1 = img1.convert('RGB')
            else:
                return None, f"Unsupported img1 type: {type(img1)}"
            
            condition_images.append(pil_img1)
            logger.info(f"Added img1 to condition_images: {type(pil_img1)}, size: {pil_img1.size if pil_img1 else 'N/A'}")
        else:
            return None, "First image is required"
        
        # 如果提供了第二张图像，将其转换为 PIL 图像并添加到条件图像列表
        if img2 is not None:
            if isinstance(img2, str):  # 文件路径
                pil_img2 = Image.open(img2).convert('RGB')
            elif isinstance(img2, np.ndarray):
                if img2.dtype != np.uint8:
                    img2 = (img2 * 255).astype(np.uint8)
                pil_img2 = Image.fromarray(img2, 'RGB')
            elif hasattr(img2, 'convert'):  # 如果已经是 PIL 图像
                pil_img2 = img2.convert('RGB')
            else:
                return None, f"Unsupported img2 type: {type(img2)}"
            
            condition_images.append(pil_img2)
            logger.info(f"Added img2 to condition_images: {type(pil_img2)}, size: {pil_img2.size if pil_img2 else 'N/A'}")
        else:
            logger.info("Only one image provided")

        logger.info(f"Final condition_images count: {len(condition_images)}")
        
        # 获取第一张图像的尺寸作为生成尺寸
        if condition_images:
            width, height = condition_images[0].size
            logger.info(f"Using image dimensions: {width}x{height}")
        else:
            # 如果没有图像，使用默认尺寸
            width, height = 1024, 768
            logger.warning(f"No input images found, using default dimensions: {width}x{height}")
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        
        # 使用 Fluxe2 Klein 的图像条件生成功能
        # 传入图像列表作为条件，并指定输出尺寸为原始图像尺寸
        logger.info("Attempting image-conditioned generation with FLUX_2-klein")
        result_images = pipe(
            prompt=prompt,
            image=condition_images,  # 传入条件图像列表
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size,
            width=width,
            height=height
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
    使用 FLUX.2-klein 模型对蒙版区域进行图像编辑
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
        
        # 如果启用了 LoRA，加载 LoRA 权重
        if lora_enable and lora_model and lora_model != "":
            success = load_lora_with_conversion(pipe, lora_model, os.path.basename(lora_model), lora_weight)
            if not success:
                logger.warning("LoRA loading failed, continuing without LoRA")
        
        # 处理图像和蒙版数据
        if isinstance(image_with_mask, dict):
            # ImageMask 组件返回字典格式，需要提取图像和蒙版
            if 'image' not in image_with_mask or 'mask' not in image_with_mask:
                return None, "Invalid image_with_mask format: missing image or mask"
            
            image = image_with_mask['image']
            mask = image_with_mask['mask']
            
            # 确保图像和蒙版是 PIL 格式
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
        
        # 使用 Flux2 Klein 的 inpainting 功能
        # 将蒙版转换为黑白图像，白色表示需要重绘的区域
        mask = mask.convert('L')  # 确保蒙版是灰度图
        
        # 对蒙版进行二值化处理，使白色区域更清晰
        mask = mask.point(lambda x: 255 if x > 128 else 0, mode='1')
        
        # 将蒙版转换回 L 模式
        mask = mask.convert('L')
        
        logger.info(f"Attempting inpainting with image size: {image.size}, mask size: {mask.size}")
        
        # 使用原始图像尺寸进行生成
        width, height = image.size
        
        result_images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,  # 传递蒙版图像
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size,
            width=width,
            height=height
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
    使用 FLUX.2-klein 模型扩展图像
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
        
        # 如果启用了 LoRA，加载 LoRA 权重
        if lora_enable and lora_model and lora_model != "":
            success = load_lora_with_conversion(pipe, lora_model, os.path.basename(lora_model), lora_weight)
            if not success:
                logger.warning("LoRA loading failed, continuing without LoRA")
        
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
        width, height = extended_image.size
        logger.info(f"Using extended image dimensions: {width}x{height}")
        
        result_images = pipe(
            prompt=prompt,
            image=extended_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=batch_size,
            width=width,
            height=height
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
