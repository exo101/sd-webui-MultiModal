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
import gc  # 添加垃圾回收模块
import cv2  # 添加 OpenCV 用于蒙版处理和裁剪区域计算

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


def cleanup_pipeline(pipe):
    """
    清理模型管道以释放显存
    :param pipe: 要清理的管道对象
    """
    if pipe is None:
        return
    
    try:
        logger.info("Starting pipeline cleanup...")
        
        # 1. 卸载 LoRA（如果已加载）
        try:
            if hasattr(pipe, 'unload_lora_weights'):
                pipe.unload_lora_weights()
                logger.info("Unloaded LoRA weights")
        except Exception as e:
            logger.warning(f"Failed to unload LoRA: {e}")
        
        # 2. 将模型组件移至 CPU
        try:
            if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                pipe.transformer.to('cpu')
                logger.info("Moved transformer to CPU")
        except Exception as e:
            logger.warning(f"Failed to move transformer to CPU: {e}")
        
        try:
            if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
                pipe.text_encoder.to('cpu')
                logger.info("Moved text encoder to CPU")
        except Exception as e:
            logger.warning(f"Failed to move text encoder to CPU: {e}")
        
        try:
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                pipe.vae.to('cpu')
                logger.info("Moved VAE to CPU")
        except Exception as e:
            logger.warning(f"Failed to move VAE to CPU: {e}")
        
        # 3. 删除管道引用
        try:
            del pipe
            logger.info("Deleted pipeline reference")
        except Exception as e:
            logger.warning(f"Failed to delete pipeline: {e}")
        
        # 4. 清空 CUDA 缓存
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Emptied CUDA cache")
        except Exception as e:
            logger.warning(f"Failed to empty CUDA cache: {e}")
        
        # 5. 重置 CUDA 种子
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logger.info("Reset CUDA memory stats")
        except Exception as e:
            logger.warning(f"Failed to reset CUDA memory stats: {e}")
        
        # 6. 执行垃圾回收
        try:
            gc.collect()
            logger.info("Executed garbage collection")
        except Exception as e:
            logger.warning(f"Failed to execute garbage collection: {e}")
            
    except Exception as e:
        logger.error(f"Error during pipeline cleanup: {e}")


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
    pipe = None
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
        import traceback
        logger.error(traceback.format_exc())
        # 发生错误时清理资源
        cleanup_pipeline(pipe)
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
    pipe = None
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
        
        # 获取输入图像的尺寸 - 根据规范必须保持输入输出尺寸一致性
        if condition_images and len(condition_images) > 0:
            input_width, input_height = condition_images[0].size
            logger.info(f"Input image size: {input_width}x{input_height}")
        else:
            logger.warning("No input images provided, using default size")
            input_width, input_height = 1024, 768
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        
        # 使用 Fluxe2 Klein 的图像条件生成功能
        # 传入图像列表作为条件，并显式传递尺寸参数以保持输入输出一致性
        logger.info("Attempting image-conditioned generation with FLUX_2-klein")
        result_images = pipe(
            prompt=prompt,
            image=condition_images,  # 传入条件图像列表
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=input_width,  # 显式传递宽度参数
            height=input_height,  # 显式传递高度参数
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
    image_with_mask,
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
    使用 FLUX.2-klein 模型进行局部编辑
    
    核心原理:
    1. 从 Gradio ImageMask 组件提取背景图像和蒙版区域信息
    2. 将蒙版区域位置转换为文本描述，增强提示词
    3. 将完整图像传递给 FLUX.2-klein，通过增强的提示词指导模型只编辑蒙版区域
    4. 模型基于语义理解自主识别并编辑指定区域
    
    Args:
        image_with_mask: Gradio ImageMask 返回的 dict 对象，包含:
            - 'background': 原始背景图像 (PIL.Image)
            - 'layers': 蒙版图层的列表 (List[PIL.Image])
            - 'composite': 合成图像
        prompt: 编辑提示词
        steps: 推理步数
        guidance_scale: 引导系数
        seed: 随机种子
        model_path: 模型路径
        batch_size: 批量大小
        lora_enable: 是否启用 LoRA
        lora_model: LoRA 模型路径
        lora_weight: LoRA 权重
        
    Returns:
        result_images: 编辑后的图像列表
        message: 状态消息
    """
    pipe = None
    try:
        # 加载模型管道
        pipe = load_flux_klein_pipeline(model_path)
        if pipe is None:
            return None, f"Failed to load model from {model_path}"
        
        # 记录生成参数
        optimization_used = "Flash Attention" if FLASH_ATTENTION_AVAILABLE else ("SageAttention" if SAGE_ATTENTION_AVAILABLE else "None")
        logger.info(f"Starting local edit with parameters: "
                   f"Prompt='{prompt}', Steps={steps}, Guidance={guidance_scale}, "
                   f"Seed={seed}, Model={model_path}, Batch size={batch_size}, "
                   f"LoRA={lora_model if lora_enable else 'Disabled'}, "
                   f"Optimization={optimization_used}")
        
        # 应用注意力优化
        apply_attention_optimizations(pipe)
        
        # 如果启用了 LoRA，加载 LoRA 权重
        if lora_enable and lora_model and lora_model != "":
            success = load_lora_with_conversion(pipe, lora_model, os.path.basename(lora_model), lora_weight)
            if not success:
                logger.warning("LoRA loading failed, continuing without LoRA")
        
        # 从 Gradio ImageMask 中提取图像和蒙版
        logger.info(f"Received image_with_mask dict with keys: {list(image_with_mask.keys())}")
        
        # 提取背景图像
        if isinstance(image_with_mask, dict) and 'background' in image_with_mask:
            background_image = image_with_mask.get('background')
            if background_image is None:
                error_msg = "Background image is missing or None"
                logger.error(error_msg)
                return None, error_msg
            
            logger.info(f"Successfully extracted background image, type: {type(background_image)}")
            
            # 转换为 PIL.Image 格式
            if hasattr(background_image, 'convert'):
                original_image = background_image.convert('RGB')
            else:
                original_image = Image.fromarray(np.array(background_image).astype(np.uint8)).convert('RGB')
            
            # 提取蒙版图层
            mask_layer = None
            if 'layers' in image_with_mask and len(image_with_mask['layers']) > 0:
                # layers[0] 通常是蒙版图層
                mask_layer = image_with_mask['layers'][0]
                logger.info(f"Extracted mask from layers[0], type: {type(mask_layer)}")
                
                # 转换为 PIL.Image 格式
                if not isinstance(mask_layer, Image.Image):
                    mask_layer = Image.fromarray(np.array(mask_layer).astype(np.uint8))
            else:
                logger.warning("No mask layer found, will process full image")
            
            # 如果没有找到蒙版，创建一个全黑的蒙版 (表示不编辑任何区域)
            if mask_layer is None:
                mask_layer = Image.new('L', original_image.size, 0)
                logger.info("Created blank mask for full image processing")
            
        else:
            error_msg = f"Unknown image_with_mask format. Available keys: {list(image_with_mask.keys()) if isinstance(image_with_mask, dict) else 'Not a dict'}"
            logger.error(error_msg)
            return None, error_msg
        
        # 分析蒙版区域，生成位置描述
        mask_array = np.array(mask_layer.convert('L'))
        _, binary_mask = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # 计算蒙版覆盖比例
        mask_coverage = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1])
        
        # 增强提示词：添加位置信息和编辑范围说明
        enhanced_prompt = prompt
        
        if np.sum(binary_mask) > 0:
            # 找到蒙版的边界框
            coords = cv2.findNonZero(binary_mask)
            x, y, w, h = cv2.boundingRect(coords)
            
            # 计算相对位置
            img_w, img_h = original_image.size
            center_x = x + w / 2
            center_y = y + h / 2
            rel_x = center_x / img_w  # 0-1 之间
            rel_y = center_y / img_h  # 0-1 之间
            
            # 生成位置描述
            position_desc = []
            
            # 水平位置
            if rel_x < 0.33:
                position_desc.append("左侧")
            elif rel_x > 0.67:
                position_desc.append("右侧")
            else:
                position_desc.append("中间")
            
            # 垂直位置
            if rel_y < 0.33:
                position_desc.append("上部")
            elif rel_y > 0.67:
                position_desc.append("下部")
            else:
                position_desc.append("中部")
            
            # 区域大小描述
            area_ratio = (w * h) / (img_w * img_h)
            if area_ratio < 0.05:
                size_desc = "小范围"
            elif area_ratio < 0.2:
                size_desc = "中等范围"
            else:
                size_desc = "大范围"
            
            # 构建位置描述前缀
            location_prefix = f"[{size_desc}{''.join(position_desc)}区域]"
            
            # 增强提示词：添加明确的局部编辑指令
            enhanced_prompt = f"{location_prefix}{prompt}。只修改标记区域内的内容，保持其他区域完全不变。精确编辑标记位置，不要影响周围区域。"
            
            logger.info(f"Mask coverage: {mask_coverage:.2%}, position: {''.join(position_desc)}, area: {size_desc}")
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
        else:
            logger.info("No mask detected, using original prompt")
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        
        # 生成图像
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        start_time = time.time()
        
        # 使用 Flux2 Klein 的图像编辑功能
        # 传递完整图像，通过增强的提示词指导模型进行局部编辑
        logger.info(f"Editing full image with enhanced prompt: {enhanced_prompt}")
        logger.info(f"Input image size: {original_image.width}x{original_image.height}")
        
        result_images = pipe(
            prompt=enhanced_prompt,
            image=original_image,  # 传递完整图像
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=original_image.width,  # 使用原图宽度
            height=original_image.height,  # 使用原图高度
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        
        end_time = time.time()
        
        # 记录性能信息
        logger.info(f"Generated {len(result_images)} edited images in {end_time - start_time:.2f}s using {optimization_used} optimization")
        
        # 保存结果图像
        final_images = []
        for i, edited_image in enumerate(result_images):
            # 确保输出尺寸与输入一致
            if edited_image.size != original_image.size:
                logger.warning(f"Edited image size {edited_image.size} != original size {original_image.size}, resizing...")
                edited_image = edited_image.resize(original_image.size, Image.Resampling.LANCZOS)
            
            # 保存图像
            unique_filename = f"flux_klein_local_edit_{uuid.uuid4().hex}_{i}.png"
            save_path = os.path.join("outputs", "flux_klein_edit", unique_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            edited_image.save(save_path)
            final_images.append(edited_image)
            
            logger.info(f"Saved edited image to {save_path}")
        
        return final_images, f"Successfully generated {len(final_images)} locally edited images"

    except Exception as e:
        error_msg = f"Error during local image editing: {str(e)}"
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
    pipe = None
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
        
        # 使用扩展后的图像进行生成，并显式传递新尺寸参数
        logger.info(f"Generating extended image with size: {new_width}x{new_height}")
        result_images = pipe(
            prompt=prompt,
            image=extended_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=new_width,  # 显式传递扩展后的宽度
            height=new_height,  # 显式传递扩展后的高度
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
        # 发生错误时清理资源
        cleanup_pipeline(pipe)
        return None, error_msg
    finally:
        # 确保资源被释放
        cleanup_pipeline(pipe)
