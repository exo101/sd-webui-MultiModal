import gradio as gr
import os
import sys
import json
import time
import traceback
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from modules import shared

# 导入发送到分镜功能
try:
    from scripts.storyboard_assistant import send_to_storyboard
    STORYBOARD_AVAILABLE = True
except Exception as e:
    print(f"⚠️ 导入分镜助手失败：{e}")
    STORYBOARD_AVAILABLE = False

# 尝试导入SageAttention和Flash Attention
try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# Flash Attention检测
FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass

from backend import attention

# 尝试导入ModelScope
try:
    from modelscope import ZImageImg2ImgPipeline as ModelScopeZImageImg2ImgPipeline
    MODELScope_AVAILABLE = True
except ImportError as e:
    MODELScope_AVAILABLE = False
    ModelScopeZImageImg2ImgPipeline = None

# 检查模型文件是否存在
models_dir = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
model_exists = models_dir.exists() and any(models_dir.iterdir())

# 模块是否可用的标志 - 仅基于 ModelScope 是否可用，不依赖模型文件是否存在
Z_IMAGE_IMG2IMG_MODULE_AVAILABLE = MODELScope_AVAILABLE

# 确保模块可以被正确导入
if __name__ != "__main__":
    # 当作为模块导入时，不需要立即检查模型是否存在，而是等到实际使用时再检查
    pass

def ensure_model_directory():
    """确保模型目录存在，如果不存在则创建并给出提示"""
    if not models_dir.exists():
        print(f"[WARNING] Z-Image-Turbo模型目录不存在: {models_dir}")
        print(f"[INFO] 请将Z-Image-Turbo模型文件下载到以下路径: {models_dir}")
        print(f"[INFO] 您可以从ModelScope或其他来源下载模型文件")
        models_dir.mkdir(parents=True, exist_ok=True)
        return False
    elif not any(models_dir.iterdir()):
        print(f"[WARNING] Z-Image-Turbo模型目录为空: {models_dir}")
        print(f"[INFO] 请将Z-Image-Turbo模型文件放置在上述路径中")
        return False
    else:
        return True

# 注释掉模块加载时的模型目录检查，只在实际使用时进行检查
# ensure_model_directory()

# 模型和输出目录
models_dir = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
output_dir = Path(shared.data_path) / "outputs" / "z-image-turbo"
output_dir.mkdir(parents=True, exist_ok=True)

# 全局变量
pipe = None
model_loaded = False
current_model_type = None  # 记录当前加载的模型类型 ('original', 'nunchaku', 'fp8')

def load_model_if_needed(model_type='original', nunchaku_precision='fp4', nunchaku_rank=128, fp8_model_name=None):
    """按需加载Z-Image-Turbo模型，增加FP8支持"""
    global pipe, model_loaded, current_model_type

    try:
        # 检查模型是否已经加载，且是相同类型的模型
        if (model_loaded and pipe is not None and
            current_model_type == model_type and
            (model_type != 'nunchaku' or (current_model_type == 'nunchaku' and
             getattr(pipe, 'nunchaku_precision', None) == nunchaku_precision and
             getattr(pipe, 'nunchaku_rank', None) == nunchaku_rank))):
            return "模型已加载"

        # 确保模型目录存在
        model_save_path = models_dir
        model_save_path.mkdir(parents=True, exist_ok=True)

        # 确保参数类型正确
        nunchaku_rank = int(nunchaku_rank) if nunchaku_rank is not None else 128

        if model_type == 'fp8':
            # FP8模型加载逻辑
            try:
                # 首先检查是否指定了特定的FP8模型文件名
                if fp8_model_name:
                    # 确保fp8_model_name是字符串类型
                    if not isinstance(fp8_model_name, str):
                        return f"FP8模型文件名必须是字符串类型，当前类型: {type(fp8_model_name)}"
                    
                    # 先在指定的模型保存路径中查找
                    fp8_model_path = model_save_path / fp8_model_name
                    if fp8_model_path.exists():
                        print(f"[INFO] 在模型目录找到指定的FP8模型文件: {fp8_model_path}")
                    else:
                        # 如果在指定路径没找到，则在整个模型路径下递归搜索
                        all_fp8_files = list(Path(shared.models_path).rglob("*.fp8")) + list(Path(shared.models_path).rglob("*fp8*.safetensors"))
                        fp8_model_path = None
                        for file_path in all_fp8_files:
                            if file_path.name == fp8_model_name:
                                fp8_model_path = file_path
                                break
                    
                    if not fp8_model_path:
                        return f"选定的FP8模型文件不存在: {fp8_model_name}"
                else:
                    # 没有指定特定模型文件，自动搜索可用的FP8模型
                    # 优先在当前模型目录中查找
                    fp8_files = list(model_save_path.glob("*.fp8")) + list(model_save_path.glob("*fp8*.safetensors"))
                    
                    # 如果当前目录没有，尝试在Tongyi-MAI/Z-Image-Turbo子目录查找
                    if not fp8_files:
                        subdir_path = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
                        fp8_files = list(subdir_path.glob("*.fp8")) + list(subdir_path.glob("*fp8*.safetensors"))
                        if fp8_files:
                            print(f"[INFO] 在子目录找到FP8模型文件: {subdir_path}")
                    
                    # 如果还是没有，搜索整个模型目录下的所有FP8文件
                    if not fp8_files:
                        all_fp8_files = list(Path(shared.models_path).rglob("*.fp8")) + list(Path(shared.models_path).rglob("*fp8*.safetensors"))
                        if all_fp8_files:
                            fp8_files = all_fp8_files
                            print(f"[INFO] 在模型路径中找到FP8模型文件")
                    
                    # 检查是否找到了任何FP8文件
                    if not fp8_files:
                        return f"FP8模型文件未找到，请确保在 {shared.models_path} 或其子目录中有.fp8或*fp8*.safetensors文件"
                    
                    # 使用第一个找到的FP8文件
                    fp8_model_path = fp8_files[0]
                
                print(f"[INFO] 找到FP8模型文件: {fp8_model_path}")
                
                print(f"[INFO] 准备加载FP8模型文件: {fp8_model_path}")
                
                # 加载FP8模型 - 使用ModelScope的量化功能
                from modelscope import ZImagePipeline
                
                # 确保基础模型路径存在
                base_model_path = models_dir
                
                if not base_model_path.exists() or not any(base_model_path.iterdir()):
                    # 如果本地没有基础模型，提示用户需要先下载
                    return (
                        f"无法加载FP8模型：本地没有找到基础模型。 "
                        f"请先下载Z-Image-Turbo基础模型到以下路径：{base_model_path}"
                    )
                
                # 从本地基础模型路径加载完整pipeline
                pipe = ZImagePipeline.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch.bfloat16,  # 使用bfloat16作为默认类型，避免FP8兼容性问题
                    low_cpu_mem_usage=False,
                    local_files_only=True  # 确保只从本地加载
                )
                
                # 应用注意力优化
                if FLASH_ATTENTION_AVAILABLE or SAGE_ATTENTION_AVAILABLE:
                    print(f"[INFO] 应用注意力优化...")
                    apply_attention_optimizations(pipe, model_type)
                
                # 使用模型自带的设备管理机制来优化内存使用
                # 启用模型CPU卸载以节省显存
                if hasattr(pipe, 'enable_model_cpu_offload'):
                    pipe.enable_model_cpu_offload()
                elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                    pipe.enable_sequential_cpu_offload()
                else:
                    # 如果没有CPU卸载功能，则尝试将模型移动到GPU
                    try:
                        pipe = pipe.to("cuda")
                    except Exception as move_error:
                        pass

                # 设置为FP8模型类型
                current_model_type = 'fp8'
                
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] 加载FP8模型时出错: {error_msg}")
                import traceback
                print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")
                return f"FP8模型加载失败: {str(e)}"
                
        elif model_type == 'nunchaku':
            # Nunchaku模型加载逻辑
            try:
                # 确保使用扩展目录中的nunchaku库
                extension_nunchaku_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nunchaku")
                if extension_nunchaku_path not in sys.path:
                    sys.path.insert(0, extension_nunchaku_path)
                
                # 从nunchaku库导入正确的类
                from nunchaku import NunchakuZImageTransformer2DModel
                from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
            except ImportError as e:
                return f"错误：未安装Nunchaku库或无法导入NunchakuZImageTransformer2DModel: {str(e)}"

            # 检查nunchaku模型文件是否存在
            nunchaku_model_filename = f"svdq-{nunchaku_precision}_r{nunchaku_rank}-z-image-turbo.safetensors"
            nunchaku_model_path = model_save_path / nunchaku_model_filename

            if not nunchaku_model_path.exists():
                return f"Nunchaku模型文件未找到: {nunchaku_model_path}"

            # 加载Nunchaku加速模型
            try:
                transformer = NunchakuZImageTransformer2DModel.from_pretrained(
                    str(nunchaku_model_path)
                )
            except Exception as e:
                return f"加载Nunchaku模型失败: {str(e)}"

            # 确保transformer成功加载
            if transformer is None:
                return "加载Nunchaku模型失败: 返回的transformer为None"

            # 使用transformer创建pipeline
            try:
                pipe = ZImagePipeline.from_pretrained(
                    str(model_save_path),
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False
                )
                
                # 应用注意力优化
                if FLASH_ATTENTION_AVAILABLE or SAGE_ATTENTION_AVAILABLE:
                    print(f"[INFO] 应用注意力优化...")
                    apply_attention_optimizations(pipe, model_type)
                    
            except Exception as e:
                return f"创建ZImagePipeline失败: {str(e)}"

            # 确保pipe成功创建
            if pipe is None:
                return "创建ZImagePipeline失败: 返回的pipe为None"

            # 使用模型自带的设备管理机制
            # 启用模型CPU卸载以节省显存
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                pipe.enable_sequential_cpu_offload()
            else:
                # 如果没有CPU卸载功能，则尝试将模型移动到GPU
                try:
                    pipe = pipe.to("cuda")
                except Exception as move_error:
                    pass

            # 记录Nunchaku参数
            if pipe is not None:
                pipe.nunchaku_precision = nunchaku_precision
                pipe.nunchaku_rank = nunchaku_rank

            current_model_type = 'nunchaku'
        else:
            # 原始模型加载逻辑
            try:
                from modelscope import ZImagePipeline
            except ImportError:
                return "错误：ModelScope不可用"
                
            if not model_save_path.exists():
                return f"本地模型路径不存在: {model_save_path}"

            # 检查目录是否包含任何文件（避免glob("*")可能引发的异常）
            try:
                if next(model_save_path.iterdir(), None) is None:
                    return f"本地模型路径为空: {model_save_path}"
            except Exception as e:
                return f"检查模型路径时出错: {model_save_path} - {str(e)}"

            # 从本地路径加载模型，确保使用force_download=False和local_files_only=True参数
            try:
                pipe = ZImagePipeline.from_pretrained(
                    str(model_save_path),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                    local_files_only=True  # 确保只使用本地文件
                )
                
                # 应用注意力优化
                if FLASH_ATTENTION_AVAILABLE or SAGE_ATTENTION_AVAILABLE:
                    print(f"[INFO] 应用注意力优化...")
                    apply_attention_optimizations(pipe, model_type)
                    
            except Exception as e:
                return f"加载原始模型失败: {str(e)}"

            # 使用模型自带的设备管理机制
            # 启用模型CPU卸载以节省显存
            if hasattr(pipe, 'enable_model_cpu_offload'):
                print("[INFO] 启用模型CPU卸载以节省显存")
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                print("[INFO] 启用顺序CPU卸载")
                pipe.enable_sequential_cpu_offload()
            else:
                # 如果没有CPU卸载功能，则尝试将模型移动到GPU
                try:
                    print("[INFO] 将模型移动到CUDA设备")
                    pipe = pipe.to("cuda")
                except Exception as move_error:
                    print(f"[WARNING] 将模型移动到CUDA设备失败: {move_error}")

            current_model_type = 'original'

        # 确保pipe不为None
        if pipe is None:
            return "模型加载失败：管道对象初始化失败"
            
        model_loaded = True
        return "模型加载成功"
    except Exception as e:
        error_msg = str(e)
        return f"模型加载失败: {error_msg}"


def apply_attention_optimizations(pipe, model_type='original'):
    """应用注意力优化到模型"""
    try:
        print(f"[DEBUG] 开始应用注意力优化，模型类型: {model_type}")
        
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            print(f"[DEBUG] 找到transformer组件，开始应用优化...")
            # 应用SageAttention或Flash Attention优化
            if SAGE_ATTENTION_AVAILABLE:
                print(f"[INFO] 为{model_type}模型应用SageAttention优化...")
                replace_transformer_attention_with_sage(pipe.transformer)
            elif FLASH_ATTENTION_AVAILABLE:
                print(f"[INFO] 为{model_type}模型应用Flash Attention优化...")
                replace_transformer_attention_with_flash(pipe.transformer)
            else:
                print(f"[INFO] 未检测到可用的注意力优化库")
        else:
            print(f"[WARNING] 无法找到pipe.transformer组件，跳过注意力优化")
            print(f"[DEBUG] pipe属性: {hasattr(pipe, 'transformer')}")
            if hasattr(pipe, 'transformer'):
                print(f"[DEBUG] transformer值: {pipe.transformer}")
    except Exception as e:
        print(f"[ERROR] 应用注意力优化失败: {str(e)}")
        import traceback
        print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")
        # 不让注意力优化的失败影响整体流程
        pass


def replace_transformer_attention_with_sage(transformer):
    """将transformer中的注意力机制替换为SageAttention"""
    try:
        for name, module in transformer.named_modules():
            # 确保name是字符串类型，避免'int' object is not iterable错误
            if not isinstance(name, str):
                continue
                
            if 'attn' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # 替换forward方法以使用SageAttention
                original_forward = module.forward
                
                def make_new_forward(orig_forward):
                    def sage_forward(hidden_states, *args, **kwargs):
                        # 原始的query/key/value投影
                        query = orig_forward.__self__.to_q(hidden_states)
                        key = orig_forward.__self__.to_k(hidden_states)
                        value = orig_forward.__self__.to_v(hidden_states)

                        # 确保维度正确
                        batch_size, seq_len, dim = query.shape
                        head_dim = dim // orig_forward.__self__.heads
                        heads = orig_forward.__self__.heads

                        # 重塑为多头形式
                        query = query.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        key = key.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

                        # 使用SageAttention进行计算
                        out = sageattn(query, key, value, 
                                     scale=head_dim**(-0.5), 
                                     attention_dropout=0.0, 
                                     causal=False)
                        
                        # 重塑回原始格式
                        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
                        
                        # 通过输出投影
                        if hasattr(orig_forward.__self__, 'to_out'):
                            if not isinstance(orig_forward.__self__.to_out, (list, tuple)):
                                out = orig_forward.__self__.to_out(out)
                            else:
                                for layer in orig_forward.__self__.to_out:
                                    out = layer(out)
                        
                        return out
                    return sage_forward
                
                module.forward = make_new_forward(original_forward).__get__(module, type(module))
        print("[INFO] SageAttention优化应用成功")
    except Exception as e:
        print(f"[ERROR] 应用SageAttention优化失败: {str(e)}")
        import traceback
        print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")


def replace_transformer_attention_with_flash(transformer):
    """将transformer中的注意力机制替换为FlashAttention"""
    try:
        import torch.nn.functional as F
        
        for name, module in transformer.named_modules():
            # 确保name是字符串类型，避免'int' object is not iterable错误
            if not isinstance(name, str):
                continue
                
            if 'attn' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # 替换forward方法以使用Flash Attention
                original_forward = module.forward
                
                def make_new_forward(orig_forward):
                    def flash_forward(hidden_states, *args, **kwargs):
                        # 原始的query/key/value投影
                        query = orig_forward.__self__.to_q(hidden_states)
                        key = orig_forward.__self__.to_k(hidden_states)
                        value = orig_forward.__self__.to_v(hidden_states)

                        # 确保维度正确
                        batch_size, seq_len, dim = query.shape
                        head_dim = dim // orig_forward.__self__.heads
                        heads = orig_forward.__self__.heads

                        # 重塑为多头形式
                        query = query.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        key = key.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

                        # 尝试使用Flash Attention
                        try:
                            # Flash Attention 2 implementation
                            from flash_attn import flash_attn_func
                            out = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
                        except Exception:
                            # 回退到PyTorch的scaled_dot_product_attention
                            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

                        # 重塑回原始格式
                        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
                        
                        # 通过输出投影
                        if hasattr(orig_forward.__self__, 'to_out'):
                            if not isinstance(orig_forward.__self__.to_out, (list, tuple)):
                                out = orig_forward.__self__.to_out(out)
                            else:
                                for layer in orig_forward.__self__.to_out:
                                    out = layer(out)
                                    
                        return out
                    return flash_forward
                
                module.forward = make_new_forward(original_forward).__get__(module, type(module))
        print("[INFO] Flash Attention优化应用成功")
    except Exception as e:
        print(f"[ERROR] 应用Flash Attention优化失败: {str(e)}")
        import traceback
        print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")


def get_lora_list():
    """获取LoRA列表"""
    try:
        lora_path = Path(shared.models_path) / "Lora"
        if lora_path.exists():
            lora_files = [f.stem for f in lora_path.glob("*.safetensors")] + \
                         [f.stem for f in lora_path.glob("*.ckpt")] + \
                         [f.stem for f in lora_path.glob("*.pt")]
            return lora_files
        return []
    except Exception as e:
        print(f"获取LoRA列表失败: {e}")
        return []


def open_folder(folder_path):
    """打开指定的文件夹"""
    import subprocess
    try:
        if os.name == 'nt':  # Windows系统
            os.startfile(folder_path)
        elif os.name == 'posix':  # Linux/Mac系统
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', folder_path])
        return "文件夹已打开"
    except Exception as e:
        return f"打开文件夹失败: {str(e)}"


def save_image(image_np, seed, index=0):
    """保存图像到输出目录"""
    try:
        # 确保图像数据在正确的范围内
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # 确保数据类型是uint8
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        
        # 创建PIL图像对象
        image = Image.fromarray(image_np)
        
        # 确保seed是整数类型
        seed = int(seed) if seed is not None else int(time.time())
        
        # 生成文件名
        timestamp = int(time.time())
        filename = f"zimage_{timestamp}_s{seed}_{index}_img2img.png"
        filepath = output_dir / filename
        
        # 保存图像
        image.save(filepath)
        
        return str(filepath)
    except Exception as e:
        print(f"保存图像失败: {str(e)}")
        return None


def generate_image_img2img(init_image, prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size=1,
                           strength=0.6, enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7,
                           use_nunchaku=False, nunchaku_precision='fp4', nunchaku_rank=128,
                           use_fp8=False, fp8_model_name=None, lora_enable=False, lora_model_1=None, lora_weight_1=0.8, lora_model_2=None, lora_weight_2=0.8):
    """使用Z-Image-Turbo进行图像到图像转换，添加FP8模型支持"""
    
    global pipe, model_loaded

    try:
        # 添加详细的参数类型检查日志
        print(f"[DEBUG] generate_image_img2img 参数类型检查:")
        print(f"  - prompt类型: {type(prompt)} = {repr(prompt)}")
        print(f"  - seed类型: {type(seed)} = {seed}")
        print(f"  - steps类型: {type(steps)} = {steps}")
        print(f"  - width类型: {type(width)} = {width}")
        print(f"  - height类型: {type(height)} = {height}")
        print(f"  - batch_size类型: {type(batch_size)} = {batch_size}")
        print(f"  - strength类型: {type(strength)} = {strength}")
        print(f"  - init_image类型: {type(init_image)}")
        
        # 检查提示词
        if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
            return "错误：请输入正向提示词", None

        # 检查初始图像
        if init_image is None:
            return "错误：请输入初始图像", None

        # 如果种子为-1，则生成随机种子
        if seed == -1 or seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # 确保所有数值参数都是正确的类型
        try:
            seed = int(seed) if seed is not None else random.randint(0, 2**32 - 1)
            steps = int(steps) if steps is not None else 8
            width = int(width) if width is not None else 1024
            height = int(height) if height is not None else 1024
            batch_size = int(batch_size) if batch_size is not None else 1
            strength = float(strength) if strength is not None else 0.6
            cfg_scale = float(cfg_scale) if cfg_scale is not None else 0.0
        except (ValueError, TypeError) as e:
            error_msg = f"参数类型转换错误: {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] 类型信息 - seed:{type(seed)}, steps:{type(steps)}, width:{type(width)}, height:{type(height)}")
            return error_msg, None

        print(f"[DEBUG] 转换后的参数:")
        print(f"  - seed: {seed} (类型: {type(seed)})")
        print(f"  - steps: {steps} (类型: {type(steps)})")
        print(f"  - width: {width} (类型: {type(width)})")
        print(f"  - height: {height} (类型: {type(height)})")


        # 确定模型类型
        if use_fp8:
            model_type = 'fp8'
        elif use_nunchaku:
            model_type = 'nunchaku'
        else:
            model_type = 'original'

        # 按需加载模型
        if model_type == 'fp8':
            status = load_model_if_needed(model_type, fp8_model_name=fp8_model_name)
        elif model_type == 'nunchaku':
            # 确保nunchaku参数类型正确
            nunchaku_rank = int(nunchaku_rank) if nunchaku_rank is not None else 128
            status = load_model_if_needed(model_type, nunchaku_precision, nunchaku_rank)
        else:
            status = load_model_if_needed(model_type)

        if "失败" in str(status):  # 更安全的字符串检查
            return status, None

        # 确保pipe不为None
        if pipe is None:
            error_msg = f"错误：管道未正确初始化。模型类型: {model_type}, 加载状态: {status}"
            print(f"[ERROR] {error_msg}")
            return error_msg, None

        # 为Turbo模型设置合适的参数
        # 根据用户设置的最大步数限制，而不是固定为10
        max_allowed_steps = getattr(shared.opts, 'zimage_max_inference_steps', 10)  # 从设置中获取最大步数限制
        actual_steps = min(steps, max_allowed_steps)
        actual_guidance = 0.0  # 根据官方示例，Turbo模型应该使用0.0的guidance_scale

        # 处理LoRA
        lora_applied = False
        if lora_enable and (lora_model_1 or lora_model_2):
            # 检查当前模型是否支持LoRA
            if current_model_type == 'fp8':
                # FP8模型支持LoRA，通过transformer加载
                
                # FP8模型的transformer可能有不同的LoRA支持方式
                try:
                    # 验证LoRA文件格式是否与FP8模型兼容
                    lora_paths = []
                    if lora_model_1:
                        lora_path_1 = Path(shared.models_path) / "Lora" / f"{lora_model_1}.safetensors"
                        if not lora_path_1.exists():
                            for ext in ['.ckpt', '.pt']:
                                temp_path = Path(shared.models_path) / "Lora" / f"{lora_model_1}{ext}"
                                if temp_path.exists():
                                    lora_path_1 = temp_path
                                    break
                        if lora_path_1.exists():
                            lora_paths.append((str(lora_path_1), lora_weight_1))
                            
                    if lora_model_2:
                        lora_path_2 = Path(shared.models_path) / "Lora" / f"{lora_model_2}.safetensors"
                        if not lora_path_2.exists():
                            for ext in ['.ckpt', '.pt']:
                                temp_path = Path(shared.models_path) / "Lora" / f"{lora_model_2}{ext}"
                                if temp_path.exists():
                                    lora_path_2 = temp_path
                                    break
                        if lora_path_2.exists():
                            lora_paths.append((str(lora_path_2), lora_weight_2))
                    
                    # 应用LoRA到FP8模型的transformer
                    for lora_path, lora_weight in lora_paths:
                        print(f"[INFO] 应用LoRA到FP8 transformer: {lora_path}，权重: {lora_weight}")
                        
                        # 获取LoRA文件名（不含扩展名），作为weight_name
                        lora_name = Path(lora_path).stem
                        
                        try:
                            # 尝试加载LoRA权重，先检查是否与FP8模型兼容
                            pipe.load_lora_weights(
                                lora_path, 
                                local_files_only=True,
                                weight_name=lora_name
                            )
                            
                            # 使用fuse_lora融合LoRA权重
                            pipe.fuse_lora(lora_scale=lora_weight)
                            lora_applied = True
                            
                        except RuntimeError as e:
                            if "must match the size of tensor" in str(e):
                                return f"LoRA与FP8模型不兼容: 维度不匹配。LoRA通常是针对特定基础模型训练的，可能与量化后的FP8模型不兼容。", None
                            else:
                                raise e  # 重新抛出其他RuntimeError
                        except Exception as e:
                            return f"FP8 LoRA加载失败: {str(e)}", None
                except Exception as e:
                    return f"FP8 LoRA加载失败: {str(e)}", None
            elif current_model_type == 'nunchaku':
                # 对于Nunchaku模型，使用Nunchaku内置的LoRA支持
                
                # 检查transformer是否支持LoRA功能
                if hasattr(pipe.transformer, 'update_lora_params'):
                    try:
                        # 验证LoRA文件格式是否与Nunchaku兼容
                        lora_paths = []
                        if lora_model_1:
                            lora_path_1 = Path(shared.models_path) / "Lora" / f"{lora_model_1}.safetensors"
                            if not lora_path_1.exists():
                                for ext in ['.ckpt', '.pt']:
                                    temp_path = Path(shared.models_path) / "Lora" / f"{lora_model_1}{ext}"
                                    if temp_path.exists():
                                        lora_path_1 = temp_path
                                        break
                            if lora_path_1.exists():
                                lora_paths.append((str(lora_path_1), lora_weight_1))
                                
                        if lora_model_2:
                            lora_path_2 = Path(shared.models_path) / "Lora" / f"{lora_model_2}.safetensors"
                            if not lora_path_2.exists():
                                for ext in ['.ckpt', '.pt']:
                                    temp_path = Path(shared.models_path) / "Lora" / f"{lora_model_2}{ext}"
                                    if temp_path.exists():
                                        lora_path_2 = temp_path
                                        break
                            if lora_path_2.exists():
                                lora_paths.append((str(lora_path_2), lora_weight_2))
                        
                        # 应用LoRA - 根据nunchaku示例代码，update_lora_params只接受路径参数
                        for lora_path, lora_weight in lora_paths:
                            pipe.transformer.update_lora_params(lora_path)
                            lora_applied = True
                            
                    except Exception as e:
                        return f"Nunchaku LoRA加载失败: {str(e)}", None
                else:
                    pass
            elif current_model_type == 'original':
                # 对于原始模型，使用标准的LoRA加载方式
                try:
                    lora_paths = []
                    if lora_model_1:
                        lora_path_1 = Path(shared.models_path) / "Lora" / f"{lora_model_1}.safetensors"
                        if not lora_path_1.exists():
                            for ext in ['.ckpt', '.pt']:
                                temp_path = Path(shared.models_path) / "Lora" / f"{lora_model_1}{ext}"
                                if temp_path.exists():
                                    lora_path_1 = temp_path
                                    break
                        if lora_path_1.exists():
                            lora_paths.append((str(lora_path_1), lora_weight_1))
                            
                    if lora_model_2:
                        lora_path_2 = Path(shared.models_path) / "Lora" / f"{lora_model_2}.safetensors"
                        if not lora_path_2.exists():
                            for ext in ['.ckpt', '.pt']:
                                temp_path = Path(shared.models_path) / "Lora" / f"{lora_model_2}{ext}"
                                if temp_path.exists():
                                    lora_path_2 = temp_path
                                    break
                            if lora_path_2.exists():
                                lora_paths.append((str(lora_path_2), lora_weight_2))
                    
                    # 加载找到的LoRA
                    for lora_path, lora_weight in lora_paths:
                        # 获取LoRA文件名（不含扩展名），作为weight_name
                        lora_name = Path(lora_path).stem
                        pipe.load_lora_weights(lora_path, local_files_only=True, weight_name=lora_name)
                        pipe.fuse_lora(lora_scale=lora_weight)
                        lora_applied = True
                        
                except Exception as e:
                    return f"标准LoRA加载失败: {str(e)}", None
            else:
                pass

        # 导入Z-Image Img2Img Pipeline
        from diffusers import ZImageImg2ImgPipeline
        
        # 将当前pipe转换为img2img pipeline
        img2img_pipe = ZImageImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            transformer=pipe.transformer,
            scheduler=pipe.scheduler
        )

        # 使用模型自带的设备管理机制
        # 启用模型CPU卸载以节省显存
        if hasattr(img2img_pipe, 'enable_model_cpu_offload'):
            img2img_pipe.enable_model_cpu_offload()
        elif hasattr(img2img_pipe, 'enable_sequential_cpu_offload'):
            img2img_pipe.enable_sequential_cpu_offload()
        else:
            # 如果没有CPU卸载功能，则尝试将模型移动到GPU
            try:
                img2img_pipe = img2img_pipe.to("cuda")
            except Exception as move_error:
                pass

        # 生成图像 - 使用Z-Image Img2Img Pipeline
        try:
            # 转换输入图像为PIL格式
            if isinstance(init_image, np.ndarray):
                init_image_pil = Image.fromarray(init_image)
            else:
                init_image_pil = init_image

            # 调整图像尺寸
            init_image_pil = init_image_pil.resize((width, height))

            # 确保batch_size是有效的整数
            batch_size = int(batch_size) if batch_size is not None else 1
            
            # ZImageImg2ImgPipeline的生成过程
            generator = torch.Generator().manual_seed(seed)
            
            # 对于FP8模型，我们可能需要特别处理以确保正确的数值范围
            if model_type == 'fp8':
                images = img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image_pil,
                    strength=float(strength) if strength is not None else 0.6,
                    num_inference_steps=int(actual_steps),
                    guidance_scale=float(actual_guidance),
                    generator=generator,
                    num_images_per_prompt=batch_size,
                    # 确保输出值在正确范围内
                    output_type="pil"  # 明确指定输出类型
                ).images
            else:
                images = img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image_pil,
                    strength=float(strength) if strength is not None else 0.6,
                    num_inference_steps=int(actual_steps),
                    guidance_scale=float(actual_guidance),
                    generator=generator,
                    num_images_per_prompt=batch_size
                ).images

            # 确保生成的图像不是空的或全是黑色的
            processed_images = []
            for i, image in enumerate(images):
                # 检查图像是否有像素值
                img_array = np.array(image)
                
                # 检查图像是否全黑（所有像素值都接近0）
                if np.mean(img_array) < 5:  # 平均像素值低于5，认为是黑图
                    
                    # 对于FP8模型，尝试更多推理步骤
                    if model_type == 'fp8':
                        re_images = img2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=init_image_pil,
                            strength=float(strength) if strength is not None else 0.6,
                            num_inference_steps=min(int(actual_steps) + 4, 20),  # 增加步数，但不超过20
                            guidance_scale=float(actual_guidance),
                            generator=generator,
                            num_images_per_prompt=1,  # 只重新生成一张
                            output_type="pil"
                        ).images
                        
                        # 再次检查重新生成的图像
                        re_img_array = np.array(re_images[0])
                        if np.mean(re_img_array) < 5:
                            
                            # 如果仍然是黑图，尝试使用不同的随机种子
                            alt_generator = torch.Generator().manual_seed(seed + 12345)
                            re_images = img2img_pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                image=init_image_pil,
                                strength=float(strength) if strength is not None else 0.6,
                                num_inference_steps=min(int(actual_steps) + 8, 25),  # 进一步增加步数
                                guidance_scale=float(actual_guidance),
                                generator=alt_generator,
                                num_images_per_prompt=1,
                                output_type="pil"
                            ).images
                            
                            # 最终检查
                            final_img_array = np.array(re_images[0])
                            if np.mean(final_img_array) < 5:
                                processed_images.append(image)  # 使用原图像
                            else:
                                processed_images.append(re_images[0])  # 使用新图像
                        else:
                            processed_images.append(re_images[0])  # 使用新图像
                    else:
                        # 非FP8模型，按原方式处理
                        alt_images = img2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=init_image_pil,
                            strength=float(strength) if strength is not None else 0.6,
                            num_inference_steps=max(int(actual_steps), 8),
                            guidance_scale=float(actual_guidance),
                            generator=torch.Generator().manual_seed(seed + 1000),
                            num_images_per_prompt=1,
                            output_type="pil"
                        ).images
                        
                        alt_img_array = np.array(alt_images[0])
                        if np.mean(alt_img_array) < 5:
                            processed_images.append(image)  # 使用原图像
                        else:
                            processed_images.append(alt_images[0])  # 使用新图像
                else:
                    processed_images.append(image)  # 图像正常，直接添加

            # 保存图像
            output_images = []
            for idx, image in enumerate(processed_images):
                output_image_path = output_dir / f"{seed}_{idx}_img2img.png"
                image.save(output_image_path)
                output_images.append(str(output_image_path))

            return "转换成功", output_images
        except Exception as e:
            return f"转换失败: {str(e)}", None

    except Exception as e:
        error_details = traceback.format_exc()
        return f"图像到图像转换失败: {str(e)}", None


def create_z_image_img2img_ui():
    """创建Z-Image-Turbo Img2Img UI界面"""

    # 检查ModelScope是否可用
    if not MODELScope_AVAILABLE:
        with gr.Blocks() as demo:
            gr.Markdown("# Z-Image-Turbo 图像到图像转换 (不可用)")
            gr.Markdown("错误：ModelScope 不可用，请检查安装")
        return demo

    # 导入采样器模块
    try:
        from modules import sd_samplers
        sampler_names = [sampler.name for sampler in sd_samplers.visible_samplers()]
        default_sampler = sampler_names[0] if sampler_names else "euler"
    except:
        sampler_names = [
            "Euler",
            "Euler Ancestral",
            "Heun",
            "DPM++ 2M"
        ]
        default_sampler = "Euler"

    with gr.Blocks() as demo:
        gr.Markdown("# Z-Image-Turbo 图像到图像转换")
        gr.Markdown("基于 ModelScope 的超快速图生图模型")

        with gr.Row():
            with gr.Column():  # 左半边 - 参数设置
                init_image = gr.Image(label="输入图像", type="pil", height=400, width=700)

                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="输入您的提示词，例如：一只可爱的猫"
                )

                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    placeholder="输入您不希望出现在图像中的内容"
                )

                with gr.Row():
                    width = gr.Slider(
                        minimum=256, maximum=2048, step=64, value=1024, label="宽度"
                    )
                    height = gr.Slider(
                        minimum=256, maximum=2048, step=64, value=1024, label="高度"
                    )
                    
                # 添加自动设置图像尺寸的按钮
                auto_set_dimensions = gr.Button("从上传图像自动设置尺寸")
                
                # 定义Python函数来获取图像尺寸
                def get_image_dimensions(image):
                    if image is not None:
                        # 确保返回的是整数类型
                        return int(image.width), int(image.height)
                    else:
                        return 1024, 1024  # 默认值
                
                # 绑定按钮点击事件到Python函数
                auto_set_dimensions.click(
                    fn=get_image_dimensions,
                    inputs=[init_image],
                    outputs=[width, height]
                )

                with gr.Row():
                    strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.6, label="重绘强度"
                    )
                    steps = gr.Slider(
                        minimum=1, maximum=50, step=1, value=8, label="推理步数"
                    )
                    cfg_scale = gr.Slider(
                        minimum=0.0, maximum=20.0, step=0.1, value=0.0, label="CFG Scale"
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="随机种子 (-1为随机)", value=-1, precision=0
                    )
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, value=1, label="生成批次"
                    )

                sampler = gr.Dropdown(
                    choices=sampler_names,
                    value=default_sampler,
                    label="采样方法"
                )

                # 添加Nunchaku模型选项
                use_nunchaku = gr.Checkbox(label="使用 Nunchaku 加速模型", value=False)
                with gr.Group(visible=False) as nunchaku_options:
                    gr.Markdown("提示：50系显卡推荐使用 fp4，其他显卡推荐使用 int4")
                    nunchaku_precision = gr.Dropdown(
                        choices=["int4", "fp4"],
                        value="fp4",
                        label="量化精度",
                        interactive=True
                    )
                    nunchaku_rank = gr.Dropdown(
                        choices=[32, 64, 128],
                        value=128,
                        label="Rank值 (较低值速度更快，较高值质量更好)",
                        interactive=True
                    )

                # 添加FP8模型选项
                use_fp8 = gr.Checkbox(label="使用 FP8 量化模型", value=False)
                with gr.Group(visible=False) as fp8_options_group:
                    # 动态获取FP8模型列表
                    fp8_model_choices = []
                    try:
                        # 只检查 Tongyi-MAI/Z-Image-Turbo 目录下的FP8文件
                        model_path = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
                        
                        # 搜索当前目录和子目录下的所有FP8文件
                        fp8_files = list(model_path.rglob("*.fp8")) + list(model_path.rglob("*fp8*.safetensors"))
                        
                        fp8_model_choices = [f.name for f in fp8_files]
                    except Exception as e:
                        print(f"[ERROR] 获取FP8模型列表时出错: {e}")
                    
                    fp8_model_name = gr.Dropdown(
                        choices=fp8_model_choices,
                        value=fp8_model_choices[0] if fp8_model_choices else None,
                        label="FP8模型文件",
                        interactive=True
                    )
                    gr.Markdown("注意：FP8模型支持LoRA功能")

                # 添加LoRA支持选项
                lora_enable = gr.Checkbox(label="启用 LoRA", value=False)
                with gr.Group(visible=False) as lora_options_group:
                    # 获取LoRA列表
                    lora_choices = get_lora_list()
                    
                    with gr.Row():
                        # 支持多选的下拉框
                        lora_model_1 = gr.Dropdown(
                            choices=lora_choices,
                            label="LoRA 模型 1",
                            interactive=True
                        )
                        lora_model_2 = gr.Dropdown(
                            choices=lora_choices,
                            label="LoRA 模型 2",
                            interactive=True
                        )
                    
                    with gr.Row():
                        lora_weight_1 = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="LoRA 权重 1", value=0.8)
                        lora_weight_2 = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="LoRA 权重 2", value=0.8)
                        
                    with gr.Row():
                        refresh_lora_btn = gr.Button("刷新LoRA列表", size="sm")

                # 刷新LoRA列表的函数
                def refresh_lora_list():
                    try:
                        lora_choices = get_lora_list()
                        return [gr.update(choices=lora_choices), gr.update(choices=lora_choices)]
                    except Exception as e:
                        print(f"刷新LoRA列表失败: {e}")
                        return [gr.update(), gr.update()]

                refresh_lora_btn.click(
                    fn=refresh_lora_list,
                    inputs=[],
                    outputs=[lora_model_1, lora_model_2]
                )

                # 添加高分辨率修复(Hires.fix)选项
                enable_hr = gr.Checkbox(label="启用高分辨率修复", value=False)
                with gr.Group(visible=False) as hr_options_group:
                    hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="放大倍数", value=2.0)
                    hr_upscaler = gr.Dropdown(
                        label="放大算法",
                        choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]],
                        value=shared.latent_upscale_default_mode
                    )
                    hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label="高分辨率修复步数", value=0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="重绘幅度", value=0.7)

                # 切换高级选项可见性
                use_nunchaku.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_nunchaku],
                    outputs=[nunchaku_options]
                )

                use_fp8.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_fp8],
                    outputs=[fp8_options_group]
                )

                lora_enable.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[lora_enable],
                    outputs=[lora_options_group]
                )

                enable_hr.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_hr],
                    outputs=[hr_options_group]
                )

            with gr.Column():  # 右半边 - 输出
                output_info = gr.Textbox(label="输出信息")
                output_images = gr.Gallery(label="生成的图像")
                
                # 添加到分镜按钮
                if STORYBOARD_AVAILABLE:
                    with gr.Row():
                        send_to_storyboard_btn = gr.Button(
                            "📤 发送到分镜",
                            variant="secondary",
                            visible=True
                        )
                    send_status = gr.Textbox(label="发送状态", interactive=False, visible=True)

                with gr.Row():
                    generate_btn = gr.Button("图像到图像转换", variant="primary")
                    open_folder_btn = gr.Button("打开输出目录", variant="secondary")

                # 添加按钮点击事件
                open_folder_btn.click(
                    fn=lambda: open_folder(output_dir),
                    inputs=[],
                    outputs=[output_info]
                )

                generate_btn.click(
                    fn=generate_image_img2img,
                    inputs=[
                        init_image, prompt, negative_prompt, width, height,
                        steps, cfg_scale, seed, sampler, batch_size,
                        strength, enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                        use_nunchaku, nunchaku_precision, nunchaku_rank,  # 添加Nunchaku相关参数
                        use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2  # 添加FP8和LoRA相关参数
                    ],
                    outputs=[output_info, output_images]
                )
                
                # 发送到分镜功能
                if STORYBOARD_AVAILABLE:
                    def send_z_image_img2img_gallery_to_storyboard(images):
                        """将画廊中的图片发送到分镜助手"""
                        if not images or len(images) == 0:
                            return "❌ 没有可发送的图片"
                        
                        messages = []
                        last_index = -1
                        last_target_page = 1
                        
                        for img in images:
                            # PIL Image 需要保存为临时文件
                            import tempfile
                            if hasattr(img, 'save'):
                                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                                    img.convert('RGB').save(tmp, quality=85)
                                    img_path = tmp.name
                            else:
                                img_path = img
                            
                            result = send_to_storyboard(img_path)
                            success = result.get('success', False)
                            message = result.get('message', '')
                            index = result.get('index', -1)
                            last_target_page = result.get('target_page', 1)
                            
                            if success:
                                messages.append(f"✅ {message}")
                                last_index = index
                            else:
                                messages.append(f"❌ {message}")
                        
                        if last_index >= 0:
                            return f"已处理 {len(images)} 张图片，最后添加到分镜 #{last_index + 1}（第 {last_target_page} 页）"
                        else:
                            return "处理失败，请查看控制台日志"
                    
                    send_to_storyboard_btn.click(
                        fn=send_z_image_img2img_gallery_to_storyboard,
                        inputs=[output_images],
                        outputs=[send_status]
                    )

        return demo
