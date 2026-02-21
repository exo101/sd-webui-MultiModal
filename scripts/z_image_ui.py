import gradio as gr
import os
import sys
import json
import time
import traceback
import random
import shutil
import queue  # 添加队列模块
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from modules import shared

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

# 尝试导入angle_selector模块
try:
    import importlib.util
    import os
    from pathlib import Path
    
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    angle_selector_path = current_dir / "z_image_angle_selector.py"

    if angle_selector_path.exists():
        spec = importlib.util.spec_from_file_location("z_image_angle_selector", str(angle_selector_path))
        angle_selector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(angle_selector_module)
        create_angle_visualization_component = angle_selector_module.create_z_image_angle_visualization_component
        ANGLE_SELECTOR_AVAILABLE = True
    else:
        create_angle_visualization_component = None
        ANGLE_SELECTOR_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] 多角度提示词可视化选择器模块导入失败: {e}")
    create_angle_visualization_component = None
    ANGLE_SELECTOR_AVAILABLE = False

# 尝试导入ModelScope
try:
    from modelscope import ZImagePipeline as ModelScopeZImagePipeline
    MODELScope_AVAILABLE = True
except ImportError as e:
    MODELScope_AVAILABLE = False
    ModelScopeZImagePipeline = None

# 尝试导入图生图功能
try:
    import importlib.util
    import os
    from pathlib import Path
    
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    img2img_path = current_dir / "z_image_img2img_ui.py"

    if img2img_path.exists():
        spec = importlib.util.spec_from_file_location("z_image_img2img_ui", str(img2img_path))
        img2img_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(img2img_module)
        create_z_image_img2img_ui = img2img_module.create_z_image_img2img_ui
        IMG2IMG_UI_AVAILABLE = True
    else:
        create_z_image_img2img_ui = None
        IMG2IMG_UI_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] 图生图功能模块导入失败: {e}")
    create_z_image_img2img_ui = None
    IMG2IMG_UI_AVAILABLE = False


def generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size=1,
                   enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7,
                   use_nunchaku=False, nunchaku_precision='fp4', nunchaku_rank=128,
                   use_fp8=False, fp8_model_name=None, lora_enable=False, lora_model_1=None, lora_weight_1=0.8, lora_model_2=None, lora_weight_2=0.8):
    """生成图像"""
    try:
        if MODELScope_AVAILABLE:
            pipeline = ModelScopeZImagePipeline(
                model='Qwen/Qwen1.5-7B-Chat',
                torch_dtype=torch.float16,
                device_map="auto",
                use_nunchaku=use_nunchaku,
                nunchaku_precision=nunchaku_precision,
                nunchaku_rank=nunchaku_rank,
                use_fp8=use_fp8,
                fp8_model_name=fp8_model_name,
                lora_enable=lora_enable,
                lora_model_1=lora_model_1,
                lora_weight_1=lora_weight_1,
                lora_model_2=lora_model_2,
                lora_weight_2=lora_weight_2,
            )
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                seed=seed,
                scheduler=sampler,
                num_images_per_prompt=batch_size,
                enable_hr=enable_hr,
                hr_scale=hr_scale,
                hr_upscaler=hr_upscaler,
                hr_second_pass_steps=hr_second_pass_steps,
                denoising_strength=denoising_strength,
            ).images[0]
            return image, None
        else:
            return "错误：无法加载ModelScope模块", None
    except Exception as e:
        error_details = traceback.format_exc()
        return f"图像生成失败: {str(e)}\n详细错误信息:\n{error_details}", None


def generate_image_img2img(init_image, prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size=1,
                           strength=0.6, enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7,
                           use_nunchaku=False, nunchaku_precision='fp4', nunchaku_rank=128,
                           use_fp8=False, fp8_model_name=None, lora_enable=False, lora_model_1=None, lora_weight_1=0.8, lora_model_2=None, lora_weight_2=0.8):
    """这是一个占位函数，实际实现在z_image_img2img_ui.py中"""
    # 导入独立模块中的实际实现
    import importlib.util
    img2img_path = os.path.join(os.path.dirname(__file__), 'z_image_img2img_ui.py')
    spec = importlib.util.spec_from_file_location("z_image_img2img_ui", img2img_path)
    img2img_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(img2img_module)
    generate_image_img2img_impl = img2img_module.generate_image_img2img
    
    return generate_image_img2img_impl(init_image, prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size,
                                      strength, enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                                      use_nunchaku, nunchaku_precision, nunchaku_rank,
                                      use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2)


"""尝试导入图生图功能（如果文件存在）"""
import importlib.util
import os
import sys
from pathlib import Path

# 获取当前文件所在目录
current_dir = Path(__file__).parent
img2img_path = current_dir / "z_image_img2img_ui.py"

# 检查图生图文件是否存在
if img2img_path.exists():
    try:
        # 动态导入图生图模块
        spec = importlib.util.spec_from_file_location("z_image_img2img_ui", str(img2img_path))
        img2img_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(img2img_module)
        generate_image_img2img = img2img_module.generate_image_img2img
    except Exception as e:
        print(f"[WARNING] 图生图功能模块导入失败: {e}")
        
        # 定义一个占位函数
        def generate_image_img2img(*args, **kwargs):
            return "错误：图生图功能模块导入失败", None
else:
    print(f"[WARNING] 图生图功能模块未找到: {img2img_path}")
    
    # 定义一个占位函数
    def generate_image_img2img(*args, **kwargs):
        return "错误：图生图功能模块未找到", None


    ModelScopeZImagePipeline = None

# 检查模型文件是否存在
models_dir = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
model_exists = models_dir.exists() and any(models_dir.iterdir())

# 模块是否可用的标志 - 仅基于ModelScope是否可用，不依赖模型文件是否存在
Z_IMAGE_MODULE_AVAILABLE = MODELScope_AVAILABLE

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

        if model_type == 'fp8':
            # FP8模型加载逻辑
            try:
                # 首先检查是否指定了特定的FP8模型文件名
                if fp8_model_name:
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
                # 注意：这里不指定float8_e4m3fn类型，因为可能导致兼容性问题
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

                # 设置为FP8模型类型
                current_model_type = 'fp8'
                
            except Exception as e:
                error_msg = str(e)
                import traceback
                print(f"[ERROR] 加载FP8模型时出错: {error_msg}\n{traceback.format_exc()}")
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
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                pipe.enable_sequential_cpu_offload()
            else:
                # 如果没有CPU卸载功能，则尝试将模型移动到GPU
                try:
                    pipe = pipe.to("cuda")
                except Exception as move_error:
                    pass

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
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 应用SageAttention或Flash Attention优化
            if SAGE_ATTENTION_AVAILABLE:
                print(f"[INFO] 为{model_type}模型应用SageAttention优化...")
                replace_transformer_attention_with_sage(pipe.transformer)
            elif FLASH_ATTENTION_AVAILABLE:
                print(f"[INFO] 为{model_type}模型应用Flash Attention优化...")
                replace_transformer_attention_with_flash(pipe.transformer)
        else:
            print(f"[WARNING] 无法找到pipe.transformer组件，跳过注意力优化")
    except Exception as e:
        print(f"[ERROR] 应用注意力优化失败: {str(e)}")


def replace_transformer_attention_with_sage(transformer):
    """将transformer中的注意力机制替换为SageAttention"""
    try:
        for name, module in transformer.named_modules():
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


def replace_transformer_attention_with_flash(transformer):
    """将transformer中的注意力机制替换为FlashAttention"""
    try:
        import torch.nn.functional as F
        
        for name, module in transformer.named_modules():
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
        
        # 生成文件名
        timestamp = int(time.time())
        filename = f"zimage_{timestamp}_s{seed}_{index}.png"
        filepath = output_dir / filename
        
        # 保存图像
        image.save(filepath)
        
        return str(filepath)
    except Exception as e:
        print(f"保存图像失败: {str(e)}")
        return None


def generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size=1,
                   enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7,
                   use_nunchaku=False, nunchaku_precision='fp4', nunchaku_rank=128,
                   use_fp8=False, fp8_model_name=None, lora_enable=False, lora_model_1=None, lora_weight_1=0.8, lora_model_2=None, lora_weight_2=0.8):
    """使用Z-Image-Turbo生成图像，添加FP8模型支持"""
    global pipe, model_loaded

    try:
        # 检查提示词
        if not prompt or prompt.strip() == "":
            return "错误：请输入正向提示词", None

        # 如果种子为-1，则生成随机种子
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

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
            status = load_model_if_needed(model_type, nunchaku_precision, nunchaku_rank)
        else:
            status = load_model_if_needed(model_type)

        if "失败" in status or "失败" in status:
            return status, None

        # 确保pipe不为None
        if pipe is None:
            return f"错误：管道未正确初始化。模型类型: {model_type}, 加载状态: {status}", None

        # 移除尺寸必须为16的倍数的限制
        # 根据官方示例，Z-Image-Turbo模型支持任意尺寸输入
        # 不再进行尺寸调整

        # 为Turbo模型设置合适的参数
        # 根据用户设置的最大步数限制，而不是固定为10
        max_allowed_steps = getattr(shared.opts, 'zimage_max_inference_steps', 10)  # 从设置中获取最大步数限制
        actual_steps = min(steps, max_allowed_steps)
        actual_guidance = 0.0  # 根据官方示例，Turbo模型应该使用0.0的guidance_scale

        # 处理LoRA
        lora_applied = False
        if lora_enable and (lora_model_1 or lora_model_2):
            print(f"[INFO] 开始应用LoRA: lora_model_1={lora_model_1}, lora_weight_1={lora_weight_1}, lora_model_2={lora_model_2}, lora_weight_2={lora_weight_2}")
            
            # 检查当前模型是否支持LoRA
            if current_model_type == 'fp8':
                # FP8模型支持LoRA，通过transformer加载
                print(f"[INFO] 检测到FP8模型，准备应用LoRA")
                
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
                                print(f"[WARNING] LoRA与FP8模型不兼容: {str(e)}")
                                print(f"[INFO] FP8量化模型与LoRA的维度不匹配，无法加载此LORA")
                                return f"LoRA与FP8模型不兼容: 维度不匹配。LoRA通常是针对特定基础模型训练的，可能与量化后的FP8模型不兼容。", None
                            else:
                                raise e  # 重新抛出其他RuntimeError
                        except Exception as e:
                            print(f"[ERROR] 加载LoRA到FP8模型时出错: {str(e)}")
                            import traceback
                            print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")
                            return f"FP8 LoRA加载失败: {str(e)}", None
                except Exception as e:
                    print(f"[ERROR] FP8 LoRA加载失败: {str(e)}")
                    import traceback
                    print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")
                    return f"FP8 LoRA加载失败: {str(e)}", None
            elif current_model_type == 'nunchaku':
                # 对于Nunchaku模型，使用Nunchaku内置的LoRA支持
                print(f"[INFO] 检测到Nunchaku transformer，准备应用LoRA")
                
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
                            print(f"[INFO] 应用LoRA: {lora_path}")
                            pipe.transformer.update_lora_params(lora_path)
                            lora_applied = True
                            
                    except Exception as e:
                        print(f"[ERROR] Nunchaku LoRA加载失败: {str(e)}")
                        import traceback
                        print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")
                        return f"Nunchaku LoRA加载失败: {str(e)}", None
                else:
                    print("[WARNING] 当前Nunchaku版本不支持LoRA功能")
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
                        print(f"[INFO] 应用LoRA (标准diffusers方式): {lora_path}，权重: {lora_weight}")
                        # 获取LoRA文件名（不含扩展名），作为weight_name
                        lora_name = Path(lora_path).stem
                        pipe.load_lora_weights(lora_path, local_files_only=True, weight_name=lora_name)
                        pipe.fuse_lora(lora_scale=lora_weight)
                        lora_applied = True
                        
                except Exception as e:
                    print(f"[ERROR] 标准LoRA加载失败: {str(e)}")
                    return f"标准LoRA加载失败: {str(e)}", None
            else:
                print(f"[WARNING] 当前模型 ({current_model_type}) 不支持 LoRA 功能 或 LoRA 功能受限")

            if lora_applied:
                print(f"[INFO] LoRA应用成功")
            else:
                print(f"[WARNING] 没有找到有效的LoRA模型文件或当前模型不支持LoRA")

        # 生成图像 - 使用ModelScope官方示例方式
        try:
            generator = torch.Generator().manual_seed(seed)
            
            print(f"[INFO] 开始生成图像，参数: 提示词='{prompt[:50]}...', 尺寸={width}x{height}, 步数={actual_steps}, 批次大小={batch_size}")
            if lora_applied:
                print(f"[INFO] LoRA已应用，模型类型: {current_model_type}")
            
            # 高分辨率修复逻辑
            if enable_hr:
                print(f"[INFO] 启用高分辨率修复，缩放比例: {hr_scale}")
                
                # 第一阶段：生成低分辨率图像
                low_res_width = width
                low_res_height = height
                
                # 计算高分辨率尺寸，使用原始尺寸乘以缩放因子
                high_res_width = int(width * hr_scale)
                high_res_height = int(height * hr_scale)
                
                print(f"[INFO] 低分辨率: {low_res_width}x{low_res_height}, 高分辨率: {high_res_width}x{high_res_height}")
                
                # 生成低分辨率图像
                if model_type == 'fp8':
                    low_res_images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=low_res_width,
                        height=low_res_height,
                        num_inference_steps=actual_steps,
                        guidance_scale=actual_guidance,
                        generator=generator,
                        num_images_per_prompt=batch_size,
                        output_type="pil"
                    ).images
                else:
                    low_res_images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=low_res_width,
                        height=low_res_height,
                        num_inference_steps=actual_steps,
                        guidance_scale=actual_guidance,
                        generator=generator,
                        num_images_per_prompt=batch_size
                    ).images
                
                # 对每张低分辨率图像进行高分辨率修复
                processed_images = []
                for i, low_res_image in enumerate(low_res_images):
                    # 从低分辨率图像开始进行高分辨率修复
                    # 使用ZImageImg2ImgPipeline进行高分辨率修复
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
                    if hasattr(img2img_pipe, 'enable_model_cpu_offload'):
                        print("[INFO] 启用模型CPU卸载以节省显存")
                        img2img_pipe.enable_model_cpu_offload()
                    elif hasattr(img2img_pipe, 'enable_sequential_cpu_offload'):
                        print("[INFO] 启用顺序CPU卸载")
                        img2img_pipe.enable_sequential_cpu_offload()
                    else:
                        # 如果没有CPU卸载功能，则尝试将模型移动到GPU
                        try:
                            print("[INFO] 将模型移动到CUDA设备")
                            img2img_pipe = img2img_pipe.to("cuda")
                        except Exception as move_error:
                            print(f"[WARNING] 将模型移动到CUDA设备失败: {move_error}")

                    # 调整低分辨率图像到高分辨率尺寸
                    upscaled_image = low_res_image.resize((high_res_width, high_res_height), resample=Image.LANCZOS)
                    
                    # 使用高分辨率参数进行第二次生成
                    hr_steps = hr_second_pass_steps if hr_second_pass_steps > 0 else actual_steps
                    hr_strength = denoising_strength if denoising_strength > 0 else 0.6
                    
                    if model_type == 'fp8':
                        hr_image = img2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=upscaled_image,
                            strength=hr_strength,
                            num_inference_steps=hr_steps,
                            guidance_scale=actual_guidance,
                            generator=generator,
                            num_images_per_prompt=1,
                            output_type="pil"
                        ).images[0]
                    else:
                        hr_image = img2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=upscaled_image,
                            strength=hr_strength,
                            num_inference_steps=hr_steps,
                            guidance_scale=actual_guidance,
                            generator=generator,
                            num_images_per_prompt=1
                        ).images[0]
                    
                    processed_images.append(hr_image)
            else:
                # 非高分辨率修复模式，直接生成图像
                if model_type == 'fp8':
                    images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=actual_steps,
                        guidance_scale=actual_guidance,
                        generator=generator,
                        num_images_per_prompt=batch_size,
                        # 确保输出值在正确范围内
                        output_type="pil"  # 明确指定输出类型
                    ).images
                else:
                    images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=actual_steps,
                        guidance_scale=actual_guidance,
                        generator=generator,
                        num_images_per_prompt=batch_size
                    ).images

                # 确保生成的图像不是空的或全是黑色的
                processed_images = []
                for i, image in enumerate(images):
                    # 检查图像是否有像素值
                    import numpy as np
                    img_array = np.array(image)
                    
                    # 检查图像是否全黑（所有像素值都接近0）
                    if np.mean(img_array) < 5:  # 平均像素值低于5，认为是黑图
                        print(f"[WARNING] 检测到可能是黑图（平均像素值: {np.mean(img_array):.2f}），尝试重新生成")
                        
                        # 对于FP8模型，尝试更多推理步骤
                        if model_type == 'fp8':
                            print(f"[INFO] 使用FP8模型重新生成，增加推理步数")
                            re_images = pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                width=width,
                                height=height,
                                num_inference_steps=min(actual_steps + 4, 20),  # 增加步数，但不超过20
                                guidance_scale=actual_guidance,
                                generator=generator,
                                num_images_per_prompt=1,  # 只重新生成一张
                                output_type="pil"
                            ).images
                            
                            # 再次检查重新生成的图像
                            re_img_array = np.array(re_images[0])
                            if np.mean(re_img_array) < 5:
                                print(f"[WARNING] 重新生成后仍是黑图（平均像素值: {np.mean(re_img_array):.2f}），尝试调整参数")
                                
                                # 如果仍然是黑图，尝试使用不同的随机种子
                                alt_generator = torch.Generator().manual_seed(seed + 12345)
                                re_images = pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    width=width,
                                    height=height,
                                    num_inference_steps=min(actual_steps + 8, 25),  # 进一步增加步数
                                    guidance_scale=actual_guidance,
                                    generator=alt_generator,
                                    num_images_per_prompt=1,
                                    output_type="pil"
                                ).images
                                
                                # 最终检查
                                final_img_array = np.array(re_images[0])
                                if np.mean(final_img_array) < 5:
                                    print(f"[WARNING] 多次尝试后仍是黑图（平均像素值: {np.mean(final_img_array):.2f}），使用原图像")
                                    processed_images.append(image)  # 使用原图像
                                else:
                                    print(f"[INFO] 重新生成成功（平均像素值: {np.mean(final_img_array):.2f}）")
                                    processed_images.append(re_images[0])  # 使用新图像
                            else:
                                print(f"[INFO] 重新生成成功（平均像素值: {np.mean(re_img_array):.2f}）")
                                processed_images.append(re_images[0])  # 使用新图像
                        else:
                            # 非FP8模型，按原方式处理
                            alt_images = pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                width=width,
                                height=height,
                                num_inference_steps=max(actual_steps, 8),
                                guidance_scale=actual_guidance,
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
                output_image_path = output_dir / f"{seed}_{idx}.png"
                image.save(output_image_path)
                output_images.append(output_dir / f"{seed}_{idx}.png")

            return "生成成功", output_images
        except Exception as e:
            import traceback
            print(f"[ERROR] 生成失败: {str(e)}\n{traceback.format_exc()}")
            return f"生成失败: {str(e)}", None

    except Exception as e:
        error_details = traceback.format_exc()
        return f"图像生成失败: {str(e)}\n详细错误信息:\n{error_details}", None


def create_z_image_ui():
    """创建Z-Image-Turbo UI界面"""

    # 检查ModelScope是否可用
    if not MODELScope_AVAILABLE:
        with gr.Blocks() as demo:
            gr.Markdown("# Z-Image-Turbo 图像生成 (不可用)")
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
        gr.Markdown("# Z-Image-Turbo 图像生成")
        gr.Markdown("基于 ModelScope 的超快速文生图/图生图模型")

        # 创建选项卡用于切换文生图和图生图
        with gr.Tabs():
            with gr.TabItem("文生图 (Text-to-Image)"):
                with gr.Row():
                    with gr.Column():  # 左半边 - 参数设置
                        prompt = gr.Textbox(
                            label="提示词",
                            placeholder="输入您的提示词，例如：一只可爱的猫"
                        )

                        negative_prompt = gr.Textbox(
                            label="负面提示词",
                            placeholder="输入您不希望出现在图像中的内容"
                        )

                        # 添加多角度提示词可视化选择器（如果可用）
                        if ANGLE_SELECTOR_AVAILABLE:
                            with gr.Accordion("多角度提示词可视化选择器", open=False):
                                angle_selector_component = create_angle_visualization_component(prompt)

                        with gr.Row():
                            width = gr.Slider(
                                minimum=256, maximum=2048, step=64, value=1024, label="宽度"
                            )
                            height = gr.Slider(
                                minimum=256, maximum=2048, step=64, value=1024, label="高度"
                            )

                        with gr.Row():
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

                        with gr.Row():
                            generate_btn = gr.Button("生成图像", variant="primary")
                            open_folder_btn = gr.Button("打开输出目录", variant="secondary")

                        # 添加队列功能区域
                        with gr.Accordion("任务队列", open=False):
                            with gr.Group():
                                queue_status_text = gr.Textbox(label="队列状态", value="当前队列大小: 0", interactive=False)
                                
                                with gr.Row():
                                    # 添加到队列按钮
                                    add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                                    process_queue_btn = gr.Button("执行队列任务", variant="primary")
                                    clear_queue_btn = gr.Button("清空队列", variant="stop")
                                
                                # 队列操作状态
                                queue_operation_status = gr.Textbox(label="队列操作状态", interactive=False)
                                
                                # 详细队列状态显示
                                detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, lines=5, max_lines=10)

                        # 添加按钮点击事件
                        open_folder_btn.click(
                            fn=lambda: open_folder(output_dir),
                            inputs=[],
                            outputs=[output_info]
                        )

                        generate_btn.click(
                            fn=generate_image,
                            inputs=[
                                prompt, negative_prompt, width, height,
                                steps, cfg_scale, seed, sampler, batch_size,
                                enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                                use_nunchaku, nunchaku_precision, nunchaku_rank,  # 添加Nunchaku相关参数
                                use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2  # 添加FP8和LoRA相关参数，移除4位量化参数
                            ],
                            outputs=[output_info, output_images]
                        )
                        
                        # 添加到队列的事件绑定
                        add_to_queue_btn.click(
                            fn=lambda prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size, \
                               enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength, \
                               use_nunchaku, nunchaku_precision, nunchaku_rank, \
                               use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2: \
                               add_to_queue('zimage', 
                                   prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size,
                                   enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                                   use_nunchaku, nunchaku_precision, nunchaku_rank,
                                   use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2),
                            inputs=[
                                prompt, negative_prompt, width, height,
                                steps, cfg_scale, seed, sampler, batch_size,
                                enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                                use_nunchaku, nunchaku_precision, nunchaku_rank,
                                use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2
                            ],
                            outputs=[queue_operation_status]
                        )
                        
                        # 更新队列状态
                        def update_queue_status():
                            return get_queue_status()
                        
                        def update_detailed_queue_status():
                            return get_detailed_queue_status()
                        
                        # 添加按钮点击事件来更新队列状态
                        add_to_queue_btn.click(
                            fn=update_queue_status,
                            inputs=[],
                            outputs=[queue_status_text]
                        )
                        
                        add_to_queue_btn.click(
                            fn=update_detailed_queue_status,
                            inputs=[],
                            outputs=[detailed_queue_status]
                        )
                        
                        process_queue_btn.click(
                            fn=process_queue,
                            inputs=[],
                            outputs=[output_info, output_images]
                        )
                        
                        process_queue_btn.click(
                            fn=update_queue_status,
                            inputs=[],
                            outputs=[queue_status_text]
                        )
                        
                        process_queue_btn.click(
                            fn=update_detailed_queue_status,
                            inputs=[],
                            outputs=[detailed_queue_status]
                        )
                        
                        # 清空队列按钮事件
                        def clear_queue():
                            global task_queue
                            import queue
                            task_queue = queue.Queue()  # 重新创建空队列
                            return "队列已清空"
                        
                        clear_queue_btn.click(
                            fn=clear_queue,
                            inputs=[],
                            outputs=[queue_operation_status]
                        )
                        
                        clear_queue_btn.click(
                            fn=update_queue_status,
                            inputs=[],
                            outputs=[queue_status_text]
                        )
                        
                        clear_queue_btn.click(
                            fn=update_detailed_queue_status,
                            inputs=[],
                            outputs=[detailed_queue_status]
                        )

            # 从独立模块加载图生图选项卡
            if IMG2IMG_UI_AVAILABLE and create_z_image_img2img_ui:
                with gr.TabItem("图生图 (Image-to-Image)"):
                    img2img_interface = create_z_image_img2img_ui()
                    gr.components.HTML("<p>图生图功能已加载</p>")
            else:
                with gr.TabItem("图生图 (Image-to-Image)"):
                    gr.components.HTML("<p>图生图功能模块未找到</p>")

        return demo

# 添加队列相关的全局变量和函数
task_queue = queue.Queue()


def add_to_queue(task_type, *args):
    """将任务添加到队列"""
    # 根据任务类型解析参数
    if task_type == 'zimage':
        # zimage任务参数: prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size, 
        # enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
        # use_nunchaku, nunchaku_precision, nunchaku_rank, 
        # use_fp8, fp8_model_name, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2
        task_info = {
            'type': task_type,
            'params': {
                'prompt': args[0],
                'negative_prompt': args[1],
                'width': args[2],
                'height': args[3],
                'steps': args[4],
                'cfg_scale': args[5],
                'seed': args[6],
                'sampler': args[7],
                'batch_size': args[8],
                'enable_hr': args[9],
                'hr_scale': args[10],
                'hr_upscaler': args[11],
                'hr_second_pass_steps': args[12],
                'denoising_strength': args[13],
                'use_nunchaku': args[14],
                'nunchaku_precision': args[15],
                'nunchaku_rank': args[16],
                'use_fp8': args[17],
                'fp8_model_name': args[18],
                'lora_enable': args[19],
                'lora_model_1': args[20] if args[19] else 'N/A',
                'lora_weight_1': args[21] if args[19] else 0.0,
                'lora_model_2': args[22] if args[19] else 'N/A',
                'lora_weight_2': args[23] if args[19] else 0.0
            }
        }
    
    task = {
        'info': task_info,
        'args': args
    }
    task_queue.put(task)
    
    # 返回当前队列大小和任务信息摘要
    queue_size = task_queue.qsize()
    task_summary = f"生成任务: {args[2]}x{args[3]}, 步数: {args[4]}, 提示词: {args[0][:30]}{'...' if len(args[0]) > 30 else ''}"
    
    return f"任务已添加 - {task_summary}，当前队列大小: {queue_size}"


def process_queue():
    """处理队列中的所有任务"""
    results = []
    statuses = []
    task_num = 1
    
    while not task_queue.empty():
        task = task_queue.get()
        task_info = task['info']
        args = task['args']
        task_type = task_info['type']
        
        try:
            if task_type == 'zimage':
                result, images = generate_image(*args)
                results.extend(images if images else [])
                statuses.append(f"任务{task_num}: {result}")
            else:
                result = f"未知的任务类型: {task_type}"
                statuses.append(f"任务{task_num}: {result}")
                
            task_num += 1
        except Exception as e:
            results.append(None)
            statuses.append(f"任务{task_num}执行失败: {str(e)}")
            task_num += 1
    
    if results:
        return "所有任务已完成: " + "; ".join(statuses), results
    else:
        return "队列为空，没有任务需要执行", []


def get_queue_status():
    """获取当前队列状态"""
    size = task_queue.qsize()
    return f"当前队列大小: {size}"


# 存储任务详情的函数
def get_detailed_queue_status():
    """获取详细的队列状态，包括任务参数"""
    import copy
    temp_queue = queue.Queue()
    details = []
    idx = 1
    
    # 临时取出所有任务，记录详情，并放回原队列
    while not task_queue.empty():
        task = task_queue.get()
        temp_queue.put(task)
        
        task_info = task['info']
        task_type = task_info['type']
        
        if task_type == 'zimage':
            detail = f"任务{idx}: 生成图像 - {task_info['params']['width']}x{task_info['params']['height']}"
            detail += f", 步数: {task_info['params']['steps']}, CFG: {task_info['params']['cfg_scale']}"
            detail += f", 提示词: {task_info['params']['prompt'][:30]}{'...' if len(task_info['params']['prompt']) > 30 else ''}"
            if task_info['params']['lora_enable']:
                detail += f", LoRA: {task_info['params']['lora_model_1']}/{task_info['params']['lora_model_2']}"
        
        details.append(detail)
        idx += 1
    
    # 将任务放回原队列
    while not temp_queue.empty():
        task_queue.put(temp_queue.get())
    
    if details:
        return "\n".join(details)
    else:
        return "队列为空"
