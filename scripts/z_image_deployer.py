import os
import sys
import gradio as gr
from modules import shared, paths, images
import subprocess
import torch
from pathlib import Path
from modelscope import ZImagePipeline
from PIL import Image
import numpy as np
import time
import random
import json
import safetensors.torch
from diffusers import DiffusionPipeline

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

# 将当前脚本目录添加到Python路径，以便导入同目录下的其他模块
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 导入图生图和队列功能 - 使用绝对导入方式
try:
    from z_image_img2img import create_z_image_img2img_ui, generate_image_with_zimage_img2img
except ImportError as e:
    print(f"无法导入 z_image_img2img 模块: {e}")
    # 提供一个占位函数以防止完全崩溃
    def create_z_image_img2img_ui():
        with gr.Group():
            gr.Markdown("图生图功能加载失败，请检查依赖项。")

try:
    from z_image_queue import add_to_queue, process_queue, get_queue_status, get_detailed_queue_status, clear_queue
except ImportError as e:
    print(f"无法导入 z_image_queue 模块: {e}")
    # 为队列功能提供基础占位实现
    def add_to_queue(*args, **kwargs):
        return "队列功能不可用"
    def process_queue(*args, **kwargs):
        return "队列功能不可用", None
    def get_queue_status():
        return "队列状态: 不可用"
    def get_detailed_queue_status():
        return "详细队列: 不可用"
    def clear_queue():
        return "队列已清空 (占位)"

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


def get_model_list():
    """获取模型列表"""
    try:
        model_path = Path(shared.models_path) / "Stable-diffusion"
        if model_path.exists():
            model_files = [f.stem for f in model_path.glob("*.safetensors")] + \
                         [f.stem for f in model_path.glob("*.ckpt")] + \
                         [f.stem for f in model_path.glob("*.pth")]
            return model_files
        return []
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return []


def get_zimage_model_list():
    """获取Z-Image目录下的模型列表"""
    try:
        zimage_path = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image"
        if zimage_path.exists():
            model_files = []
            # 查找所有safetensors和ckpt文件
            for ext in ['.safetensors', '.ckpt', '.pt']:
                model_files.extend([f.stem for f in zimage_path.glob(f"*{ext}")])
            # 去重并排序
            unique_models = sorted(list(set(model_files)))
            
            # 如果有模型文件，返回带默认选项的列表
            if unique_models:
                # 将"Z-Image (default)"放在第一位，后面跟实际模型
                return ["Z-Image (default)"] + unique_models
            else:
                return ["Z-Image (default)"]
        return ["Z-Image (default)"]
    except Exception as e:
        print(f"获取Z-Image模型列表失败: {e}")
        return ["Z-Image (default)"]


def apply_attention_optimizations(pipe):
    """应用注意力优化到模型"""
    try:
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 应用SageAttention或Flash Attention优化
            if SAGE_ATTENTION_AVAILABLE:
                print(f"[INFO] 为模型应用SageAttention优化...")
                replace_transformer_attention_with_sage(pipe.transformer)
            elif FLASH_ATTENTION_AVAILABLE:
                print(f"[INFO] 为模型应用Flash Attention优化...")
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


def generate_image_with_zimage(prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2, enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7, selected_model=None):
    """
    使用Z-Image正式版模型生成图像
    """
    try:
        # 输入参数验证
        if not prompt.strip():
            return "错误：提示词不能为空", None

        # 使用本地Z-Image模型路径
        zimage_dir = os.path.join(shared.models_path, "Tongyi-MAI", "Z-Image")
        
        # 检查本地Z-Image模型是否存在
        zimage_path = Path(zimage_dir)
        if not (zimage_path.exists() and any(zimage_path.iterdir())):
            return "错误：Z-Image正式版模型未找到，请确保模型已下载至 models/Tongyi-MAI/Z-Image 目录", None
        
        print(f"[INFO] 开始加载本地Z-Image模型...")
        
        # 创建模型实例
        pipe = ZImagePipeline.from_pretrained(
            str(zimage_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            local_files_only=True
        )
        
        # 应用注意力优化
        if FLASH_ATTENTION_AVAILABLE or SAGE_ATTENTION_AVAILABLE:
            print(f"[INFO] 应用注意力优化...")
            apply_attention_optimizations(pipe)
        
        # 启用优化的注意力机制
        if hasattr(pipe, 'transformer'):
            print("[INFO] 检测到Z-Image Transformer，尝试启用优化的注意力机制...")
            
            # 检查是否支持PyTorch的Scaled Dot Product Attention
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("[INFO] PyTorch Scaled Dot Product Attention可用，启用中...")
                try:
                    # 尝试启用三种SDP后端
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    print("[INFO] 已启用Flash Attention、Math Attention和内存高效Attention")
                except Exception as e:
                    print(f"[WARNING] 无法完全启用SDP后端: {e}")
        
        # 启用模型CPU卸载功能，按需将组件移动到GPU进行处理
        if hasattr(pipe, 'enable_model_cpu_offload'):
            print("[INFO] 启用模型CPU卸载功能以节省显存...")
            pipe.enable_model_cpu_offload()
        else:
            # 如果没有CPU卸载功能，则将管道移动到GPU
            pipe.to("cuda")
            print("[INFO] 模型已成功加载并移至CUDA设备")
        
        # 初始化LoRA状态
        lora_applied = False
        
        # 处理LoRA
        if lora_enable and (lora_model_1 or lora_model_2):
            print(f"[INFO] 启用LoRA支持，准备加载模型...")

            # 解融合任何之前可能存在的LoRA权重，确保干净的环境
            try:
                pipe.unfuse_lora()
                pipe.unload_lora_weights()
                print("[INFO] 已清除之前的LoRA权重")
            except:
                pass  # 忽略解融合时的错误，可能是首次运行
            
            lora_paths = []
            # 查找第一个LoRA模型
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
                    print(f"[INFO] 找到LoRA模型1: {lora_path_1}")
                else:
                    print(f"[WARNING] 未找到LoRA模型1: {lora_model_1}")
                    
            # 查找第二个LoRA模型
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
                    print(f"[INFO] 找到LoRA模型2: {lora_path_2}")
                else:
                    print(f"[WARNING] 未找到LoRA模型2: {lora_model_2}")

            # 应用所有找到的LoRA
            for lora_path, lora_weight in lora_paths:
                print(f"[INFO] 正在加载并融合LoRA: {lora_path}，缩放权重: {lora_weight}")
                try:
                    pipe.load_lora_weights(lora_path, local_files_only=True)
                    pipe.fuse_lora(lora_scale=lora_weight)
                    lora_applied = True
                except Exception as e:
                    error_msg = f"无法为 {lora_path} 加载LoRA权重: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    return f"LoRA加载失败: {error_msg}", None
            
            if lora_applied:
                print(f"[INFO] LoRA加载和融合完成")
            else:
                print(f"[INFO] 本次生成未应用任何LoRA模型")
        
        print(f"[INFO] 开始生成图像...")
        print(f"[INFO] 参数: 提示词='{prompt[:50]}...', 尺寸={width}x{height}, 步数={steps}, CFG Scale={cfg_scale}, 批次={batch_size}")
        
        # 设置随机种子
        if int(seed) == -1:
            seed = random.randint(0, 2**32 - 1)
        print(f"[INFO] 使用随机种子: {seed}")
        
        generator = torch.Generator("cuda").manual_seed(int(seed))
        
        # 高清修复处理
        if enable_hr:
            print(f"[INFO] 启用高清修复，缩放比例: {hr_scale}, 重绘强度: {denoising_strength}")
            
            # 计算高清修复后的目标尺寸
            target_width = int(width * hr_scale)
            target_height = int(height * hr_scale)
            
            # 确保尺寸是8的倍数
            target_width = max(64, target_width - target_width % 8)
            target_height = max(64, target_height - target_height % 8)
            
            print(f"[INFO] 目标尺寸: {target_width}x{target_height}")
            
            # 第一阶段：生成用户指定尺寸的图像
            print(f"[INFO] 第一阶段：生成基础图像 {width}x{height}")
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                cfg_normalization=False,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                num_images_per_prompt=batch_size
            )
            
            # 获取基础图像
            images_list = output.images
            
            # 第二阶段：使用选定的放大器进行上采样
            print(f"[INFO] 第二阶段：将图像放大到 {target_width}x{target_height}")
            
            # 将基础图像调整到目标尺寸
            upscaled_images = []
            for img in images_list:
                upscaled_img = images.resize_image(0, img, target_width, target_height, upscaler_name=hr_upscaler)
                upscaled_images.append(upscaled_img)
            
            # 生成最终高清图像
            final_images = []
            for idx, upscaled_img in enumerate(upscaled_images):
                print(f"[INFO] 处理第 {idx+1}/{len(upscaled_images)} 张图像的高清修复")
                upscaled_img = upscaled_img.convert("RGB")
                upscaled_tensor = torch.tensor(np.array(upscaled_img)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32) / 255.0
                
                # 由于Z-Image可能不直接支持img2img，我们使用基础的diffusion pipeline进行第二阶段处理
                # 为了兼容Z-Image模型，我们暂时只进行上采样，不进行第二阶段扩散
                final_images.append(upscaled_img)
                
        else:
            # 标准生成流程
            print("[INFO] 正在调用Z-ImagePipeline进行图像生成...")
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                cfg_normalization=False,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                num_images_per_prompt=batch_size
            )
            
            # 确保输出包含图像
            if not hasattr(output, 'images') or not output.images:
                return "错误：模型返回结果无效，未生成图像", None
                
            final_images = output.images

        print(f"[INFO] 图像生成完成，共生成 {len(final_images)} 张图像")
        
        # 保存图像
        output_dir = Path(paths.data_path) / "outputs" / "z-image"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, image in enumerate(final_images):
            timestamp = int(time.time())
            filename = f"zimage_{'hr' if enable_hr else 'std'}_{timestamp}_s{seed}_{i}.png"
            filepath = output_dir / filename
            image.save(filepath)
            saved_paths.append(str(filepath))
        
        # 自动从GPU卸载模型以释放显存
        try:
            # 尝试先将模型移到CPU
            pipe = pipe.to("cpu")
            # 删除模型对象
            del pipe
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            # 进行垃圾回收
            import gc
            gc.collect()
            print("[INFO] 模型已从GPU卸载，显存已释放")
        except Exception as e:
            print(f"[WARNING] 无法卸载模型: {e}")
            
        return f"图像生成成功! {'启用高清修复' if enable_hr else '标准生成'}, 批次: {batch_size}, Seed: {seed}", saved_paths
        
    except Exception as e:
        error_msg = str(e)
        import traceback
        print(f"[ERROR] 图像生成过程中发生异常: {error_msg}")
        print(f"[ERROR] 详细堆栈跟踪:\n{traceback.format_exc()}")
        # 在发生异常时也尝试清理显存
        try:
            if 'pipe' in locals():
                pipe = pipe.to("cpu")
                del pipe
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                print("[INFO] 因生成错误，模型已从GPU卸载，显存已释放")
        except Exception as e:
            print(f"[WARNING] 错误处理时无法卸载模型: {e}")
        return f"图像生成失败: {error_msg}\n详细错误信息:\n{traceback.format_exc()}", None


def create_z_image_deploy_ui():
    """创建Z-Image生成界面"""
    # 获取采样器列表
    try:
        from modules import sd_samplers
        sampler_names = [sampler.name for sampler in sd_samplers.visible_samplers()]
        default_sampler = sampler_names[0] if sampler_names else "Euler"
    except:
        sampler_names = ["Euler", "Euler a", "DPM++ 2M"]
        default_sampler = "Euler"
    
    # 获取模型列表
    model_choices = get_model_list()
    
    with gr.Blocks() as ui:
        gr.Markdown("# Z-Image 图像生成")
        gr.Markdown("使用Z-Image正式版模型进行图像生成")
        
        # 创建选项卡用于切换文生图和图生图
        with gr.Tabs():
            with gr.TabItem("文生图 (Text-to-Image)"):
                with gr.Row():
                    with gr.Column():  # 左半边 - 参数设置
                        # 根据项目规范，移除模型选择下拉列表
                        gr.Markdown("注意：使用官方Z-Image模型")
                        
                        prompt = gr.Textbox(
                            label="提示词",
                            placeholder="输入您的提示词...",
                            lines=3
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="负面提示词",
                            placeholder="输入您不希望出现在图像中的内容",
                            value="blurry, distorted, malformed, bad anatomy, extra limbs, fused fingers, bad hands, bad feet, deformed, ugly, low quality, artifact, noise",
                            lines=2
                        )
                        
                        with gr.Row():
                            width = gr.Slider(minimum=256, maximum=2048, step=64, value=1280, label="宽度")
                            height = gr.Slider(minimum=256, maximum=2048, step=64, value=720, label="高度")
                        
                        with gr.Row():
                            steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="推理步数")
                            cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=4.0, label="CFG Scale")
                        
                        with gr.Row():
                            seed = gr.Number(label="随机种子 (-1为随机)", value=-1, precision=0)
                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="生成批次")
                        
                        with gr.Row():
                            sampler = gr.Dropdown(
                                choices=sampler_names,
                                value=default_sampler,
                                label="采样器"
                            )
                            
                            # 注意：Z-Image模型使用内置调度器，当前采样器选择暂不生效，为预留接口
                            
                        # 添加高清修复选项
                        with gr.Accordion("高清修复 (Hires. fix)", open=False):
                            enable_hr = gr.Checkbox(label="启用高清修复", value=False)
                            with gr.Row():
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, value=2.0, label="放大倍数", elem_id="txt2img_hr_scale")
                                hr_upscaler = gr.Dropdown(label="放大算法", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode, elem_id="txt2img_hr_upscaler")
                            with gr.Row():
                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, value=0, label="高清阶段步数", elem_id="txt2img_hires_steps")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, label="重绘强度", elem_id="txt2img_denoising_strength")
                        
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
                                    print(f"刷新LoRA列表失败: {e}")
                                    return [gr.update(), gr.update()]

                            refresh_lora_btn.click(
                                fn=refresh_lora_list,
                                inputs=[],
                                outputs=[lora_model_1, lora_model_2]
                            )
                        
                        # 切换LoRA选项可见性
                        lora_enable.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[lora_enable],
                            outputs=[lora_options_group]
                        )

                    with gr.Column():  # 右半边 - 输出和按钮
                        with gr.Row():
                            generate_btn = gr.Button("生成图像", variant="primary")
                            add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                            open_folder_btn = gr.Button("打开输出目录", variant="secondary")
                        
                        with gr.Column():
                            output_status = gr.Textbox(label="输出状态", interactive=False)
                            output_gallery = gr.Gallery(label="生成的图像", show_label=True, elem_id="zimage_gallery", columns=2, height="auto")
                            
                        # 队列功能区域
                        with gr.Accordion("任务队列管理", open=False):
                            with gr.Group():
                                queue_status_text = gr.Textbox(label="队列状态", value="当前队列大小: 0", interactive=False)
                                
                                with gr.Row():
                                    process_queue_btn = gr.Button("执行队列任务", variant="primary")
                                    clear_queue_btn = gr.Button("清空队列", variant="stop")
                                
                                # 队列操作状态
                                queue_operation_status = gr.Textbox(label="队列操作状态", interactive=False)
                                
                                # 详细队列状态显示
                                detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, lines=5, max_lines=10)
                        
                        # 绑定所有事件
                        generate_btn.click(
                            fn=generate_image_with_zimage,
                            inputs=[
                                prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size,
                                lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2,
                                enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength
                            ],
                            outputs=[output_status, output_gallery]
                        )
                        
                        # 添加到队列的事件绑定
                        add_to_queue_btn.click(
                            fn=lambda prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size, \
                               lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2, \
                               enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength: \
                               add_to_queue('zimage_txt2img', 
                                   prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size,
                                   lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2,
                                   enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength),
                            inputs=[
                                prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size,
                                lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2,
                                enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength
                            ],
                            outputs=[queue_operation_status]
                        )
                        
                        open_folder_btn.click(
                            fn=lambda: open_folder(Path(shared.data_path) / "outputs" / "z-image"),
                            inputs=[],
                            outputs=[output_status]
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
                            fn=lambda: process_queue(generate_image_with_zimage, generate_image_with_zimage_img2img),
                            inputs=[],
                            outputs=[output_status, output_gallery]
                        )
                        
                        # 为文生图界面的process_queue_btn也提供相同的img2img函数，以保证接口统一
                        process_queue_btn.click(
                            fn=lambda: process_queue(generate_image_with_zimage, generate_image_with_zimage_img2img),
                            inputs=[],
                            outputs=[output_status, output_gallery]
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
            
            # 图生图选项卡
            with gr.TabItem("图生图 (Image-to-Image)"):
                img2img_interface = create_z_image_img2img_ui()

    return ui


# 检查模型是否可用
def check_zimage_availability():
    """检查Z-Image模型和相关依赖是否可用"""
    try:
        import modelscope
        from modelscope import ZImagePipeline
        return True
    except:
        return False


# 模块是否可用的标志
Z_IMAGE_MODULE_AVAILABLE = check_zimage_availability()
