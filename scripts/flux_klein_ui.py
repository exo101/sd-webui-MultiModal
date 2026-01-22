import gradio as gr
import torch
import os
import gc
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from modules import shared
from modules import sd_samplers
from modules.ui_components import ToolButton
from modules import ui_common  # 导入ui_common模块
import requests
from PIL import Image

# 初始化全局变量
pipe = None
FLUX_KLEIN_LOADED = False

# 尝试导入diffusers相关模块
try:
    from diffusers import FluxPipeline
    from diffusers.schedulers import (
        FlowMatchEulerDiscreteScheduler,
        EulerDiscreteScheduler,
        DDIMScheduler,
        DDPMScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("请安装diffusers库: pip install diffusers")

# 尝试导入modelscope相关模块
try:
    from modelscope import Flux2KleinPipeline
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("请安装modelscope库: pip install modelscope")

# 尝试导入transformers相关模块
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("请安装transformers库: pip install transformers")


# 根据依赖库是否可用决定插件是否可用
FLUX_KLEIN_AVAILABLE = (DIFFUSERS_AVAILABLE or MODELSCOPE_AVAILABLE) and TRANSFORMERS_AVAILABLE

def download_flux_klein_model(model_path, model_type):
    """从魔搭社区下载FLUX.2-klein模型"""
    try:
        from modelscope import snapshot_download
        from pathlib import Path
        
        # 根据模型类型选择模型ID
        if model_type == "FLUX.2-klein-9B":
            model_id = "black-forest-labs/FLUX.2-klein-9B"
        else:  # 包括 "FLUX.2-klein-base-4B" 和其他情况
            model_id = "black-forest-labs/FLUX.2-klein-base-4B"  # 默认模型
            
        print(f"正在使用魔搭社区下载FLUX.2-klein模型，模型ID: {model_id} 到 {model_path}...")
        
        # 创建模型目录
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # 使用魔搭社区下载模型文件
        snapshot_download(
            model_id=model_id,
            cache_dir=model_path,
            revision='master'
        )
        print(f"FLUX.2-klein模型已成功从魔搭社区下载到: {model_path}")
        return True
    except Exception as e:
        print(f"使用魔搭社区下载FLUX.2-klein模型失败: {e}")
        # 如果魔搭社区下载失败，尝试Hugging Face方式作为备选
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            from pathlib import Path
            
            # 根据模型类型选择模型ID
            if model_type == "FLUX.2-klein-9B":
                model_id = "black-forest-labs/FLUX.2-klein-9B"
            else:  # 包括 "FLUX.2-klein-base-4B" 和其他情况
                model_id = "black-forest-labs/FLUX.2-klein-base-4B"  # 默认模型
                
            print(f"正在使用Hugging Face Hub下载FLUX.2-klein模型，模型ID: {model_id} 到 {model_path}...")
            
            # 创建模型目录
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            # 使用Hugging Face下载模型文件
            hf_snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"FLUX.2-klein模型已成功使用Hugging Face Hub下载到: {model_path}")
            return True
        except Exception as e2:
            print(f"下载FLUX.2-klein模型失败: {e2}")
            return False

def load_flux_klein_pipeline(model_path, model_type, device="cuda"):
    """加载FLUX.2-klein模型管道"""
    global pipe, FLUX_KLEIN_LOADED
    try:
        print(f"正在加载FLUX.2-klein模型，路径: {model_path}，类型: {model_type}...")
        
        # 确保model_type是字符串类型
        if isinstance(model_type, int):
            # 将索引转换为实际的模型类型字符串
            model_types = ["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"]
            if 0 <= model_type < len(model_types):
                model_type = model_types[model_type]
            else:
                model_type = "FLUX.2-klein-base-4B"  # 默认模型类型
        elif not isinstance(model_type, str):
            model_type = "FLUX.2-klein-base-4B"  # 默认模型类型
        
        # 检查模型路径是否存在，如果不存在则尝试构造完整路径
        base_model_path = Path(model_path)
        
        # 检查基础路径是否直接包含模型文件（检查是否有model_index.json）
        if (base_model_path / "model_index.json").exists():
            full_model_path = base_model_path
        else:
            # 基础路径不包含模型文件，尝试构造完整路径
            print(f"基础模型路径不包含模型文件，正在尝试查找实际模型目录...")
            if model_type == "FLUX.2-klein-9B":
                full_model_path = base_model_path / "FLUX_2-klein-9B"
            else:  # 包括 "FLUX.2-klein-base-4B" 和其他情况
                full_model_path = base_model_path / "FLUX_2-klein-base-4B"
        
        print(f"实际使用的模型路径: {full_model_path}")
        
        if not full_model_path.exists():
            print(f"模型路径不存在: {full_model_path}")
            return None

        # 检查是否可以使用modelscope的专用pipeline
        if MODELSCOPE_AVAILABLE and str(model_type).startswith("FLUX.2-klein-"):
            dtype = torch.bfloat16
            pipe = Flux2KleinPipeline.from_pretrained(
                str(full_model_path), 
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            # 启用模型CPU卸载以节省显存 - 这是关键优化
            pipe.enable_model_cpu_offload()
            
            # 启用切片注意力和VAE切片以进一步节省显存
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
        else:
            # 使用diffusers加载
            from diffusers import DiffusionPipeline
            
            pipe = DiffusionPipeline.from_pretrained(
                str(full_model_path),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # 启用显存优化
            pipe.enable_model_cpu_offload()  # 启用CPU卸载
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
        
        # 由于启用了CPU卸载，不需要手动移动到设备
        print(f"模型已配置CPU卸载以节省显存")
        
        FLUX_KLEIN_LOADED = True
        print("FLUX.2-klein模型加载完成！")
        return pipe
    except Exception as e:
        print(f"加载FLUX.2-klein模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def list_lora_models():
    """列出可用的LoRA模型"""
    lora_dir = os.path.join(shared.models_path, "Lora")
    if not os.path.exists(lora_dir):
        return []
    
    lora_files = []
    for file in os.listdir(lora_dir):
        if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
            lora_files.append(file)
    
    return lora_files

def apply_lora(pipe, lora_model, lora_weight):
    """应用LoRA模型"""
    try:
        if not lora_model:
            return
        
        lora_path = os.path.join(shared.models_path, "Lora", lora_model)
        if os.path.exists(lora_path):
            # 应用LoRA模型
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)
            print(f"LoRA模型已应用: {lora_model}, 权重: {lora_weight}")
        else:
            print(f"LoRA模型不存在: {lora_path}")
    except Exception as e:
        print(f"应用LoRA模型失败: {e}")

def remove_lora(pipe):
    """移除LoRA模型"""
    try:
        pipe.unfuse_lora()
        pipe.disable_lora()
        print("LoRA模型已移除")
    except Exception as e:
        print(f"移除LoRA模型失败: {e}")

def generate_flux_klein_image(prompt, steps, guidance_scale, height, width, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein生成图像"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe
        
    try:
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 生成图像 (FLUX模型不支持negative_prompt参数)
        images = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 移除LoRA
        if lora_enable and lora_model:
            remove_lora(pipe)
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_klein_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"图像生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"生成FLUX.2-klein图像失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存并提示用户
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"图像生成失败: {e}"


def multi_img_flux_klein(img1, img2, prompt, steps, guidance_scale, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行双图像结合"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if img1 is None:
            return None, "第一张图像不能为空"
            
        # 处理图像尺寸 - 保持原始尺寸，不强制调整
        img1 = Image.fromarray(img1).convert("RGB")
        # 不再调整为固定尺寸，保持原始宽高比
        
        # 如果提供了第二张图像，将其与第一张图像结合
        if img2 is not None:
            img2 = Image.fromarray(img2).convert("RGB")
            # 不再调整为固定尺寸，保持原始宽高比
            # 将两张图像作为列表传递给管道
            image_input = [img1, img2]
        else:
            # 只有一张图像，直接使用
            image_input = img1
        
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 使用图像作为条件进行生成，FLUX模型不使用negative_prompt
        images = pipe(
            prompt=prompt,
            image=image_input,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 移除LoRA
        if lora_enable and lora_model:
            remove_lora(pipe)
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_multi_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"双图像结合生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"双图像结合生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"双图像结合生成失败: {e}"

def inpaint_flux_klein(image_with_mask, prompt, steps, guidance_scale, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行局部编辑 - 使用蒙版区域作为编辑位置的指导"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if image_with_mask is None:
            return None, "输入图像不能为空"
        
        # 确保处理的是正确的数据类型
        image = None
        mask = None
        
        # 检查是否是ImageMask返回的字典格式
        if isinstance(image_with_mask, dict):
            # 根据webui实现，ImageMask返回包含'background'和'composite'键的字典
            if 'background' in image_with_mask and 'composite' in image_with_mask:
                # 获取图像和蒙版数据
                image_data = image_with_mask['background']
                mask_data = image_with_mask['composite']
                
                # 从复合图像中提取蒙版部分
                if isinstance(mask_data, np.ndarray):
                    # 如果是RGBA数组，提取alpha通道作为蒙版
                    if mask_data.ndim == 3 and mask_data.shape[2] == 4:
                        # Alpha channel is the 4th channel
                        mask_raw = mask_data[:, :, 3]
                        # 转换为L模式的PIL图像
                        mask = Image.fromarray(mask_raw.astype('uint8'), mode='L')
                    else:
                        # 如果不是RGBA，则使用灰度转换
                        mask_gray = np.dot(mask_data[...,:3], [0.2989, 0.5870, 0.1140])
                        mask = Image.fromarray(mask_gray.astype('uint8'), mode='L')
                elif isinstance(mask_data, Image.Image):
                    # 如果已经是PIL图像，提取alpha通道作为蒙版
                    mask = mask_data.split()[-1] if mask_data.mode in ('RGBA', 'LA') else mask_data.convert("L")
                else:
                    return None, f"蒙版数据格式不支持: {type(mask_data)}"
                
                # 确保图像是PIL RGB格式
                if isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data.astype('uint8'))
                elif isinstance(image_data, Image.Image):
                    image = image_data
                else:
                    # 如果是其他类型，尝试转换
                    try:
                        image = Image.fromarray(np.asarray(image_data).astype('uint8'))
                    except:
                        return None, f"图像数据格式不支持: {type(image_data)}"
                image = image.convert("RGB")
            else:
                return None, f"字典格式不正确，缺少必要键: {image_with_mask.keys() if hasattr(image_with_mask, 'keys') else type(image_with_mask)}"
        else:
            # 如果不是字典格式，说明没有蒙版，创建一个全黑的蒙版
            if isinstance(image_with_mask, np.ndarray):
                image = Image.fromarray(image_with_mask.astype('uint8'))
            elif isinstance(image_with_mask, Image.Image):
                image = image_with_mask
            else:
                # 尝试其他转换方式
                try:
                    image = Image.fromarray(np.asarray(image_with_mask).astype('uint8'))
                except:
                    return None, f"图像数据格式不支持: {type(image_with_mask)}"
            image = image.convert("RGB")
            # 创建一个全黑的蒙版（表示没有要编辑的区域）
            mask = Image.new("L", image.size, 0)
        
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 由于Flux2KleinPipeline不支持mask_image参数，我们只能使用图像作为条件进行生成
        # 我们可以尝试使用蒙版信息预处理图像，或者简单地依赖文本提示来指导编辑
        images = pipe(
            prompt=prompt,
            image=image,  # 使用图像作为条件
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 移除LoRA
        if lora_enable and lora_model:
            remove_lora(pipe)
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_inpaint_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"局部编辑生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"局部编辑生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"局部编辑生成失败: {e}"

def create_flux_klein_ui():
    """创建FLUX.2-klein的UI界面"""
    if not FLUX_KLEIN_AVAILABLE:
        with gr.Column():
            gr.Markdown("FLUX.2-klein模块当前不可用，可能是因为缺少依赖项。")
            gr.Markdown("- 需要安装 `diffusers` 库")
            gr.Markdown("- 需要安装 `modelscope` 库")
            gr.Markdown("- 需要安装 `transformers` 库")
        return

    with gr.Tabs():
        with gr.TabItem("文生图"):
            # 文生图界面组件
            with gr.Row():
                with gr.Column():
                    # 提示词输入区域
                    prompt = gr.Textbox(label="正面提示词 (Prompt)", lines=3, value="一只可爱的小猫")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 生成参数
                        steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 尺寸参数
                        height = gr.Slider(label="高度", minimum=256, maximum=1536, value=1024, step=64)
                        width = gr.Slider(label="宽度", minimum=256, maximum=1536, value=768, step=64)
                    
                    with gr.Row():
                        # 随机种子
                        seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    with gr.Row():
                        # 模型类型选择
                        model_type = gr.Dropdown(
                            choices=["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"],
                            value="FLUX.2-klein-base-4B",
                            label="模型类型"
                        )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=0.5,
                                    info="控制LoRA模型的影响强度"
                                )
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_lora_interactive(enable_lora):
                                return gr.update(interactive=enable_lora)
                            
                            lora_enable.change(
                                fn=update_lora_interactive,
                                inputs=lora_enable,
                                outputs=lora_model
                            )
                            
                            # 刷新LoRA模型列表的函数
                            def refresh_lora_models():
                                updated_choices = list_lora_models()
                                default_value = updated_choices[0] if updated_choices else ""
                                return gr.update(choices=updated_choices, value=default_value)
                            
                            # 绑定刷新按钮事件
                            refresh_lora_button.click(
                                fn=refresh_lora_models,
                                inputs=[],
                                outputs=lora_model
                            )
                    
                    with gr.Row():
                        # 生成按钮和打开目录按钮
                        gen_btn = gr.Button("生成图像", variant="primary")
                        open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                
                with gr.Column():
                    # 结果展示画廊
                    result_gallery = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="flux_klein_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    result_status = gr.Textbox(label="状态信息", interactive=False)
            
            # 事件绑定 - 文生图部分
            gen_btn.click(
                fn=generate_flux_klein_image,
                inputs=[prompt, steps, guidance_scale, height, width, seed, model_type, batch_size, lora_enable, lora_model, lora_weight],
                outputs=[result_gallery, result_status]
            )
            
            # 打开输出目录事件
            open_outputs_btn.click(
                fn=lambda: ui_common.open_folder("outputs"),
                inputs=[],
                outputs=[]
            )

        with gr.TabItem("双图像结合"):
            # 双图像结合界面组件
            with gr.Row():
                with gr.Column():
                    # 双图像结合输入区域
                    with gr.Row():  # 新增行容器，使两个图像组件并排
                        with gr.Column():
                            multi_img1 = gr.Image(label="第一张图像", type="numpy", height=300)
                        with gr.Column():
                            multi_img2 = gr.Image(label="第二张图像 (可选，留空则仅处理第一张图像)", type="numpy", height=300)
                    
                    # 提示词输入区域
                    multi_prompt = gr.Textbox(label="提示词", lines=3, value="结合两张图像特征生成")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 双图像结合参数
                        multi_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        multi_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 随机种子
                        multi_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        multi_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    with gr.Row():
                        # 模型类型选择
                        multi_model_type = gr.Dropdown(
                            choices=["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"],
                            value="FLUX.2-klein-base-4B",
                            label="模型类型"
                        )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                multi_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                multi_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                multi_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=0.5,
                                    info="控制LoRA模型的影响强度"
                                )
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                multi_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_multi_lora_interactive(enable_lora):
                                return gr.update(interactive=enable_lora)
                            
                            multi_lora_enable.change(
                                fn=update_multi_lora_interactive,
                                inputs=multi_lora_enable,
                                outputs=multi_lora_model
                            )
                            
                            # 刷新LoRA模型列表的函数
                            def refresh_multi_lora_models():
                                updated_choices = list_lora_models()
                                default_value = updated_choices[0] if updated_choices else ""
                                return gr.update(choices=updated_choices, value=default_value)
                            
                            # 绑定刷新按钮事件
                            multi_refresh_lora_button.click(
                                fn=refresh_multi_lora_models,
                                inputs=[],
                                outputs=multi_lora_model
                            )
                    
                    with gr.Row():
                        # 生成按钮和打开目录按钮
                        multi_btn = gr.Button("双图像结合生成", variant="primary")
                        multi_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                
                with gr.Column():
                    # 结果展示画廊
                    multi_result_gallery = gr.Gallery(
                        label="双图像结合结果",
                        show_label=True,
                        elem_id="flux_multi_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    multi_result_status = gr.Textbox(label="状态信息", interactive=False)
            
            # 事件绑定 - 双图像结合部分
            multi_btn.click(
                fn=multi_img_flux_klein,
                inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_model_type, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight],
                outputs=[multi_result_gallery, multi_result_status]
            )
            
            # 打开输出目录事件
            multi_open_outputs_btn.click(
                fn=lambda: ui_common.open_folder("outputs"),
                inputs=[],
                outputs=[]
            )

        with gr.TabItem("局部重绘"):
            # 局部重绘界面组件
            with gr.Row():
                with gr.Column():
                    # 图像+蒙版输入区域（使用Gradio的ImageMask组件）
                    inpaint_image = gr.ImageMask(
                        label="上传图像并绘制蒙版区域",
                        sources=['upload'],
                        interactive=True,
                        type="pil"  # 使用pil类型以更好地兼容处理流程
                    )
                    
                    # 提示词输入区域
                    inpaint_prompt = gr.Textbox(label="提示词", lines=3, value="修复这个区域，画一只小狗")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 局部重绘参数
                        inpaint_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        inpaint_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 随机种子
                        inpaint_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        inpaint_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    with gr.Row():
                        # 模型类型选择
                        inpaint_model_type = gr.Dropdown(
                            choices=["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"],
                            value="FLUX.2-klein-base-4B",
                            label="模型类型"
                        )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                inpaint_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                inpaint_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                inpaint_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=0.5,
                                    info="控制LoRA模型的影响强度"
                                )
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                inpaint_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_inpaint_lora_interactive(enable_lora):
                                return gr.update(interactive=enable_lora)
                            
                            inpaint_lora_enable.change(
                                fn=update_inpaint_lora_interactive,
                                inputs=inpaint_lora_enable,
                                outputs=inpaint_lora_model
                            )
                            
                            # 刷新LoRA模型列表的函数
                            def refresh_inpaint_lora_models():
                                updated_choices = list_lora_models()
                                default_value = updated_choices[0] if updated_choices else ""
                                return gr.update(choices=updated_choices, value=default_value)
                            
                            # 绑定刷新按钮事件
                            inpaint_refresh_lora_button.click(
                                fn=refresh_inpaint_lora_models,
                                inputs=[],
                                outputs=inpaint_lora_model
                            )
                    
                    with gr.Row():
                        # 生成按钮和打开目录按钮
                        inpaint_btn = gr.Button("局部重绘", variant="primary")
                        inpaint_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                
                with gr.Column():
                    # 结果展示画廊
                    inpaint_result_gallery = gr.Gallery(
                        label="局部重绘结果",
                        show_label=True,
                        elem_id="flux_inpaint_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    inpaint_result_status = gr.Textbox(label="状态信息", interactive=False)
            
            # 事件绑定 - 局部重绘部分
            inpaint_btn.click(
                fn=inpaint_flux_klein,
                inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_model_type, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
                outputs=[inpaint_result_gallery, inpaint_result_status]
            )
            
            # 打开输出目录事件
            inpaint_open_outputs_btn.click(
                fn=lambda: ui_common.open_folder("outputs"),
                inputs=[],
                outputs=[]
            )

    # 返回组件列表以便在其他地方使用（如果需要）
    return {
        "prompt": prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "seed": seed,
        "model_type": model_type,
        "gen_btn": gen_btn,
        "result_gallery": result_gallery,
        "result_status": result_status,
        "multi_img1": multi_img1,
        "multi_img2": multi_img2,
        "multi_prompt": multi_prompt,
        "multi_steps": multi_steps,
        "multi_guidance_scale": multi_guidance_scale,
        "multi_seed": multi_seed,
        "multi_model_type": multi_model_type,
        "multi_btn": multi_btn,
        "multi_result_gallery": multi_result_gallery,
        "multi_result_status": multi_result_status,
        "inpaint_image": inpaint_image,
        "inpaint_prompt": inpaint_prompt,
        "inpaint_steps": inpaint_steps,
        "inpaint_guidance_scale": inpaint_guidance_scale,
        "inpaint_seed": inpaint_seed,
        "inpaint_model_type": inpaint_model_type,
        "inpaint_btn": inpaint_btn,
        "inpaint_result_gallery": inpaint_result_gallery,
        "inpaint_result_status": inpaint_result_status
    }