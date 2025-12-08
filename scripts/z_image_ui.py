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

# 尝试导入ModelScope
try:
    from modelscope import ZImagePipeline
    MODELScope_AVAILABLE = True
    print("Z-Image-Turbo: ModelScope库可用")
except ImportError as e:
    MODELScope_AVAILABLE = False
    print(f"Z-Image-Turbo: ModelScope库不可用: {e}")

# 模型和输出目录
models_dir = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
output_dir = Path(shared.data_path) / "outputs" / "z-image-turbo"
output_dir.mkdir(parents=True, exist_ok=True)

# 全局变量
pipe = None
model_loaded = False
current_model_type = None  # 记录当前加载的模型类型 ('original' 或 'gguf')
current_gguf_model = None  # 记录当前加载的GGUF模型文件名

def get_gguf_models():
    """获取目录中所有的GGUF模型文件"""
    gguf_files = list(models_dir.glob("*.gguf"))
    return [f.name for f in gguf_files]

def load_model_if_needed(model_type='original', gguf_model=None):
    """按需加载Z-Image-Turbo模型"""
    global pipe, model_loaded, current_model_type, current_gguf_model
    
    try:
        # 检查模型是否已经加载，且是相同类型的模型
        if (model_loaded and pipe is not None and 
            current_model_type == model_type and 
            (model_type != 'gguf' or current_gguf_model == gguf_model)):
            print("Z-Image-Turbo: 模型已加载，跳过重复加载")
            return "模型已加载"
        
        print("Z-Image-Turbo: 开始加载模型...")
        print(f"Z-Image-Turbo: 开始加载 {model_type} 类型的 Z-Image-Turbo 模型...")
        
        # 确保模型目录存在
        model_save_path = models_dir
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 使用ModelScope官方示例方式加载模型
        from modelscope import ZImagePipeline
        
        # 检查模型是否已经存在于指定路径
        model_files = list(model_save_path.glob("*"))
        print(f"Z-Image-Turbo: 模型目录中的文件: {model_files}")
        
        if model_type == 'gguf' and gguf_model:
            # GGUF模型加载逻辑
            gguf_path = model_save_path / gguf_model
            if not gguf_path.exists():
                return f"GGUF模型文件未找到: {gguf_model}"
            
            print(f"Z-Image-Turbo: 加载GGUF模型: {gguf_path}")
            
            # 按照示例代码正确加载GGUF模型
            from modelscope import ZImageTransformer2DModel, GGUFQuantizationConfig
            
            # 使用from_single_file方法加载GGUF模型文件，与官方示例保持一致
            # 指定transformer目录中的配置文件
            transformer_config_path = model_save_path / "transformer"
            transformer = ZImageTransformer2DModel.from_single_file(
                str(gguf_path),
                config=str(transformer_config_path),  # 指定transformer配置路径
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            
            # 使用transformer创建pipeline，指定本地路径避免网络请求
            pipe = ZImagePipeline.from_pretrained(
                str(model_save_path),
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            
            # 将模型移到CUDA设备
            pipe = pipe.to("cuda")
            
            current_model_type = 'gguf'
            current_gguf_model = gguf_model
        else:
            # 原始模型加载逻辑
            if model_save_path.exists() and model_files:
                # 从本地路径加载模型
                status_text = "正在加载本地模型..."
                print(status_text)
                print(f"Z-Image-Turbo: 从本地路径加载ModelScope模型: {model_save_path}")
                # 根据ModelScope专用管道规范，不使用load_in_8bit参数
                pipe = ZImagePipeline.from_pretrained(
                    str(model_save_path),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
            else:
                # 指定本地保存路径
                status_text = "正在下载模型，请稍候..."
                print(status_text)
                print(f"Z-Image-Turbo: ModelScope模型将保存到: {model_save_path}")
                # 根据ModelScope专用管道规范，不使用load_in_8bit参数
                pipe = ZImagePipeline.from_pretrained(
                    'Tongyi-MAI/Z-Image-Turbo',
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
                
                # 保存模型到本地
                try:
                    print(f"Z-Image-Turbo: 正在保存模型到: {model_save_path}")
                    pipe.save_pretrained(str(model_save_path))
                    print("Z-Image-Turbo: 模型保存完成")
                except Exception as save_error:
                    print(f"Z-Image-Turbo: 保存ModelScope模型时出错: {save_error}")
                    traceback.print_exc()
            
            # 使用模型自带的设备管理机制
            # 参考FLUX Kontext的做法，启用模型CPU卸载而不是强制放到GPU
            if hasattr(pipe, 'enable_model_cpu_offload'):
                print("Z-Image-Turbo: 启用模型CPU卸载")
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                print("Z-Image-Turbo: 启用序列化CPU卸载")
                pipe.enable_sequential_cpu_offload()
            else:
                # 如果没有CPU卸载功能，则尝试将模型移动到GPU
                try:
                    pipe = pipe.to("cuda")
                    print("Z-Image-Turbo: 模型已移动到CUDA设备")
                except Exception as move_error:
                    print(f"Z-Image-Turbo: 将模型移动到CUDA设备时出错: {move_error}")
                    # 如果无法移动到GPU，则保持在CPU上运行
            
            current_model_type = 'original'
            current_gguf_model = None
        
        model_loaded = True
        print("Z-Image-Turbo: 模型加载成功")
        return "模型加载成功"
    except Exception as e:
        error_msg = str(e)
        print(f"Z-Image-Turbo: ModelScope加载错误详情: {error_msg}")
        traceback.print_exc()
        return f"模型加载失败: {error_msg}"

def save_image(image_np, seed, index=0):
    """保存图像到output目录"""
    try:
        # 生成文件名
        timestamp = int(time.time())
        if index > 0:
            filename = f"z-image-turbo-{timestamp}-{seed}-{index}.png"
        else:
            filename = f"z-image-turbo-{timestamp}-{seed}.png"
        
        # 完整路径
        image_path = output_dir / filename
        
        # 保存图像
        if isinstance(image_np, np.ndarray):
            # 确保数组数据类型正确
            if image_np.dtype != np.uint8:
                if image_np.dtype in [np.float32, np.float64] and image_np.max() <= 1.0:
                    # 处理NaN和inf值
                    image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    # 处理NaN和inf值
                    image_np = np.nan_to_num(image_np, nan=0, posinf=255, neginf=0)
                    image_np = image_np.astype(np.uint8)
            
            # 转换为PIL图像并保存
            if len(image_np.shape) == 2:
                image = Image.fromarray(image_np, mode='L')
            elif len(image_np.shape) == 3:
                if image_np.shape[2] == 1:
                    image = Image.fromarray(image_np.squeeze(), mode='L')
                elif image_np.shape[2] == 3:
                    image = Image.fromarray(image_np, mode='RGB')
                elif image_np.shape[2] == 4:
                    image = Image.fromarray(image_np, mode='RGBA')
                    image = image.convert('RGB')
                else:
                    image = Image.fromarray(image_np).convert('RGB')
            else:
                image = Image.fromarray(image_np).convert('RGB')
            
            image.save(image_path, 'PNG')
        else:
            # 如果已经是PIL图像
            image_np.save(image_path, 'PNG')
        
        print(f"Z-Image-Turbo: 图像已保存到: {image_path}")
        return str(image_path)
    except Exception as e:
        print(f"Z-Image-Turbo: 保存图像时出错: {e}")
        traceback.print_exc()
        return None

def open_folder(folder_path):
    """打开文件夹"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder_path)
        elif sys.platform.startswith("darwin"):
            os.system(f'open "{folder_path}"')
        else:
            os.system(f'xdg-open "{folder_path}"')
        return f"已打开文件夹: {folder_path}"
    except Exception as e:
        print(f"打开文件夹时出错: {e}")
        return f"打开文件夹失败: {e}"

def create_z_image_ui():
    """创建Z-Image-Turbo UI界面"""
    print("Z-Image-Turbo: 开始创建 UI...")
    
    # 检查ModelScope是否可用
    if not MODELScope_AVAILABLE:
        print("Z-Image-Turbo: ModelScope 不可用，创建错误提示界面")
        with gr.Blocks() as demo:
            gr.Markdown("# Z-Image-Turbo 图像生成 (不可用)")
            gr.Markdown("错误：ModelScope 不可用，请检查安装")
        print("Z-Image-Turbo: ModelScope 不可用，返回简化UI")
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
    
    # 获取GGUF模型列表
    gguf_model_choices = get_gguf_models()
    default_gguf_model = gguf_model_choices[0] if gguf_model_choices else ""
    
    print("Z-Image-Turbo: 创建主界面")
    with gr.Blocks() as demo:
        gr.Markdown("# Z-Image-Turbo 图像生成")
        gr.Markdown("基于 ModelScope 的超快速文生图模型")
        
        with gr.Row():
            with gr.Column():
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
                
                # 添加GGUF模型选项
                with gr.Accordion("GGUF 模型选项", open=False):
                    use_gguf = gr.Checkbox(label="使用 GGUF 模型", value=False)
                    gguf_model = gr.Dropdown(
                        choices=gguf_model_choices,
                        value=default_gguf_model,
                        label="GGUF 模型",
                        interactive=True
                    )
                    # 添加刷新按钮
                    refresh_gguf_btn = gr.Button("刷新 GGUF 模型列表")
                    refresh_gguf_btn.click(
                        fn=lambda: gr.update(choices=get_gguf_models()),
                        inputs=[],
                        outputs=[gguf_model]
                    )
                
                # 添加高分辨率修复(Hires.fix)选项
                with gr.Accordion("高分辨率修复", open=False):
                    enable_hr = gr.Checkbox(label="启用高分辨率修复", value=False)
                    with gr.Group(visible=False) as hr_options:
                        hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="放大倍数", value=2.0)
                        hr_upscaler = gr.Dropdown(
                            label="放大算法",
                            choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]],
                            value=shared.latent_upscale_default_mode
                        )
                        hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label="高分辨率修复步数", value=0)
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="重绘幅度", value=0.7)
                        
                    enable_hr.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[enable_hr],
                        outputs=[hr_options]
                    )
                
                generate_btn = gr.Button("生成图像")
            
            with gr.Column():
                # 使用Gallery组件支持多图像显示
                output_images = gr.Gallery(
                    label="生成结果", 
                    interactive=False, 
                    height=512, 
                    object_fit="contain",
                    columns=3
                )
                output_info = gr.Textbox(label="生成信息", interactive=False)
                
                # 添加打开输出目录按钮
                open_folder_btn = gr.Button("打开输出目录", elem_id="open_zimage_folder")
                open_folder_btn.click(
                    fn=lambda: open_folder(str(output_dir)),
                    inputs=[],
                    outputs=[]
                )
        
        print("Z-Image-Turbo: 设置事件处理器")
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, negative_prompt, width, height, 
                steps, cfg_scale, seed, sampler, batch_size,
                enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                use_gguf, gguf_model  # 添加GGUF相关参数
            ],
            outputs=[output_info, output_images]
        )
    
    print("Z-Image-Turbo: UI 创建完成")
    return demo

# 模块是否可用的标志
Z_IMAGE_MODULE_AVAILABLE = MODELScope_AVAILABLE
print(f"Z-Image-Turbo: 模块可用性: {Z_IMAGE_MODULE_AVAILABLE}")

# 确保模块可以被正确导入
if __name__ != "__main__":
    # 当作为模块导入时，进行简单的初始化检查
    if Z_IMAGE_MODULE_AVAILABLE:
        print("Z-Image-Turbo: 模块初始化完成，准备就绪")
    else:
        print("Z-Image-Turbo: 模块初始化失败，功能不可用")

def generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size=1,
                   enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7,
                   use_gguf=False, gguf_model=None):
    """使用Z-Image-Turbo生成图像"""
    global pipe, model_loaded
    
    try:
        # 检查提示词
        if not prompt or prompt.strip() == "":
            return "错误：请输入正向提示词", None
        
        # 如果种子为-1，则生成随机种子
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # 确定模型类型
        model_type = 'gguf' if use_gguf and gguf_model else 'original'
        
        # 按需加载模型（首次会自动下载）
        status = load_model_if_needed(model_type, gguf_model)
        if "失败" in status:
            return status, None
        
        # 为Turbo模型设置合适的参数
        actual_steps = min(steps, 10)  # Turbo模型通常只需要很少的步数
        actual_guidance = 0.0  # 根据官方示例，Turbo模型应该使用0.0的guidance_scale
            
        # 生成图像 - 使用ModelScope官方示例方式
        images = []  # 用于存储批量生成的图像
        seeds = []   # 用于存储每个图像的种子
        saved_paths = []  # 用于存储保存的图像路径
        
        # 生成batch_size张图像
        for i in range(batch_size):
            # 为每张图像计算种子
            image_seed = seed + i if seed != -1 else random.randint(0, 2**32 - 1)
            seeds.append(image_seed)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 创建生成器，不指定设备让PyTorch自动处理
            generator = torch.Generator().manual_seed(image_seed)
            
            try:
                result = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=actual_steps,
                    guidance_scale=actual_guidance,
                    generator=generator,
                )
            except torch.cuda.OutOfMemoryError:
                # 如果GPU内存不足，尝试使用CPU执行
                generator = torch.Generator(device='cpu').manual_seed(image_seed)
                result = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=actual_steps,
                    guidance_scale=actual_guidance,
                    generator=generator,
                )
            
            # 获取生成的图像
            if hasattr(result, "images") and result.images:
                image = result.images[0]
            else:
                image = result if isinstance(result, (Image.Image, np.ndarray)) else None
                
            if image is None:
                return "错误：未能生成有效图像", None
            
            # 统一转换为 numpy 数组并确保是 uint8 类型
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            elif isinstance(image, torch.Tensor):
                # 处理tensor数据
                image_tensor = image.detach().cpu()
                # 处理不同的维度情况
                if image_tensor.dim() == 4:  # (1, C, H, W)
                    image_tensor = image_tensor.squeeze(0)
                if image_tensor.dim() == 3:  # (C, H, W)
                    # 转换为 (H, W, C) 格式
                    image_tensor = image_tensor.permute(1, 2, 0)
                
                # 检查是否有NaN或无穷大值
                if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                    print("警告：检测到模型输出包含NaN或无穷大值")
                    # 尝试用均值填充NaN和无穷大
                    image_tensor = torch.nan_to_num(image_tensor, nan=0.5, posinf=1.0, neginf=0.0)
                
                # 如果张量值域在[-1, 1]之间，需要转换到[0, 1]
                if image_tensor.min() < 0:
                    image_tensor = (image_tensor + 1) / 2
                # 确保值域在[0, 1]之间
                image_tensor = torch.clamp(image_tensor, 0, 1)
                # 转换为numpy数组
                image_np = image_tensor.numpy()
            elif isinstance(image, np.ndarray):
                image_np = image.copy()  # 创建副本避免修改原始数据
            else:
                image_np = np.array(image)
            
            # 确保数值范围在 [0,1] 并转换为 uint8
            # 处理NaN和inf值
            image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 确保正确的形状 (H, W, C) 或 (H, W)
            if len(image_np.shape) == 3 and image_np.shape[0] < image_np.shape[2]:
                # 可能是 (C, H, W) 格式，需要转换为 (H, W, C)
                image_np = np.transpose(image_np, (1, 2, 0))
            
            # 检查图像是否全黑或全白
            img_min, img_max = image_np.min(), image_np.max()
            if img_min == img_max:
                print(f"警告：图像数据异常，最小值={img_min}, 最大值={img_max}")
                # 如果是常量图像，生成一个测试图像
                image_np = np.ones_like(image_np) * 0.5  # 灰色图像用于测试
            
            # 确保值在正确范围内
            if image_np.max() <= 1.0:
                image_np = image_np * 255
            else:
                # 如果已经在0-255范围内，确保数值有效
                image_np = np.clip(image_np, 0, 255)
            
            image_np = image_np.astype(np.uint8)
            
            # 特别处理GGUF模型可能产生的问题
            if use_gguf and (np.all(image_np == 0) or np.all(image_np == 255)):
                print("警告：检测到GGUF模型生成的图像全黑或全白，尝试修复...")
                # 对于全黑或全白图像，创建一个测试图像以便诊断问题
                image_np = np.ones_like(image_np) * 127  # 灰色图像用于测试
            
            # 如果启用了高分辨率修复，则进行处理
            if enable_hr and hr_scale > 1.0:
                # 使用WebUI的上采样器来处理图像放大
                upscaler = next(iter([x for x in shared.sd_upscalers if x.name == hr_upscaler]), None)
                if upscaler:
                    hr_image = Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np
                    # 使用上采样器放大图像
                    upsampled_image = upscaler.scaler.upscale(hr_image, hr_scale, upscaler.data_path)
                    image_np = np.array(upsampled_image)
            
            images.append(image_np)
            
            # 保存图像
            saved_path = save_image(image_np, image_seed, i if batch_size > 1 else 0)
            if saved_path:
                saved_paths.append(saved_path)
        
        # 统一返回格式，Gallery组件会自动处理单张或多张图像
        model_info = f"GGUF模型: {gguf_model}" if use_gguf and gguf_model else "原始模型"
        result_info = f"""图像生成完成!
参数详情:
- Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
- Negative Prompt: {negative_prompt if negative_prompt and negative_prompt.strip() != '' else '无'}
- 尺寸: {width}x{height}
- 步数: {actual_steps}
- CFG: {actual_guidance}
- {'种子: ' + str(seeds[0]) if batch_size == 1 else '种子范围: ' + str(seeds[0]) + '-' + str(seeds[-1])}
- 采样方法: {sampler}
- 高分辨率修复: {'启用' if enable_hr else '禁用'}
- 模型类型: {model_info}
- 保存路径: {saved_paths[0] if saved_paths else '未保存'}"""
        
        # Gradio Gallery 组件可以自动处理单张图像的列表
        return result_info, images
            
    except Exception as e:
        error_details = traceback.format_exc()
        return f"图像生成失败: {str(e)}\n详细错误信息:\n{error_details}", None
