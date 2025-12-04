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

# 尝试导入WebUI的采样器模块
try:
    from modules import sd_samplers
    WEBUI_SAMPLERS_AVAILABLE = True
except ImportError:
    WEBUI_SAMPLERS_AVAILABLE = False

# 尝试导入WebUI的调度器模块
try:
    from modules import sd_schedulers
    WEBUI_SCHEDULERS_AVAILABLE = True
except ImportError:
    WEBUI_SCHEDULERS_AVAILABLE = False

# 模型和输出目录
models_dir = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
output_dir = Path(shared.data_path) / "outputs" / "z-image-turbo"
output_dir.mkdir(parents=True, exist_ok=True)

# 全局变量
pipe = None
model_loaded = False

def load_model_if_needed():
    """按需加载Z-Image-Turbo模型"""
    global pipe, model_loaded
    
    try:
        # 检查模型是否已经加载
        if model_loaded and pipe is not None:
            print("Z-Image-Turbo: 模型已加载，跳过重复加载")
            return "模型已加载"
        
        print("Z-Image-Turbo: 开始加载模型...")
        print("Z-Image-Turbo: 开始加载 Z-Image-Turbo 模型...")
        
        # 确保模型目录存在
        model_save_path = models_dir
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 使用ModelScope官方示例方式加载模型
        from modelscope import ZImagePipeline
        
        # 检查模型是否已经存在于指定路径
        model_files = list(model_save_path.glob("*"))
        print(f"Z-Image-Turbo: 模型目录中的文件: {model_files}")
        
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
                    image_np = (image_np * 255).astype(np.uint8)
                else:
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
                
                # 获取WebUI内置的采样器选项
                if WEBUI_SAMPLERS_AVAILABLE:
                    try:
                        sampler_choices = [(sampler.name, sampler.name) for sampler in sd_samplers.visible_samplers()]
                    except:
                        sampler_choices = [
                            ("Euler", "euler"),
                            ("Euler Ancestral", "euler_ancestral"),
                            ("Heun", "heun"),
                            ("DPM++ 2M", "dpmpp_2m")
                        ]
                else:
                    sampler_choices = [
                        ("Euler", "euler"),
                        ("Euler Ancestral", "euler_ancestral"),
                        ("Heun", "heun"),
                        ("DPM++ 2M", "dpmpp_2m")
                    ]
                
                sampler = gr.Dropdown(
                    choices=sampler_choices,
                    value=sampler_choices[0][1] if sampler_choices else "euler",
                    label="采样方法"
                )
                
                # 添加调度器选项
                if WEBUI_SCHEDULERS_AVAILABLE:
                    try:
                        scheduler_choices = [(scheduler.label, scheduler.name) for scheduler in sd_schedulers.schedulers]
                    except:
                        scheduler_choices = [
                            ("Automatic", "automatic"),
                            ("Karras", "karras"),
                            ("Exponential", "exponential"),
                            ("SGM Uniform", "sgm_uniform"),
                            ("Simple", "simple"),
                            ("Normal", "normal"),
                            ("DDIM", "ddim_uniform")
                        ]
                else:
                    scheduler_choices = [
                        ("Automatic", "automatic"),
                        ("Karras", "karras"),
                        ("Exponential", "exponential"),
                        ("SGM Uniform", "sgm_uniform"),
                        ("Simple", "simple"),
                        ("Normal", "normal"),
                        ("DDIM", "ddim_uniform")
                    ]
                
                scheduler = gr.Dropdown(
                    choices=scheduler_choices,
                    value=scheduler_choices[0][1] if scheduler_choices else "automatic",
                    label="调度器"
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
        
        print("Z-Image-Turbo: 设置事件处理器")
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, negative_prompt, width, height, 
                steps, cfg_scale, seed, sampler, scheduler, batch_size,
                enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength
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

def generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, scheduler, batch_size=1,
                   enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7):
    """使用Z-Image-Turbo生成图像"""
    global pipe, model_loaded
    
    try:
        # 检查提示词
        if not prompt or prompt.strip() == "":
            return "错误：请输入正向提示词", None
        
        # 如果种子为-1，则生成随机种子
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # 按需加载模型（首次会自动下载）
        status = load_model_if_needed()
        if "失败" in status:
            return status, None
        
        # 为Turbo模型设置合适的参数
        actual_steps = min(steps, 10)  # Turbo模型通常只需要很少的步数
        actual_guidance = 0.0  # 根据官方示例，Turbo模型应该使用0.0的guidance_scale
        
        # 尝试根据选择的调度器设置参数
        scheduler_config = {}
        try:
            if scheduler == "karras":
                scheduler_config["use_karras_sigmas"] = True
            elif scheduler == "exponential":
                scheduler_config["use_exponential_sigmas"] = True
            elif scheduler == "sgm_uniform":
                scheduler_config["use_beta_sigmas"] = True
            # 其他调度器使用默认配置
        except:
            pass  # 如果有任何错误，使用默认配置
            
        # 创建调度器配置（如果模型支持）
        if pipe and hasattr(pipe, 'scheduler') and scheduler_config:
            try:
                pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config, **scheduler_config)
            except:
                pass  # 如果配置失败，继续使用默认调度器
            
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
                image_np = image.cpu().numpy()
            elif isinstance(image, np.ndarray):
                image_np = image
            else:
                image_np = np.array(image)
            
            # 确保数值范围在 [0,255] 并转换为 uint8
            image_np = np.clip(image_np, 0, 1) * 255 if image_np.max() <= 1 else image_np
            image_np = image_np.astype(np.uint8)
            
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
        result_info = f"""图像生成完成!
参数详情:
- Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
- Negative Prompt: {negative_prompt if negative_prompt and negative_prompt.strip() != '' else '无'}
- 尺寸: {width}x{height}
- 步数: {actual_steps}
- CFG: {actual_guidance}
- {'种子: ' + str(seeds[0]) if batch_size == 1 else '种子范围: ' + str(seeds[0]) + '-' + str(seeds[-1])}
- 采样方法: {sampler}
- 调度器: {scheduler}
- 高分辨率修复: {'启用' if enable_hr else '禁用'}
- 保存路径: {saved_paths[0] if saved_paths else '未保存'}"""
        
        # Gradio Gallery 组件可以自动处理单张图像的列表
        return result_info, images
            
    except Exception as e:
        error_details = traceback.format_exc()
        return f"图像生成失败: {str(e)}\n详细错误信息:\n{error_details}", None

