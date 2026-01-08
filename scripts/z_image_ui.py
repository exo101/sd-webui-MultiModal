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
    from modelscope import ZImagePipeline as ModelScopeZImagePipeline
    MODELScope_AVAILABLE = True
except ImportError as e:
    MODELScope_AVAILABLE = False

# 模块是否可用的标志
Z_IMAGE_MODULE_AVAILABLE = MODELScope_AVAILABLE

# 模型和输出目录
models_dir = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo"
output_dir = Path(shared.data_path) / "outputs" / "z-image-turbo"
output_dir.mkdir(parents=True, exist_ok=True)

# 全局变量
pipe = None
model_loaded = False
current_model_type = None  # 记录当前加载的模型类型 ('original', 'nunchaku')

def load_model_if_needed(model_type='original', nunchaku_precision='fp4', nunchaku_rank=128):
    """按需加载Z-Image-Turbo模型，使用SDNQ特定的加载方式"""
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

        if model_type == 'sdnq':
            # SDNQ量化模型加载逻辑 - 使用项目示例代码
            try:
                # SDNQ模型路径
                sdnq_model_path = Path(shared.models_path) / "Tongyi-MAI" / "Z-Image-Turbo-SDNQ-uint4-svd-r32"
                
                if not sdnq_model_path.exists():
                    return f"SDNQ模型路径不存在: {sdnq_model_path}"

                # 检查目录是否包含必要的文件
                try:
                    if not (sdnq_model_path / "transformer").exists():
                        return f"SDNQ模型缺少transformer目录: {sdnq_model_path}"
                    if not (sdnq_model_path / "text_encoder").exists():
                        return f"SDNQ模型缺少text_encoder目录: {sdnq_model_path}"
                    if not (sdnq_model_path / "vae").exists():
                        return f"SDNQ模型缺少vae目录: {sdnq_model_path}"
                except Exception as e:
                    return f"检查SDNQ模型路径时出错: {sdnq_model_path} - {str(e)}"

                # 导入SDNQ相关库
                import diffusers
                try:
                    # 尝试从系统导入sdnq库
                    from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
                    from sdnq.common import use_torch_compile as triton_is_available
                    from sdnq.loader import apply_sdnq_options_to_model
                except ImportError:
                    # 如果系统中没有sdnq库，尝试从本地扩展路径导入
                    sdnq_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sdnq")
                    if sdnq_path not in sys.path:
                        sys.path.insert(0, sdnq_path)
                    
                    try:
                        from sdnq import SDNQConfig
                        from sdnq.common import use_torch_compile as triton_is_available
                        from sdnq.loader import apply_sdnq_options_to_model
                    except ImportError:
                        return "错误：未安装SDNQ库或无法导入SDNQ相关模块。请确保已安装SDNQ库。"

                # 从本地路径加载SDNQ模型
                pipe = diffusers.pipelines.ZImagePipeline.from_pretrained(
                    str(sdnq_model_path),
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )

                # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
                if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
                    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
                    # 可选的编译优化，可能需要更多时间初始化
                    # pipe.transformer = torch.compile(pipe.transformer)

            except Exception as e:
                return f"加载SDNQ量化模型失败: {str(e)}"

            # 使用模型自带的设备管理机制
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

            current_model_type = 'sdnq'
        elif model_type == 'nunchaku':
            # Nunchaku模型加载逻辑
            try:
                # 确保使用扩展目录中的nunchaku库
                import sys
                import os
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
            except Exception as e:
                return f"创建ZImagePipeline失败: {str(e)}"

            # 确保pipe成功创建
            if pipe is None:
                return "创建ZImagePipeline失败: 返回的pipe为None"

            # 将模型移到CUDA设备
            try:
                pipe = pipe.to("cuda")
            except Exception as e:
                return f"将模型移动到CUDA设备失败: {str(e)}"

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

            # 从本地路径加载模型
            try:
                pipe = ZImagePipeline.from_pretrained(
                    str(model_save_path),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
            except Exception as e:
                return f"加载原始模型失败: {str(e)}"

            # 使用模型自带的设备管理机制
            # 参考FLUX Kontext的做法，启用模型CPU卸载而不是强制放到GPU
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

def generate_image(prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, batch_size=1,
                   enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7,
                   use_nunchaku=False, nunchaku_precision='fp4', nunchaku_rank=128,
                   use_sdnq=False):
    """使用Z-Image-Turbo生成图像，添加SDNQ量化模型支持"""
    global pipe, model_loaded

    try:
        # 检查提示词
        if not prompt or prompt.strip() == "":
            return "错误：请输入正向提示词", None

        # 如果种子为-1，则生成随机种子
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # 确定模型类型
        if use_sdnq:
            model_type = 'sdnq'
        elif use_nunchaku:
            model_type = 'nunchaku'
        else:
            model_type = 'original'

        # 按需加载模型
        if model_type == 'sdnq':
            status = load_model_if_needed(model_type)
        elif model_type == 'nunchaku':
            status = load_model_if_needed(model_type, nunchaku_precision, nunchaku_rank)
        else:
            status = load_model_if_needed(model_type)

        if "失败" in status or "失败" in status:
            return status, None

        # 确保pipe不为None
        if pipe is None:
            return f"错误：管道未正确初始化。模型类型: {model_type}, 加载状态: {status}", None

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

            # 根据模型类型使用不同的调用方式
            if current_model_type == 'sdnq':
                # 对于SDNQ模型，可能需要使用ModelScope的pipeline接口
                try:
                    # 如果pipe是ModelScope pipeline类型
                    if hasattr(pipe, 'call'):
                        result = pipe(
                            input={
                                'text': prompt,
                                'height': height,
                                'width': width,
                                'num_inference_steps': actual_steps,
                                'guidance_scale': actual_guidance,
                                'generator': generator
                            }
                        )
                    else:
                        # 否则尝试使用diffusers风格的调用
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
                    if hasattr(pipe, 'call'):
                        result = pipe(
                            input={
                                'text': prompt,
                                'height': height,
                                'width': width,
                                'num_inference_steps': actual_steps,
                                'guidance_scale': actual_guidance,
                                'generator': generator
                            }
                        )
                    else:
                        result = pipe(
                            prompt=prompt,
                            height=height,
                            width=width,
                            num_inference_steps=actual_steps,
                            guidance_scale=actual_guidance,
                            generator=generator,
                        )
            else:
                # 原始模型和Nunchaku模型的调用方式
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

            # 获取生成的图像 - 处理不同的返回格式
            if hasattr(result, "images") and result.images:
                image = result.images[0]
            elif isinstance(result, dict) and "images" in result:
                # ModelScope pipeline可能返回字典
                image = result["images"][0] if result["images"] else None
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                # 如果返回列表，取第一项
                image = result[0] if len(result) > 0 else None
            else:
                image = result if isinstance(result, (Image.Image, np.ndarray)) else None

            if image is None:
                return "错误：未能生成有效图像", None

            # 统一转换为 numpy 数组并确保是 uint8 类型
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            elif isinstance(image, torch.Tensor):
                # 转换为numpy数组
                image_tensor = image.detach().cpu()
                image_np = image_tensor.numpy()
            elif isinstance(image, np.ndarray):
                image_np = image.copy()  # 创建副本避免修改原始数据
            else:
                image_np = np.array(image)

            # 确保数值范围在 [0,1] 并转换为 uint8
            # 处理NaN和inf值
            image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)

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
        if use_sdnq:
            model_info = "SDNQ量化模型 (uint4, svd-rank 32)"
        elif use_nunchaku:
            model_info = f"Nunchaku模型 ({nunchaku_precision}, rank={nunchaku_rank})"
        else:
            model_info = "原始模型"
            
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
        gr.Markdown("基于 ModelScope 的超快速文生图模型")

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
                with gr.Accordion("Nunchaku 加速模型选项", open=False):
                    use_nunchaku = gr.Checkbox(label="使用 Nunchaku 加速模型", value=False)
                    with gr.Group() as nunchaku_options:
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
                
                # 添加SDNQ量化模型选项
                with gr.Accordion("SDNQ 量化模型选项", open=False):
                    use_sdnq = gr.Checkbox(label="使用 SDNQ 量化模型", value=False)
                    with gr.Group() as sdnq_options:
                        gr.Markdown("SDNQ (Stochastic Distilled Non-uniform Quantization) 量化模型，uint4精度，svd-rank 32")

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

            with gr.Column():  # 右半边 - 生成结果和按钮
                generate_btn = gr.Button("生成图像")
                output_info = gr.Textbox(label="生成信息", interactive=False)
                output_images = gr.Gallery(
                    label="生成结果",
                    interactive=False,
                    height=512,
                    object_fit="contain",
                    columns=3
                )

                # 添加打开输出目录按钮
                open_folder_btn = gr.Button("打开输出目录", elem_id="open_zimage_folder")
                open_folder_btn.click(
                    fn=lambda: open_folder(str(output_dir)),
                    inputs=[],
                    outputs=[]
                )

                generate_btn.click(
                    fn=generate_image,
                    inputs=[
                        prompt, negative_prompt, width, height,
                        steps, cfg_scale, seed, sampler, batch_size,
                        enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength,
                        use_nunchaku, nunchaku_precision, nunchaku_rank,  # 添加Nunchaku相关参数
                        use_sdnq  # 添加SDNQ相关参数
                    ],
                    outputs=[output_info, output_images]
                )

    return demo
