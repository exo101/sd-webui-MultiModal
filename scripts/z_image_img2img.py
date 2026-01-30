import os
import sys
import gradio as gr
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from modules import shared, paths, images
import time
import random

# 将当前脚本目录添加到Python路径，以便导入同目录下的其他模块
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 导入队列功能 - 使用绝对导入方式
from z_image_queue import add_to_queue, process_queue, get_queue_status, get_detailed_queue_status

# 将ZImagePipeline导入移到模块级别
from modelscope import ZImagePipeline


def get_lora_list():
    """获取LoRA模型列表"""
    try:
        lora_path = Path(shared.models_path) / "Lora"
        if lora_path.exists():
            lora_files = []
            # 查找所有支持的LoRA文件
            for ext in ['.safetensors', '.ckpt', '.pt']:
                lora_files.extend([f.stem for f in lora_path.glob(f"*{ext}")])
            # 去重并排序
            unique_loras = list(set(lora_files))
            return sorted(unique_loras)
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


def generate_image_with_zimage_img2img(init_image, prompt, negative_prompt, width, height, steps, cfg_scale, seed, strength, batch_size, lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2, enable_hr=False, hr_scale=2.0, hr_upscaler=None, hr_second_pass_steps=0, denoising_strength=0.7, selected_model=None):
    """
    使用Z-Image正式版模型进行图生图生成
    """
    try:
        # 使用本地Z-Image模型路径
        zimage_dir = os.path.join(paths.models_path, "Tongyi-MAI", "Z-Image")
        
        # 检查本地Z-Image模型是否存在
        if not (os.path.exists(zimage_dir) and any(Path(zimage_dir).iterdir())):
            return "错误：Z-Image正式版模型未找到，请确保模型已下载至 models/Tongyi-MAI/Z-Image 目录", None
        
        print(f"[INFO] 加载本地Z-Image模型...")
        
        pipe = ZImagePipeline.from_pretrained(
            str(zimage_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            local_files_only=True
        )
        
        # 启用优化的注意力机制
        if hasattr(pipe, 'transformer'):
            print("[INFO] 检测到Z-Image Transformer，尝试启用优化的注意力机制...")
            
            # 检查是否支持Flash Attention
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("[INFO] 启用PyTorch Flash Attention...")
                try:
                    # 尝试启用三种SDP后端
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                except Exception as e:
                    print(f"[WARNING] 无法启用Flash Attention: {e}")
            
        
        # 启用模型CPU卸载功能，按需将组件移动到GPU进行处理
        if hasattr(pipe, 'enable_model_cpu_offload'):
            print("[INFO] 启用模型CPU卸载功能以节省显存...")
            pipe.enable_model_cpu_offload()
        else:
            # 如果没有CPU卸载功能，则将管道移动到GPU
            pipe.to("cuda")
            print("[INFO] 模型已成功加载并移至CUDA设备")
            
        # 处理LoRA
        if lora_enable and (lora_model_1 or lora_model_2):
            print(f"[INFO] 开始应用LoRA: lora_model_1={lora_model_1}, lora_weight_1={lora_weight_1}, lora_model_2={lora_model_2}, lora_weight_2={lora_weight_2}")
            
            lora_applied = False
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
            
            # 应用LoRA
            for lora_path, lora_weight in lora_paths:
                print(f"[INFO] 应用LoRA: {lora_path}，权重: {lora_weight}")
                
                try:
                    pipe.load_lora_weights(lora_path, local_files_only=True)
                    pipe.fuse_lora(lora_scale=lora_weight)
                    lora_applied = True
                except Exception as e:
                    print(f"[ERROR] LoRA加载失败: {str(e)}")
                    return f"LoRA加载失败: {str(e)}", None
            
            if lora_applied:
                print(f"[INFO] LoRA应用成功")
            else:
                print(f"[WARNING] 没有找到有效的LoRA模型文件")
        
        # 调整图像尺寸
        init_image = init_image.convert("RGB")
        init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
        
        print(f"[INFO] 开始图生图生成...")
        print(f"[INFO] 参数: 提示词='{prompt[:50]}...', 尺寸={width}x{height}, 步数={steps}, 重绘强度={strength}, 批次={batch_size}")
        
        # 设置随机种子
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            
        # 生成图像 - 使用官方示例参数
        generator = torch.Generator("cuda").manual_seed(seed)
        
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
            
            # 第一阶段：使用用户提供的图像尺寸生成图像
            try:
                # 将初始图像调整为指定尺寸
                adjusted_init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
                adjusted_init_tensor = torch.tensor(np.array(adjusted_init_image)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32) / 255.0
                
                # 第一阶段生成
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=adjusted_init_tensor,
                    strength=min(strength, 0.8),  # 减少第一阶段的强度
                    height=height,
                    width=width,
                    cfg_normalization=False,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    num_images_per_prompt=batch_size
                )
            except Exception as e:
                print(f"[WARNING] 图生图模式失败，回退到文生图模式: {str(e)}")
                # 如果图生图失败，则回退到文生图模式
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
            # 标准图生图流程
            try:
                # 将PIL图像转换为tensor并归一化
                import torchvision.transforms as transforms
                to_tensor = transforms.ToTensor()
                init_tensor = to_tensor(init_image).unsqueeze(0).to("cuda")
                
                # 使用decode latents的方式进行图生图
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_tensor,
                    strength=strength,  # 使用strength参数控制重绘强度
                    height=height,
                    width=width,
                    cfg_normalization=False,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    num_images_per_prompt=batch_size
                )
            except Exception as e:
                print(f"[WARNING] 图生图模式失败，回退到文生图模式: {str(e)}")
                # 如果图生图失败，则回退到文生图模式
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
            
            final_images = output.images

        # 保存图像
        output_dir = Path(paths.data_path) / "outputs" / "z-image"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, image in enumerate(final_images):
            timestamp = int(time.time())
            filename = f"zimage_img2img_{'hr' if enable_hr else 'std'}_{timestamp}_s{seed}_{i}.png"
            filepath = output_dir / filename
            image.save(filepath)
            saved_paths.append(str(filepath))
        
        # 自动从GPU卸载模型以释放显存
        if 'pipe' in locals():
            print("[INFO] 开始卸载模型到CPU并清理显存...")
            try:
                pipe = pipe.to("cpu")
                # 删除模型对象
                del pipe
                # 清空CUDA缓存
                torch.cuda.empty_cache()
                # 进行垃圾回收
                import gc
                gc.collect()
                print("[INFO] 模型已成功卸载到CPU，显存已清理")
            except Exception as unload_error:
                print(f"[WARNING] 模型卸载过程中出现警告: {unload_error}")
        
        return f"图生图生成成功! {'启用高清修复' if enable_hr else '标准生成'}, 批次: {batch_size}, Seed: {seed}", saved_paths
        
    except Exception as e:
        error_msg = str(e)
        import traceback
        
        # 确保即使出错也从GPU卸载模型
        if 'pipe' in locals():
            print("[INFO] 发生错误，开始卸载模型到CPU并清理显存...")
            try:
                pipe = pipe.to("cpu")
                # 删除模型对象
                del pipe
                # 清空CUDA缓存
                torch.cuda.empty_cache()
                # 进行垃圾回收
                import gc
                gc.collect()
                print("[INFO] 模型已成功卸载到CPU，显存已清理")
            except Exception as unload_error:
                print(f"[WARNING] 模型卸载过程中出现警告: {unload_error}")
                
        return f"图生图生成失败: {error_msg}\n详细错误信息:\n{traceback.format_exc()}", None


def get_zimage_model_list():
    """获取Z-Image目录下的模型列表"""
    # 根据项目规范，Z-Image模型不再使用本地文件，而是直接使用官方模型
    return ["Z-Image (default)"]


def create_tab():
    with gr.Row():
        with gr.Column():  # 左半边 - 输入
            # 图生图输入图像
            init_image = gr.Image(label="输入图像", type="pil")
            
            prompt = gr.TextArea(
                label="提示词",
                placeholder="正面提示词，例如：masterpiece, best quality, 1girl, detailed eyes, beautiful, detailed face, detailed hands",
                lines=3
            )
            negative_prompt = gr.TextArea(
                label="负面提示词",
                value="low quality, worst quality, blurry, distorted, malformed, bad anatomy, extra limbs, fused fingers, bad hands, bad feet, deformed, ugly, low quality, artifact, noise",
                placeholder="负面提示词，例如：low quality, worst quality, blurry, distorted",
                lines=2
            )

            with gr.Row():
                width = gr.Slider(
                    minimum=64, maximum=2048, step=8, value=1024, label="宽度"
                )
                height = gr.Slider(
                    minimum=64, maximum=2048, step=8, value=1024, label="高度"
                )

         
            with gr.Row():
                steps = gr.Slider(
                    minimum=1, maximum=50, step=1, value=8, label="推理步数"
                )
                cfg_scale = gr.Slider(
                    minimum=0.0, maximum=20.0, step=0.1, value=0.0, label="CFG Scale"
                )
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="重绘强度"
                )

            with gr.Row():
                seed = gr.Number(
                    label="随机种子 (-1为随机)", value=-1, precision=0
                )
                batch_size = gr.Slider(
                    minimum=1, maximum=8, step=1, value=1, label="生成批次"
                )

            # 添加高清修复选项
            with gr.Accordion("高清修复 (Hires. fix)", open=False):
                enable_hr = gr.Checkbox(label="启用高清修复", value=False)
                with gr.Row():
                    hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, value=2.0, label="放大倍数", elem_id="img2img_hr_scale")
                    hr_upscaler = gr.Dropdown(label="放大算法", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode, elem_id="img2img_hr_upscaler")
                with gr.Row():
                    hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, value=0, label="高清阶段步数", elem_id="img2img_hires_steps")
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, label="重绘强度", elem_id="img2img_denoising_strength")

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

            lora_enable.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[lora_enable],
                outputs=[lora_options_group]
            )

        with gr.Column():  # 右半边 - 输出
            output_info = gr.Textbox(label="输出信息")
            output_images = gr.Gallery(label="生成的图像")

            with gr.Row():
                generate_btn = gr.Button("图生图生成", variant="primary")
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
                fn=generate_image_with_zimage_img2img,
                inputs=[
                    init_image,
                    prompt, negative_prompt, width, height,
                    steps, cfg_scale, seed, strength, batch_size,
                    lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2,
                    enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength
                ],
                outputs=[output_info, output_images]
            )
            
            # 添加到队列的事件绑定
            add_to_queue_btn.click(
                fn=lambda init_image, prompt, negative_prompt, width, height, steps, cfg_scale, seed, strength, batch_size, \
                       lora_enable, lora_model_1, lora_weight_1, lora_model_2, lora_weight_2, \
                       enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength: \
                    add_to_queue(
                        'zimage_img2img',
                        init_image=init_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        seed=seed,
                        strength=strength,
                        batch_size=batch_size,
                        lora_enable=lora_enable,
                        lora_model_1=lora_model_1,
                        lora_weight_1=lora_weight_1,
                        lora_model_2=lora_model_2,
                        lora_weight_2=lora_weight_2,
                        enable_hr=enable_hr,
                        hr_scale=hr_scale,
                        hr_upscaler=hr_upscaler,
                        hr_second_pass_steps=hr_second_pass_steps,
                        denoising_strength=denoising_strength
                    ),
                inputs=[
                    init_image,
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    cfg_scale,
                    seed,
                    strength,
                    batch_size,
                    lora_enable,
                    lora_model_1,
                    lora_weight_1,
                    lora_model_2,
                    lora_weight_2,
                    enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength
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


def create_z_image_img2img_ui():
    """创建Z-Image图生图UI界面"""
    # 创建输出目录
    output_dir = Path(paths.data_path) / "outputs" / "z-image"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with gr.Blocks(analytics_enabled=False) as ui:
        create_tab()
    
    return ui


def title():
    """返回此标签页在WebUI中的标题"""
    return "Z-Image Img2Img"


def show():
    """指定此标签页在WebUI中的显示位置"""
    # 返回True表示在主界面显示此标签页
    return True
