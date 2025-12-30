import os
import torch
import diffusers
import gradio as gr
from modules import devices
from PIL import Image
from pathlib import Path
import time

def create_qwen_image_sdnq_ui():
    """
    创建Qwen-Image SDNQ量化模型UI界面
    """
    # 尝试导入SDNQ相关模块
    try:
        from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
        from sdnq.common import use_torch_compile as triton_is_available_func
        from sdnq.loader import apply_sdnq_options_to_model
        SDNQ_AVAILABLE = True
    except ImportError as e:
        SDNQ_AVAILABLE = False
        print(f"SDNQ相关模块不可用: {e}")
        with gr.Group():
            gr.Markdown("## Qwen-Image SDNQ模型不可用")
            gr.Markdown("请安装SDNQ库: `pip install git+https://github.com/bghira/sdnq.git`")
        return {}
   
    with gr.Row():
        # 左侧参数设置列
        with gr.Column(scale=4):
            # 使用固定的模型路径
            model_base_path = "models"
            model_path = os.path.join(model_base_path, "Qwen-Image-Layered-SDNQ-uint4-svd-r32")
            model_path_hidden = gr.Textbox(value=model_path, visible=False)  # 隐藏的模型路径

            with gr.Row():
                resolution = gr.Dropdown(
                    label="分辨率",
                    choices=[640, 1024],
                    value=640,
                    info="选择分辨率，640为推荐值",
                    interactive=True
                )
                layers = gr.Slider(
                    label="图层数量",
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                    info="指定图像分解的图层数量",
                    interactive=True
                )

            with gr.Row():
                true_cfg_scale = gr.Slider(
                    label="CFG缩放",
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.1,
                    info="分类器自由引导缩放系数",
                    interactive=True
                )
                num_inference_steps = gr.Slider(
                    label="推理步数",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                    info="生成图像的推理步数",
                    interactive=True
                )

            with gr.Row():
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    value=" ",
                    placeholder="输入负面提示词",
                    interactive=True
                )
                prompt = gr.Textbox(
                    label="正面提示词",
                    value=" ",
                    placeholder="输入正面提示词",
                    interactive=True
                )

            with gr.Row():
                use_en_prompt = gr.Checkbox(
                    label="使用英文提示",
                    value=True,
                    info="是否使用英文提示词，关闭则为中文",
                    interactive=True
                )
                cfg_normalize = gr.Checkbox(
                    label="CFG归一化",
                    value=True,
                    info="是否启用CFG归一化",
                    interactive=True
                )

            with gr.Row():
                input_image = gr.Image(
                    type="filepath",
                    label="输入图像",
                    interactive=True,
                    height=300
                )

            with gr.Row():
                generate_btn = gr.Button("生成图像", variant="primary")
                open_output_folder = gr.Button("打开输出目录", variant="secondary")

        # 右侧结果展示列
        with gr.Column(scale=6):
            with gr.Row():
                output_gallery = gr.Gallery(
                    label="输出图像",
                    show_label=True,
                    columns=2,
                    object_fit="contain",
                    height="auto"
                )

        def process_image(image_path, resolution, layers, true_cfg_scale, 
                         num_inference_steps, negative_prompt, prompt, 
                         use_en_prompt, cfg_normalize):
            """
            处理图像的主要函数
            """
            if not image_path:
                return ["请上传输入图像"], []
            
            # 使用预设的模型路径
            model_path = os.path.join(model_base_path, "Qwen-Image-Layered-SDNQ-uint4-svd-r32")
            
            # 验证模型路径是否存在
            if not os.path.exists(model_path):
                print(f"模型路径不存在，将从Hugging Face下载: {model_path}")
                model_path = "Disty0/Qwen-Image-Layered-SDNQ-uint4-svd-r32"
            
            try:
                # 首先导入SDNQ，这会将QwenImageLayeredPipeline注册到diffusers中
                import sdnq
                from sdnq import SDNQConfig
                from sdnq.common import use_torch_compile as triton_is_available_func
                from sdnq.loader import apply_sdnq_options_to_model
                
                # 现在可以安全地导入QwenImageLayeredPipeline
                from diffusers import QwenImageLayeredPipeline
                
                print(f"正在从 {model_path} 加载模型...")
                
                # 加载QwenImageLayeredPipeline
                pipe = QwenImageLayeredPipeline.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16
                )
                
                # 确保加载的确实是QwenImageLayeredPipeline
                print(f"加载的模型类型: {type(pipe)}")
                
                # 检查是否包含QwenImageLayeredPipeline的必要组件
                required_attrs = ['transformer', 'text_encoder', 'vae']
                for attr in required_attrs:
                    if not hasattr(pipe, attr):
                        print(f"警告: 模型缺少预期的组件: {attr}")
                
                print("模型加载成功")

                # 启用量化选项（如果支持）
                if triton_is_available_func and (
                    torch.cuda.is_available() or torch.xpu.is_available()
                ):
                    try:
                        if hasattr(pipe, 'transformer'):
                            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                        if hasattr(pipe, 'text_encoder'):
                            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
                        if hasattr(pipe, 'vae'):
                            pipe.vae = apply_sdnq_options_to_model(pipe.vae, use_quantized_matmul=True)
                        print("✅ 成功应用SDNQ量化优化")
                    except Exception as e:
                        print(f"⚠️ 应用SDNQ量化优化时出错: {e}")
                        # 继续执行，不中断处理
                else:
                    print("⚠️ 跳过量化优化（Triton不可用或选项未启用）")

                # 启用模型CPU卸载以节省显存
                pipe.enable_model_cpu_offload()
                pipe.set_progress_bar_config(disable=None)

                # 加载输入图像
                print(f"正在加载输入图像: {image_path}")
                image = Image.open(image_path).convert("RGBA")
                print(f"图像尺寸: {image.size}")

                # 执行推理 - 按照官方示例
                print("开始执行推理...")
                with torch.inference_mode():
                    result = pipe(
                        image=image,
                        generator=torch.manual_seed(777),
                        true_cfg_scale=true_cfg_scale,
                        prompt=prompt,  # 使用正面提示词
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=1,
                        layers=layers,
                        resolution=resolution,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
                        cfg_normalize=cfg_normalize,  # Whether enable cfg normalization.
                        use_en_prompt=use_en_prompt,  # 是否使用英文提示词
                    )
                
                # 处理输出 - 根据返回类型正确获取图像
                if hasattr(result, 'images'):
                    output = result.images
                else:
                    output = result  # 假设直接返回图像列表
                
                print("推理完成")

                # 生成输出图像路径
                timestamp = int(time.time())
                output_images_paths = []
                
                # 确保输出目录存在
                os.makedirs("outputs", exist_ok=True)
                
                # 保存图像 - 处理可能的列表或单个图像
                if isinstance(output, list):
                    # 如果输出是图像列表，保存所有图像
                    for i, img in enumerate(output):
                        # 检查img是否还是列表，如果是，则展开它
                        if isinstance(img, list):
                            for j, sub_img in enumerate(img):
                                output_image_path = f"outputs/qwen_image_sdnq_{timestamp}_{i}_{j}.png"
                                sub_img.save(output_image_path)
                                output_images_paths.append(output_image_path)
                        else:
                            output_image_path = f"outputs/qwen_image_sdnq_{timestamp}_{i}.png"
                            img.save(output_image_path)
                            output_images_paths.append(output_image_path)
                elif hasattr(output, 'save'):
                    # 如果输出是单个图像对象，直接保存
                    output_image_path = f"outputs/qwen_image_sdnq_{timestamp}_0.png"
                    output.save(output_image_path)
                    output_images_paths.append(output_image_path)
                else:
                    # 尝试将输出视为可迭代对象
                    try:
                        for i, img in enumerate(output):
                            # 检查img是否还是列表，如果是，则展开它
                            if isinstance(img, list):
                                for j, sub_img in enumerate(img):
                                    output_image_path = f"outputs/qwen_image_sdnq_{timestamp}_{i}_{j}.png"
                                    sub_img.save(output_image_path)
                                    output_images_paths.append(output_image_path)
                            else:
                                output_image_path = f"outputs/qwen_image_sdnq_{timestamp}_{i}.png"
                                img.save(output_image_path)
                                output_images_paths.append(output_image_path)
                    except (TypeError, StopIteration):
                        raise ValueError(f"无法处理模型输出格式: {type(output)}")
                
                print(f"输出图像已保存: {output_images_paths}")
                
                # 返回结果 - 状态文本和图像路径列表
                status_text = f"✅ 处理完成！输出图像: {len(output_images_paths)}张"
                return [status_text], output_images_paths

            except Exception as e:
                error_msg = f"❌ 处理过程中出现错误: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return [error_msg], []

        def open_folder():
            """打开输出目录"""
            import subprocess
            import platform
            output_path = "outputs"  # 修正：打开输出图像的目录
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_path])
            elif platform.system() == "Darwin":
                subprocess.run(["open", output_path])
            else:
                subprocess.run(["xdg-open", output_path])
            return gr.update()

        generate_btn.click(
            fn=process_image,
            inputs=[
                input_image, resolution, layers, true_cfg_scale,
                num_inference_steps, negative_prompt, prompt,
                use_en_prompt, cfg_normalize
            ],
            outputs=[output_gallery, output_gallery]  # 第一个是状态文本，第二个是图像路径列表
        )
        
        open_output_folder.click(
            fn=open_folder,
            inputs=[],
            outputs=[]
        )

    return {
        "resolution": resolution,
        "layers": layers,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": num_inference_steps,
        "negative_prompt": negative_prompt,
        "prompt": prompt,
        "use_en_prompt": use_en_prompt,
        "cfg_normalize": cfg_normalize,
        "input_image": input_image,
        "generate_btn": generate_btn,
        "output_gallery": output_gallery
    }

# 导出模块可用性标志
QWEN_IMAGE_SDNQ_MODULE_AVAILABLE = False
try:
    # 检查SDNQ模块
    from sdnq import SDNQConfig
    QWEN_IMAGE_SDNQ_MODULE_AVAILABLE = True
except ImportError:
    pass
