import gradio as gr
from PIL import Image
import os

# 使用webui环境中的rembg
from modules import shared

def create_image_matting_module():
    """创建智能抠图模块并返回组件结构"""
    result = {}

    # 移除Accordion包装，直接创建UI组件以适应新的布局
    with gr.Row():
        rm_upload = gr.Files(
            label="上传图片", 
            file_types=["image"],
            file_count="multiple",
            scale=3,
            height=400
        )
        
        with gr.Column(scale=1):
            rm_preview = gr.Image(
                label="图片预览",
                visible=False,
                height=300,
                interactive=False
            )
            
            rm_bg_color = gr.ColorPicker(
                label="背景颜色", 
                value="#FFFFFF",
                interactive=True,
                visible=True,
                show_label=True,
                container=True
            )
            rm_bg_transparent = gr.Button(
                "透明背景",
                size="sm",
                variant="secondary"
            )

    rm_process_btn = gr.Button(
        "开始处理",
        size="lg",
        variant="primary",
        elem_classes="orange-button",
        scale=1
    )
    
    rm_output = gr.Gallery(
        label="处理结果",
        columns=4,
        height=300,
        visible=True,
        object_fit="contain"
    )
    
    rm_progress = gr.Textbox(
        label="处理进度",
        value="等待处理...",
        interactive=False,
        scale=1
    )
    
    # 将关键组件保存到result中供外部调用
    result["rm_upload"] = rm_upload
    result["rm_preview"] = rm_preview
    result["rm_bg_color"] = rm_bg_color
    result["rm_bg_transparent"] = rm_bg_transparent
    result["rm_process_btn"] = rm_process_btn
    result["rm_output"] = rm_output
    result["rm_progress"] = rm_progress

    def update_preview(files):
        """根据上传的文件数量更新预览"""
        if files and len(files) == 1:
            # 只有一张图片时显示预览
            return gr.update(value=files[0].name, visible=True)
        else:
            # 多张图片或没有图片时不显示预览
            return gr.update(value=None, visible=False)

    def process_images(files, bg_color):
        if not files:
            raise gr.Error("请先上传图片")

        # 延迟导入rembg，确保在webui环境中使用
        try:
            from rembg import remove
        except ImportError:
            raise gr.Error("缺少依赖: rembg，请在webui环境中安装")

        # 检查CUDA是否可用，如果不可用则使用CPU模式
        use_cpu = False
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in providers:
                use_cpu = True
                print("[WARNING] CUDA不可用，将使用CPU模式处理图像")
        except Exception as e:
            use_cpu = True
            print(f"[WARNING] 检查ONNX Runtime提供程序时出错，将使用CPU模式: {str(e)}")

        processed_images = []
        save_dir = os.path.join(os.path.dirname(__file__), "rm_background_results")
        os.makedirs(save_dir, exist_ok=True)

        total = len(files)
        for i, file in enumerate(files):
            try:
                img = Image.open(file.name).convert("RGBA")
                
                # 根据CUDA可用性决定是否使用CPU模式
                if use_cpu:
                    img_no_bg = remove(img, session_opts={
                        'providers': ['CPUExecutionProvider']
                    })
                else:
                    img_no_bg = remove(img)

                if bg_color != "transparent":
                    bg = Image.new("RGBA", img.size, bg_color)
                    bg.paste(img_no_bg, (0, 0), img_no_bg)
                    img_final = bg.convert("RGB")
                else:
                    img_final = img_no_bg

                filename = os.path.splitext(os.path.basename(file.name))[0] + "_processed.png"
                save_path = os.path.join(save_dir, filename)
                img_final.save(save_path)

                img.close()
                img_no_bg.close()
                img_final.close()

                processed_images.append(save_path)

            except Exception as e:
                print(f"[ERROR] 处理失败: {str(e)}")
                # 不添加任何内容到processed_images，跳过失败的图像

        # 只返回处理成功的图像路径
        return processed_images, gr.update(visible=True), f"处理完成，共处理 {len(processed_images)} 张图片"

    def set_transparent():
        return "transparent"

    # 绑定上传事件到预览更新函数
    rm_upload.change(
        fn=update_preview,
        inputs=[rm_upload],
        outputs=[rm_preview]
    )

    rm_bg_transparent.click(
        fn=set_transparent,
        inputs=[],
        outputs=[rm_bg_color]
    )

    rm_process_btn.click(
        fn=process_images,
        inputs=[rm_upload, rm_bg_color],
        outputs=[rm_output, rm_progress]
    )

    return result  # 返回组件集合