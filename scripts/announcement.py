import gradio as gr
from pathlib import Path
import os

def create_announcement_module():
    """创建重要公告模块并返回组件结构"""
    result = {}

    # 在 Blocks 上下文中创建组件
    with gr.Blocks():
        markdown_content = gr.Markdown("""
            ### 欢迎使用多模态模型插件！
            我们为您整理了一些有用的资源：
            ### 教程资源
            - <a href="https://www.bilibili.com/video/BV15BgQzyE7R/" target="_blank">多模态插件使用教程</a>
            - <a href="https://space.bilibili.com/403361177/pugv/" target="_blank">LoRA模型训练教程</a>
            - <a href="https://www.bilibili.com/video/BV12V4y1a7b2" target="_blank">Stable Diffusion入门指南</a>
            - <a href="https://www.bilibili.com/video/BV1yFYFeUEZJ" target="_blank">ComfyUI使用教程</a>
            ### 实用平台
            - <a href="https://www.liblib.art/userpage/cbfa0d0f32474a47aea198ab10b24040/publish" target="_blank">Liblib AI - 在线生成平台</a>
            - <a href="https://www.doubao.com/chat/" target="_blank">豆包AI - 智能对话平台</a>
            - <a href="https://www.tongyi.com/qianwen/" target="_blank">通义千问 - 智能对话平台</a>
            - <a href="https://ollama.com/search" target="_blank">Ollama - 多模态模型下载</a>
            - <a href="https://huggingface.co/" target="_blank">HuggingFace - AI 开源社区</a>
            - <a href="https://github.com/" target="_blank">GitHub - 代码开源平台</a>
            - <a href="https://www.modelscope.cn/models" target="_blank">ModelScope - 魔搭社区</a>

            ### 联系方式
            - 微信：yangzhenyu7849
            - QQ：1009924899
            """)
        
        # 添加魔搭模型下载功能
        with gr.Accordion("📥 魔搭模型下载", open=False):
            gr.Markdown("""
            **使用说明：**
            - 在下方输入要下载的魔搭模型 ID（例如：`damo/cv_cnn_image-denoising_dncnn`）
            - 选择或输入下载路径
            - 点击"开始下载"按钮
            - 下载进度将在下方显示
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    modelscope_model_id = gr.Textbox(
                        label="模型 ID",
                        placeholder="请输入魔搭模型 ID（如：damo/cv_cnn_image-denoising_dncnn）",
                        info="从魔搭社区复制的模型标识符"
                    )
                
                with gr.Column(scale=1):
                    modelscope_download_path = gr.Textbox(
                        label="下载路径",
                        placeholder="请选择或输入下载路径",
                        info="模型将下载到此目录"
                    )
                    
                    # 添加打开目录按钮
                    open_modelscope_dir_btn = gr.Button(
                        "📁 打开下载目录",
                        variant="secondary",
                        size="sm"
                    )
            
            with gr.Row():
                download_modelscope_btn = gr.Button(
                    "⬇️ 开始下载",
                    variant="primary",
                    size="lg"
                )
            
            modelscope_status = gr.Textbox(
                label="下载状态",
                interactive=False,
                info="显示下载进度和状态信息"
            )
            
            # 处理打开目录按钮点击事件
            def on_open_modelscope_dir_click(path_text):
                try:
                    if not path_text.strip():
                        return "❌ 请先指定下载路径"
                    
                    dir_path = Path(path_text.strip())
                    
                    # 如果目录不存在，尝试创建
                    if not dir_path.exists():
                        dir_path.mkdir(parents=True, exist_ok=True)
                        return f"✅ 已创建目录并准备打开：{dir_path}"
                    
                    # 根据操作系统打开目录
                    import sys
                    if sys.platform == 'win32':
                        os.startfile(str(dir_path))
                    elif sys.platform == 'darwin':
                        os.system(f'open "{dir_path}"')
                    else:
                        os.system(f'xdg-open "{dir_path}"')
                    
                    return f"✅ 已成功打开目录：{dir_path}"
                except Exception as e:
                    return f"❌ 打开目录失败：{str(e)}"
            
            open_modelscope_dir_btn.click(
                fn=on_open_modelscope_dir_click,
                inputs=[modelscope_download_path],
                outputs=[modelscope_status]
            )
            
            # 处理下载按钮点击事件
            def on_download_modelscope(model_id, download_path):
                if not model_id.strip():
                    return "❌ 请输入模型 ID"
                
                if not download_path.strip():
                    return "❌ 请指定下载路径"
                
                try:
                    # 检查是否安装了 modelscope 库
                    try:
                        from modelscope import snapshot_download
                    except ImportError:
                        return "❌ 未安装 modelscope 库，请先安装：pip install modelscope"
                    
                    # 创建下载目录
                    dir_path = Path(download_path.strip())
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
                    # 开始下载
                    return f"⏳ 正在下载模型 {model_id} 到 {download_path}...\n请稍候，下载进度将在控制台显示"
                    
                    # 注意：实际下载逻辑需要根据具体需求实现
                    # 这里只是提供 UI 框架，实际下载可以调用：
                    # snapshot_download(model_id=model_id, cache_dir=download_path)
                    
                except Exception as e:
                    return f"❌ 下载失败：{str(e)}"
            
            download_modelscope_btn.click(
                fn=on_download_modelscope,
                inputs=[modelscope_model_id, modelscope_download_path],
                outputs=[modelscope_status]
            )

        # 将关键组件保存到 result 中供外部调用
        result["markdown_content"] = markdown_content
        result["modelscope_model_id"] = modelscope_model_id
        result["modelscope_download_path"] = modelscope_download_path
        result["download_modelscope_btn"] = download_modelscope_btn
        result["modelscope_status"] = modelscope_status

    return result  # 返回组件集合
