import gradio as gr
from pathlib import Path
def create_announcement_module():
    """创建重要公告模块并返回组件结构"""
    result = {}

    # 在Blocks上下文中创建组件
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
            - <a href="https://huggingface.co/" target="_blank">HuggingFace - AI开源社区</a>
            - <a href="https://github.com/" target="_blank">GitHub - 代码开源平台</a>
            - <a href="https://www.modelscope.cn/models" target="_blank">ModelScope - 魔搭社区</a>

            ### 联系方式
            - 微信：yangzhenyu7849
            - QQ：1009924899
            """)

        # 将关键组件保存到result中供外部调用
        result["markdown_content"] = markdown_content

    return result  # 返回组件集合
