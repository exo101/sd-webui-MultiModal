import os
import sys
from modules import util
from modules import shared
import gradio as gr

def open_folder(folder_path):
    """打开指定文件夹"""
    folder_abs_path = os.path.abspath(folder_path)
    if os.path.exists(folder_abs_path):
        if sys.platform == 'win32':
            os.startfile(folder_abs_path)
        elif sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.run(['open', folder_abs_path])
        else:  # Linux
            import subprocess
            subprocess.run(['xdg-open', folder_abs_path])
    return gr.update()

def create_flux_klein_angle_visualization_component(prompt_component):
    """创建3D角度可视化选择器组件，导入已有的flux_klein_angle_selector组件"""
    try:
        # 导入现有的flux_klein_angle_selector组件
        from .flux_klein_angle_selector import create_flux_klein_angle_visualization_component as selector_component
        return selector_component(prompt_component)
    except ImportError:
        # 如果导入失败，创建一个简单的替代方案
        with gr.Row():
            gr.Markdown("### 3D视角预设")
            with gr.Column():
                isometric_right = gr.Button("右前等距视角", interactive=True)
                side_view = gr.Button("左侧面图", interactive=True)
            with gr.Column():
                isometric_left = gr.Button("左前等距视角", interactive=True)
                top_down = gr.Button("俯视图", interactive=True)

        # 定义按钮点击事件
        def select_isometric_right():
            return "ISOMETRIC ↘ Front-right view"

        def select_isometric_left():
            return "ISOMETRIC ↙ Front-left view"

        def select_side_view():
            return "SIDE VIEW ← Profile from left"

        def select_top_down():
            return "TOP-DOWN ↑ Bird's eye view"

        # 绑定按钮事件到提示词组件
        isometric_right.click(fn=select_isometric_right, outputs=prompt_component)
        isometric_left.click(fn=select_isometric_left, outputs=prompt_component)
        side_view.click(fn=select_side_view, outputs=prompt_component)
        top_down.click(fn=select_top_down, outputs=prompt_component)
        
        return prompt_component

def refresh_lora_models():
    """刷新LoRA模型列表"""
    from .flux_klein_model_loader import list_lora_models
    updated_choices = list_lora_models()
    default_value = updated_choices[0] if updated_choices else ""
    return gr.update(choices=updated_choices, value=default_value)

def update_lora_interactive(enable_lora):
    """更新LoRA模型选择框的交互状态"""
    return gr.update(interactive=not (enable_lora is None or enable_lora is False))

def clear_queue():
    """清空队列"""
    global task_queue
    import queue
    task_queue = queue.Queue()  # 重新创建空队列
    return "队列已清空"