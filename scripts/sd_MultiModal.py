import os
import torch
from modules import devices
import gradio as gr
import numpy as np
import datetime
from pathlib import Path
from modules import script_callbacks
import webbrowser
import subprocess
import sys
import time
import importlib

# 添加scripts目录到系统路径，确保模块可以被正确加载
scripts_dir = Path(__file__).parent
if str(scripts_dir) is not None and str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# 尝试导入各个功能模块
def import_modules():
    """尝试导入所有必要的模块，并返回包含这些模块的命名空间对象"""
    def _import_and_register_modules():
        # 确保当前脚本目录在Python路径中
        script_dir = str(scripts_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
            
        try: from prompt_templates import create_prompt_template_ui
        except ImportError: create_prompt_template_ui = None
        
        try: from quick_description import create_quick_description
        except ImportError: create_quick_description = None
        
        try: from image_matting import create_image_matting_module
        except ImportError: create_image_matting_module = None
        
        try: from image_management import create_image_management_module
        except ImportError: create_image_management_module = None
        
        try: from tag_management import create_tag_management_module
        except ImportError: create_tag_management_module = None
        
        try: from announcement import create_announcement_module
        except ImportError: create_announcement_module = None
          
        try: from flux_kontext_ui import create_flux_kontext_ui, FLUX_KONTEXT_AVAILABLE
        except ImportError: 
            create_flux_kontext_ui = None
            FLUX_KONTEXT_AVAILABLE = False
            
        try: 
            from cleaner_ui import create_cleaner_module, CLEANER_AVAILABLE
        except ImportError: 
            create_cleaner_module = None
            CLEANER_AVAILABLE = False
            
        try: 
            from qwen_image_ui import create_qwen_image_ui, QWEN_IMAGE_MODULE_AVAILABLE
        except ImportError: 
            create_qwen_image_ui = None
            QWEN_IMAGE_MODULE_AVAILABLE = False
            
        try: 
            from qwen_image_edit import create_qwen_image_edit_ui, QWEN_IMAGE_EDIT_MODULE_AVAILABLE
        except ImportError: 
            create_qwen_image_edit_ui = None
            QWEN_IMAGE_EDIT_MODULE_AVAILABLE = False
            
        try: 
            from segment_anything_ui import create_sam_ui, SAM_AVAILABLE
        except ImportError: 
            create_sam_ui = None
            SAM_AVAILABLE = False
        
        # 尝试导入FLUX.1-krea模块
        try: 
            from flux_krea_ui import create_flux_krea_ui, FLUX_KREA_AVAILABLE
        except ImportError: 
            create_flux_krea_ui = None
            FLUX_KREA_AVAILABLE = False

        # 尝试导入FLUX.2-klein模块
        try: 
            from flux_klein_ui import create_flux_klein_ui, FLUX_KLEIN_AVAILABLE
        except ImportError: 
            create_flux_klein_ui = None
            FLUX_KLEIN_AVAILABLE = False

         # 添加Z-Image模块导入
        try: 
            from z_image_ui import create_z_image_ui as z_image_create_func, Z_IMAGE_MODULE_AVAILABLE
        except ImportError: 
            z_image_create_func = None
            Z_IMAGE_MODULE_AVAILABLE = False
            
            
        # 添加Qwen API模块导入
        try: 
            from qwen_api_ui import create_qwen_api_ui
        except ImportError: 
            create_qwen_api_ui = None

        # 注释掉已移动的功能模块导入
        # try: 
        #     from qwen_video import create_qwen_video_gen_ui, QWEN_VIDEO_GEN_AVAILABLE
        # except ImportError: 
        #     create_qwen_video_gen_ui = None
        #     QWEN_VIDEO_GEN_AVAILABLE = False

        # 返回命名空间对象
        import types
        namespace = types.SimpleNamespace()
        namespace.create_prompt_template_ui = create_prompt_template_ui
        namespace.create_quick_description = create_quick_description
        # namespace.create_video_frame_extractor = create_video_frame_extractor
        namespace.create_image_matting_module = create_image_matting_module
        namespace.create_image_management_module = create_image_management_module
        namespace.create_tag_management_module = create_tag_management_module
        namespace.create_announcement_module = create_announcement_module
        # namespace.create_latent_sync_ui = create_latent_sync_ui
        # namespace.create_index_tts_ui = create_index_tts_ui
        # namespace.INDEX_TTS_AVAILABLE = INDEX_TTS_AVAILABLE
        namespace.create_flux_kontext_ui = create_flux_kontext_ui
        namespace.FLUX_KONTEXT_AVAILABLE = FLUX_KONTEXT_AVAILABLE
        namespace.create_cleaner_module = create_cleaner_module
        namespace.CLEANER_AVAILABLE = CLEANER_AVAILABLE
        namespace.create_qwen_image_ui = create_qwen_image_ui
        namespace.QWEN_IMAGE_MODULE_AVAILABLE = QWEN_IMAGE_MODULE_AVAILABLE
        namespace.create_qwen_image_edit_ui = create_qwen_image_edit_ui
        namespace.QWEN_IMAGE_EDIT_MODULE_AVAILABLE = QWEN_IMAGE_EDIT_MODULE_AVAILABLE
        namespace.create_sam_ui = create_sam_ui
        namespace.SAM_AVAILABLE = SAM_AVAILABLE
        namespace.create_flux_krea_ui = create_flux_krea_ui
        namespace.FLUX_KREA_AVAILABLE = FLUX_KREA_AVAILABLE
        namespace.create_flux_klein_ui = create_flux_klein_ui
        namespace.FLUX_KLEIN_AVAILABLE = FLUX_KLEIN_AVAILABLE

        # 添加 Z-Image 模块到命名空间
        try:
            from z_image_ui import create_z_image_ui as z_image_create_func, Z_IMAGE_MODULE_AVAILABLE
        except ImportError: 
            z_image_create_func = None
            Z_IMAGE_MODULE_AVAILABLE = False

        namespace.create_z_image_ui = z_image_create_func
        namespace.Z_IMAGE_MODULE_AVAILABLE = Z_IMAGE_MODULE_AVAILABLE
        
        # 添加Qwen API模块到命名空间
        try: 
            from qwen_api_ui import create_qwen_api_ui, QWEN_API_AVAILABLE
        except ImportError: 
            create_qwen_api_ui = None
            QWEN_API_AVAILABLE = False

        namespace.create_qwen_api_ui = create_qwen_api_ui
        namespace.QWEN_API_AVAILABLE = QWEN_API_AVAILABLE
        
        # 注释掉已移动的功能模块到命名空间
        # try: 
        #     from qwen_video import create_qwen_video_gen_ui, QWEN_VIDEO_GEN_AVAILABLE
        # except ImportError: 
        #     create_qwen_video_gen_ui = None
        #     QWEN_VIDEO_GEN_AVAILABLE = False

        # namespace.create_qwen_video_gen_ui = create_qwen_video_gen_ui
        # namespace.QWEN_VIDEO_GEN_AVAILABLE = QWEN_VIDEO_GEN_AVAILABLE
        
        return namespace
        
    return _import_and_register_modules()

# 尝试导入所有模块
imported_modules = import_modules()

# 将导入的模块赋值给变量，方便在后续代码中使用
create_prompt_template_ui = imported_modules.create_prompt_template_ui
create_quick_description = imported_modules.create_quick_description
create_image_matting_module = imported_modules.create_image_matting_module
create_image_management_module = imported_modules.create_image_management_module
create_tag_management_module = imported_modules.create_tag_management_module
create_announcement_module = imported_modules.create_announcement_module

create_sam_segmentation = imported_modules.create_sam_ui
SAM_AVAILABLE = imported_modules.SAM_AVAILABLE

create_flux_kontext_ui = imported_modules.create_flux_kontext_ui
FLUX_KONTEXT_AVAILABLE = imported_modules.FLUX_KONTEXT_AVAILABLE

# 添加 cleaner 模块变量赋值
create_cleaner_module = imported_modules.create_cleaner_module
CLEANER_AVAILABLE = imported_modules.CLEANER_AVAILABLE


# 确保 SAM、Cleaner 和 Qwen Image 模块变量正确赋值
create_sam_ui = imported_modules.create_sam_ui
SAM_AVAILABLE = imported_modules.SAM_AVAILABLE
create_cleaner_module = imported_modules.create_cleaner_module
CLEANER_AVAILABLE = imported_modules.CLEANER_AVAILABLE
create_qwen_image_ui = imported_modules.create_qwen_image_ui
QWEN_IMAGE_MODULE_AVAILABLE = imported_modules.QWEN_IMAGE_MODULE_AVAILABLE
create_qwen_image_edit_ui = imported_modules.create_qwen_image_edit_ui
QWEN_IMAGE_EDIT_MODULE_AVAILABLE = imported_modules.QWEN_IMAGE_EDIT_MODULE_AVAILABLE


# 添加 Z-Image 模块变量赋值
create_z_image_ui = imported_modules.create_z_image_ui
Z_IMAGE_MODULE_AVAILABLE = imported_modules.Z_IMAGE_MODULE_AVAILABLE

# 添加FLUX KREA模块变量赋值
create_flux_krea_ui = imported_modules.create_flux_krea_ui
FLUX_KREA_AVAILABLE = imported_modules.FLUX_KREA_AVAILABLE

# 添加Qwen API模块变量赋值
create_qwen_api_ui = imported_modules.create_qwen_api_ui
QWEN_API_AVAILABLE = imported_modules.QWEN_API_AVAILABLE

# 注释掉已移动的模块变量赋值
# create_qwen_video_gen_ui = imported_modules.create_qwen_video_gen_ui
# QWEN_VIDEO_GEN_AVAILABLE = imported_modules.QWEN_VIDEO_GEN_AVAILABLE

# 添加FLUX_KLEIN模块变量赋值
create_flux_klein_ui = imported_modules.create_flux_klein_ui
FLUX_KLEIN_AVAILABLE = imported_modules.FLUX_KLEIN_AVAILABLE

current_dir = os.path.abspath(os.getcwd())
python_interpreter = sys.executable
ollama_api_script_path = os.path.join(scripts_dir.parent, "ollama", "ollama_api.py")

# 规范化路径
python_interpreter = os.path.normpath(python_interpreter)
ollama_api_script_path = os.path.abspath(ollama_api_script_path)

# 确保使用正确的Python解释器路径
if not os.path.exists(python_interpreter):
    # 如果当前python解释器路径不存在，则使用webui的python解释器
    python_interpreter = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "python.exe")
    if not os.path.exists(python_interpreter):
        # Fallback到系统Python
        python_interpreter = "python"

class ModelProcessor:
    """模型处理器类,封装模型相关操作"""
    @staticmethod
    def build_args(mode, model_name, user_input, file_path=None):
        """构建命令行参数"""
        args = [mode, model_name, user_input]
        if file_path:
            args.append(file_path)
        return args
        
    @staticmethod
    def run_model(args, script_path):
        """运行模型并获取输出"""
        full_cmd = [python_interpreter, script_path] + args
        result = subprocess.run(full_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            return f"模型执行失败: {error_msg}"
            
        return result.stdout

    @staticmethod
    def process_image(model_name, image_path, script_path, user_input, is_batch=False, save_dir=None):
        """处理单张或批量图片"""
        args = ModelProcessor.build_args("vision", model_name, user_input, image_path)
        result = ModelProcessor.run_model(args, script_path)
        
        if is_batch and save_dir:
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0]) + ".txt"
            FileHandler.save_text(result, save_path)
            return f"已保存: {save_path}"
            
        return result

    @staticmethod
    def process_text(model_name, script_path, user_input):
        """处理纯文本对话"""
        args = ModelProcessor.build_args("text", model_name, user_input)
        return ModelProcessor.run_model(args, script_path)

class FileHandler:
    """文件处理类"""
    @staticmethod
    def save_text(content, path):
        """保存文本内容到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    @staticmethod
    def save_chat_history(chat_history):
        """保存聊天记录"""
        save_dir = os.path.join(os.path.dirname(__file__), "chat_history")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"chat_history_{timestamp}.txt")
        
        with open(filename, "w", encoding='utf-8') as f:
            for human_message, ai_message in chat_history:
                f.write(f"用户: {human_message}\n")
                f.write(f"AI: {ai_message}\n\n")
        return f"聊天记录已保存到: {filename}"

class UIHelper:
    """UI辅助类"""
    @staticmethod
    def get_upload_visibility(is_single):
        """获取上传组件可见性"""
        return [is_single, not is_single]  # [image, multi_image]

    @staticmethod
    def switch_upload(upload_method):
        """切换上传方式"""
        is_single = upload_method == "single"
        visibilities = UIHelper.get_upload_visibility(is_single)
        save_path_info = "已锁定 无需填写" if is_single else "结果保存路径"
        
        return (*[gr.update(visible=v) for v in visibilities],
                gr.update(info=save_path_info, interactive=not is_single))

    @staticmethod
    def get_model_updates(model_type):
        is_vision = model_type == "vision"
        return [
            gr.update(interactive=not is_vision),  # language_model
            gr.update(interactive=is_vision),      # vision_model
            gr.update(visible=is_vision),          # image_components
            gr.update(label="AI 聊天")             # chat_history label（可选）
        ]

class ChatProcessor:
    """聊天处理类"""
    @staticmethod
    def process_model_task(model_name, message, upload_method, script_path, chat_history,
                          input_data, batch_save_path=None, model_type="vision"):
        """处理模型任务"""
        if not model_name:
            chat_history.append(("模型", "模型不能为空"))
            return chat_history

        if model_type == "vision":
            if not input_data:
                chat_history.append(("错误", "未选择图片文件"))
                return chat_history
                
            if upload_method == "single":
                input_path = input_data if isinstance(input_data, str) else input_data.name
                output = ModelProcessor.process_image(model_name, input_path, script_path, message)
                user_message = f"{model_name}:{message} ![]({input_path})"
                chat_history.append((user_message, output))
                
            elif upload_method == "batch" and os.path.isdir(batch_save_path):
                results = []
                for file_path in [f.name for f in input_data]:
                    result = ModelProcessor.process_image(model_name, file_path, script_path, message, 
                                                        True, batch_save_path)
                    results.append(result)
                chat_history.append((f"{model_name}:批量任务", "\n".join(results)))
        else:
            # 处理语言模型对话
            output = ModelProcessor.process_text(model_name, script_path, message)
            chat_history.append((message, output))
            
        return chat_history

    @staticmethod
    def extract_prompt(chat_history):
        """提取提示词"""
        if not chat_history:
            return ""
        
        for msg in reversed(chat_history):
            user_msg, ai_msg = msg
            if isinstance(user_msg, str) and not user_msg.startswith("![]"):
                return user_msg
            if isinstance(ai_msg, str):
                return ai_msg
        return ""

    @staticmethod
    def extract_image_and_prompt(chat_history):
        """提取图片路径和提示词"""
        for msg in reversed(chat_history):
            if isinstance(msg[0], str) and msg[0].startswith("![]"):
                start_idx = msg[0].find("(") + 1
                end_idx = msg[0].find(")")
                if start_idx > 0 and end_idx > start_idx:
                    return msg[0][start_idx:end_idx], msg[1] if isinstance(msg[1], str) else ""
        return None, ""

# 定义支持的视觉模型
vision_model_names = [
     "qwen3-vl:8b",
    "qwen3-vl:4b",
    "qwen2.5vl:3b",    
    "qwen3-vl:2b",
    "qwen3-vl-abliterated:8b",
    "qwen3-vl-abliterated:4b",
    "qwen3-vl-abliterated:b",
]

# 定义支持的语言模型
language_model_names = [
    "qwen3:latest",
    "qwen3:1.7b",
    "deepseek-r1:8b",
]

# 支持的图片格式
image_format = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

def chat(message, chat_history, vision_model, language_model, model_type, upload_method, batch_save_path,
         image_input, multi_images_input):
    # 添加处理状态反馈
    if "[处理中，请稍候" in message:
        # 这是一个快捷描述按钮的点击，不需要特殊处理
        pass
    elif "stable diffusion" in message.lower() or "sd prompt" in message.lower():
        # 添加处理提示到聊天历史
        chat_history.append(("用户", f"[处理中] 正在生成Stable Diffusion提示词，请稍候..."))
    
    script_path = ollama_api_script_path
    model_name = vision_model if model_type == "vision" else language_model
    input_data = image_input if upload_method == "single" else multi_images_input
    
    chat_history = ChatProcessor.process_model_task(
        model_name, message, upload_method, script_path, chat_history,
        input_data, batch_save_path, model_type
    )
    
    return "", chat_history, image_input

def MultiModal_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            # 重要公告标签页
            with gr.TabItem("1资源汇总"):
                # 使用延迟渲染避免出现空模块问题   
                announcement_ui = create_announcement_module()
                if "markdown_content" in announcement_ui:
                    announcement_ui["markdown_content"]
            
            
            # 图像识别与语言交互标签页
            with gr.TabItem("2图像识别与语言交互"):
                with gr.Row():
                    # 左侧区域：标签管理、图像识别与语言交互、模型选择作为一个整体
                    with gr.Column(scale=1):
                        # 标签管理模块
                        try:
                            if create_tag_management_module is not None:
                                tag_management_components = create_tag_management_module()
                                if tag_management_components:
                                    with gr.Box():
                                        if "refresh_button" in tag_management_components:
                                            tag_management_components["refresh_button"]
                                        if "folder_path" in tag_management_components:
                                            tag_management_components["folder_path"].elem_classes = ["xykc-accordion"]
                            else:
                                gr.Markdown("标签管理模块当前不可用。")
                        except Exception as e:
                            print(f"标签管理模块加载失败: {e}")
                        
                        # 图像管理模块
                        try:
                            if create_image_management_module is not None:
                                image_management_ui = create_image_management_module()
                                if image_management_ui:
                                    with gr.Box():
                                        if "dir_input" in image_management_ui:
                                            image_management_ui["dir_input"]
                                        if "load_dir_btn" in image_management_ui:
                                            image_management_ui["load_dir_btn"]
                                        if "gallery" in image_management_ui:
                                            image_management_ui["gallery"]
                            else:
                                gr.Markdown("图像管理模块当前不可用。")
                        except Exception as e:
                            print(f"图像管理模块加载失败: {e}")
                        
                        # 模型选择区域
                        with gr.Group():
                            model_type = gr.Radio(
                                [("图像识别", "vision"), ("语言交互", "text")],
                                value="vision",
                                label="模型类型",
                                interactive=True,
                                info="只有图像识别模型才可以与图片进行交互和批量操作"
                            )
                            
                            gr.Markdown("📌 **模型选择建议**：8GB显存选择1.7B或3B模型获得更快响应速度，16GB显存可选择latest或7B模型")
                            
                            vision_model = gr.Dropdown(
                                label="视觉模型",
                                choices=vision_model_names,
                                value=vision_model_names[0] if vision_model_names else None,
                                interactive=True,
                                info="选择视觉模型",
                                scale=2,
                                elem_classes="larger-text",
                                container=True
                            )
                            
                            language_model = gr.Dropdown(
                                label="语言模型",
                                choices=language_model_names,
                                value=language_model_names[0] if language_model_names else None,
                                interactive=False,
                                info="选择语言模型",
                                scale=2,
                                elem_classes="larger-text",
                                container=True
                            )
                        
                        # 图像识别与语言交互区域
                        with gr.Group():
                            gr.Markdown("### 图像识别与语言交互")
                            with gr.Row(visible=True) as image_components:
                                upload_method = gr.Radio(
                                    [("单张图片", "single"), ("批量图片", "batch")],
                                    value="single",
                                    label="上传方式",
                                    interactive=True,
                                    scale=2,
                                    elem_classes="larger-text",
                                    container=True
                                )
                                batch_save_path = gr.Textbox(
                                    label="结果保存路径",                       
                                    interactive=False,
                                    info="已锁定 无需填写",
                                    scale=2,
                                    elem_classes="larger-text",
                                    container=True
                                )
                            
                            with gr.Box(visible=True) as image_container:
                                image_input = gr.Image(
                                    type="filepath",
                                    label="单张图片输入",
                                    visible=True, 
                                    height=300,
                                    scale=1,
                                    min_width=300,
                                    show_label=True,
                                    container=True
                                )
                                multi_images_input = gr.Files(
                                    type="filepath",
                                    label="多张图片输入",
                                    visible=False,
                                    height=300,
                                    scale=1,
                                    min_width=300,
                                    file_count="multiple",
                                    file_types=image_format
                                )
                    
                    # 右侧区域：关键词辅助模板和聊天区域作为一个整体
                    with gr.Column(scale=1):
                        # 关键词辅助模板区域
                        with gr.Accordion("关键词辅助模板", open=False):
                            if create_prompt_template_ui is not None:
                                template_ui = create_prompt_template_ui()
                                with gr.Row():
                                    with gr.Column():
                                        template_ui["expression_template"]
                                    with gr.Column():
                                        template_ui["story_template"]
                                    with gr.Column():
                                        template_ui["shot_template"]
                            else:
                                gr.Markdown("关键词辅助模板模块当前不可用。")
                        
                        # 聊天区域
                        chat_history = gr.Chatbot(
                            elem_id="chatbot", 
                            label="聊天记录", 
                            height=300,
                            render=True
                        )
                        chat_message = gr.Textbox(
                            show_label=False,
                            placeholder="输入消息或上传图片",
                            container=True,
                            scale=1,
                            min_width=300,
                            lines=3
                        )
                        with gr.Row(equal_height=True):
                            submit_button = gr.Button(
                                "发送",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )
                            clear_button = gr.Button(
                                "清空聊天",
                                size="lg", 
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )
                            save_button = gr.Button(
                                "保存聊天记录",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )
                            copy_button = gr.Button(
                                "复制最新回复",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )

                        # 快捷描述区域
                        with gr.Group():
                            # 创建并添加快捷描述按钮
                            if create_quick_description is not None:
                                quick_description_buttons = create_quick_description(chat_message)
                            else:
                                quick_description_buttons = {}
                            
                            # 将快捷描述按钮点击事件绑定到聊天输入框

                chat_inputs = [
                    chat_message, chat_history, vision_model, language_model,
                    model_type, upload_method, batch_save_path,
                    image_input, multi_images_input
                ]
                chat_outputs = [chat_message, chat_history, image_input]

                chat_message.submit(chat, inputs=chat_inputs, outputs=chat_outputs)
                submit_button.click(chat, inputs=chat_inputs, outputs=chat_outputs)
                clear_button.click(lambda: [[], ""], outputs=[chat_history, chat_message])
                save_button.click(
                    FileHandler.save_chat_history,
                    inputs=[chat_history],
                    outputs=[gr.Textbox(visible=True, value="", label="保存状态")]
                )
                # 使用 JavaScript 实现复制功能，复制聊天历史中的最新AI回复
                copy_button.click(
                    None,
                    inputs=[chat_history],
                    outputs=[],
                    _js="""
                    (chat_history) => {
                        if (chat_history && chat_history.length > 0) {
                            // 获取最新的AI回复（聊天历史中的最后一个条目）
                            const lastMessage = chat_history[chat_history.length - 1];
                            if (lastMessage && lastMessage.length >= 2) {
                                // 第二个元素是AI的回复
                                const aiResponse = lastMessage[1];
                                if (aiResponse && aiResponse.length > 0) {
                                    navigator.clipboard.writeText(aiResponse).then(() => {
                                        alert("最新回复已复制到剪贴板！");
                                    }).catch(err => {
                                        console.error('复制失败: ', err);
                                        // 降级方案：创建临时textarea元素
                                        const textArea = document.createElement("textarea");
                                        textArea.value = aiResponse;
                                        document.body.appendChild(textArea);
                                        textArea.focus();
                                        textArea.select();
                                        try {
                                            document.execCommand('copy');
                                            alert("最新回复已复制到剪贴板！");
                                        } catch (err) {
                                            console.error('复制失败: ', err);
                                            alert("复制失败，请手动选择文本进行复制");
                                        }
                                        document.body.removeChild(textArea);
                                    });
                                    return;
                                }
                            }
                        }
                        alert("没有可复制的回复内容");
                    }
                    """
                )
                # 模型类型切换事件
                model_type.change(
                    fn=UIHelper.get_model_updates,
                    inputs=[model_type],
                    outputs=[
                        language_model,     # 输出1: 控制语言模型是否可交互
                        vision_model,       # 输出2: 控制视觉模型是否可交互
                        image_components,   # 输出3: 控制图片上传区域是否显示
                        chat_history        # 输出4: 更新聊天记录标签名（可选）
                    ]
                )
                upload_method.change(
                    UIHelper.switch_upload,
                    inputs=[upload_method],
                    outputs=[image_input, multi_images_input, batch_save_path]
                )

                ui.load(lambda: "single", outputs=[upload_method])
          
            
            # 图像分割/图像抠图/图像清理标签页
            with gr.TabItem("3.图像分割/图像抠图/图像清理"):
                with gr.Tabs():
                    with gr.TabItem("智能抠图"):
                        if create_image_matting_module is not None:
                            try:
                                create_image_matting_module()
                            except Exception as e:
                                gr.Markdown(f"智能抠图模块加载失败：{e}")
                        else:
                            gr.Markdown("智能抠图模块当前不可用。")
                    
                    with gr.TabItem("图像分割"):
                        if SAM_AVAILABLE and create_sam_ui is not None:
                            try:
                                sam_ui_components = create_sam_ui()
                            except Exception as e:
                                with gr.Group():
                                    gr.Markdown("## 图像分割")
                                    gr.Markdown(f"图像分割模块加载时出现错误：{str(e)}")
                                    gr.Markdown("请检查控制台输出以获取详细错误信息。")
                                import traceback
                                traceback.print_exc()
                        else:
                            gr.Markdown("图像分割模块不可用。请确保已安装segment-anything库。")
                    
                    with gr.TabItem("图像清理"):
                        if CLEANER_AVAILABLE and create_cleaner_module is not None:
                            try:
                                cleaner_ui_components = create_cleaner_module()
                            except Exception as e:
                                with gr.Group():
                                    gr.Markdown("## 图像清理")
                                    gr.Markdown(f"图像清理模块加载时出现错误：{str(e)}")
                                    gr.Markdown("请检查控制台输出以获取详细错误信息。")
                                import traceback
                                traceback.print_exc()
                        else:
                            gr.Markdown("图像清理模块不可用。请确保已安装litelama库。")
            
            # 添加 FLUX 系列标签页
            with gr.TabItem("4.FLUX加速系列图像生成与编辑"):
                with gr.Tabs():
                    with gr.TabItem("kontext图像编辑"):
                        try:
                            if 'FLUX_KONTEXT_AVAILABLE' in globals() and FLUX_KONTEXT_AVAILABLE:
                                flux_kontext_components = create_flux_kontext_ui()
                                
                                if not flux_kontext_components:
                                    gr.Markdown("FLUX.1-Kontext模块加载失败")
                            else:
                                gr.Markdown("FLUX.1-Kontext模块当前不可用，可能是因为缺少模型文件或依赖项。")
                        except Exception as e:
                            gr.Markdown(f"FLUX.1-Kontext模块初始化错误: {e}")
                            import traceback
                            traceback.print_exc()
                
                    with gr.TabItem("FLUX.1-图像生成"):
                        try:
                            # 检查FLUX.1-krea模块是否可用
                            flux_krea_available = globals().get('FLUX_KREA_AVAILABLE', False)
                            if flux_krea_available:
                                # 创建FLUX.1-krea UI组件
                                flux_krea_components = create_flux_krea_ui()
                                
                                if not flux_krea_components:
                                    gr.Markdown("FLUX.1-krea模块加载失败")
                            else:
                                gr.Markdown("FLUX.1-krea模块当前不可用，可能是因为缺少模型文件或依赖项。")
                        except Exception as e:
                            gr.Markdown(f"FLUX.1-krea模块初始化错误: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    with gr.TabItem("FLUX.2-klein-图像生成"):
                        try:
                            # 检查FLUX.2-klein模块是否可用
                            flux_klein_available = globals().get('FLUX_KLEIN_AVAILABLE', False)
                            if flux_klein_available:
                                # 创建FLUX.2-klein UI组件
                                flux_klein_components = create_flux_klein_ui()
                                
                                if not flux_klein_components:
                                    gr.Markdown("FLUX.2-klein模块加载失败")
                            else:
                                gr.Markdown("FLUX.2-klein模块当前不可用，可能是因为缺少模型文件或依赖项。")
                        except Exception as e:
                            gr.Markdown(f"FLUX.2-klein模块初始化错误: {e}")
                            import traceback
                            traceback.print_exc()
            # 添加 Qwen Image 标签页（如果可用）
            if 'QWEN_IMAGE_MODULE_AVAILABLE' in globals() and QWEN_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("5.Qwen Image图像生成与编辑"):
                    try:
                        with gr.Tabs():
                            with gr.TabItem("文生图"):
                                # 创建 Qwen Image 文生图 UI 组件
                                qwen_image_components = create_qwen_image_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_image_components:
                                    gr.Markdown("Qwen Image文生图模块加载失败")
                            
                            with gr.TabItem("图像编辑"):
                                # 创建 Qwen Image 图像编辑 UI 组件
                                qwen_image_edit_components = create_qwen_image_edit_ui() if 'create_qwen_image_edit_ui' in globals() and QWEN_IMAGE_EDIT_MODULE_AVAILABLE else None
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_image_edit_components:
                                    gr.Markdown("Qwen Image图像编辑模块加载失败")
                            
                            with gr.TabItem("Qwen图像生成与编辑API调用"):
                                # 创建 Qwen API UI 组件
                                qwen_api_components = create_qwen_api_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_api_components:
                                    gr.Markdown("Qwen API模块加载失败")
                    except Exception as e:
                        gr.Markdown(f"Qwen Image模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'QWEN_IMAGE_MODULE_AVAILABLE' in globals() and not QWEN_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("5.nunchaku加速-Qwen Image图像生成与编辑"):
                    gr.Markdown("Qwen Image模块当前不可用，可能是因为缺少模型文件或依赖项。")
            
            # 添加 Z-Image-Turbo 标签页（如果可用）
            if 'Z_IMAGE_MODULE_AVAILABLE' in globals() and Z_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("6.Z-Image-Turbo图像生成"):
                    try:
                        with gr.Tabs():
                            with gr.TabItem("文生图"):
                                # 创建 Z-Image-Turbo 文生图 UI 组件
                                z_image_components = create_z_image_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not z_image_components:
                                    gr.Markdown("Z-Image-Turbo文生图模块加载失败")
                    except Exception as e:
                        gr.Markdown(f"Z-Image-Turbo模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'Z_IMAGE_MODULE_AVAILABLE' in globals() and not Z_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("6.Z-Image-Turbo图像生成"):
                    gr.Markdown("Z-Image-Turbo模块当前不可用，可能是因为缺少模型文件或依赖项。")
            
            # 注释掉已移动的标签页：wan系列视频生成API调用 (10 -> 7)
    return [(ui, "多模态图像处理15", "MultiModal_vision_tab")]
                  
script_callbacks.on_ui_tabs(MultiModal_tab)

# 在WebUI启动时在后台日志中显示插件信息和使用声明
def on_app_started(*args, **kwargs):
    print("=" * 60)
    print("多模态图像处理插件 - forge版本专用")
    print("开发者：鸡肉爱土豆")
    print("网址：https://space.bilibili.com/403361177")
    print("声明：为创作者提供更便捷更强大无复杂工作流的插件")
    print()
    print("集成功能：")
    print("- 智能抠图")
    print("- FLUX加速系列图像生成与编辑")
    print("- Z-Image-Turbo图像生成")
    print("- Qwen图像生成与编辑")
    print("- Qwen与wan系列api调用")
    print()
    print("使用须知：使用此插件者请合法使用AI，不得发表不正当言论，作假新闻，二次销售，二次改装等违法行为，之后的一切行为与插件开发者无关。")
    print("=" * 60)

script_callbacks.on_app_started(on_app_started)

# 检查模块状态
modules_status = {
    'flux_kontext': FLUX_KONTEXT_AVAILABLE,
    'cleaner': CLEANER_AVAILABLE,
    'sam': SAM_AVAILABLE,
    'qwen_image': QWEN_IMAGE_MODULE_AVAILABLE,
    'qwen_image_edit': QWEN_IMAGE_EDIT_MODULE_AVAILABLE,
    'qwen_api': QWEN_API_AVAILABLE,
    'flux_krea': FLUX_KREA_AVAILABLE,
    'z_image': Z_IMAGE_MODULE_AVAILABLE,
}