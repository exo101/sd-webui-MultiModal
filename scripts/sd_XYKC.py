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
        
        try: from video_frame_extractor import create_video_frame_extractor
        except ImportError: create_video_frame_extractor = None
        
        try: from image_matting import create_image_matting_module
        except ImportError: create_image_matting_module = None
        
        try: from image_management import create_image_management_module
        except ImportError: create_image_management_module = None
        
        try: from tag_management import create_tag_management_module
        except ImportError: create_tag_management_module = None
        
        try: from announcement import create_announcement_module
        except ImportError: create_announcement_module = None
        
        try: from latent_sync_ui import create_latent_sync_ui
        except ImportError: create_latent_sync_ui = None
        
        try: from index_tts_ui import create_index_tts_ui, INDEX_TTS_AVAILABLE
        except ImportError: 
            create_index_tts_ui = None
            INDEX_TTS_AVAILABLE = False
        
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
        
        # 添加对qwen3_vl_4b_instruct的导入
        try:
            # 添加XYKC_AI_PyScripts目录到系统路径
            xykc_ai_pyscripts_dir = os.path.join(scripts_dir.parent, "XYKC_AI", "XYKC_AI_PyScripts")
            if xykc_ai_pyscripts_dir not in sys.path:
                sys.path.append(xykc_ai_pyscripts_dir)
            
            # 导入qwen3_vl_4b_instruct模块
            import qwen3_vl_4b_instruct
            QWEN3_VL_4B_AVAILABLE = True
        except ImportError:
            qwen3_vl_4b_instruct = None
            QWEN3_VL_4B_AVAILABLE = False
        
        # 返回命名空间对象
        import types
        namespace = types.SimpleNamespace()
        namespace.create_prompt_template_ui = create_prompt_template_ui
        namespace.create_quick_description = create_quick_description
        namespace.create_video_frame_extractor = create_video_frame_extractor
        namespace.create_image_matting_module = create_image_matting_module
        namespace.create_image_management_module = create_image_management_module
        namespace.create_tag_management_module = create_tag_management_module
        namespace.create_announcement_module = create_announcement_module
        namespace.create_latent_sync_ui = create_latent_sync_ui
        namespace.create_index_tts_ui = create_index_tts_ui
        namespace.INDEX_TTS_AVAILABLE = INDEX_TTS_AVAILABLE
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
        # 添加qwen3_vl_4b_instruct相关变量
        namespace.qwen3_vl_4b_instruct = qwen3_vl_4b_instruct
        namespace.QWEN3_VL_4B_AVAILABLE = QWEN3_VL_4B_AVAILABLE
        
        return namespace
    
    return _import_and_register_modules()

# 尝试导入所有模块
imported_modules = import_modules()

# 将导入的模块赋值给变量，方便在后续代码中使用
create_prompt_template_ui = imported_modules.create_prompt_template_ui
create_quick_description = imported_modules.create_quick_description
create_video_frame_extractor = imported_modules.create_video_frame_extractor
create_image_matting_module = imported_modules.create_image_matting_module
create_image_management_module = imported_modules.create_image_management_module
create_tag_management_module = imported_modules.create_tag_management_module
create_announcement_module = imported_modules.create_announcement_module
create_latent_sync_ui = imported_modules.create_latent_sync_ui
create_index_tts_ui = imported_modules.create_index_tts_ui
INDEX_TTS_AVAILABLE = imported_modules.INDEX_TTS_AVAILABLE

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

# 添加qwen3_vl_4b_instruct模块变量赋值
qwen3_vl_4b_instruct = imported_modules.qwen3_vl_4b_instruct
QWEN3_VL_4B_AVAILABLE = imported_modules.QWEN3_VL_4B_AVAILABLE

current_dir = os.path.abspath(os.getcwd())
# 修改：不再使用插件自带的Python解释器，而是使用系统Python
# python_interpreter = os.path.join(current_dir, "extensions\\sd-webui-XYKC\\XYKC_AI\\python.exe")
python_interpreter = sys.executable
# 修改：使用正确的MultiModal扩展路径
ollama_api_script_path = os.path.join(current_dir, "extensions\\sd-webui-MultiModal\\XYKC_AI\\XYKC_AI_PyScripts\\ollama_api.py")

# 规范化路径
python_interpreter = os.path.normpath(python_interpreter)
ollama_api_script_path = os.path.abspath(ollama_api_script_path)

class ModelProcessor:
    """模型处理器类,封装模型相关操作"""
    @staticmethod
    def build_args(mode, model_name, user_input, file_path=None):
        """构建命令行参数"""
        # 对于Qwen3-VL-4B-Instruct模型，使用不同的参数格式
        if model_name == "Qwen3-VL-4B-Instruct":
            args = [model_name, mode, user_input]
            if file_path:
                args.append(file_path)
            return args
        
        args = [mode, model_name, user_input]
        if file_path:
            args.append(file_path)
        return args
        
    @staticmethod
    def run_model(args, script_path):
        """运行模型并获取输出"""
        # 检查是否是Qwen3-VL-4B-Instruct模型
        if "qwen3_vl_4b_instruct.py" in script_path:
            # 对于Qwen3-VL-4B-Instruct模型，直接调用模块函数而不是通过子进程
            try:
                # 导入模块并调用处理函数
                if QWEN3_VL_4B_AVAILABLE and qwen3_vl_4b_instruct:
                    if args[1] == "vision" and len(args) >= 4:  # vision模式且有图片路径
                        result = qwen3_vl_4b_instruct.process_with_model(
                            image_path=args[3], 
                            prompt=args[2]
                        )
                    else:  # text模式
                        result = qwen3_vl_4b_instruct.process_with_model(
                            prompt=args[2]
                        )
                    return result
                else:
                    return "错误: Qwen3-VL-4B-Instruct模型不可用"
            except Exception as e:
                return f"模型执行失败: {str(e)}"
        else:
            # 原有的处理逻辑
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
]

# 如果Qwen3-VL-4B-Instruct模型可用，则添加到视觉模型列表中
if QWEN3_VL_4B_AVAILABLE:
    vision_model_names.append("Qwen3-VL-4B-Instruct")

# 定义支持的语言模型
language_model_names = [
    "qwen3:latest",
    "qwen3:1.7b",
    "deepseek-r1:8b",
]

# 支持的图片格式
image_format = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

def open_url():
    webbrowser.open("https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key/")

def show_qwen_api_box(model):
    return (gr.update(visible=False, open=False),
            gr.update(label=f'和 {model} 聊天'))

def get_script_path(model_name):
    # 检查是否是Qwen3-VL-4B-Instruct模型
    if model_name == "Qwen3-VL-4B-Instruct" and QWEN3_VL_4B_AVAILABLE:
        # 返回qwen3_vl_4b_instruct.py的路径
        return os.path.join(scripts_dir.parent, "XYKC_AI", "XYKC_AI_PyScripts", "qwen3_vl_4b_instruct.py")
    return ollama_api_script_path

def chat(message, chat_history, vision_model, language_model, model_type, upload_method, batch_save_path,
         image_input, multi_images_input):
    # 添加处理状态反馈
    if "[处理中，请稍候" in message:
        # 这是一个快捷描述按钮的点击，不需要特殊处理
        pass
    elif "stable diffusion" in message.lower() or "sd prompt" in message.lower():
        # 添加处理提示到聊天历史
        chat_history.append(("用户", f"[处理中] 正在生成Stable Diffusion提示词，请稍候..."))
    
    script_path = get_script_path(vision_model if model_type == "vision" else language_model)
    model_name = vision_model if model_type == "vision" else language_model
    input_data = image_input if upload_method == "single" else multi_images_input
    
    chat_history = ChatProcessor.process_model_task(
        model_name, message, upload_method, script_path, chat_history,
        input_data, batch_save_path, model_type
    )
    
    return "", chat_history, image_input

def XYKC_tab():
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
                        tag_management_components = create_tag_management_module()
                        tag_management_components["folder_path"].elem_classes = ["xykc-accordion"]
                        
                        # 图像管理模块
                        try:
                            image_management_ui = create_image_management_module()
                            if image_management_ui:
                                with gr.Box():
                                    if "dir_input" in image_management_ui:
                                        image_management_ui["dir_input"]
                                    if "load_dir_btn" in image_management_ui:
                                        image_management_ui["load_dir_btn"]
                                    if "gallery" in image_management_ui:
                                        image_management_ui["gallery"]
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
                            template_ui = create_prompt_template_ui()
                            with gr.Row():
                                with gr.Column():
                                    template_ui["expression_template"]
                                with gr.Column():
                                    template_ui["story_template"]
                                with gr.Column():
                                    template_ui["shot_template"]
                        
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

                        # 快捷描述区域
                        with gr.Group():
                            # 创建并添加快捷描述按钮
                            quick_description_buttons = create_quick_description(chat_message)
                            
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
                        # 简化调用方式并统一结构
                        try:
                            create_image_matting_module()
                        except Exception as e:
                            gr.Markdown(f"智能抠图模块加载失败：{e}")
                    
                    with gr.TabItem("图像分割"):
                        # 检查并显示图像分割模块
                        if SAM_AVAILABLE and create_sam_segmentation is not None:
                            try:
                                sam_ui_components = create_sam_segmentation()
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
                        # 检查并显示图像清理模块
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
            
            # 视频关键帧提取标签页
            with gr.TabItem("4.视频关键帧提取"):
                # 创建并添加视频分帧组件
                video_frame_components = create_video_frame_extractor()                   

                # 将视频分帧组件解包
                video_input = video_frame_components["video_input"]
                frame_output = video_frame_components["frame_output"]
                frame_quality = video_frame_components["frame_quality"]
                frame_mode = video_frame_components["frame_mode"]
                frame_preview = video_frame_components["frame_preview"]
                extract_video_frames = video_frame_components["extract_video_frames"]
                
                # 绑定按钮点击事件
                extract_button = gr.Button("提取关键帧")
                extract_button.click(
                    fn=extract_video_frames,
                    inputs=[video_input, frame_output, frame_quality, frame_mode],
                    outputs=[gr.File(label="提取的帧文件"), frame_preview]
                )
            
            # 数字人视频生成标签页
            with gr.TabItem("5.数字人对口型生成"):
                # 创建并添加数字人视频生成功能
                latent_sync_components = create_latent_sync_ui()
                
                # 将组件解包以供引用（如果需要）
                latent_video_input = latent_sync_components["video_input"]
                latent_audio_input = latent_sync_components["audio_input"]
                latent_guidance_scale = latent_sync_components["guidance_scale"]
                latent_inference_steps = latent_sync_components["inference_steps"]
                latent_seed = latent_sync_components["seed"]
                latent_process_btn = latent_sync_components["process_btn"]
                latent_video_output = latent_sync_components["video_output"]

            # Index-TTS语音合成标签页（如果可用）
            if 'INDEX_TTS_AVAILABLE' in globals() and INDEX_TTS_AVAILABLE:
                with gr.TabItem("6.Index-TTS语音合成"):
                    try:
                        # 创建并添加Index-TTS功能
                        index_tts_components = create_index_tts_ui()
                    except Exception as e:
                        gr.Markdown(f"Index-TTS模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'INDEX_TTS_AVAILABLE' in globals() and not INDEX_TTS_AVAILABLE:
                with gr.TabItem("6.Index-TTS语音合成"):
                    gr.Markdown("Index-TTS模块当前不可用，可能是因为缺少模型文件或依赖项。")

            # FLUX.1-Kontext标签页（如果可用）
            if 'FLUX_KONTEXT_AVAILABLE' in globals() and FLUX_KONTEXT_AVAILABLE:
                with gr.TabItem("7.FLUX.1-Kontext图像编辑"):
                    try:
                        # 直接创建FLUX.1-Kontext UI组件
                        flux_kontext_components = create_flux_kontext_ui()
                        
                        # 如果组件创建成功，它们已经在create_flux_kontext_ui函数中被正确创建和显示
                        # 不需要额外的处理
                        if not flux_kontext_components:
                            gr.Markdown("FLUX.1-Kontext模块加载失败")
                    except Exception as e:
                        gr.Markdown(f"FLUX.1-Kontext模块初始化错误: {e}")
 
            # 添加 Qwen Image 标签页（如果可用）
            if 'QWEN_IMAGE_MODULE_AVAILABLE' in globals() and QWEN_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("8.Qwen Image图像生成"):
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
                                qwen_image_edit_components = create_qwen_image_edit_ui() if 'create_qwen_image_edit_ui' in globals() else None
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_image_edit_components:
                                    gr.Markdown("Qwen Image图像编辑模块加载失败")
                    except Exception as e:
                        gr.Markdown(f"Qwen Image模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'QWEN_IMAGE_MODULE_AVAILABLE' in globals() and not QWEN_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("8.Qwen Image图像生成"):
                    gr.Markdown("Qwen Image模块当前不可用，可能是因为缺少模型文件或依赖项。")
               
        
    return [(ui, "多模态插件12", "XYKC_vision_tab")]

script_callbacks.on_ui_tabs(XYKC_tab)

import modules.scripts as scripts
import gradio as gr
from modules import script_callbacks

# 在WebUI启动时在后台日志中显示插件信息和使用声明
def on_app_started(*args, **kwargs):
    print("=" * 60)
    print("多模态webui插件12 - forge版本专用")
    print("开发者：鸡肉爱土豆")
    print("网址：https://space.bilibili.com/403361177")
    print("声明：为创作者提供更便捷更强大无复杂工作流的插件")
    print()
    print("集成功能：")
    print("- 图像分割")
    print("- 图像编辑")
    print("- 图像清理")
    print("- 批量标注")
    print("- 大语言模型交互")
    print("- 智能抠图")
    print("- 视频提取关键帧")
    print("- 关键词辅助模板")
    print("- 数字人视频生成")
    print("- Qwen图像生成")
    print()
    print("使用须知：使用此插件者请合法使用AI，不得发表不正当言论，作假新闻，二次销售，二次改装等违法行为，之后的一切行为与插件开发者无关。")
    print("=" * 60)

script_callbacks.on_app_started(on_app_started)

# 检查模块状态
modules_status = {
    'index_tts': INDEX_TTS_AVAILABLE,
    'flux_kontext': FLUX_KONTEXT_AVAILABLE,
    'cleaner': CLEANER_AVAILABLE,
    'sam': SAM_AVAILABLE,
}