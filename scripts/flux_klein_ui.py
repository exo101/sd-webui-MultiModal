import gradio as gr
import torch
import os
import subprocess
import platform
import gc
import logging
import json
import datetime
import time  # 添加缺失的时间模块导入
import random  # 添加缺失的随机模块导入
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import gradio as gr  # 添加gradio导入
from modules import shared
from modules import sd_samplers
from modules.ui_components import ToolButton
from modules import ui_common  # 导入ui_common模块
from modules import util  # 导入util模块
import requests
from PIL import Image
from modules import sd_models, processing, shared, sd_samplers, images, devices
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui_components import InputAccordion
from modules.ui_gradio_extensions import reload_javascript
import scripts.flux_klein_model_loader as model_loader
import scripts.flux_klein_generators as generators
import scripts.flux_klein_queue_manager as queue_manager
from scripts.flux_klein_angle_selector import create_flux_klein_angle_visualization_component
from scripts.flux_klein_generators import (
    generate_flux_klein_image,
    multi_img_flux_klein,
    inpaint_flux_klein,
    extend_flux_klein
)
from scripts.flux_klein_queue_manager import (
    add_to_queue,
    process_queue,
    get_queue_status,
    clear_queue,
    get_detailed_queue_status
)
import scripts.flux_klein_model_loader as model_loader
from scripts.flux_klein_model_loader import get_bf16_models, get_fp8_models, list_lora_models, _is_fp8_model, _is_sdnq_model, _identify_model_type, _scan_model_directory

# 创建logger实例
logger = logging.getLogger(__name__)

# 自定义关键词缓存
CUSTOM_KEYWORDS_CACHE = []

def load_custom_keywords():
    """加载自定义关键词"""
    global CUSTOM_KEYWORDS_CACHE
    # 修正路径：使用插件目录下的config子目录
    plugin_dir = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(plugin_dir, "config", "custom_keywords.json")
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                CUSTOM_KEYWORDS_CACHE = config_data.get('keywords', [])
    except Exception as e:
        print(f"加载自定义关键词失败: {e}")
        CUSTOM_KEYWORDS_CACHE = []

def save_custom_keywords(keywords_list):
    """保存自定义关键词到标准配置文件格式"""
    # 修正路径：使用插件目录下的config子目录
    plugin_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(plugin_dir, "config")
    config_file = os.path.join(config_dir, "custom_keywords.json")
    try:
        # 确保配置目录存在
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # 创建标准配置数据结构
        config_data = {
            'keywords': keywords_list,
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # 保存到配置文件，使用UTF-8编码和美化格式
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 已保存 {len(keywords_list)} 个关键词到配置文件")
        print(f"📁 配置文件位置: {config_file}")
        
    except Exception as e:
        print(f"❌ 保存自定义关键词失败: {e}")
        raise  # 重新抛出异常以便上层处理

# 加载关键词数据
load_custom_keywords()

# 检查模块是否可用
FLUX_KLEIN_AVAILABLE = True
MODULE_IMPORT_ERROR = ""

try:
    # 检查是否可以导入必要的模块和函数
    from scripts.flux_klein_generators import generate_flux_klein_image
    from scripts.flux_klein_model_loader import load_flux_klein_pipeline
except ImportError as e:
    FLUX_KLEIN_AVAILABLE = False
    MODULE_IMPORT_ERROR = str(e)
except Exception as e:
    FLUX_KLEIN_AVAILABLE = False
    MODULE_IMPORT_ERROR = str(e)

def create_flux_klein_ui():
    """创建FLUX.2-klein的UI界面"""
    with gr.Tabs():
        with gr.TabItem("文生图"):
            # 文生图界面组件
            with gr.Row():
                with gr.Column():
                    # 提示词输入区域
                    prompt = gr.Textbox(label="正面提示词 (Prompt)", lines=3, value="一只可爱的小猫")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 生成参数 - 确保这些组件是可交互的
                        steps = gr.Slider(
                            label="步数", 
                            minimum=1, 
                            maximum=50, 
                            value=4, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                        guidance_scale = gr.Slider(
                            label="CFG Scale", 
                            minimum=1.0, 
                            maximum=10.0, 
                            value=1.0, 
                            step=0.1,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 尺寸参数 - 确保这些组件是可交互的
                        height = gr.Slider(
                            label="高度", 
                            minimum=256, 
                            maximum=1536, 
                            value=1024, 
                            step=64,
                            interactive=True  # 明确设置为可交互
                        )
                        width = gr.Slider(
                            label="宽度", 
                            minimum=256, 
                            maximum=1536, 
                            value=768, 
                            step=64,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 随机种子 - 确保这个组件是可交互的
                        seed = gr.Number(
                            label="种子 (Seed)", 
                            value=-1, 
                            precision=0,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 生成批次 - 确保这些组件是可交互的
                        batch_count = gr.Slider(
                            label="批次数量", 
                            minimum=1, 
                            maximum=8, 
                            value=1, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                        batch_size = gr.Slider(
                            label="每批数量", 
                            minimum=1, 
                            maximum=8, 
                            value=1, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    # 添加推荐参数提示
                    gr.Markdown("""
                    **参数推荐：推理步数15以上，引导数4**
                    """)
                    
                    # 指令收藏 - 极简版本
                    with gr.Accordion("📝 指令收藏", open=False):
                        # 简化的收藏功能
                        with gr.Row():
                            keyword_input = gr.Textbox(
                                label="关键词",
                                placeholder="输入要保存的关键词...",
                                interactive=True
                            )
                            save_btn = gr.Button("收藏", variant="primary")
                            delete_btn = gr.Button("删除", variant="secondary")
                        
                        # 显示已保存的关键词
                        saved_keywords = gr.HTML()
                        keywords_state = gr.State(CUSTOM_KEYWORDS_CACHE)
                        
                        # 简化的操作函数
                        def simple_save_keyword(keyword_text, current_list):
                            if not keyword_text.strip():
                                return current_list, refresh_minimal_display(current_list)
                            
                            # 解析格式：显示名称|实际提示词
                            if '|' in keyword_text:
                                parts = keyword_text.split('|', 1)
                                display_name = parts[0].strip()
                                prompt_value = parts[1].strip()
                            else:
                                display_name = keyword_text.strip()
                                prompt_value = keyword_text.strip()
                            
                            # 检查重复
                            for item in current_list:
                                if item['display'] == display_name:
                                    return current_list, refresh_minimal_display(current_list)
                            
                            # 添加新关键词
                            new_item = {
                                'display': display_name,
                                'prompt': prompt_value,
                                'created_at': datetime.now().isoformat()
                            }
                            current_list.append(new_item)
                            
                            # 保存到配置文件
                            global CUSTOM_KEYWORDS_CACHE
                            CUSTOM_KEYWORDS_CACHE = current_list.copy()
                            save_custom_keywords(current_list)
                            
                            return current_list, refresh_minimal_display(current_list)
                        
                        def simple_delete_keyword(keyword_text, current_list):
                            if not keyword_text.strip():
                                return current_list, refresh_minimal_display(current_list)
                            
                            display_name = keyword_text.strip()
                            
                            # 查找并删除
                            for i, item in enumerate(current_list):
                                if item['display'] == display_name:
                                    current_list.pop(i)
                                    # 保存到配置文件
                                    global CUSTOM_KEYWORDS_CACHE
                                    CUSTOM_KEYWORDS_CACHE = current_list.copy()
                                    save_custom_keywords(current_list)
                                    break
                            
                            return current_list, refresh_minimal_display(current_list)
                        
                        def refresh_minimal_display(keywords_list):
                            if not keywords_list:
                                return "<small>暂无收藏关键词</small>"
                            
                            html = ""
                            for item in keywords_list:
                                display_text = item.get('display', '')
                                prompt_text = item.get('prompt', '')
                                # 使用更可靠的元素查找方式
                                escaped_prompt = prompt_text.replace("'", "\\'").replace('"', '\\"')
                                # 使用querySelector按label查找元素，更稳定
                                html += f'<button onclick="var elem=document.querySelector(\'[data-testid=\\\'textbox\\\'\'][aria-label=\\\'正面提示词 (Prompt)\\\']\')||document.querySelector(\'input[type=\\\'text\\\'],textarea\').closest(\'div\').querySelector(\'textarea,input\');if(elem){{elem.value+=(elem.value?\' \':\'\')+\'{escaped_prompt}\';}}" style="margin: 2px; padding: 4px 8px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">{display_text}</button>'
                            return html
                        
                        # 绑定事件
                        save_btn.click(
                            fn=simple_save_keyword,
                            inputs=[keyword_input, keywords_state],
                            outputs=[keywords_state, saved_keywords]
                        )
                        
                        delete_btn.click(
                            fn=simple_delete_keyword,
                            inputs=[keyword_input, keywords_state],
                            outputs=[keywords_state, saved_keywords]
                        )
                        
                        # 初始化显示
                        saved_keywords.value = refresh_minimal_display(CUSTOM_KEYWORDS_CACHE)
                    
                    # 模型选择下拉列表 - 分别显示BF16、FP8和SDNQ模型
                    with gr.Row():
                        with gr.Column(scale=1):
                            bf16_model_choice = gr.Dropdown(
                                choices=["无"] + get_bf16_models(),  # 添加"无"选项以避免冲突
                                value=get_bf16_models()[0] if get_bf16_models() else "FLUX_2-klein-base-4B",
                                label="BF16模型选择 (基础模型)"
                            )
                        with gr.Column(scale=1):
                            fp8_model_choice = gr.Dropdown(
                                choices=["无"] + get_fp8_models(),  # 添加"无"选项
                                value=None,
                                label="FP8模型选择 (量化模型)",
                                info="选择FP8量化模型进行加速"
                            )
                    
                    # 添加SDNQ模型选择
                    with gr.Row():
                        sdnq_model_choice = gr.Dropdown(
                            choices=["无"] + get_sdnq_models(),  # 添加SDNQ模型选项
                            value=None,
                            label="SDNQ 4bit模型选择 (超量化模型)",
                            info="选择SDNQ 4bit动态SVD量化模型进行极致加速"
                        )
                    
                    with gr.Row():
                        # 添加刷新模型列表按钮
                        refresh_model_btn = gr.Button("刷新模型列表")
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                refresh_lora_button = gr.Button("刷新LoRA模型列表")
                
                with gr.Column():
                    # 结果展示画廊
                    result_gallery = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="flux_klein_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 确保prompt变量可用于JavaScript访问
                    prompt.elem_id = "txt2img_prompt"
                    
                    # 任务队列
                    with gr.Accordion("任务队列", open=False):
                        queue_status = gr.Textbox(label="队列状态", interactive=False)
                        with gr.Row():
                            add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                            process_queue_btn = gr.Button("处理队列任务", variant="primary")
                            clear_queue_btn = gr.Button("清空队列", variant="stop")
                        queue_result = gr.Textbox(label="队列操作结果", interactive=False)
                        detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, max_lines=10)
                    
                    # 生成按钮和打开输出目录按钮
                    with gr.Row():
                        gen_btn = gr.Button("生成结果", variant="primary")
                        open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
            
            # 模块可用性检查和事件绑定
            if not FLUX_KLEIN_AVAILABLE:
                # 如果模块不可用，显示错误信息并禁用相关功能
                error_message = f"⚠️ FLUX.2-klein模块当前不可用\n错误详情: {MODULE_IMPORT_ERROR}\n\n请确保已安装以下依赖项：\n- diffusers 库\n- modelscope 库\n- transformers 库"
                result_status.value = error_message
                
                # 禁用生成按钮
                gen_btn.variant = "secondary"
                gen_btn.interactive = False
                
                # 显示警告信息
                with gr.Row():
                    gr.Markdown(f"**⚠️ 警告**: {error_message}")
            else:
                # 设置prompt元素ID以便JavaScript访问
                prompt.elem_id = "txt2img_prompt"
                
                # 正常的功能绑定
                # 事件绑定 - 文生图部分
                def update_lora_interactive_fn(enable_lora):
                    return update_lora_interactive(enable_lora)
                
                lora_enable.change(
                    fn=update_lora_interactive_fn,
                    inputs=[lora_enable],
                    outputs=[lora_model]
                )
                
                # 绑定刷新按钮事件
                refresh_lora_button.click(
                    fn=refresh_lora_models,
                    inputs=[],
                    outputs=[lora_model]
                )
                
                gen_btn.click(
                    fn=lambda prompt, steps, guidance_scale, height, width, seed, bf16_choice, fp8_choice, sdnq_choice, batch_size, lora_enable, lora_model, lora_weight: 
                        generate_flux_klein_image(
                            prompt, steps, guidance_scale, height, width, seed, 
                            get_selected_model(bf16_choice, fp8_choice, sdnq_choice),  # 使用合并的模型选择
                            batch_size, lora_enable, lora_model, lora_weight
                        ),
                    inputs=[
                        prompt, steps, guidance_scale, 
                        height, width, seed, bf16_model_choice, fp8_model_choice, sdnq_model_choice,  # 传入三个模型选择
                        batch_size, lora_enable, lora_model, lora_weight
                    ],
                    outputs=[result_gallery, result_status]
                )
                
                # 刷新模型列表按钮事件
                refresh_model_btn.click(
                    fn=lambda: (
                        ["无"] + get_bf16_models(), 
                        ["无"] + get_fp8_models(),
                        ["无"] + get_sdnq_models()  # 返回元组而不是列表，修复Content-Length错误
                    ),
                    inputs=[],
                    outputs=[bf16_model_choice, fp8_model_choice, sdnq_model_choice]  # 更新三个下拉框
                )
                
                # 打开输出目录事件
                open_outputs_btn.click(
                    fn=lambda: open_folder("outputs"),
                    inputs=[],
                    outputs=[]
                )
                
                # 任务队列相关事件
                add_to_queue_btn.click(
                    fn=lambda prompt, width, height, steps, guidance_scale, seed, bf16_choice, fp8_choice, sdnq_choice, batch_size, lora_enable, lora_model, lora_weight: (
                        add_to_queue('txt2img', prompt, width, height, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice, sdnq_choice), batch_size, lora_enable, lora_model, lora_weight)
                    ),
                    inputs=[
                        prompt, width, height, 
                        steps, guidance_scale, seed,
                        bf16_model_choice, fp8_model_choice, sdnq_model_choice,  # 使用三个模型选择
                        batch_size, lora_enable, lora_model, lora_weight
                    ],
                    outputs=[queue_result]
                ).then(
                    fn=get_queue_status,
                    inputs=[],
                    outputs=[queue_status]
                ).then(
                    fn=get_detailed_queue_status,
                    inputs=[],
                    outputs=[detailed_queue_status]
                )
                
                process_queue_btn.click(
                    fn=process_queue,
                    inputs=[],
                    outputs=[result_gallery, result_status]
                ).then(
                    fn=get_queue_status,
                    inputs=[],
                    outputs=[queue_status]
                ).then(
                    fn=get_detailed_queue_status,
                    inputs=[],
                    outputs=[detailed_queue_status]
                )

                clear_queue_btn.click(
                    fn=clear_queue,
                    inputs=[],
                    outputs=[queue_result]
                ).then(
                    fn=get_queue_status,
                    inputs=[],
                    outputs=[queue_status]
                ).then(
                    fn=get_detailed_queue_status,
                    inputs=[],
                    outputs=[detailed_queue_status]
                )

        with gr.TabItem("图像编辑"):
            # 双图像结合界面组件
            with gr.Row():
                with gr.Column():
                    # 双图像结合输入区域
                    with gr.Row():  # 新增行容器，使两个图像组件并排
                        with gr.Column():
                            multi_img1 = gr.Image(
                                label="第一张图像",
                                type="filepath",  # 使用filepath类型以避免Content-Length错误
                                height=256,
                                interactive=True,
                                sources=['upload']  # 仅允许上传
                            )
                        
                        with gr.Column():
                            multi_img2 = gr.Image(
                                label="第二张图像 (可选)",
                                type="filepath",  # 使用filepath类型以避免Content-Length错误
                                height=256,
                                interactive=True,
                                sources=['upload']  # 仅允许上传
                            )
                    
                    # 提示词输入区域
                    multi_prompt = gr.Textbox(label="提示词", lines=3, value="编辑这张图像，让它看起来更好")
                    
                    # 3D角度可视化选择器折叠模块
                    with gr.Accordion("3D角度可视化选择器", open=False):
                        create_flux_klein_angle_visualization_component(multi_prompt)
                    
                    # 指令收藏 - 极简版本
                    with gr.Accordion("📝 指令收藏", open=False):
                        # 简化的收藏功能
                        with gr.Row():
                            multi_keyword_input = gr.Textbox(
                                label="关键词",
                                placeholder="输入要保存的关键词...",
                                interactive=True
                            )
                            multi_save_btn = gr.Button("收藏", variant="primary")
                            multi_delete_btn = gr.Button("删除", variant="secondary")
                        
                        # 显示已保存的关键词
                        multi_saved_keywords = gr.HTML()
                        multi_keywords_state = gr.State(CUSTOM_KEYWORDS_CACHE)
                        
                        # 简化的操作函数
                        def multi_simple_save_keyword(keyword_text, current_list):
                            if not keyword_text.strip():
                                return current_list, refresh_minimal_display(current_list)
                            
                            # 解析格式：显示名称|实际提示词
                            if '|' in keyword_text:
                                parts = keyword_text.split('|', 1)
                                display_name = parts[0].strip()
                                prompt_value = parts[1].strip()
                            else:
                                display_name = keyword_text.strip()
                                prompt_value = keyword_text.strip()
                            
                            # 检查重复
                            for item in current_list:
                                if item['display'] == display_name:
                                    return current_list, refresh_minimal_display(current_list)
                            
                            # 添加新关键词
                            new_item = {
                                'display': display_name,
                                'prompt': prompt_value,
                                'created_at': datetime.now().isoformat()
                            }
                            current_list.append(new_item)
                            
                            # 保存到配置文件
                            global CUSTOM_KEYWORDS_CACHE
                            CUSTOM_KEYWORDS_CACHE = current_list.copy()
                            save_custom_keywords(current_list)
                            
                            return current_list, refresh_minimal_display(current_list)
                        
                        def multi_simple_delete_keyword(keyword_text, current_list):
                            if not keyword_text.strip():
                                return current_list, refresh_minimal_display(current_list)
                            
                            display_name = keyword_text.strip()
                            
                            # 查找并删除
                            for i, item in enumerate(current_list):
                                if item['display'] == display_name:
                                    current_list.pop(i)
                                    # 保存到配置文件
                                    global CUSTOM_KEYWORDS_CACHE
                                    CUSTOM_KEYWORDS_CACHE = current_list.copy()
                                    save_custom_keywords(current_list)
                                    break
                            
                            return current_list, refresh_minimal_display(current_list)
                        
                        def refresh_minimal_display(keywords_list):
                            if not keywords_list:
                                return "<small>暂无收藏关键词</small>"
                            
                            html = ""
                            for item in keywords_list:
                                display_text = item.get('display', '')
                                prompt_text = item.get('prompt', '')
                                # 使用更可靠的元素查找方式
                                escaped_prompt = prompt_text.replace("'", "\\'").replace('"', '\\"')
                                # 使用querySelector按label查找元素，更稳定
                                html += f'<button onclick="var elem=document.querySelector(\'[data-testid=\\\'textbox\\\'\'][aria-label=\\\'提示词\\\']\')||document.querySelector(\'input[type=\\\'text\\\'],textarea\').closest(\'div\').querySelector(\'textarea,input\');if(elem){{elem.value+=(elem.value?\' \':\'\')+\'{escaped_prompt}\';}}" style="margin: 2px; padding: 4px 8px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">{display_text}</button>'
                            return html
                        
                        # 绑定事件
                        multi_save_btn.click(
                            fn=multi_simple_save_keyword,
                            inputs=[multi_keyword_input, multi_keywords_state],
                            outputs=[multi_keywords_state, multi_saved_keywords]
                        )
                        
                        multi_delete_btn.click(
                            fn=multi_simple_delete_keyword,
                            inputs=[multi_keyword_input, multi_keywords_state],
                            outputs=[multi_keywords_state, multi_saved_keywords]
                        )
                        
                        # 初始化显示
                        multi_saved_keywords.value = refresh_minimal_display(CUSTOM_KEYWORDS_CACHE)
                    
                    with gr.Row():
                        # 双图像结合参数 - 确保这些组件是可交互的
                        multi_steps = gr.Slider(
                            label="步数", 
                            minimum=1, 
                            maximum=50, 
                            value=15, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                        multi_guidance_scale = gr.Slider(
                            label="CFG Scale", 
                            minimum=1.0, 
                            maximum=10.0, 
                            value=1.0, 
                            step=0.1,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 随机种子 - 确保这个组件是可交互的
                        multi_seed = gr.Number(
                            label="种子 (Seed)", 
                            value=-1, 
                            precision=0,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 生成批次 - 确保这个组件是可交互的
                        multi_batch_size = gr.Slider(
                            label="生成批次", 
                            minimum=1, 
                            maximum=8, 
                            value=1, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    # 模型选择下拉列表 - 分别显示BF16、FP8和SDNQ模型
                    with gr.Row():
                        with gr.Column(scale=1):
                            multi_bf16_model_choice = gr.Dropdown(
                                choices=["无"] + get_bf16_models(),  # 添加"无"选项以避免冲突
                                value=get_bf16_models()[0] if get_bf16_models() else "FLUX_2-klein-base-4B",
                                label="BF16模型选择 (基础模型)"
                            )
                        with gr.Column(scale=1):
                            multi_fp8_model_choice = gr.Dropdown(
                                choices=["无"] + get_fp8_models(),  # 添加"无"选项
                                value=None,
                                label="FP8模型选择 (量化模型)",
                                info="选择FP8量化模型进行加速"
                            )
                    
                    # 添加SDNQ模型选择
                    with gr.Row():
                        multi_sdnq_model_choice = gr.Dropdown(
                            choices=["无"] + get_sdnq_models(),  # 添加SDNQ模型选项
                            value=None,
                            label="SDNQ 4bit模型选择 (超量化模型)",
                            info="选择SDNQ 4bit动态SVD量化模型进行极致加速"
                        )
                    
                    with gr.Row():
                        # 添加刷新模型列表按钮
                        multi_refresh_model_btn = gr.Button("刷新模型列表")
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                multi_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                multi_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                multi_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                multi_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_multi_lora_interactive(enable_lora):
                                return update_lora_interactive(enable_lora)
                            
                            multi_lora_enable.change(
                                fn=update_multi_lora_interactive,
                                inputs=multi_lora_enable,
                                outputs=multi_lora_model
                            )
                            
                            # 绑定刷新按钮事件
                            multi_refresh_lora_button.click(
                                fn=refresh_lora_models,
                                inputs=[],
                                outputs=multi_lora_model
                            )
                    
                    # 移除了左侧的生成按钮
                    # 原来的 multi_btn 已经在右侧重新定义
                    # 只保留右侧统一的操作按钮
                
                with gr.Column():
                    # 结果展示画廊
                    multi_result_gallery = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="flux_multi_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    multi_result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 队列功能区域
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
                    
                    # 生成按钮和打开输出目录按钮
                    with gr.Row():
                        multi_btn = gr.Button("生成结果", variant="primary")
                        multi_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")

            # 图像编辑Tab的事件绑定
            if not FLUX_KLEIN_AVAILABLE:
                # 如果模块不可用，禁用相关功能
                multi_btn.variant = "secondary"
                multi_btn.interactive = False
                multi_result_status.value = f"⚠️ FLUX.2-klein模块不可用: {MODULE_IMPORT_ERROR}"
            else:
                # 事件绑定 - 双图结合部分
                multi_btn.click(
                    fn=lambda img1, img2, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, sdnq_choice, batch_size, lora_enable, lora_model, lora_weight:
                        multi_img_flux_klein(
                            Image.open(img1) if img1 else None,  # 从文件路径加载图像
                            Image.open(img2) if img2 else None,  # 从文件路径加载图像
                            prompt, steps, guidance_scale, seed,
                            get_selected_model(bf16_choice, fp8_choice, sdnq_choice),  # 使用合并的模型选择
                            batch_size, lora_enable, lora_model, lora_weight
                        ),
                    inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_bf16_model_choice, multi_fp8_model_choice, multi_sdnq_model_choice, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight],
                    outputs=[multi_result_gallery, multi_result_status]
                )
                
                # 刷新模型列表按钮事件 - 更新本地的BF16、FP8和SDNQ模型选择
                multi_refresh_model_btn.click(
                    fn=lambda: (
                        ["无"] + get_bf16_models(), 
                        ["无"] + get_fp8_models(),
                        ["无"] + get_sdnq_models()  # 返回元组而不是列表，修复Content-Length错误
                    ),
                    inputs=[],
                    outputs=[multi_bf16_model_choice, multi_fp8_model_choice, multi_sdnq_model_choice]  # 更新本地的三个下拉框
                )

                # 统一处理所有事件绑定
                def setup_events():
                    # 添加到队列的事件绑定
                    add_to_queue_btn.click(
                        fn=lambda img1, img2, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, sdnq_choice, batch_size, lora_enable, lora_model, lora_weight: (
                            add_to_queue('multi', Image.open(img1) if img1 else None, Image.open(img2) if img2 else None, prompt, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice, sdnq_choice), batch_size, lora_enable, lora_model, lora_weight)
                        ),
                        inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_bf16_model_choice, multi_fp8_model_choice, multi_sdnq_model_choice, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight],
                        outputs=[queue_operation_status]
                    ).then(
                        fn=get_queue_status,
                        inputs=[],
                        outputs=[queue_status_text]
                    ).then(
                        fn=get_detailed_queue_status,
                        inputs=[],
                        outputs=[detailed_queue_status]
                    )
                    
                    # 处理队列任务事件
                    process_queue_btn.click(
                        fn=process_queue,
                        inputs=[],
                        outputs=[multi_result_gallery, multi_result_status]
                    ).then(
                        fn=get_queue_status,
                        inputs=[],
                        outputs=[queue_status_text]
                    ).then(
                        fn=get_detailed_queue_status,
                        inputs=[],
                        outputs=[detailed_queue_status]
                    )
                    
                    # 清空队列按钮事件
                    clear_queue_btn.click(
                        fn=clear_queue,
                        inputs=[],
                        outputs=[queue_operation_status]
                    ).then(
                        fn=get_queue_status,
                        inputs=[],
                        outputs=[queue_status_text]
                    ).then(
                        fn=get_detailed_queue_status,
                        inputs=[],
                        outputs=[detailed_queue_status]
                    )
                
                # 执行事件绑定设置
                setup_events()
                
                # 打开输出目录按钮事件
                multi_open_outputs_btn.click(
                    fn=lambda: open_folder("outputs"),
                    inputs=[],
                    outputs=[]
                )

        # 局部重绘Tab
        with gr.TabItem("局部编辑"):
            # 局部重绘界面组件
            with gr.Row():
                with gr.Column():
                    # 图像+蒙版输入区域（使用Gradio的ImageMask组件）
                    # 按规范显式配置画笔工具
                    inpaint_image = gr.ImageMask(
                        label="上传图像并绘制蒙版区域",
                        sources=['upload'],
                        type="pil",
                        show_label=True,
                        brush=gr.Brush(
                            colors=["#FFFFFF", "#000000", "#FF0000", "#00FF00"],  # 白、黑、红、绿
                            default_color="#FFFFFF",  # 白色默认，符合编辑语义
                            default_size=25
                        ),
                        eraser=gr.Eraser(
                            default_size=25
                        )
                    )
                    
                    
                    # 提示词输入区域
                    inpaint_prompt = gr.Textbox(label="提示词", lines=3, value="修复这个区域，画一只小狗")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 局部重绘参数 - 确保这些组件是可交互的
                        inpaint_steps = gr.Slider(
                            label="步数", 
                            minimum=1, 
                            maximum=50, 
                            value=4, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                        inpaint_guidance_scale = gr.Slider(
                            label="CFG Scale", 
                            minimum=1.0, 
                            maximum=10.0, 
                            value=1.0, 
                            step=0.1,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 随机种子
                        inpaint_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        inpaint_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    # 模型选择下拉列表 - 分别显示BF16和FP8模型
                    with gr.Row():
                        with gr.Column(scale=1):
                            inpaint_bf16_model_choice = gr.Dropdown(
                                choices=["无"] + get_bf16_models(),  # 添加"无"选项以避免冲突
                                value=get_bf16_models()[0] if get_bf16_models() else "FLUX_2-klein-base-4B",
                                label="BF16模型选择 (基础模型)"
                            )
                        with gr.Column(scale=1):
                            inpaint_fp8_model_choice = gr.Dropdown(
                                choices=["无"] + get_fp8_models(),  # 添加"无"选项
                                value=None,
                                label="FP8模型选择 (量化模型)",
                                info="选择FP8量化模型进行加速"
                            )
                    
                    # 添加SDNQ模型选择
                    with gr.Row():
                        inpaint_sdnq_model_choice = gr.Dropdown(
                            choices=["无"] + get_sdnq_models(),  # 添加SDNQ模型选项
                            value=None,
                            label="SDNQ 4bit模型选择 (超量化模型)",
                            info="选择SDNQ 4bit动态SVD量化模型进行极致加速"
                        )
                    
                    with gr.Row():
                        # 添加刷新模型列表按钮
                        inpaint_refresh_model_btn = gr.Button("刷新模型列表")
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                inpaint_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                inpaint_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                inpaint_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                inpaint_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_inpaint_lora_interactive(enable_lora):
                                return update_lora_interactive(enable_lora)
                            
                            inpaint_lora_enable.change(
                                fn=update_inpaint_lora_interactive,
                                inputs=inpaint_lora_enable,
                                outputs=inpaint_lora_model
                            )
                            
                            # 绑定刷新按钮事件
                            inpaint_refresh_lora_button.click(
                                fn=refresh_lora_models,
                                inputs=[],
                                outputs=inpaint_lora_model
                            )
                    
                    # 严格遵守功能模块与UI组件的职责隔离规范
                    # 局部重绘Tab专注于图像修复功能，不包含指令收藏功能
                    # 仅保留核心的LoRA模型和图像处理相关组件
                    
                with gr.Column():
                    # 结果展示画廊
                    inpaint_result_gallery = gr.Gallery(
                        label="局部重绘结果",
                        show_label=True,
                        elem_id="flux_inpaint_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    inpaint_result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 队列功能区域
                    with gr.Accordion("任务队列", open=False):
                        with gr.Group():
                            queue_status_text = gr.Textbox(label="队列状态", value="当前队列大小: 0", interactive=False)
                            
                            with gr.Row():
                                # 添加到队列按钮
                                inpaint_add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                                inpaint_process_queue_btn = gr.Button("执行队列任务", variant="primary")
                                inpaint_clear_queue_btn = gr.Button("清空队列", variant="stop")
                            
                            # 队列操作状态
                            inpaint_queue_operation_status = gr.Textbox(label="队列操作状态", interactive=False)
                            
                            # 详细队列状态显示
                            inpaint_detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, lines=5, max_lines=10)
                    
                    # 生成按钮和打开输出目录按钮
                    with gr.Row():
                        inpaint_btn = gr.Button("局部重绘", variant="primary")
                        inpaint_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                    
                    # 局部重绘Tab的事件绑定
            if not FLUX_KLEIN_AVAILABLE:
                # 如果模块不可用，禁用相关功能
                inpaint_btn.variant = "secondary"
                inpaint_btn.interactive = False
                inpaint_result_status.value = f"⚠️ FLUX.2-klein模块不可用: {MODULE_IMPORT_ERROR}"
            else:
                # 事件绑定 - 局部重绘部分（使用预处理函数）
                inpaint_btn.click(
                    fn=preprocess_inpaint_data,
                    inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_bf16_model_choice, inpaint_fp8_model_choice, inpaint_sdnq_model_choice, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
                    outputs=[inpaint_result_gallery, inpaint_result_status]
                )
                
                # 刷新模型列表按钮事件 - 更新本地的BF16、FP8和SDNQ模型选择
                inpaint_refresh_model_btn.click(
                    fn=lambda: (
                        ["无"] + get_bf16_models(), 
                        ["无"] + get_fp8_models(),
                        ["无"] + get_sdnq_models()  # 返回元组而不是列表，修复Content-Length错误
                    ),
                    inputs=[],
                    outputs=[inpaint_bf16_model_choice, inpaint_fp8_model_choice, inpaint_sdnq_model_choice]  # 更新本地的三个下拉框
                )

                # 统一处理所有事件绑定
                def setup_inpaint_events():
                    # 添加到队列的事件绑定
                    inpaint_add_to_queue_btn.click(
                        fn=lambda image_with_mask, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, sdnq_choice, batch_size, lora_enable, lora_model, lora_weight: (
                            add_to_queue('inpaint', 
                                        image_with_mask,  # 直接传递图像和蒙版，不需要压缩
                                        prompt, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice, sdnq_choice), batch_size, lora_enable, lora_model, lora_weight)
                        ),
                        inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_bf16_model_choice, inpaint_fp8_model_choice, inpaint_sdnq_model_choice, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
                        outputs=[inpaint_queue_operation_status]
                    ).then(
                        fn=get_queue_status,
                        inputs=[],
                        outputs=[queue_status_text]
                    ).then(
                        fn=get_detailed_queue_status,
                        inputs=[],
                        outputs=[inpaint_detailed_queue_status]
                    )
                    
                    # 处理队列任务事件
                    inpaint_process_queue_btn.click(
                        fn=process_queue,
                        inputs=[],
                        outputs=[inpaint_result_gallery, inpaint_result_status]
                    ).then(
                        fn=get_queue_status,
                        inputs=[],
                        outputs=[queue_status_text]
                    ).then(
                        fn=get_detailed_queue_status,
                        inputs=[],
                        outputs=[inpaint_detailed_queue_status]
                    )
                    
                    # 清空队列按钮事件
                    inpaint_clear_queue_btn.click(
                        fn=clear_queue,
                        inputs=[],
                        outputs=[inpaint_queue_operation_status]
                    ).then(
                        fn=get_queue_status,
                        inputs=[],
                        outputs=[queue_status_text]
                    ).then(
                        fn=get_detailed_queue_status,
                        inputs=[],
                        outputs=[inpaint_detailed_queue_status]
                    )
                
                # 执行事件绑定设置
                setup_inpaint_events()
                
                # 打开输出目录按钮事件
                inpaint_open_outputs_btn.click(
                    fn=lambda: open_folder("outputs"),
                    inputs=[],
                    outputs=[]
                )

        # 图像扩展Tab
        with gr.TabItem("图像扩展"):
            with gr.Row():
                with gr.Column():  # 左侧面板 - 参数设置
                    # 图像扩展输入
                    extend_input = gr.Image(
                        label="上传要扩展的图像",
                        type="filepath",  # 使用filepath类型以避免Content-Length错误
                        height=512
                    )
                    
                    # 提示词输入区域
                    extend_prompt = gr.Textbox(label="提示词", lines=3, value="高质量扩展图像边缘，与原图无缝衔接，保持原有风格和细节")
                    
                    # 扩展参数设置
                    gr.Markdown("### 扩展参数")
                    
                    # 添加推荐参数提示
                    gr.Markdown("""
                    **参数推荐：推理步数15以上，引导数4**
                    """)
                    
                    with gr.Row():
                        extend_left = gr.Slider(
                            label="向左扩展像素", 
                            minimum=0, 
                            maximum=512, 
                            value=64, 
                            step=8,
                            interactive=True  # 明确设置为可交互
                        )
                        extend_right = gr.Slider(
                            label="向右扩展像素", 
                            minimum=0, 
                            maximum=512, 
                            value=64, 
                            step=8,
                            interactive=True  # 明确设置为可交互
                        )
                        
                    with gr.Row():
                        extend_top = gr.Slider(
                            label="向上扩展像素", 
                            minimum=0, 
                            maximum=512, 
                            value=64, 
                            step=8,
                            interactive=True  # 明确设置为可交互
                        )
                        extend_bottom = gr.Slider(
                            label="向下扩展像素", 
                            minimum=0, 
                            maximum=512, 
                            value=64, 
                            step=8,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        extend_steps = gr.Slider(
                            label="步数", 
                            minimum=1, 
                            maximum=50, 
                            value=4, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                        extend_guidance_scale = gr.Slider(
                            label="CFG Scale", 
                            minimum=1.0, 
                            maximum=10.0, 
                            value=1.0, 
                            step=0.1,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 随机种子 - 确保这个组件是可交互的
                        extend_seed = gr.Number(
                            label="种子 (Seed)", 
                            value=-1, 
                            precision=0,
                            interactive=True  # 明确设置为可交互
                        )
                    
                    with gr.Row():
                        # 生成批次 - 确保这个组件是可交互的
                        extend_batch_size = gr.Slider(
                            label="生成批次", 
                            minimum=1, 
                            maximum=8, 
                            value=1, 
                            step=1,
                            interactive=True  # 明确设置为可交互
                        )
                    # 模型选择下拉列表 - 分别显示BF16和FP8模型
                    with gr.Row():
                        with gr.Column(scale=1):
                            extend_bf16_model_choice = gr.Dropdown(
                                choices=["无"] + get_bf16_models(),  # 添加"无"选项以避免冲突
                                value=get_bf16_models()[0] if get_bf16_models() else "FLUX_2-klein-base-4B",
                                label="BF16模型选择 (全参数模型)"
                            )
                        with gr.Column(scale=1):
                            extend_fp8_model_choice = gr.Dropdown(
                                choices=["无"] + get_fp8_models(),  # 添加"无"选项
                                value=None,
                                label="FLUX.2-klein-9B-FP8模型选择 ",
                            )
                    
                    # 添加SDNQ模型选择
                    with gr.Row():
                        extend_sdnq_model_choice = gr.Dropdown(
                            choices=["无"] + get_sdnq_models(),  # 添加SDNQ模型选项
                            value=None,
                            label="FLUX.2-klein-9B-SDNQ 4bit模型选择",
                            info="(量化模型)不支持lora"
                        )
                    
                    with gr.Row():
                        # 添加刷新模型列表按钮
                        extend_refresh_model_btn = gr.Button("刷新模型列表")
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                extend_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                extend_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),  # 修复：使用list_lora_models()而不是refresh_lora_models()
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                extend_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,
                                    step=0.01,
                                    value=1.0,
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                extend_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_extend_lora_interactive(enable_lora):
                                return update_lora_interactive(enable_lora)
                            
                            extend_lora_enable.change(
                                fn=update_extend_lora_interactive,
                                inputs=extend_lora_enable,
                                outputs=extend_lora_model
                            )
                            
                            # 绑定刷新按钮事件
                            extend_refresh_lora_button.click(
                                fn=refresh_lora_models,
                                inputs=[],
                                outputs=extend_lora_model
                            )
                    
                    # 严格遵守功能模块与UI组件的职责隔离规范
                    # 图像扩展Tab专注于图像扩增功能，不包含指令收藏功能
                    # 仅保留核心的LoRA模型和图像处理相关组件
                
                with gr.Column():  # 右侧面板 - 操作与输出
                    # 结果展示画廊
                    extend_result_gallery = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="flux_extend_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    extend_result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 队列功能区域
                    with gr.Accordion("任务队列", open=False):
                        with gr.Group():
                            extend_queue_status = gr.Textbox(label="队列状态", value="当前队列大小: 0", interactive=False)
                            
                            with gr.Row():
                                # 添加到队列按钮
                                extend_add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                                extend_process_queue_btn = gr.Button("执行队列任务", variant="primary")
                                extend_clear_queue_btn = gr.Button("清空队列", variant="stop")
                            
                            # 队列操作状态
                            extend_queue_result = gr.Textbox(label="队列操作状态", interactive=False)
                            
                            # 详细队列状态显示
                            extend_detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, lines=5, max_lines=10)
                    
                    # 生成按钮和打开输出目录按钮
                    with gr.Row():
                        extend_gen_btn = gr.Button("生成图像", variant="primary")
                        extend_open_outputs_btn = gr.Button("打开输出目录")
                    
                    # 图像扩展Tab的事件绑定
                    if not FLUX_KLEIN_AVAILABLE:
                        # 如果模块不可用，禁用相关功能
                        extend_gen_btn.variant = "secondary"
                        extend_gen_btn.interactive = False
                        extend_result_status.value = f"⚠️ FLUX.2-klein模块不可用: {MODULE_IMPORT_ERROR}"
                    else:
                        # 任务队列相关事件
                        extend_add_to_queue_btn.click(
                            fn=lambda image, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight, left, right, top, bottom: (
                                add_to_queue('extend', Image.open(image) if image else None, prompt, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice), batch_size, lora_enable, lora_model, lora_weight, left, right, top, bottom)
                            ),
                            inputs=[
                                extend_input, extend_prompt, 
                                extend_steps, extend_guidance_scale, extend_seed,
                                extend_bf16_model_choice, extend_fp8_model_choice, extend_batch_size,
                                extend_lora_enable, extend_lora_model, extend_lora_weight,
                                extend_left, extend_right, extend_top, extend_bottom
                            ],
                            outputs=[extend_queue_result]
                        ).then(
                            fn=get_queue_status,
                            inputs=[],
                            outputs=[extend_queue_status]
                        ).then(
                            fn=get_detailed_queue_status,
                            inputs=[],
                            outputs=[extend_detailed_queue_status]
                        )
                        
                        extend_process_queue_btn.click(
                            fn=process_queue,
                            inputs=[],
                            outputs=[extend_result_gallery, extend_result_status]
                        ).then(
                            fn=get_queue_status,
                            inputs=[],
                            outputs=[extend_queue_status]
                        ).then(
                            fn=get_detailed_queue_status,
                            inputs=[],
                            outputs=[extend_detailed_queue_status]
                        )
                        
                        # 清空队列按钮事件
                        extend_clear_queue_btn.click(
                            fn=clear_queue,
                            inputs=[],
                            outputs=[extend_queue_result]
                        ).then(
                            fn=get_queue_status,
                            inputs=[],
                            outputs=[extend_queue_status]
                        ).then(
                            fn=get_detailed_queue_status,
                            inputs=[],
                            outputs=[extend_detailed_queue_status]
                        )
                        
                        # 生成图像事件
                        extend_gen_btn.click(
                            fn=lambda image, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, sdnq_choice, batch_size, lora_enable, lora_model, lora_weight, left, right, top, bottom:
                                extend_flux_klein(
                                    Image.open(image) if image else None,  # 从文件路径加载图像
                                    prompt, steps, guidance_scale, seed,
                                    get_selected_model(bf16_choice, fp8_choice, sdnq_choice),  # 使用合并的模型选择
                                    batch_size, lora_enable, lora_model, lora_weight,
                                    left, right, top, bottom
                                ),
                            inputs=[
                                extend_input, extend_prompt,
                                extend_steps, extend_guidance_scale, extend_seed,
                                extend_bf16_model_choice, extend_fp8_model_choice, extend_sdnq_model_choice, extend_batch_size,
                                extend_lora_enable, extend_lora_model, extend_lora_weight,
                                extend_left, extend_right, extend_top, extend_bottom
                            ],
                            outputs=[extend_result_gallery, extend_result_status]
                        )
                        
                        # 打开输出目录事件
                        extend_open_outputs_btn.click(
                            fn=lambda: open_folder("outputs"),
                            inputs=[],
                            outputs=[]
                        )
                        
                        # 刷新模型列表按钮事件
                        extend_refresh_model_btn.click(
                            fn=lambda: (
                                ["无"] + get_bf16_models(), 
                                ["无"] + get_fp8_models(),
                                ["无"] + get_sdnq_models()  # 返回元组而不是列表，修复Content-Length错误
                            ),
                            inputs=[],
                            outputs=[extend_bf16_model_choice, extend_fp8_model_choice, extend_sdnq_model_choice]  # 更新本地的三个下拉框
                        )

    # 返回组件列表以便在其他地方使用（如果需要）
    return locals()  # 返回所有局部变量

def preprocess_inpaint_data(*args, **kwargs):
    """预处理局部重绘数据的包装函数"""
    try:
        return inpainting(*args, **kwargs)
    except Exception as e:
        logger.error(f"局部重绘处理出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 返回错误信息和空图像列表
        return [], f"❌ 处理失败: {str(e)}"

# 局部重绘功能函数 - 修正蒙版处理逻辑
def inpainting(image_with_mask, prompt, steps, guidance_scale, seed, bf16_model_choice, fp8_model_choice, sdnq_model_choice, batch_size, lora_enable, lora_model, lora_weight):
    """
    处理局部重绘任务
    """
    # 获取选中的模型
    model_name = get_selected_model(bf16_model_choice, fp8_model_choice, sdnq_model_choice)
    
    logger = logging.getLogger(__name__)
    
    logger.info("=== 局部重绘数据预处理 ===")
    logger.info(f"image_with_mask 类型: {type(image_with_mask)}")
    logger.info(f"选中模型: {model_name}")
    
    if isinstance(image_with_mask, dict):
        logger.info(f"字典键: {list(image_with_mask.keys())}")
        if 'image' in image_with_mask:
            img_data = image_with_mask['image']
            logger.info(f"图像数据类型: {type(img_data)}")
            if hasattr(img_data, 'shape'):
                logger.info(f"图像形状: {img_data.shape}")
            elif hasattr(img_data, 'size'):
                logger.info(f"图像尺寸: {img_data.size}")
        if 'mask' in image_with_mask:
            mask_data = image_with_mask['mask']
            logger.info(f"蒙版数据类型: {type(mask_data)}")
            if hasattr(mask_data, 'shape'):
                logger.info(f"蒙版形状: {mask_data.shape}")
            elif hasattr(mask_data, 'size'):
                logger.info(f"蒙版尺寸: {mask_data.size}")
    else:
        logger.info("image_with_mask 不是字典格式")
    
    # 调用实际的生成函数
    return inpaint_flux_klein(
        image_with_mask, prompt, steps, guidance_scale, seed,
        get_selected_model(bf16_model_choice, fp8_model_choice, sdnq_model_choice),
        batch_size, lora_enable, lora_model, lora_weight
    )

def get_selected_model(bf16_model_choice, fp8_model_choice, sdnq_model_choice=None):
    """
    根据BF16、FP8和SDNQ模型选择返回最终模型选择
    优先级：SDNQ > FP8 > BF16
    """
    # 首先检查SDNQ模型
    if sdnq_model_choice is not None and sdnq_model_choice != "" and sdnq_model_choice != "无":
        return sdnq_model_choice
    # 然后检查FP8模型
    elif fp8_model_choice is not None and fp8_model_choice != "" and fp8_model_choice != "无":
        return fp8_model_choice
    # 最后使用BF16模型
    elif bf16_model_choice is not None and bf16_model_choice != "无":
        return bf16_model_choice
    # 如果都没有有效选择，返回默认模型
    else:
        return "FLUX_2-klein-base-4B"

def open_folder(folder_path):
    """打开指定的文件夹"""
  
    abs_path = os.path.abspath(folder_path)
    
    if platform.system() == "Windows":
        os.startfile(abs_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", abs_path])
    else:  # Linux
        subprocess.run(["xdg-open", abs_path])

def update_lora_interactive(enable_lora):
    """
    更新LoRA模型选择组件的交互状态
    """
    return gr.update(interactive=enable_lora)

def refresh_lora_models():
    """
    刷新LoRA模型列表
    """
    try:
        lora_choices = list_lora_models()
        return gr.update(choices=lora_choices)
    except Exception as e:
        print(f"刷新LoRA模型列表时出错: {e}")
        return gr.update()

# 添加一个特殊函数来处理图像上传，减少Content-Length错误的可能性
def handle_image_upload(image):
    """
    专门处理图像上传的函数，避免Content-Length错误
    """
    if image is None:
        return None
    # 确保图像数据完整
    try:
        # 尝试处理图像以验证它是否完整
        if hasattr(image, 'convert'):
            # 如果是PIL图像，转换为RGB
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"图像处理错误: {e}")
        return None

# 添加获取SDNQ模型列表的函数
def get_sdnq_models():
    """获取SDNQ模型列表 - 支持动态识别SDNQ 4bit模型"""
    model_dir = os.path.join("models", "FLUX.2-klein")
    return _scan_model_directory(model_dir, 'sdnq')

# 在模型扫描函数中添加SDNQ支持
def _scan_model_directory(model_dir, model_type_filter):
    """
    扫描模型目录，获取模型列表
    :param model_dir: 模型目录路径
    :param model_type_filter: 模型类型过滤器 ('bf16', 'fp8' 或 'sdnq')
    :return: 模型列表
    """
    import os
    from pathlib import Path
    
    model_choices = ["无"]  # 添加"无"选项
    
    if not os.path.exists(model_dir):
        return model_choices
    
    for item in os.listdir(model_dir):
        item_path = Path(model_dir) / item
        
        # 检查是否为单独的模型文件
        if item_path.is_file() and item_path.suffix in ['.safetensors', '.bin']:
            if model_type_filter == 'fp8' and _is_fp8_model(str(item_path)):
                model_choices.append(f"{item}")  # 直接显示文件名
            elif model_type_filter == 'sdnq' and _is_sdnq_model(str(item_path)):
                model_choices.append(f"{item} (SDNQ-4bit)")
        # 检查是否为模型目录
        elif item_path.is_dir():
            # 检查目录中是否包含模型文件
            has_model_files = (
                any(file.suffix in [".bin", ".safetensors", ".pt", ".ckpt"] for file in item_path.iterdir() if file.is_file()) or
                (item_path / "model_index.json").exists()
            )
            
            if not has_model_files:
                continue
                
            # 检查模型类型
            is_fp8 = _is_fp8_model(str(item_path))
            is_sdnq = _is_sdnq_model(str(item_path))
            
            # 根据过滤器类型决定是否添加到列表
            if model_type_filter == 'fp8' and is_fp8:
                model_choices.append(f"{item} (FP8)")
            elif model_type_filter == 'sdnq' and is_sdnq:
                model_choices.append(f"{item} (SDNQ-4bit-Dynamic-SVD-R32)")
            elif model_type_filter == 'bf16' and not is_fp8 and not is_sdnq:
                # 动态识别模型类型，不再硬编码4B/9B
                model_type_info = _identify_model_type(item)
                model_choices.append(f"{item} {model_type_info}")
    
    return model_choices


