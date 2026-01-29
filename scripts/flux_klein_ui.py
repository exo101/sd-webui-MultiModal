import gradio as gr
import torch
import os
import gc
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from modules import shared
from modules import sd_samplers
from modules.ui_components import ToolButton
from modules import ui_common  # 导入ui_common模块
from modules import util  # 导入util模块
import requests
from PIL import Image

# 导入自定义模块
from scripts.flux_klein_model_loader import FLUX_KLEIN_AVAILABLE, list_lora_models, list_flux_klein_models, get_bf16_models, get_fp8_models
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
    get_detailed_queue_status
)
from scripts.flux_klein_utils import create_flux_klein_angle_visualization_component, refresh_lora_models, update_lora_interactive, clear_queue, open_folder

def create_flux_klein_ui():
    """创建FLUX.2-klein的UI界面"""
    if not FLUX_KLEIN_AVAILABLE:
        with gr.Column():
            gr.Markdown("FLUX.2-klein模块当前不可用，可能是因为缺少依赖项。")
            gr.Markdown("- 需要安装 `diffusers` 库")
            gr.Markdown("- 需要安装 `modelscope` 库")
            gr.Markdown("- 需要安装 `transformers` 库")
        return

    with gr.Tabs():
        with gr.TabItem("文生图"):
            # 文生图界面组件
            with gr.Row():
                with gr.Column():
                    # 提示词输入区域
                    prompt = gr.Textbox(label="正面提示词 (Prompt)", lines=3, value="一只可爱的小猫")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 生成参数
                        steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 尺寸参数
                        height = gr.Slider(label="高度", minimum=256, maximum=1536, value=1024, step=64)
                        width = gr.Slider(label="宽度", minimum=256, maximum=1536, value=768, step=64)
                    
                    with gr.Row():
                        # 随机种子
                        seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        batch_count = gr.Slider(label="批次数量", minimum=1, maximum=8, value=1, step=1)
                        batch_size = gr.Slider(label="每批数量", minimum=1, maximum=8, value=1, step=1)
                    
                    # 模型选择下拉列表 - 分别显示BF16和FP8模型
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
                fn=lambda prompt, steps, guidance_scale, height, width, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight: 
                    generate_flux_klein_image(
                        prompt, steps, guidance_scale, height, width, seed, 
                        get_selected_model(bf16_choice, fp8_choice),  # 使用合并的模型选择
                        batch_size, lora_enable, lora_model, lora_weight
                    ),
                inputs=[
                    prompt, steps, guidance_scale, 
                    height, width, seed, bf16_model_choice, fp8_model_choice,  # 传入两个模型选择
                    batch_size, lora_enable, lora_model, lora_weight
                ],
                outputs=[result_gallery, result_status]
            )
            
            # 刷新模型列表按钮事件
            refresh_model_btn.click(
                fn=lambda: [["无"] + get_bf16_models(), ["无"] + get_fp8_models()],  # 返回两个列表，都包含"无"选项
                inputs=[],
                outputs=[bf16_model_choice, fp8_model_choice]  # 更新两个下拉框
            )
            
            # 打开输出目录事件
            open_outputs_btn.click(
                fn=lambda: open_folder("outputs"),
                inputs=[],
                outputs=[]
            )
            
            # 任务队列相关事件
            add_to_queue_btn.click(
                fn=lambda prompt, width, height, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight: (
                    add_to_queue('txt2img', prompt, width, height, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice), batch_size, lora_enable, lora_model, lora_weight)
                ),
                inputs=[
                    prompt, width, height, 
                    steps, guidance_scale, seed,
                    bf16_model_choice, fp8_model_choice,  # 使用BF16和FP8两个模型选择
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
                                type="numpy", 
                                height=300,
                                interactive=True,
                                show_label=True,
                                sources=['upload'],  # 明确指定来源
                                show_share_button=False  # 隐藏分享按钮
                            )
                        with gr.Column():
                            multi_img2 = gr.Image(
                                label="第二张图像 (可选，留空则仅处理第一张图像)", 
                                type="numpy", 
                                height=300,
                                interactive=True,
                                show_label=True,
                                sources=['upload'],  # 明确指定来源
                                show_share_button=False  # 隐藏分享按钮
                            )
                    
                    # 提示词输入区域
                    multi_prompt = gr.Textbox(label="提示词", lines=3, value="结合两张图像特征生成")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    # 3D角度可视化选择器折叠模块
                    with gr.Accordion("3D角度可视化选择器", open=False):
                        create_flux_klein_angle_visualization_component(multi_prompt)
                    
                    with gr.Row():
                        # 双图像结合参数
                        multi_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        multi_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 随机种子
                        multi_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        multi_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    # 模型选择下拉列表 - 分别显示BF16和FP8模型
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

            # 事件绑定 - 双图像结合部分
            multi_btn.click(
                fn=lambda img1, img2, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight:
                    multi_img_flux_klein(
                        img1, img2, prompt, steps, guidance_scale, seed,
                        get_selected_model(bf16_choice, fp8_choice),  # 使用合并的模型选择
                        batch_size, lora_enable, lora_model, lora_weight
                    ),
                inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_bf16_model_choice, multi_fp8_model_choice, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight],
                outputs=[multi_result_gallery, multi_result_status]
            )
            
            # 刷新模型列表按钮事件 - 更新本地的BF16和FP8模型选择
            multi_refresh_model_btn.click(
                fn=lambda: [["无"] + get_bf16_models(), ["无"] + get_fp8_models()],  # 返回两个列表，都包含"无"选项
                inputs=[],
                outputs=[multi_bf16_model_choice, multi_fp8_model_choice]  # 更新本地的两个下拉框
            )
            
            # 统一处理所有事件绑定
            def setup_events():
                # 添加到队列的事件绑定
                add_to_queue_btn.click(
                    fn=lambda img1, img2, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight: (
                        add_to_queue('multi', img1, img2, prompt, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice), batch_size, lora_enable, lora_model, lora_weight)
                    ),
                    inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_bf16_model_choice, multi_fp8_model_choice, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight],
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
            
            # 打开输出目录事件
            multi_open_outputs_btn.click(
                fn=lambda: open_folder("outputs"),
                inputs=[],
                outputs=[]
            )
            

        with gr.TabItem("局部重绘"):
            # 局部重绘界面组件
            with gr.Row():
                with gr.Column():
                    # 图像+蒙版输入区域（使用Gradio的ImageMask组件）
                    inpaint_image = gr.ImageMask(
                        label="上传图像并绘制蒙版区域",
                        sources=['upload'],
                        interactive=True,
                        type="pil"  # 使用pil类型以更好地兼容处理流程
                    )
                    
                    # 提示词输入区域
                    inpaint_prompt = gr.Textbox(label="提示词", lines=3, value="修复这个区域，画一只小狗")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 局部重绘参数
                        inpaint_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        inpaint_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
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
            
            # 事件绑定 - 局部重绘部分
            inpaint_btn.click(
                fn=lambda image_with_mask, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight:
                    inpaint_flux_klein(
                        image_with_mask, prompt, steps, guidance_scale, seed,
                        get_selected_model(bf16_choice, fp8_choice),  # 使用合并的模型选择
                        batch_size, lora_enable, lora_model, lora_weight
                    ),
                inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_bf16_model_choice, inpaint_fp8_model_choice, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
                outputs=[inpaint_result_gallery, inpaint_result_status]
            )
            
            # 刷新模型列表按钮事件 - 更新本地的BF16和FP8模型选择
            inpaint_refresh_model_btn.click(
                fn=lambda: [["无"] + get_bf16_models(), ["无"] + get_fp8_models()],  # 返回两个列表，都包含"无"选项
                inputs=[],
                outputs=[inpaint_bf16_model_choice, inpaint_fp8_model_choice]  # 更新本地的两个下拉框
            )

            # 统一处理所有事件绑定
            def setup_inpaint_events():
                # 添加到队列的事件绑定
                inpaint_add_to_queue_btn.click(
                    fn=lambda image_with_mask, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight: (
                        add_to_queue('inpaint', image_with_mask, prompt, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice), batch_size, lora_enable, lora_model, lora_weight)
                    ),
                    inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_bf16_model_choice, inpaint_fp8_model_choice, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
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


        # 图像扩展界面
        with gr.TabItem("FLUX.2-klein图像扩展"):
            with gr.Row():
                with gr.Column():  # 左侧面板 - 参数设置
                    # 图像扩展输入
                    extend_input = gr.Image(
                        label="上传要扩展的图像",
                        type="numpy",
                        height=512
                    )
                    
                    # 提示词输入区域
                    extend_prompt = gr.Textbox(label="提示词", lines=3, value="高质量扩展图像边缘")
                    
                    # 3D角度可视化选择器折叠模块
                    with gr.Accordion("3D角度可视化选择器", open=False):
                        create_flux_klein_angle_visualization_component(extend_prompt)
                    
                    with gr.Row():
                        extend_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        extend_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 随机种子
                        extend_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        extend_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    # 模型选择下拉列表 - 分别显示BF16和FP8模型
                    with gr.Row():
                        with gr.Column(scale=1):
                            extend_bf16_model_choice = gr.Dropdown(
                                choices=["无"] + get_bf16_models(),  # 添加"无"选项以避免冲突
                                value=get_bf16_models()[0] if get_bf16_models() else "FLUX_2-klein-base-4B",
                                label="BF16模型选择 (基础模型)"
                            )
                        with gr.Column(scale=1):
                            extend_fp8_model_choice = gr.Dropdown(
                                choices=["无"] + get_fp8_models(),  # 添加"无"选项
                                value=None,
                                label="FP8模型选择 (量化模型)",
                                info="选择FP8量化模型进行加速"
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
                    
                    # 任务队列相关事件
                    extend_add_to_queue_btn.click(
                        fn=lambda image, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight: (
                            add_to_queue('extend', image, prompt, steps, guidance_scale, seed, get_selected_model(bf16_choice, fp8_choice), batch_size, lora_enable, lora_model, lora_weight)
                        ),
                        inputs=[
                            extend_input, extend_prompt, 
                            extend_steps, extend_guidance_scale, extend_seed,
                            extend_bf16_model_choice, extend_fp8_model_choice, extend_batch_size,
                            extend_lora_enable, extend_lora_model, extend_lora_weight
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
                        fn=lambda image, prompt, steps, guidance_scale, seed, bf16_choice, fp8_choice, batch_size, lora_enable, lora_model, lora_weight:
                            extend_flux_klein(
                                image, prompt, steps, guidance_scale, seed,
                                get_selected_model(bf16_choice, fp8_choice),  # 使用合并的模型选择
                                batch_size, lora_enable, lora_model, lora_weight
                            ),
                        inputs=[
                            extend_input, extend_prompt,
                            extend_steps, extend_guidance_scale, extend_seed,
                            extend_bf16_model_choice, extend_fp8_model_choice, extend_batch_size,
                            extend_lora_enable, extend_lora_model, extend_lora_weight
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
                        fn=lambda: [["无"] + get_bf16_models(), ["无"] + get_fp8_models()],  # 返回两个列表，都包含"无"选项
                        inputs=[],
                        outputs=[extend_bf16_model_choice, extend_fp8_model_choice]  # 更新本地的两个下拉框
                    )

    # 返回组件列表以便在其他地方使用（如果需要）
    return {
        "prompt": prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "seed": seed,
        "bf16_model_choice": bf16_model_choice,      # 使用新的BF16模型选择
        "fp8_model_choice": fp8_model_choice,        # 添加FP8模型选择
        "gen_btn": gen_btn,
        "result_gallery": result_gallery,
        "result_status": result_status,
        "multi_img1": multi_img1,
        "multi_img2": multi_img2,
        "multi_prompt": multi_prompt,
        "multi_steps": multi_steps,
        "multi_guidance_scale": multi_guidance_scale,
        "multi_seed": multi_seed,
        "multi_bf16_model_choice": bf16_model_choice,  # 使用全局BF16模型选择
        "multi_fp8_model_choice": fp8_model_choice,    # 添加全局FP8模型选择
        "multi_btn": multi_btn,
        "multi_result_gallery": multi_result_gallery,
        "multi_result_status": multi_result_status,
        "inpaint_image": inpaint_image,
        "inpaint_prompt": inpaint_prompt,
        "inpaint_steps": inpaint_steps,
        "inpaint_guidance_scale": inpaint_guidance_scale,
        "inpaint_seed": inpaint_seed,
        "inpaint_bf16_model_choice": bf16_model_choice,  # 使用全局BF16模型选择
        "inpaint_fp8_model_choice": fp8_model_choice,    # 添加全局FP8模型选择
        "inpaint_btn": inpaint_btn,
        "inpaint_result_gallery": inpaint_result_gallery,
        "inpaint_result_status": inpaint_result_status
    }


def create_image_gallery(image_paths):
    """
    创建一个包含多个图像的Gradio界面组件
    """
    html_images = ""
    for path in image_paths:
        html_images += f'<img src="{path}" style="width: 200px; height: auto; margin: 5px;">'
    
    with gr.Blocks() as image_gallery:
        # 使用HTML iframe显示图像
        html_content = f'''
        <div id="image-container" style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
            {html_images}
        </div>
        <script>
            // 添加点击事件监听器
            document.getElementById('image-container').addEventListener('click', function(event) {{
                if (event.target.tagName === 'IMG') {{
                    // 在新窗口中打开图像
                    window.open(event.target.src, '_blank');
                }}
            }});
        </script>
        '''
        gr.HTML(html_content)
    
    return image_gallery


def get_selected_model(bf16_choice, fp8_choice):
    """
    根据BF16和FP8模型选择返回最终模型选择
    如果FP8模型被选择且不是"无"，则使用FP8模型；
    否则使用BF16模型（如果不是"无"）
    """
    if fp8_choice is not None and fp8_choice != "" and fp8_choice != "无":
        return fp8_choice
    elif bf16_choice is not None and bf16_choice != "无":
        return bf16_choice
    else:
        # 默认返回一个基础模型
        bf16_models = get_bf16_models()
        if bf16_models:
            return bf16_models[0]
        else:
            return "FLUX_2-klein-base-4B"


def open_folder(folder_path):
    """打开指定的文件夹"""
    import os
    import subprocess
    import platform
    
    abs_path = os.path.abspath(folder_path)
    
    if platform.system() == "Windows":
        os.startfile(abs_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", abs_path])
    else:  # Linux
        subprocess.run(["xdg-open", abs_path])


def update_lora_interactive(enable_lora):
    """更新LoRA组件的交互状态"""
    return gr.update(interactive=bool(enable_lora))
