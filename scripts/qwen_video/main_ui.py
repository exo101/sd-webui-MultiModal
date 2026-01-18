"""
Qwen视频生成主UI模块
整合所有子模块，提供统一的UI界面
"""

import gradio as gr
import os
from modules import shared

from .api_handler import set_api_key
from .video_models import (
    generate_video_with_wan26,
    generate_video_with_wan25_i2v,
    generate_video_with_wan22_kf2v,
    generate_video_with_wan25_t2v
)
from .task_query import query_video_task, get_recent_tasks
from .utils import open_video_output_dir, create_html_video_player, download_video_to_local


def create_qwen_video_gen_ui():
    """
    创建Qwen视频生成UI界面
    """
    with gr.Blocks() as qwen_video_interface:
        gr.Markdown("# Qwen视频生成 API")
        gr.Markdown("使用阿里云百炼平台的wan2.5/wan2.6模型进行视频生成")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="请输入您的百炼API Key",
                    info="输入API Key后点击下方按钮设置"
                )
                set_api_key_btn = gr.Button("设置API Key", variant="secondary")
                api_key_status = gr.Textbox(label="状态", interactive=False)
                
                set_api_key_btn.click(
                    fn=set_api_key,
                    inputs=api_key_input,
                    outputs=api_key_status
                )
        
        with gr.Tabs():
            with gr.TabItem("图生视频 (wan2.6)"):
                with gr.Row():
                    with gr.Column():
                        video_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...",
                            lines=4,
                            max_lines=6
                        )
                        
                        image_url = gr.Image(
                            label="输入图像",
                            type="filepath",
                            interactive=True
                        )
                        
                        audio_file = gr.Audio(
                            label="音频文件（可选）",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            resolution = gr.Dropdown(
                                label="分辨率",
                                choices=["720P", "1080P"],  # 只提供受支持的分辨率选项
                                value="720P",
                                interactive=True
                            )
                            
                            duration = gr.Slider(
                                label="时长（秒）",
                                minimum=1,
                                maximum=30,
                                step=1,
                                value=10
                            )
                        
                        with gr.Row():
                            audio_enabled = gr.Checkbox(
                                label="包含音频",
                                value=True
                            )
                            
                            shot_type = gr.Dropdown(
                                label="镜头类型",
                                choices=["single", "multi"],
                                value="multi"
                            )
                        
                        video_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        # 分离进度显示和视频显示
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            progress_output = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            video_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def process_wan26_request(img, audio_file, prompt, resolution, duration, audio_enabled, shot_type):
                    if img is None:
                        return "❌ 请上传图像文件。", None
                    
                    result = generate_video_with_wan26(prompt, img, audio_file, resolution, duration, audio_enabled, shot_type)
                    
                    # 检查是否包含任务ID（表示异步任务）
                    if "任务ID:" in result:
                        # 只返回进度信息，视频输出为None
                        return result, None
                    # 检查是否包含视频URL（表示同步完成）
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            # 尝试下载视频到本地
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                # 本地下载成功，创建HTML视频播放器
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                # 如果下载失败，使用远程URL创建HTML视频播放器
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        # 其他情况返回进度信息，视频输出为None
                        return result, None
                
                video_gen_btn.click(
                    fn=process_wan26_request,
                    inputs=[image_url, audio_file, video_prompt, resolution, duration, audio_enabled, shot_type],
                    outputs=[progress_output, video_output]
                )
            
            with gr.TabItem("图生视频 (wan2.5)"):
                with gr.Row():
                    with gr.Column():
                        video_prompt_25 = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...",
                            lines=4,
                            max_lines=6
                        )
                        
                        image_url_25 = gr.Image(
                            label="输入图像",
                            type="filepath",
                            interactive=True
                        )
                        
                        audio_file_25 = gr.Audio(
                            label="音频文件（可选）",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            resolution_25 = gr.Dropdown(
                                label="分辨率",
                                choices=["720P", "1080P"],  # 只提供受支持的分辨率选项
                                value="720P",
                                interactive=True
                            )
                            
                            duration_25 = gr.Slider(
                                label="时长（秒）",
                                minimum=1,
                                maximum=30,
                                step=1,
                                value=10
                            )
                        
                        with gr.Row():
                            audio_enabled_25 = gr.Checkbox(
                                label="包含音频",
                                value=True
                            )
                        
                        video_gen_btn_25 = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        # 分离进度显示和视频显示
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            progress_output_25 = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            video_output_25 = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def process_i2v_request(img, audio_file, prompt, resolution, duration, audio_enabled):
                    if img is None:
                        return "❌ 请上传图像文件。", None
                    
                    result = generate_video_with_wan25_i2v(prompt, img, audio_file, resolution, duration, audio_enabled)
                    
                    # 检查是否包含任务ID（表示异步任务）
                    if "任务ID:" in result:
                        # 只返回进度信息，视频输出为None
                        return result, None
                    # 检查是否包含视频URL（表示同步完成）
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            # 尝试下载视频到本地
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                # 本地下载成功，创建HTML视频播放器
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                # 如果下载失败，使用远程URL创建HTML视频播放器
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        # 其他情况返回进度信息，视频输出为None
                        return result, None
                
                video_gen_btn_25.click(
                    fn=process_i2v_request,
                    inputs=[image_url_25, audio_file_25, video_prompt_25, resolution_25, duration_25, audio_enabled_25],
                    outputs=[progress_output_25, video_output_25]
                )
            
            
            with gr.TabItem("首尾帧生成 (wan2.2)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 步骤 1: 上传首尾帧图像")
                        first_frame_input = gr.Image(
                            label="首帧图像",
                            type="filepath",
                            interactive=True
                        )
                        last_frame_input = gr.Image(
                            label="尾帧图像",
                            type="filepath",
                            interactive=True
                        )
                        
                        prompt_input = gr.Textbox(
                            label="提示词",
                            placeholder="输入视频生成的提示词",
                            lines=3
                        )
                        resolution_input = gr.Dropdown(
                            label="分辨率",
                            choices=["720P", "1080P"],
                            value="720P",
                            interactive=True
                        )
                        
                        kf2v_submit_btn = gr.Button("生成首尾帧视频", variant="primary")
                    
                    with gr.Column():
                        # 分离进度显示和视频显示
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            kf2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            kf2v_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
            
                def process_kf2v_request(f_img, l_img, prompt, resolution):
                    # 检查是否上传了首帧和尾帧图像
                    if f_img is None:
                        return "❌ 请上传首帧图像。", None
                    
                    if l_img is None:
                        return "❌ 请上传尾帧图像。", None
                    
                    result = generate_video_with_wan22_kf2v(prompt, f_img, l_img, resolution)
                    
                    # 检查是否包含视频URL
                    if "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            # 尝试下载视频到本地
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                # 本地下载成功，创建HTML视频播放器
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                # 如果下载失败，使用远程URL创建HTML视频播放器
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        # 其他情况返回进度信息，视频输出为None
                        return result, None
            
                kf2v_submit_btn.click(
                    fn=process_kf2v_request,
                    inputs=[first_frame_input, last_frame_input, prompt_input, resolution_input],
                    outputs=[kf2v_progress, kf2v_output]
                )
            
            with gr.TabItem("文生视频 (wan2.5)"):
                with gr.Row():
                    with gr.Column():
                        t2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...",
                            lines=4,
                            max_lines=6
                        )
                        
                        t2v_audio_file = gr.Audio(
                            label="音频文件（可选）",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            t2v_resolution = gr.Dropdown(
                                label="分辨率",
                                choices=["832*480", "720P", "1080P"],  # 根据文档提供选项
                                value="832*480",
                                interactive=True
                            )
                            
                            t2v_duration = gr.Slider(
                                label="时长（秒）",
                                minimum=1,
                                maximum=30,
                                step=1,
                                value=10
                            )
                        
                        with gr.Row():
                            t2v_audio_enabled = gr.Checkbox(
                                label="包含音频",
                                value=True
                            )
                        
                        t2v_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        # 分离进度显示和视频显示
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            t2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            t2v_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def process_t2v_request(audio_file, prompt, resolution, duration, audio_enabled):
                    result = generate_video_with_wan25_t2v(prompt, audio_file, resolution, duration, audio_enabled)
                    
                    # 检查是否包含任务ID（表示异步任务）
                    if "任务ID:" in result:
                        # 只返回进度信息，视频输出为None
                        return result, None
                    # 检查是否包含视频URL（表示同步完成）
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                    
                        if video_url:
                            # 尝试下载视频到本地
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                # 本地下载成功，创建HTML视频播放器
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                # 如果下载失败，使用远程URL创建HTML视频播放器
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        # 其他情况返回进度信息，视频输出为None
                        return result, None
                
                t2v_gen_btn.click(
                    fn=process_t2v_request,
                    inputs=[t2v_audio_file, t2v_prompt, t2v_resolution, t2v_duration, t2v_audio_enabled],
                    outputs=[t2v_progress, t2v_output]
                )
            
            with gr.TabItem("任务状态监控"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 任务状态监控")
                        gr.Markdown("实时查看视频生成任务的进度和状态")
                        
                        task_id_monitor = gr.Textbox(
                            label="任务ID",
                            placeholder="请输入要监控的任务ID...",
                            lines=1
                        )
                        
                        monitor_start_btn = gr.Button("开始监控", variant="primary")
                        monitor_stop_btn = gr.Button("停止监控", variant="secondary")
                    
                    with gr.Column():
                        monitor_result = gr.Textbox(
                            label="监控结果",
                            lines=10,
                            interactive=False
                        )
                
                def start_monitoring(task_id):
                    if not task_id or not task_id.strip():
                        return "❌ 请输入有效的任务ID。"
                    
                    return query_video_task(task_id)
                
                monitor_start_btn.click(
                    fn=start_monitoring,
                    inputs=task_id_monitor,
                    outputs=monitor_result
                )
                
                # 添加自动刷新功能
                refresh_btn = gr.Button("刷新状态", variant="secondary")
                
                refresh_btn.click(
                    fn=query_video_task,
                    inputs=task_id_monitor,
                    outputs=monitor_result
                )
                
                # 显示最近任务
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 最近的任务")
                        recent_tasks_btn = gr.Button("加载最近任务", variant="secondary")
                    
                    with gr.Column():
                        recent_tasks_display = gr.Dataframe(
                            headers=["任务ID", "状态", "提交时间", "模型"],
                            datatype=["str", "str", "str", "str"],
                            label="最近任务列表",
                            interactive=False
                        )
                
                def load_recent_tasks():
                    tasks = get_recent_tasks()
                    data = []
                    for task in tasks:
                        data.append([
                            task['task_id'],
                            task['status'],
                            task['submit_time'],
                            task['model']
                        ])
                    return data
                
                recent_tasks_btn.click(
                    fn=load_recent_tasks,
                    inputs=[],
                    outputs=recent_tasks_display
                )
        
        # 添加打开输出目录按钮
        open_output_dir_btn = gr.Button("打开输出目录", variant="secondary")
        open_output_dir_btn.click(
            fn=open_video_output_dir,
            inputs=[],
            outputs=[]
        )

    return qwen_video_interface


# 定义模块可用性标志
QWEN_VIDEO_GEN_AVAILABLE = True  # 根据实际需求设定，如果功能完整则为True