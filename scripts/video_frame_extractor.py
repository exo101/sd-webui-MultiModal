import gradio as gr
import os
import cv2
import numpy as np
from datetime import datetime

def create_video_frame_extractor():
    """创建视频分帧提取功能组件"""
    
    def extract_video_frames(video, num_frames, quality, mode):
        """提取视频关键帧并保存为图片"""
        if video is None:
            return [], []
        
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), "video_frames")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join(save_dir, f"video_{timestamp}")
        os.makedirs(video_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算要提取的帧位置
        frames_to_extract = []
        if mode == "uniform":
            for i in range(int(num_frames)):
                frame_pos = int(i * total_frames / num_frames)
                frames_to_extract.append(frame_pos)
        elif mode == "interval":
            interval = int(total_frames / num_frames)
            frames_to_extract = [i * interval for i in range(num_frames)]
        elif mode == "change_detection":
            prev_frame = None
            change_frames = []
            frame_count = 0
            while True and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    non_zero = np.count_nonzero(diff)
                    if non_zero > 1000:  # 阈值可根据需求调整
                        change_frames.append(frame_count)
                
                prev_frame = frame.copy()
                frame_count += 1
            
            # 取变化帧的前几帧作为关键帧
            frames_to_extract = change_frames[:int(num_frames)]
        
        # 重置视频读取器
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 提取并保存帧
        extracted_images = []
        preview_images = []
        saved_count = 0
        for frame_pos in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 构建文件路径
                filename = os.path.join(video_dir, f"frame_{saved_count:04d}.jpg")
                
                # 保存帧
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                # 添加到结果列表
                extracted_images.append(filename)
                
                # 缩小用于预览
                height, width = frame.shape[:2]
                preview_size = (800, int(800 * height / width)) if width > height else (int(800 * width / height), 800)
                preview = cv2.resize(frame, preview_size)
                preview_images.append(preview)
                
                saved_count += 1
        
        cap.release()
        
        return extracted_images, preview_images
    
    # 创建视频输入组件
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="上传视频")  # 删除 type="filepath" 避免警告
            
            # 创建参数输入区域
            with gr.Row():
                frame_output = gr.Number(label="提取关键帧数量", value=10, precision=0)
                frame_quality = gr.Slider(label="帧质量", minimum=1, maximum=100, value=85, step=1)
            
            # 创建模式选择
            frame_mode = gr.Radio(
                label="提取模式",
                choices=[
                    ("均匀分布", "uniform"), 
                    ("固定间隔", "interval"), 
                    ("变化检测", "change_detection")
                ],
                value="uniform"
            )
    
    # 创建预览区域
    frame_preview = gr.Gallery(label="帧预览", columns=5, height=200, visible=True, object_fit="contain")
    
    # 返回所有创建的组件和函数
    return {
        "video_input": video_input,
        "frame_output": frame_output,
        "frame_quality": frame_quality,
        "frame_mode": frame_mode,
        "frame_preview": frame_preview,
        "extract_video_frames": extract_video_frames
    }