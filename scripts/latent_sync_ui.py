import os
import sys
import gradio as gr
from pathlib import Path
from datetime import datetime
import time
import subprocess
import warnings

# 设置初始化完成标记
LATENT_SYNC_UI_INITIALIZED = True

# 添加LatentSync到Python路径
current_dir = Path(__file__).parent.parent
latentsync_path = current_dir / "LatentSync"
latentsync_module_path = latentsync_path / "latentsync"
checkpoints_path = latentsync_path / "checkpoints"
auxiliary_path = checkpoints_path / "auxiliary"
# 插件Python环境路径已删除，因为使用webui环境

# 设置InsightFace模型路径
def setup_insightface_path():
    """设置InsightFace模型路径以确保人脸检测功能正常工作"""
    # 设置INSIGHTFACE_ROOT环境变量
    if auxiliary_path.exists():
        os.environ["INSIGHTFACE_ROOT"] = str(auxiliary_path.resolve())
    sys.path.insert(0, str(auxiliary_path / "facedetector"))

# 在初始化时设置InsightFace路径（在任何导入之前）
setup_insightface_path()

# FFmpeg路径设置
def setup_ffmpeg_path():
    """设置FFmpeg路径以确保Gradio和系统可以找到ffmpeg和ffprobe"""
    # 检查ffmpeg是否已经在PATH中
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, check=True)
        if result.returncode == 0:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # 常见的FFmpeg安装路径 (已删除XYKC_AI相关路径)
    common_ffmpeg_paths = [
        "C:\\ffmpeg\\bin",
        "C:\\Program Files\\ffmpeg\\bin",
        "C:\\Program Files (x86)\\ffmpeg\\bin",
    ]
    
    # 尝试添加常见路径
    system_path = os.environ.get("PATH", "")
    for ffmpeg_path in common_ffmpeg_paths:
        if os.path.exists(ffmpeg_path):
            ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
            ffprobe_exe = os.path.join(ffmpeg_path, "ffprobe.exe")
            if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
                # 将路径添加到系统PATH
                if system_path:
                    os.environ["PATH"] = ffmpeg_path + os.pathsep + system_path
                else:
                    os.environ["PATH"] = ffmpeg_path
                return True
    
    return False

# 在初始化时设置FFmpeg路径
setup_ffmpeg_path()

# 将LatentSync和其子模块添加到Python路径开头以确保优先级
sys.path.insert(0, str(latentsync_path))
sys.path.insert(0, str(latentsync_module_path))

# 导入LatentSync的推理函数
try:
    from LatentSync.scripts.inference import main as inference_main
    main_func = inference_main
except ImportError as e:
    main_func = None

# 模型配置
MODEL_CONFIGS = {
    "LatentSync": {
        "config_path": latentsync_path / "configs/unet/stage2.yaml",
        "checkpoint_path": latentsync_path / "checkpoints/latentsync_unet.pt"
    }
}

TEMP_DIR = latentsync_path / "temp"

def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int, model_name: str
):
    """创建参数对象"""
    model_config = MODEL_CONFIGS[model_name]
    checkpoint_path = model_config["checkpoint_path"]
    config_path = model_config["config_path"]
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2_512.yaml")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            checkpoint_path.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            str(TEMP_DIR),
            "--unet_config_path",
            config_path.absolute().as_posix(),
        ]
    )


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
    model_name,
):
    """处理视频和音频生成数字人视频"""
    try:
        # 检查输入文件是否存在
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError("视频文件不存在，请选择有效的视频文件")
        
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError("音频文件不存在，请选择有效的音频文件")
        
        # 创建临时目录
        output_dir = latentsync_path / "temp"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"无法创建临时目录: {str(e)}。请检查磁盘空间和写入权限") from e

        # 转换路径为绝对路径
        video_file_path = Path(video_path)
        video_path = video_file_path.absolute().as_posix()
        audio_path = Path(audio_path).absolute().as_posix()

        # 设置输出路径
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")
        
        # 获取模型配置
        model_config = MODEL_CONFIGS[model_name]
        
        # 创建参数对象而不是直接构建命令行参数
        args = create_args(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model_name=model_name
        )
        
        # 导入推理函数并执行
        try:
            from LatentSync.scripts.inference import main as inference_main
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError("无法导入必要的模块，请确保LatentSync正确安装") from e
        
        # 确保使用正确的配置路径
        args.unet_config_path = (latentsync_path / args.unet_config_path).absolute().as_posix()
        
        try:
            config = OmegaConf.load(args.unet_config_path)
            inference_main(config, args)
        except Exception as e:
            error_msg = str(e)
            # 提供更具体的错误信息
            if "Face not detected" in error_msg:
                error_msg = ("处理视频时出错: 未检测到人脸。\n"
                            "请确保视频中包含清晰可见的正面人脸，避免以下情况：\n"
                            "1. 人脸太小或距离摄像头太远\n"
                            "2. 人脸被遮挡（如手、头发等）\n"
                            "3. 人脸处于极端角度（如侧脸、低头、仰头等）\n"
                            "4. 视频光线太暗或对比度太低\n"
                            "5. 人脸在视频中出现时间太短\n"
                            "建议使用包含清晰正面人脸的视频进行处理。")
            elif "CUDA out of memory" in error_msg:
                error_msg = ("显存不足：\n"
                            "1. 尝试降低视频分辨率\n"
                            "2. 减少推理步数\n"
                            "3. 使用更小的模型版本\n"
                            "4. 关闭其他占用显存的应用程序")
            elif "FFmpeg" in error_msg or "ffmpeg" in error_msg:
                error_msg = ("FFmpeg错误：\n"
                            "1. 确保正确安装了FFmpeg\n"
                            "2. 检查视频文件格式是否支持\n"
                            "3. 确保视频文件未损坏")
            elif "Invalid audio format" in error_msg:
                error_msg = ("音频格式错误：\n"
                            "1. 请使用标准音频格式（如WAV、MP3）\n"
                            "2. 确保音频文件未损坏\n"
                            "3. 尝试转换音频格式后重试")
            elif "Video and audio duration mismatch" in error_msg:
                error_msg = ("视频和音频时长不匹配：\n"
                            "1. 确保视频和音频文件的时长相近\n"
                            "2. 截取或裁剪较长的文件以匹配时长")
            elif "Illegal byte sequence" in error_msg:
                error_msg = ("文件路径包含特殊字符导致无法读取：\n"
                            "1. 请确保视频和音频文件路径不包含中文或特殊字符\n"
                            "2. 将文件移动到纯英文路径下再尝试处理")
            elif "Face not detected in any frame" in error_msg:
                error_msg = ("未检测到人脸：\n"
                            "请确保视频中包含清晰可见的正面人脸，避免以下情况：\n"
                            "1. 人脸太小或距离摄像头太远\n"
                            "2. 人脸被遮挡（如手、头发等）\n"
                            "3. 人脸处于极端角度（如侧脸、低头、仰头等）\n"
                            "4. 视频光线太暗或对比度太低\n"
                            "5. 人脸在视频中出现时间太短\n"
                            "建议使用包含清晰正面人脸的视频进行处理。")
            
            raise Exception(error_msg) from e
        
        # 检查输出文件是否存在
        if not os.path.exists(output_path):
            raise Exception("处理完成但未生成输出文件。可能原因：\n"
                          "1. 处理过程中发生未知错误\n"
                          "2. 程序异常终止\n"
                          "3. 磁盘空间不足\n"
                          "4. 写入权限问题")
            
        return output_path
    except Exception as e:
        error_msg = f"处理视频时出错: {str(e)}"
        # 如果是显存不足错误，添加额外提示
        if "CUDA out of memory" in str(e):
            error_msg += "\n提示：尝试降低视频分辨率或使用更小的模型版本"
        raise gr.Error(error_msg)


def create_latent_sync_ui():
    """创建LatentSync用户界面"""
    with gr.Group():
        gr.Markdown("## 数字人视频生成")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="输入视频")
                audio_input = gr.Audio(label="输入音频", type="filepath")
                
                # 添加模型选择下拉框
                model_choice = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value=list(MODEL_CONFIGS.keys())[0],  # 默认选择第一个模型
                    label="选择模型",
                    info="8GB显存选择LatentSync 1.5，18GB显存选择LatentSync 1.6"
                )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        label="引导尺度",
                    )
                    inference_steps = gr.Slider(
                        minimum=10, 
                        maximum=50, 
                        value=20, 
                        step=1, 
                        label="推理步数"
                    )

                with gr.Row():
                    seed = gr.Number(value=1247, label="随机种子", precision=0)

                process_btn = gr.Button("生成数字人视频", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="输出视频")

                # 示例
                gr.Examples(
                    examples=[
                        [str(latentsync_path / "assets" / "demo1_video.mp4"), 
                         str(latentsync_path / "assets" / "demo1_audio.wav"),
                         list(MODEL_CONFIGS.keys())[0]],
                        [str(latentsync_path / "assets" / "demo2_video.mp4"), 
                         str(latentsync_path / "assets" / "demo2_audio.wav"),
                         list(MODEL_CONFIGS.keys())[0]],
                    ],
                    inputs=[video_input, audio_input, model_choice],
                )

        # 绑定事件
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                audio_input,
                guidance_scale,
                inference_steps,
                seed,
                model_choice,
            ],
            outputs=video_output,
        )
        
        return {
            "video_input": video_input,
            "audio_input": audio_input,
            "model_choice": model_choice,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "seed": seed,
            "process_btn": process_btn,
            "video_output": video_output
        }