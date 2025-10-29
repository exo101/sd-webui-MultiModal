import os
import sys
import json
import time
import traceback
from pathlib import Path
import subprocess
import gradio as gr
import torch
import psutil
import numpy as np
from PIL import Image
from modules import shared
import shutil

# 获取当前脚本所在目录
current_dir = Path(__file__).parent
scripts_dir = current_dir
qwen_image_dir = current_dir.parent / "qwen-image"
# 修改模型路径为指定目录
models_dir = Path(shared.models_path) / "qwen-image"
qwenimage_models_dir = models_dir / "qwenimage"
qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
qwenimage_lora_dir = Path(shared.models_path) / "Lora"  # 修改为WebUI标准LoRA目录
qwenimage_controlnet_dir = Path(shared.models_path) / "ControlNet"

# 修改图片输出目录为WebUI的outputs目录，用于保存生成的图像
qwen_image_outputs_dir = Path(shared.data_path) / "outputs"
# 确保输出目录存在
qwen_image_outputs_dir.mkdir(parents=True, exist_ok=True)

# 确定主Python解释器路径
main_python = sys.executable

# 添加当前脚本目录到系统路径
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# 获取模型文件列表
def get_model_choices(model_dir):
    """获取指定目录下的模型文件列表"""
    try:
        if not model_dir.exists():
            print(f"警告: 模型目录不存在 {model_dir}")
            # 尝试创建目录
            model_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        # 直接在指定目录查找模型文件，不深入子目录
        model_files = list(model_dir.glob("*.safetensors"))
        
        # 如果没有找到模型文件，尝试查找.pt文件
        if not model_files:
            model_files = list(model_dir.glob("*.pt"))
            
        # 如果仍然没有找到模型文件，尝试查找.bin文件
        if not model_files:
            model_files = list(model_dir.glob("*.bin"))
        
        # 如果仍然没有找到模型文件，尝试查找.ckpt文件
        if not model_files:
            model_files = list(model_dir.glob("*.ckpt"))
        
        # 如果仍然没有找到模型文件，尝试查找.onnx文件
        if not model_files:
            model_files = list(model_dir.glob("*.onnx"))
        
        # 返回 (显示名称, 文件名) 的元组列表
        result = [(f.name, f.name) for f in model_files]
        return result
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        traceback.print_exc()
        return []

# 获取LoRA模型文件列表
def get_lora_choices(lora_dir):
    """获取指定目录下的LoRA模型文件列表"""
    try:
        if not lora_dir.exists():
            print(f"警告: LoRA目录不存在 {lora_dir}")
            # 尝试创建目录
            lora_dir.mkdir(parents=True, exist_ok=True)
            # 返回默认选项"无"
            return [("无", "")]
        
        # 查找LoRA模型文件
        lora_files = list(lora_dir.glob("*.safetensors"))
        
        # 如果没有找到模型文件，尝试查找.pt文件
        if not lora_files:
            lora_files = list(lora_dir.glob("*.pt"))
            
        # 如果仍然没有找到模型文件，尝试查找.bin文件
        if not lora_files:
            lora_files = list(lora_dir.glob("*.bin"))
        
        # 构建选项列表，始终包含"无"选项
        choices = [("无", "")]  # 添加"无"选项作为默认值
        choices.extend([(f.name, str(f)) for f in lora_files])
        
        return choices
    except Exception as e:
        print(f"获取LoRA模型列表时出错: {e}")
        traceback.print_exc()
        # 即使出错也返回默认选项"无"
        return [("无", "")]

# 获取基础模型和编辑模型列表
try:
    qwenimage_model_choices = get_model_choices(qwenimage_models_dir)
    qwenimage_edit_model_choices = get_model_choices(qwenimage_edit_models_dir)
    qwenimage_lora_choices = get_lora_choices(qwenimage_lora_dir)  # 获取LoRA模型列表
except Exception as e:
    print(f"加载模型列表时出错: {e}")
    traceback.print_exc()
    qwenimage_model_choices = []
    qwenimage_edit_model_choices = []
    qwenimage_lora_choices = [("无", "")]  # 默认LoRA选项

# 获取系统信息
def get_system_info():
    """获取当前系统配置信息"""
    try:
        # 获取GPU信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
        else:
            gpu_info = "CPU Only"
        
        # 获取系统内存信息
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024**3)  # GB
        
        return {
            "gpu": gpu_info,
            "memory": f"{total_memory:.0f}GB"
        }
    except:
        # 默认配置信息
        return {
            "gpu": "NVIDIA RTX 4070 Ti",
            "memory": "64GB"
        }

# 尝试导入必要的库
QWEN_IMAGE_AVAILABLE = False
try:
    from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as LightningTransformer
    from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
    from nunchaku.utils import get_gpu_memory, get_precision
    import math
    
    QWEN_IMAGE_AVAILABLE = True
except ImportError as e:
    print(f"Qwen Image 模块导入失败: {e}")
    traceback.print_exc()

def save_numpy_image(image_array, image_path):
    """将numpy数组或PIL图像保存为图像文件"""
    try:
        # 如果是PIL图像对象，直接保存
        if isinstance(image_array, Image.Image):
            image_array.save(str(image_path), 'PNG')
            return str(image_path)
        # 如果是numpy数组，转换为PIL图像后保存
        elif isinstance(image_array, np.ndarray):
            # 确保数组数据类型正确
            if image_array.dtype != np.uint8:
                # 如果是浮点数且范围在0-1之间，转换为0-255
                if image_array.dtype in [np.float32, np.float64] and image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    # 其他情况直接转换为uint8
                    image_array = image_array.astype(np.uint8)
            
            # 使用PIL处理图像转换
            if len(image_array.shape) == 2:
                # 灰度图
                image = Image.fromarray(image_array, mode='L')
            elif len(image_array.shape) == 3:
                if image_array.shape[2] == 1:
                    # 单通道图转灰度图
                    image = Image.fromarray(image_array.squeeze(), mode='L')
                elif image_array.shape[2] == 3:
                    # RGB图像
                    image = Image.fromarray(image_array, mode='RGB')
                elif image_array.shape[2] == 4:
                    # RGBA图像转RGB
                    image = Image.fromarray(image_array, mode='RGBA')
                    image = image.convert('RGB')
                else:
                    # 其他情况默认转RGB
                    image = Image.fromarray(image_array).convert('RGB')
            else:
                # 其他情况默认转RGB
                image = Image.fromarray(image_array).convert('RGB')
            
            # 保存图像
            image.save(str(image_path), 'PNG')
            return str(image_path)
        else:
            print(f"输入不是numpy数组或PIL图像: {type(image_array)}")
            return None
    except Exception as e:
        print(f"保存图像时出错: {e}")
        traceback.print_exc()
        return None

def edit_images(prompt, image1, image2, image3, steps, cfg_scale, negative_prompt,
               model_file, scheduler, lora_model_1="", lora_model_2="", 
               lora_weight_1=1.0, lora_weight_2=1.0, seed=-1):
    try:
        print("开始执行图像编辑功能...")
        if not prompt:
            return None, "编辑指令不能为空", "编辑失败"
        
        # 检查至少有一张图像
        images = [image1, image2, image3]
        uploaded_images = []
        
        # 如果没有上传图像，则从outputs目录中查找最新的图像
        has_uploaded_image = any(img is not None for img in images)
        if not has_uploaded_image:
            # 查找outputs目录中的最新图像文件
            outputs_dir = Path(shared.data_path) / "outputs"
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
            latest_images = []
            image_times = []
            
            # 查找最新的几张图像
            for extension in image_extensions:
                for image_path in outputs_dir.rglob(extension):
                    latest_images.append(str(image_path))
                    image_times.append(image_path.stat().st_mtime)
            
            # 按修改时间排序，获取最新的图像
            if latest_images:
                sorted_images = sorted(zip(latest_images, image_times), key=lambda x: x[1], reverse=True)
                for i, (image_path, _) in enumerate(sorted_images[:3]):  # 最多获取3张最新图像
                    uploaded_images.append(image_path)
                    print(f"从outputs目录自动选择图像: {image_path}")
        else:
            # 处理图像参数，如果它们是numpy数组则保存为临时文件
            for i, img in enumerate(images):
                if img is not None:
                    if isinstance(img, np.ndarray):
                        # 为numpy数组创建临时文件
                        temp_dir = qwen_image_dir / "temp"
                        temp_dir.mkdir(exist_ok=True)
                        temp_image_path = temp_dir / f"edit_image_{i}_{int(time.time() * 1000)}.png"
                        save_result = save_numpy_image(img, temp_image_path)
                        if save_result:
                            uploaded_images.append(str(temp_image_path))
                    else:
                        # 假设是文件路径
                        uploaded_images.append(img)
        
        if len(uploaded_images) == 0:
            return None, "请至少上传一张图像或确保outputs目录中有图像文件", "编辑失败"
        
        # 准备参数
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "images": uploaded_images,  # 传递所有上传的图像
            "steps": steps,
            "cfg_scale": cfg_scale,
            "model_file": model_file,  # 只传递模型文件名
            "model_dir": str(qwenimage_edit_models_dir),  # 确保传递正确的编辑模型目录路径
            "scheduler": scheduler,  # 添加采样方法参数
            "lora_model_1": lora_model_1 if lora_model_1 else None,
            "lora_model_2": lora_model_2 if lora_model_2 else None,
            "lora_weight_1": lora_weight_1,
            "lora_weight_2": lora_weight_2,
            "seed": seed,
            "output_dir": str(qwen_image_outputs_dir)
        }
        
        # 创建临时参数文件
        args_file = qwen_image_dir / "temp_args.json"
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False, indent=2)
        
        # 构建命令 - 使用原始字符串并正确处理路径
        args_file_str = str(args_file).replace('\\', '/')
        scripts_dir_str = str(scripts_dir).replace('\\', '/')
        
        cmd = [
            main_python,
            "-c",
            f"import sys; sys.path.append('{scripts_dir_str}'); from qwen_image_scripts import run_image_editing; run_image_editing('{args_file_str}')"
        ]
        
        # 执行命令
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {qwen_image_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(qwen_image_dir), timeout=300)
        
        # 删除临时参数文件
        if args_file.exists():
            args_file.unlink()
        
        print(f"返回码: {result.returncode}")
        print(f"标准输出: {result.stdout}")
        print(f"错误输出: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = f"编辑失败: 错误代码 {result.returncode}\n标准输出: {result.stdout}\n错误输出: {result.stderr}"
            print(f"编辑失败: {error_msg}")
            return None, error_msg, "编辑失败"
            
        # 解析成功输出
        output_info = parse_script_output(result.stdout)
        if "image_path" in output_info:
            output_path = output_info["image_path"]
            # 删除与生成记录表格相关的代码
            print("图像编辑成功")
            return output_path, "编辑成功", "编辑完成"  # 简化返回值
        else:
            error_msg = f"编辑失败: {result.stdout}"
            print(f"编辑失败: {error_msg}")
            return None, error_msg, "编辑失败"
            
    except Exception as e:
        error_msg = f"编辑失败: {str(e)}"
        print(f"编辑过程中出现异常: {error_msg}")
        traceback.print_exc()
        return None, error_msg, "编辑失败"

def parse_script_output(output):
    """解析脚本输出，提取图像路径和信息文件路径"""
    try:
        lines = output.strip().split('\n')
        result = {}
        
        for line in lines:
            if line.startswith("SUCCESS:"):
                result["image_path"] = line[8:].strip()
            elif line.startswith("INFO_FILE:"):
                result["info_file"] = line[10:].strip()
        
        return result
    except Exception as e:
        print(f"解析脚本输出时出错: {e}")
        traceback.print_exc()
        return {}

def create_qwen_image_edit_ui():
    """创建Qwen图像编辑UI模块"""
    try:
        print("开始创建Qwen Image Edit UI...")
        if not QWEN_IMAGE_AVAILABLE:
            print("Qwen Image 模块不可用")
            with gr.Row():
                gr.Markdown("""## Qwen Image 模型不可用
                
                请确保已安装所需的依赖项:
                ```
                pip install nunchaku diffusers>=0.36.0.dev0 transformers>=4.53.3 accelerate>=1.9.0
                ```
                """)
            return {}
        
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    edit_prompt = gr.TextArea(
                        label="编辑指令",
                        placeholder="输入您的编辑指令，描述想要进行的编辑操作..."
                    )
                    
                    # 添加负面提示词输入框到编辑指令下方
                    edit_negative_prompt = gr.Textbox(
                        label="负面提示词 (Negative Prompt)",
                        value="",
                        max_lines=3,
                        placeholder="输入不希望出现在图像中的内容，例如：丑陋、拼合、多余的肢体、畸形、变形、身体超出画面、水印、截断、对比度低、曝光不足、曝光过度、糟糕的艺术、面部扭曲、模糊、颗粒感",
                        interactive=True,
                        elem_classes=["negative_prompt"]
                    )
                    
                    with gr.Row():
                        edit_image1 = gr.Image(type="filepath", label="图像1", interactive=True)
                        edit_image2 = gr.Image(type="filepath", label="图像2", interactive=True)
                        edit_image3 = gr.Image(type="filepath", label="图像3", interactive=True)
                    
                    with gr.Row():
                        edit_steps = gr.Slider(
                            minimum=1, maximum=50, step=1, value=8,
                            label="推理步数",
                            min_width=80
                        )
                        
                        edit_cfg = gr.Slider(
                            minimum=1.0, maximum=20.0, step=0.1, value=4.0,
                            label="CFG Scale",
                            min_width=80
                        )
                        
                        # 添加采样方法选择组件
                        edit_scheduler = gr.Dropdown(
                            choices=[
                                ("Euler", "euler"),
                                ("Euler Ancestral", "euler_ancestral"),
                                ("Heun", "heun"),
                                ("DPM++ 2M", "dpmpp_2m")
                            ],
                            value="euler",
                            label="采样方法",
                            min_width=120
                        )
                        
                        # Add base model selection dropdown
                        edit_model = gr.Dropdown(
                            choices=qwenimage_edit_model_choices,
                            label="基础模型选择",
                            value=qwenimage_edit_model_choices[0][1] if qwenimage_edit_model_choices else None,
                            interactive=True,
                            min_width=150
                        )
                    
                    # 添加LoRA模型选择组件
                    with gr.Row():
                        edit_lora_1 = gr.Dropdown(
                            choices=qwenimage_lora_choices,
                            label="LoRA模型 1",
                            value="",  # 默认选择"无"
                            interactive=True,
                            min_width=150
                        )
                        
                        edit_lora_2 = gr.Dropdown(
                            choices=qwenimage_lora_choices,
                            label="LoRA模型 2",
                            value="",  # 默认选择"无"
                            interactive=True,
                            min_width=150
                        )
                    
                    # 添加LoRA权重调节滑块
                    with gr.Row():
                        edit_lora_weight_1 = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                            label="LoRA 1 强度",
                            min_width=120
                        )
                        
                        edit_lora_weight_2 = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                            label="LoRA 2 强度",
                            min_width=120
                        )
                    
                    # 添加随机种子组件
                    with gr.Row():
                        edit_seed = gr.Number(
                            label="随机种子 (-1为随机)",
                            value=-1,
                            precision=0,
                            min_width=120
                        )
                    
                    # 编辑按钮
                    edit_button = gr.Button("编辑图像")
                
                # 结束左侧列
                with gr.Column():
                    # 调整图像组件的显示尺寸
                    edit_output = gr.Image(label="编辑结果", interactive=False, height=512)
                    edit_status = gr.Textbox(label="状态", interactive=False)
        
        edit_button.click(
            fn=edit_images,  # 修复：将未定义的run_image_edit改为已定义的edit_images函数
            inputs=[
                edit_prompt,
                edit_negative_prompt,
                edit_image1,
                edit_image2,
                edit_image3,
                edit_steps,
                edit_cfg,
                edit_model,
                edit_scheduler,
                edit_lora_1,
                edit_lora_2,
                edit_lora_weight_1,
                edit_lora_weight_2,
                edit_seed
            ],
            outputs=[edit_output, edit_status]  # 移除生成记录输出
        )
        
        # 返回UI组件字典，以便在主程序中引用
        result = {
            "edit_prompt": edit_prompt,
            "edit_image1": edit_image1,
            "edit_image2": edit_image2,
            "edit_image3": edit_image3,
            "edit_steps": edit_steps,
            "edit_model": edit_model,
            "edit_cfg": edit_cfg,
            "edit_negative_prompt": edit_negative_prompt,
            "edit_scheduler": edit_scheduler,
            "edit_lora_1": edit_lora_1,
            "edit_lora_2": edit_lora_2,
            "edit_lora_weight_1": edit_lora_weight_1,
            "edit_lora_weight_2": edit_lora_weight_2,
            "edit_seed": edit_seed,
            "edit_button": edit_button,
            "edit_output": edit_output,
            "edit_status": edit_status
        }
        
        print("Qwen Image Edit UI 创建完成")
        return result
        
    except Exception as e:
        print(f"创建Qwen Image Edit UI时出错: {e}")
        traceback.print_exc()
        # 返回空字典而不是None，避免破坏UI
        return {}

# 定义模块可用性变量
QWEN_IMAGE_EDIT_MODULE_AVAILABLE = QWEN_IMAGE_AVAILABLE