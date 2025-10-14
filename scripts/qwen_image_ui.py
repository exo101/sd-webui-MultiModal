import os
import gradio as gr
import torch
import sys
from pathlib import Path
import subprocess
import json
from modules import shared
import glob
import time
import psutil
import traceback

# 添加当前脚本目录到系统路径
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# 获取模型路径
qwen_image_dir = Path(__file__).parent.parent / "qwen-image"
models_dir = qwen_image_dir / "models"
qwenimage_models_dir = models_dir / "qwenimage"
qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
qwenimage_lora_dir = qwen_image_dir / "loras"
qwen_image_outputs_dir = qwen_image_dir / "outputs"
qwen_image_outputs_dir.mkdir(exist_ok=True)

# 使用主环境的 Python 解释器
main_python = sys.executable

# 获取模型文件列表
def get_model_choices(model_dir):
    """获取指定目录下的模型文件列表"""
    try:
        if not model_dir.exists():
            print(f"警告: 模型目录不存在 {model_dir}")
            return []
        
        model_files = list(model_dir.glob("*.safetensors"))
        # 返回 (显示名称, 文件名) 的元组列表
        result = [(f.name, f.name) for f in model_files]
        return result
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        traceback.print_exc()
        return []

# 获取子目录中的模型文件列表
def get_lora_model_choices(lora_dir):
    """获取LoRA目录下的模型文件列表，包括子目录"""
    try:
        if not lora_dir.exists():
            print(f"警告: LoRA目录不存在 {lora_dir}")
            return []
        
        lora_files = []
        # 查找根目录下的LoRA文件
        for f in lora_dir.glob("*.safetensors"):
            lora_files.append((f.name, f.name))
        
        # 查找子目录中的LoRA文件
        for subdir in lora_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.glob("*.safetensors"):
                    # 显示名称包含子目录名，文件路径相对于lora_dir
                    display_name = f"{subdir.name}/{f.name}"
                    relative_path = f"{subdir.name}/{f.name}"
                    lora_files.append((display_name, relative_path))
        
        return lora_files
    except Exception as e:
        print(f"获取LoRA模型列表时出错: {e}")
        traceback.print_exc()
        return []

# 获取基础模型、编辑模型和LoRA模型列表
try:
    qwenimage_model_choices = get_model_choices(qwenimage_models_dir)
    qwenimage_edit_model_choices = get_model_choices(qwenimage_edit_models_dir)
    qwenimage_lora_choices = get_lora_model_choices(qwenimage_lora_dir)
except Exception as e:
    print(f"加载模型列表时出错: {e}")
    traceback.print_exc()
    qwenimage_model_choices = []
    qwenimage_edit_model_choices = []
    qwenimage_lora_choices = []

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

def format_generation_info(info_dict):
    """格式化生成信息为Markdown表格"""
    try:
        if not info_dict:
            return "暂无生成记录"
        
        markdown = "### 生成信息\n\n"
        markdown += "| 参数 | 值 |\n"
        markdown += "|------|-----|\n"
        
        # 显示顺序：硬件配置、生成参数、模型信息、生成时间
        key_order = [
            "GPU配置", "系统内存", 
            "推理步数", "提示词引导系数 (CFG Scale)", "宽度", "高度",
            "模型类型", "模型文件", "生成时间"
        ]
        
        # 按指定顺序显示信息
        for key in key_order:
            if key in info_dict:
                markdown += f"| {key} | {info_dict[key]} |\n"
        
        # 显示其他未排序的信息
        for key, value in info_dict.items():
            if key not in key_order:
                markdown += f"| {key} | {value} |\n"
        
        return markdown
    except Exception as e:
        print(f"格式化生成信息时出错: {e}")
        traceback.print_exc()
        return f"格式化生成信息失败: {str(e)}"

def get_latest_generation_info():
    """获取最新的生成信息"""
    try:
        # 查找最新的info文件
        info_files = list(qwen_image_outputs_dir.glob("qwen_image_info_*.json"))
        info_files.extend(qwen_image_outputs_dir.glob("qwen_image_edit_info_*.json"))
        
        if not info_files:
            return "暂无生成记录"
        
        # 按修改时间排序，获取最新的文件
        latest_file = max(info_files, key=lambda f: f.stat().st_mtime)
        
        # 读取信息
        with open(latest_file, 'r', encoding='utf-8') as f:
            info_dict = json.load(f)
        
        return format_generation_info(info_dict)
    except Exception as e:
        print(f"获取最新生成信息时出错: {e}")
        traceback.print_exc()
        return f"获取生成信息失败: {str(e)}"

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

def generate_text_to_image(prompt, width, height, steps, cfg_scale, model_file,
                          lora_enable, lora_model, lora_scale,
                          lora_enable_2, lora_model_2, lora_scale_2,
                          lora_enable_3, lora_model_3, lora_scale_3,
                          scheduler, negative_prompt):
    try:
        print("开始执行文生图功能...")
        # 准备参数
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "model_file": model_file,
            "scheduler": scheduler,
            "lora_enable": lora_enable,
            "lora_model": lora_model if lora_enable else None,
            "lora_scale": lora_scale if lora_enable else 0.0,
            "lora_enable_2": lora_enable_2,
            "lora_model_2": lora_model_2 if lora_enable_2 else None,
            "lora_scale_2": lora_scale_2 if lora_enable_2 else 0.0,
            "lora_enable_3": lora_enable_3,
            "lora_model_3": lora_model_3 if lora_enable_3 else None,
            "lora_scale_3": lora_scale_3 if lora_enable_3 else 0.0,
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
            f"import sys; sys.path.append('{scripts_dir_str}'); from qwen_image_scripts import run_text_to_image; run_text_to_image('{args_file_str}')"
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
            error_msg = f"生成失败: 错误代码 {result.returncode}\n标准输出: {result.stdout}\n错误输出: {result.stderr}"
            print(f"生成失败: {error_msg}")
            return None, error_msg, "暂无生成记录"
            
        # 解析成功输出
        output_info = parse_script_output(result.stdout)
        if "image_path" in output_info:
            output_path = output_info["image_path"]
            if "info_file" in output_info:
                # 读取生成信息
                try:
                    with open(output_info["info_file"], 'r', encoding='utf-8') as f:
                        info_dict = json.load(f)
                    info_markdown = format_generation_info(info_dict)
                except:
                    info_markdown = "获取生成信息失败"
            else:
                info_markdown = get_latest_generation_info()
            print("文生图生成成功")
            return output_path, "生成成功", info_markdown
        else:
            error_msg = f"生成失败: {result.stdout}"
            print(f"生成失败: {error_msg}")
            return None, error_msg, "暂无生成记录"
            
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"生成过程中出现异常: {error_msg}")
        traceback.print_exc()
        return None, error_msg, "暂无生成记录"

def edit_images(prompt, image1, image2, image3, steps, cfg_scale, negative_prompt,
               model_file, lora_enable, lora_model, lora_scale,
               lora_enable_2, lora_model_2, lora_scale_2,
               lora_enable_3, lora_model_3, lora_scale_3,
               scheduler):
    try:
        print("开始执行图像编辑功能...")
        if not prompt:
            return None, "编辑指令不能为空", "暂无生成记录"
        
        # 检查至少有一张图像
        images = [image1, image2, image3]
        uploaded_images = [img for img in images if img is not None]
        
        if len(uploaded_images) == 0:
            return None, "请至少上传一张图像", "暂无生成记录"
        
        # 准备参数
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "images": uploaded_images,  # 传递所有上传的图像
            "steps": steps,
            "cfg_scale": cfg_scale,
            "model_file": model_file,  # 添加模型文件参数
            "scheduler": scheduler,  # 添加采样方法参数
            "lora_enable": lora_enable,  # 添加LoRA启用参数
            "lora_model": lora_model if lora_enable else None,  # 添加LoRA模型参数
            "lora_scale": lora_scale if lora_enable else 0.0,  # 添加LoRA权重参数
            "lora_enable_2": lora_enable_2,  # 添加第二个LoRA启用参数
            "lora_model_2": lora_model_2 if lora_enable_2 else None,  # 添加第二个LoRA模型参数
            "lora_scale_2": lora_scale_2 if lora_enable_2 else 0.0,  # 添加第二个LoRA权重参数
            "lora_enable_3": lora_enable_3,  # 添加第三个LoRA启用参数
            "lora_model_3": lora_model_3 if lora_enable_3 else None,  # 添加第三个LoRA模型参数
            "lora_scale_3": lora_scale_3 if lora_enable_3 else 0.0,  # 添加第三个LoRA权重参数
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
            return None, error_msg, "暂无生成记录"
            
        # 解析成功输出
        output_info = parse_script_output(result.stdout)
        if "image_path" in output_info:
            output_path = output_info["image_path"]
            if "info_file" in output_info:
                # 读取生成信息
                try:
                    with open(output_info["info_file"], 'r', encoding='utf-8') as f:
                        info_dict = json.load(f)
                    info_markdown = format_generation_info(info_dict)
                except:
                    info_markdown = "获取生成信息失败"
            else:
                info_markdown = get_latest_generation_info()
            print("图像编辑成功")
            return output_path, "编辑成功", info_markdown
        else:
            error_msg = f"编辑失败: {result.stdout}"
            print(f"编辑失败: {error_msg}")
            return None, error_msg, "暂无生成记录"
            
    except Exception as e:
        error_msg = f"编辑失败: {str(e)}"
        print(f"编辑过程中出现异常: {error_msg}")
        traceback.print_exc()
        return None, error_msg, "暂无生成记录"

def create_qwen_image_ui():
    try:
        print("开始创建Qwen Image UI...")
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
        
        with gr.Tabs():
            # 文生图标签页
            with gr.TabItem("文生图"):
                with gr.Row():
                    with gr.Column():
                        text_to_image_prompt = gr.TextArea(
                            label="提示词",
                            placeholder="输入您的提示词，描述想要生成的图像..."
                        )
                        
                        # 添加负面提示词输入框到正面提示词下方
                        text_to_image_negative_prompt = gr.Textbox(
                            label="负面提示词 (Negative Prompt)",
                            value="",
                            max_lines=3,
                            placeholder="输入不希望出现在图像中的内容，例如：丑陋、拼贴、多余的肢体、畸形、变形、身体超出画面、水印、截断、对比度低、曝光不足、曝光过度、糟糕的艺术、面部扭曲、模糊、颗粒感",
                            interactive=True,
                            elem_classes=["negative_prompt"]
                        )
                        
                        with gr.Row():
                            text_to_image_width = gr.Slider(
                                minimum=256, maximum=2048, step=64, value=1024, label="宽度"
                            )
                            text_to_image_height = gr.Slider(
                                minimum=256, maximum=2048, step=64, value=1024, label="高度"
                            )
                        
                        with gr.Row():
                            text_to_image_steps = gr.Number(
                                value=8, 
                                label="推理步数",
                                precision=0,
                                interactive=True,
                                min_width=80
                            )
                            
                            text_to_image_cfg = gr.Number(
                                value=4.0,
                                label="CFG Scale",
                                precision=1,
                                min_width=80
                            )
                        
                        with gr.Row():
                            # 添加采样方法选择组件
                            text_to_image_scheduler = gr.Dropdown(
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
                            text_to_image_model = gr.Dropdown(
                                choices=qwenimage_model_choices,
                                label="基础模型选择",
                                value=qwenimage_model_choices[0][1] if qwenimage_model_choices else None,
                                interactive=True,
                                min_width=150
                            )
                        
                        # 生成按钮
                        text_to_image_button = gr.Button("生成图像")
                    
                    # 将输出组件放在右侧列中（在按钮点击事件之前定义）
                    with gr.Column():
                        # 调整图像组件的显示尺寸
                        text_to_image_output = gr.Image(label="生成结果", interactive=False, height=512)
                        text_to_image_status = gr.Textbox(label="状态", interactive=False)
                        
                        # 添加记录折叠模块
                        with gr.Accordion("生成记录", open=False):
                            text_to_image_info = gr.Markdown(get_latest_generation_info())
            
            # 图像编辑标签页
            with gr.TabItem("图像编辑"):
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
                            placeholder="输入不希望出现在图像中的内容，例如：丑陋、拼贴、多余的肢体、畸形、变形、身体超出画面、水印、截断、对比度低、曝光不足、曝光过度、糟糕的艺术、面部扭曲、模糊、颗粒感",
                            interactive=True,
                            elem_classes=["negative_prompt"]
                        )
                        
                        with gr.Row():
                            edit_image1 = gr.Image(type="filepath", label="图像1", interactive=True)
                            edit_image2 = gr.Image(type="filepath", label="图像2", interactive=True)
                            edit_image3 = gr.Image(type="filepath", label="图像3", interactive=True)
                        
                        with gr.Row():
                            edit_steps = gr.Number(
                                value=8, 
                                label="推理步数",
                                precision=0,
                                interactive=True,
                                min_width=80
                            )
                            
                            edit_cfg = gr.Number(
                                value=4.0,
                                label="CFG Scale",
                                precision=1,
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
                        
                        # 编辑按钮
                        edit_button = gr.Button("编辑图像")
                    
                    # 结束左侧列
                    with gr.Column():
                        # 调整图像组件的显示尺寸
                        edit_output = gr.Image(label="编辑结果", interactive=False, height=512)
                        edit_status = gr.Textbox(label="状态", interactive=False)
                        
                        # 添加记录折叠模块
                        with gr.Accordion("生成记录", open=False):
                            edit_info = gr.Markdown(get_latest_generation_info())
        
        # 设置事件处理
        text_to_image_button.click(
            fn=generate_text_to_image,
            inputs=[
                text_to_image_prompt,
                text_to_image_width,
                text_to_image_height,
                text_to_image_steps,
                text_to_image_cfg,
                text_to_image_model,
                text_to_image_scheduler,
                text_to_image_negative_prompt
            ],
            outputs=[text_to_image_output, text_to_image_status, text_to_image_info]
        )
        
        edit_button.click(
            fn=edit_images,
            inputs=[
                edit_prompt,
                edit_image1,
                edit_image2,
                edit_image3,
                edit_steps,
                edit_cfg,
                edit_negative_prompt,
                edit_model,
                edit_scheduler
            ],
            outputs=[edit_output, edit_status, edit_info]
        )
        
        # 返回UI组件字典，以便在主程序中引用
        result = {
            "text_to_image_prompt": text_to_image_prompt,
            "text_to_image_width": text_to_image_width,
            "text_to_image_height": text_to_image_height,
            "text_to_image_steps": text_to_image_steps,
            "text_to_image_model": text_to_image_model,
            "text_to_image_cfg": text_to_image_cfg,
            "text_to_image_scheduler": text_to_image_scheduler,
            "text_to_image_negative_prompt": text_to_image_negative_prompt,
            "text_to_image_button": text_to_image_button,
            "text_to_image_output": text_to_image_output,
            "text_to_image_status": text_to_image_status,
            "text_to_image_info": text_to_image_info,
            "edit_prompt": edit_prompt,
            "edit_image1": edit_image1,
            "edit_image2": edit_image2,
            "edit_image3": edit_image3,
            "edit_steps": edit_steps,
            "edit_model": edit_model,
            "edit_cfg": edit_cfg,
            "edit_negative_prompt": edit_negative_prompt,
            "edit_scheduler": edit_scheduler,
            "edit_button": edit_button,
            "edit_output": edit_output,
            "edit_status": edit_status,
            "edit_info": edit_info
        }
        
        print("Qwen Image UI 创建完成")
        return result
        
    except Exception as e:
        print(f"创建Qwen Image UI时出错: {e}")
        traceback.print_exc()
        # 返回空字典而不是None，避免破坏UI
        return {}

# 定义模块可用性变量
QWEN_IMAGE_MODULE_AVAILABLE = QWEN_IMAGE_AVAILABLE

