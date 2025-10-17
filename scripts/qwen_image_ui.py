import os
import sys
import json
import time
import traceback
from pathlib import Path
import subprocess
import gradio as gr
import torch
import glob
import psutil
from modules import script_callbacks, shared
from modules.paths import script_path
import shutil

# 获取当前脚本所在目录
current_dir = Path(__file__).parent
scripts_dir = current_dir
qwen_image_dir = current_dir.parent / "qwen-image"
qwen_image_outputs_dir = qwen_image_dir / "outputs"

# 确保输出目录存在
qwen_image_outputs_dir.mkdir(parents=True, exist_ok=True)

# 确定主Python解释器路径
main_python = sys.executable

# 添加当前脚本目录到系统路径
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# 获取模型路径
models_dir = qwen_image_dir / "models"
qwenimage_models_dir = models_dir / "qwenimage"
qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
qwenimage_lora_dir = qwen_image_dir / "loras"
qwenimage_controlnet_dir = models_dir / "controlnet"

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

# 获取基础模型和编辑模型列表
try:
    qwenimage_model_choices = get_model_choices(qwenimage_models_dir)
    qwenimage_edit_model_choices = get_model_choices(qwenimage_edit_models_dir)
except Exception as e:
    print(f"加载模型列表时出错: {e}")
    traceback.print_exc()
    qwenimage_model_choices = []
    qwenimage_edit_model_choices = []

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

# 检查ControlNet是否可用
try:
    from diffusers.models import QwenImageControlNetModel
    CONTROLNET_AVAILABLE = True
    print("ControlNet功能可用")
except ImportError:
    CONTROLNET_AVAILABLE = False
    print("ControlNet功能不可用: 无法导入QwenImageControlNetModel")

# 定义Qwen-Image-ControlNet-Union支持的预处理器选项
# 根据项目规范，只保留Qwen-Image-ControlNet-Union模型支持的类型
# 注意：这里列表中的第一个元素是内部标识符，第二个元素是UI显示名称
def get_controlnet_preprocessors():
    """动态获取WebUI支持的预处理器列表"""
    try:
        # 添加WebUI根目录到系统路径
        import sys
        from pathlib import Path
        webui_root = Path(__file__).parent.parent.parent.parent
        extensions_builtin = webui_root / "extensions-builtin"
        
        paths_to_add = [
            str(webui_root),
            str(extensions_builtin)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.append(path)
        
        # 导入WebUI的预处理器管理模块
        from modules_forge.shared import supported_preprocessors
        
        # 定义Qwen-Image-ControlNet-Union模型支持的预处理器
        supported_types = [
            "none",
            "canny", 
            "softedge_hed",
            "depth_midas",
            "depth_anything_v2",
            "dw_openpose_full",
            "openpose",
            "openpose_face",
            "openpose_hand",
            "lineart_standard",
            "lineart",
            "lineart_anime",  # 添加lineart_anime预处理器
            "lineart_anime_denoise"
            # 移除inpaint_only预处理器，因为Qwen-Image-ControlNet-Inpainting模型不需要预处理器
        ]
        
        # 构建预处理器选项列表
        preprocessors = []
        for name in supported_types:
            preprocessor = supported_preprocessors.get(name)
            if preprocessor is not None:
                # 使用预处理器的标签作为显示名称，如果没有则使用名称本身
                display_name = getattr(preprocessor, 'label', name)
                preprocessors.append((name, display_name))
            else:
                # 如果找不到预处理器，使用默认显示名称
                display_name_map = {
                    "none": "None",
                    "canny": "Canny",
                    "softedge_hed": "Soft Edge",
                    "depth_midas": "Depth",
                    "depth_anything_v2": "Depth Anything V2",
                    "dw_openpose_full": "Pose",
                    "openpose": "Openpose",
                    "openpose_face": "Openpose Face",
                    "openpose_hand": "Openpose Hand",
                    "lineart_standard": "Lineart Standard",
                    "lineart": "Lineart Realistic",
                    "lineart_anime": "Lineart Anime",  # 添加lineart_anime的默认显示名称
                    "lineart_anime_denoise": "Lineart Anime Denoise"
                    # 移除inpaint_only的默认显示名称
                }
                display_name = display_name_map.get(name, name)
                preprocessors.append((name, display_name))
        
        return preprocessors
    except Exception as e:
        print(f"获取预处理器列表时出错: {e}")
        # 出错时返回默认列表
        return [
            ("none", "None"),
            ("canny", "Canny"),
            ("softedge_hed", "Soft Edge"),
            ("depth_midas", "Depth"),
            ("depth_anything_v2", "Depth Anything V2"),
            ("dw_openpose_full", "Pose"),
            ("openpose", "Openpose"),
            ("openpose_face", "Openpose Face"),
            ("openpose_hand", "Openpose Hand"),
            ("lineart_standard", "Lineart Standard"),
            ("lineart", "Lineart Realistic"), 
            ("lineart_anime", "Lineart Anime"),  # 添加lineart_anime到默认列表
            ("lineart_anime_denoise", "Lineart Anime Denoise")
            # 移除inpaint_only到默认列表
        ]

# 获取预处理器选项
CONTROLNET_PREPROCESSORS = get_controlnet_preprocessors()

# 预处理器类型映射（UI显示名称到内部标识符）
# 注意：现在我们直接使用WebUI的预处理器管理系统，不再需要手动维护映射表
# 但为了向后兼容，保留此变量，其值通过动态方式获取
def get_preprocessor_display_to_internal():
    """动态获取预处理器显示名称到内部标识符的映射"""
    mapping = {}
    for internal_name, display_name in CONTROLNET_PREPROCESSORS:
        mapping[display_name] = internal_name
    return mapping

PREPROCESSOR_DISPLAY_TO_INTERNAL = get_preprocessor_display_to_internal()

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

def preprocess_control_image(image_path, preprocessor_display_name):
    """预处理控制图像"""
    try:
        if not image_path or not os.path.exists(image_path):
            print(f"预处理图像路径无效: {image_path}")
            return None
        
        # 加载图像
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # 将UI显示名称转换为内部标识符
        mapped_preprocessor_type = PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_display_name, "none")
        print(f"开始使用预处理器 {preprocessor_display_name} ({mapped_preprocessor_type}) 处理图像: {image_path}")
        
        # 调用预处理脚本
        args = {
            "image_path": image_path,
            "preprocessor_type": mapped_preprocessor_type  # 使用映射后的预处理器名称
        }
        
        args_file = qwen_image_dir / "temp_preprocess_args.json"
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False, indent=2)
        
        # 构建命令
        args_file_str = str(args_file).replace('\\', '/')
        scripts_dir_str = str(scripts_dir).replace('\\', '/')
        
        cmd = [
            main_python,
            "-c",
            f"import sys; sys.path.append('{scripts_dir_str}'); from qwen_image_scripts import run_preprocess_control_image; run_preprocess_control_image('{args_file_str}')"
        ]
        
        print(f"执行预处理命令: {' '.join(cmd)}")
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(qwen_image_dir), timeout=120)
        
        # 删除临时参数文件
        if args_file.exists():
            args_file.unlink()
        
        print(f"预处理命令返回码: {result.returncode}")
        if result.stdout:
            print(f"预处理命令输出: {result.stdout}")
        if result.stderr:
            print(f"预处理命令错误: {result.stderr}")
        
        if result.returncode != 0:
            print(f"预处理失败: {result.stderr}")
            # 即使预处理失败，也尝试返回原始图像
            return image
        
        # 解析输出，查找处理后的图像路径
        output_lines = result.stdout.strip().split('\n')
        processed_image_path = None
        for line in output_lines:
            if line.startswith("SUCCESS:"):
                processed_image_path = line[8:].strip()  # 移除 "SUCCESS:" 前缀
                break
        
        if processed_image_path and os.path.exists(processed_image_path):
            print(f"成功找到预处理图像: {processed_image_path}")
            # 加载并返回处理后的图像
            processed_image = Image.open(processed_image_path)
            return processed_image
        else:
            print("未找到有效的预处理图像，返回原始图像")
            return image
            
    except Exception as e:
        print(f"预处理控制图像时出错: {e}")
        traceback.print_exc()
        # 出错时返回原始图像
        try:
            if image_path and os.path.exists(image_path):
                from PIL import Image
                return Image.open(image_path).convert("RGB")
        except:
            pass
        return None

def run_text_to_image(prompt, negative_prompt, width, height, steps, cfg_scale, 
                      model_file, scheduler, controlnet_enable=False, controlnet_model=None,
                      control_image=None, controlnet_conditioning_scale=1.0,
                          controlnet_preprocessor="none", controlnet_start=0.0, controlnet_end=1.0):
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
            "controlnet_enable": controlnet_enable and CONTROLNET_AVAILABLE,
            "controlnet_model": controlnet_model if controlnet_enable and CONTROLNET_AVAILABLE else None,
            "control_image": control_image if controlnet_enable and CONTROLNET_AVAILABLE else None,
            "controlnet_conditioning_scale": controlnet_conditioning_scale if controlnet_enable and CONTROLNET_AVAILABLE else 1.0,
            "controlnet_preprocessor": controlnet_preprocessor if controlnet_enable and CONTROLNET_AVAILABLE else "none",
            "controlnet_start": controlnet_start if controlnet_enable and CONTROLNET_AVAILABLE else 0.0,
            "controlnet_end": controlnet_end if controlnet_enable and CONTROLNET_AVAILABLE else 1.0,
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
               model_file, scheduler):
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
                            placeholder="输入您的提示词，描述想要生成的图像...",
                            lines=4,  # 设置初始显示4行
                            max_lines=10,  # 最多可以显示10行
                            elem_classes=["prompt-textarea"]  # 添加CSS类以便进一步定制
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
                        
                        # 添加ControlNet相关组件 (参考WebUI中ControlNet的设计)
                        with gr.Accordion("ControlNet 控制", open=False):
                            with gr.Row():
                                controlnet_enable = gr.Checkbox(
                                    label="启用ControlNet",
                                    value=False
                                )
                                
                                # 动态获取Qwen Image ControlNet模型列表
                                def get_qwen_image_controlnet_models():
                                    """获取Qwen Image ControlNet模型列表"""
                                    try:
                                        # 添加WebUI根目录到系统路径
                                        import sys
                                        from pathlib import Path
                                        webui_root = Path(__file__).parent.parent.parent.parent
                                        extensions_builtin = webui_root / "extensions-builtin"
                                        
                                        paths_to_add = [
                                            str(webui_root),
                                            str(extensions_builtin)
                                        ]
                                        
                                        for path in paths_to_add:
                                            if path not in sys.path:
                                                sys.path.append(path)
                                        
                                        # 导入WebUI的ControlNet模型管理模块
                                        from lib_controlnet.global_state import get_all_controlnet_names
                                        
                                        # 获取所有ControlNet模型
                                        all_models = get_all_controlnet_names()
                                        
                                        # 筛选出Qwen Image相关的模型
                                        qwen_image_models = []
                                        for model in all_models:
                                            # 检查是否包含qwen（不区分大小写）
                                            if "qwen" in model.lower():
                                                qwen_image_models.append((model, model))
                                        
                                        # 如果没有找到Qwen Image模型，则添加默认列表
                                        if not qwen_image_models:
                                            # 手动添加已知的Qwen Image模型
                                            known_models = [
                                                "Qwen-Image-ControlNet-Union",
                                                "Qwen-Image-ControlNet-Inpainting"
                                            ]
                                            
                                            # 检查这些模型是否存在于模型目录中
                                            models_dir = webui_root / "models" / "ControlNet"
                                            for model_name in known_models:
                                                model_path = models_dir / model_name
                                                if model_path.exists():
                                                    display_name = model_name
                                                    qwen_image_models.append((display_name, model_name))
                                        
                                        # 如果还是没有找到任何模型，使用默认列表
                                        if not qwen_image_models:
                                            qwen_image_models = [
                                                ("Qwen-Image-ControlNet-Union", "Qwen-Image-ControlNet-Union")
                                            ]
                                        
                                        return qwen_image_models
                                    except Exception as e:
                                        print(f"获取Qwen Image ControlNet模型列表时出错: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        # 出错时返回默认列表
                                        return [
                                            ("Qwen-Image-ControlNet-Union", "Qwen-Image-ControlNet-Union"),
                                            ("Qwen-Image-ControlNet-Inpainting", "Qwen-Image-ControlNet-Inpainting")
                                        ]
                                
                                qwen_image_controlnet_models = get_qwen_image_controlnet_models()
                                
                                controlnet_model = gr.Dropdown(
                                    choices=qwen_image_controlnet_models,
                                    value=qwen_image_controlnet_models[0][0] if qwen_image_controlnet_models else "Qwen-Image-ControlNet-Union",
                                    label="ControlNet 模型",
                                    interactive=True
                                )
                            
                            with gr.Row():
                                # 使用支持绘图的图像组件，允许用户绘制蒙版
                                control_image = gr.Image(
                                    type="filepath",
                                    label="控制图像",
                                    elem_classes=["controlnet-image-container"],
                                    height=300,
                                    tool="sketch",  # 启用绘图工具
                                    interactive=True,
                                    image_mode="RGB"  # 确保图像模式为RGB
                                )
                                
                                # 预处理效果图预览 (参考WebUI中ControlNet的设计)
                                preprocess_preview = gr.Image(
                                    label="预处理效果图预览",
                                    interactive=False,
                                    elem_classes=["preprocess-preview-container"],
                                    visible=False,
                                    height=300
                                )
                            
                            # 添加图像尺寸显示在控制图像上方
                            control_image_size = gr.Textbox(
                                label="图像尺寸",
                                interactive=False,
                                value="未上传图像"
                            )
                            
                            # 添加函数来根据模型类型更新控制图像组件的绘图功能
                            def update_control_image_interactivity(model_name):
                                """根据模型类型更新控制图像组件的交互性"""
                                # 对于inpainting模型，启用绘图功能
                                if "inpaint" in model_name.lower() or "Inpaint" in model_name:
                                    return gr.update(tool="sketch", interactive=True)
                                else:
                                    # 对于其他模型，保持现有设置
                                    return gr.update()
                            
                            # 当模型选择改变时，更新控制图像组件
                            controlnet_model.change(
                                fn=update_control_image_interactivity,
                                inputs=[controlnet_model],
                                outputs=[control_image]
                            )
                            
                            with gr.Row():
                                controlnet_preprocessor = gr.Dropdown(
                                    choices=CONTROLNET_PREPROCESSORS,
                                    value="none",
                                    label="预处理器",
                                    interactive=True
                                )
                                
                                # 添加预处理按钮
                                with gr.Column():
                                    with gr.Row():
                                        preprocess_button = gr.Button("预览预处理效果", elem_classes=["preprocess-button"])
                                        preprocess_refresh = gr.Checkbox(label="启用自动预览", value=True)
                            
                            with gr.Row():
                                controlnet_conditioning_scale = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.05,
                                    label="ControlNet 强度"
                                )
                                
                                controlnet_start = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.0,
                                    step=0.05,
                                    label="开始时间步"
                                )
                                
                                controlnet_end = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05,
                                    label="结束时间步"
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
        def on_control_image_change(image_path):
            """当控制图像改变时触发"""
            if image_path:
                # 获取图像尺寸
                try:
                    from PIL import Image
                    image = Image.open(image_path)
                    width, height = image.size
                    size_text = f"{width} × {height}"
                    return size_text, gr.update(visible=False)  # 不再显示原图在预览区域
                except Exception as e:
                    print(f"读取图像尺寸时出错: {e}")
                    return "无法读取尺寸", gr.update(visible=False)
            else:
                return "未上传图像", gr.update(visible=False)
        
        def save_processed_image(processed_image):
            """保存处理后的图像到临时文件并返回路径"""
            if not processed_image:
                return None
                
            # 创建临时目录
            temp_dir = qwen_image_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            # 生成唯一文件名
            timestamp = int(time.time() * 1000)
            temp_path = temp_dir / f"preprocess_preview_{timestamp}.png"
            
            # 保存图像
            processed_image.save(temp_path)
            return str(temp_path)

        def on_preprocess_params_change(image_path, preprocessor_type, auto_refresh):
            """当预处理参数改变时触发"""
            try:
                print(f"预处理参数变更: image_path={image_path}, preprocessor_type={preprocessor_type}, auto_refresh={auto_refresh}")
                if auto_refresh and image_path and os.path.exists(image_path) and preprocessor_type != "none":
                    processed_image = preprocess_control_image(image_path, preprocessor_type)
                    if processed_image:
                        temp_path = save_processed_image(processed_image)
                        if temp_path:
                            print(f"预览图像已保存到: {temp_path}")
                            return gr.update(visible=True, value=temp_path)
                        else:
                            print("无法保存预览图像")
                    else:
                        print("预处理未返回有效图像")
                else:
                    print("不满足自动预览条件")
                return gr.update(visible=False)
            except Exception as e:
                print(f"自动预览处理失败: {e}")
                traceback.print_exc()
                return gr.update(visible=False)

        def on_preprocess_button_click(image, preprocessor_display_name):
            """预处理按钮点击事件处理函数"""
            try:
                print(f"预处理按钮点击: image_path={image}, preprocessor={preprocessor_display_name}")
                
                if image is None:
                    return "未选择图像", None
                
                # 将UI显示名称转换为内部标识符
                preprocessor_internal_name = PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_display_name, "none")
                if preprocessor_internal_name == "none":
                    return "请选择有效的预处理器", None
                print(f"映射预处理器类型: {preprocessor_display_name} -> {preprocessor_internal_name}")
                
                # 创建临时参数文件
                temp_args = {
                    "image_path": image,
                    "preprocessor_type": preprocessor_internal_name
                }
                
                temp_args_file = qwen_image_dir / "temp_preprocess_args.json"
                with open(temp_args_file, "w", encoding="utf-8") as f:
                    json.dump(temp_args, f, ensure_ascii=False, indent=2)
                
                # 构建预处理命令
                temp_args_file_str = str(temp_args_file).replace('\\', '/')
                scripts_dir_str = str(scripts_dir).replace('\\', '/')
                
                cmd = [
                    main_python,
                    "-c",
                    f"import sys; sys.path.append('{scripts_dir_str}'); from qwen_image_scripts import run_preprocess_control_image; run_preprocess_control_image('{temp_args_file_str}')"
                ]
                
                # 执行预处理命令
                print(f"执行预处理命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(qwen_image_dir), timeout=120)
                
                # 删除临时参数文件
                if temp_args_file.exists():
                    temp_args_file.unlink()
                
                print(f"预处理命令返回码: {result.returncode}")
                print(f"预处理命令输出: {result.stdout}")
                if result.stderr:
                    print(f"预处理命令错误: {result.stderr}")
                
                if result.returncode != 0:
                    return f"预处理失败: {result.stderr}", None
                
                # 解析成功输出
                output_info = parse_script_output(result.stdout)
                if "image_path" in output_info:
                    output_path = output_info["image_path"]
                    print(f"成功找到预处理图像: {output_path}")
                    return f"预处理完成: {output_path}", output_path
                else:
                    return "预处理完成，但未找到输出图像", None
                    
            except Exception as e:
                error_msg = f"预处理过程中出现异常: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return error_msg, None
        
        # 组合控制图像和预处理参数改变事件的处理函数
        def combined_control_image_handler(image_path, preprocessor_type, auto_refresh):
            # 处理尺寸显示
            size_text = "未上传图像"
            if image_path and os.path.exists(image_path):
                try:
                    from PIL import Image
                    image = Image.open(image_path)
                    width, height = image.size
                    size_text = f"{width} × {height}"
                except Exception as e:
                    print(f"读取图像尺寸时出错: {e}")
            
            # 处理预览更新
            preview_update = gr.update(visible=False)
            if auto_refresh and image_path and os.path.exists(image_path) and preprocessor_type != "none":
                print(f"组合处理: image_path={image_path}, preprocessor_type={preprocessor_type}, auto_refresh={auto_refresh}")
                processed_image = preprocess_control_image(image_path, preprocessor_type)
                if processed_image:
                    temp_path = save_processed_image(processed_image)
                    if temp_path and os.path.exists(temp_path):
                        print(f"组合处理预览图像已保存到: {temp_path}")
                        preview_update = gr.update(visible=True, value=temp_path)
                    else:
                        print("组合处理无法保存预览图像")
                else:
                    print("组合处理预处理未返回有效图像")
            
            return size_text, preview_update
        
        # 绑定组合事件处理程序
        control_image.change(
            fn=combined_control_image_handler,
            inputs=[control_image, controlnet_preprocessor, preprocess_refresh],
            outputs=[control_image_size, preprocess_preview]
        )
        
        # 当预处理器类型或自动刷新状态改变时，也触发相同的处理逻辑
        controlnet_preprocessor.change(
            fn=combined_control_image_handler,
            inputs=[control_image, controlnet_preprocessor, preprocess_refresh],
            outputs=[control_image_size, preprocess_preview]
        )
        
        preprocess_refresh.change(
            fn=combined_control_image_handler,
            inputs=[control_image, controlnet_preprocessor, preprocess_refresh],
            outputs=[control_image_size, preprocess_preview]
        )
        
        # 预处理按钮点击事件（手动预览）
        preprocess_button.click(
            fn=on_preprocess_button_click,
            inputs=[control_image, controlnet_preprocessor],
            outputs=[control_image_size, preprocess_preview]
        )

        text_to_image_button.click(
            fn=run_text_to_image,
            inputs=[
                text_to_image_prompt,
                text_to_image_negative_prompt,
                text_to_image_width,
                text_to_image_height,
                text_to_image_steps,
                text_to_image_cfg,
                text_to_image_model,
                text_to_image_scheduler,
                controlnet_enable,
                controlnet_model,
                control_image,
                controlnet_conditioning_scale,
                controlnet_preprocessor,
                controlnet_start,
                controlnet_end
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
            "controlnet_enable": controlnet_enable,
            "controlnet_model": controlnet_model,
            "control_image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
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

# 添加CSS样式以增强负面提示词输入框的可见性
custom_css = """
.negative_prompt input, .negative_prompt textarea {
    background-color: #111827 !important;
    color: #ffffff !important;
    font-weight: normal !important;
    font-size: 14px !important;
    border: 1px solid #4b5563 !important;
    border-radius: 4px !important;
    padding: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) inset !important;
}

.negative_prompt label {
    color: #f9fafb !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}

/* 限制ControlNet图像的最大显示尺寸 */
.controlnet-image-container {
    max-width: 300px;
    max-height: 300px;
    overflow: hidden;
    border: 1px solid #4b5563;
    border-radius: 4px;
    margin: 10px 0;
    position: relative;
}

/* 尺寸预览容器 */
.size-preview-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    position: relative;
    background-color: rgba(0, 0, 0, 0.1);
    border: 1px dashed #6b7280;
    border-radius: 4px;
}

/* 尺寸预览边框 */
.size-preview-border {
    position: absolute;
    top: 0;
    left: 0;
    border: 2px solid #3b82f6;
    border-radius: 4px;
    pointer-events: none;
}

/* 尺寸预览文本 */
.size-preview-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #3b82f6;
    font-size: 12px;
    font-weight: bold;
    text-align: center;
    white-space: nowrap;
}

/* 预处理效果图容器 */
.preprocess-preview-container {
    max-width: 300px;
    max-height: 300px;
    margin: 10px 0;
    border: 1px solid #4b5563;
    border-radius: 4px;
    padding: 5px;
}
"""

# 添加JavaScript代码来处理尺寸预览
custom_js = """
<script>
function updateSizePreview(width, height) {
    const container = document.querySelector('.size-preview-container');
    if (!container) return;
    
    // 获取容器的实际尺寸
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    // 计算缩放比例
    const scale = Math.min(containerWidth / width, containerHeight / height);
    
    // 设置边框尺寸和位置
    const border = document.querySelector('.size-preview-border');
    if (border) {
        border.style.width = `${width * scale}px`;
        border.style.height = `${height * scale}px`;
        border.style.left = `${(containerWidth - width * scale) / 2}px`;
        border.style.top = `${(containerHeight - height * scale) / 2}px`;
    }
    
    // 更新文本内容
    const text = document.querySelector('.size-preview-text');
    if (text) {
        text.textContent = `${width}×${height}`;
    }
}
</script>
"""
