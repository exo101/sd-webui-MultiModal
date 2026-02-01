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
import urllib.parse
from modules import shared
from modules.call_queue import wrap_gradio_gpu_call  # 保留原来的导入
import queue  # 添加队列模块

from modules.progress import create_task_id  # 导入创建任务ID的函数

# 尝试导入WebUI的采样器模块
try:
    from modules import sd_samplers
    WEBUI_SAMPLERS_AVAILABLE = True
except ImportError:
    WEBUI_SAMPLERS_AVAILABLE = False
    
# 尝试导入WebUI的调度器模块
try:
    from modules import sd_schedulers
    WEBUI_SCHEDULERS_AVAILABLE = True
except ImportError:
    WEBUI_SCHEDULERS_AVAILABLE = False

# 尝试导入angle_selector模块 - 延迟导入
def get_angle_selector():
    try:
        import importlib.util
        import os
        from pathlib import Path
        
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        angle_selector_path = current_dir / "qwen_angle_selector.py"

        if angle_selector_path.exists():
            spec = importlib.util.spec_from_file_location("qwen_angle_selector", str(angle_selector_path))
            angle_selector_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(angle_selector_module)
            return angle_selector_module.create_qwen_angle_visualization_component, True
        else:
            return None, False
    except Exception as e:
        print(f"[WARNING] 多角度提示词可视化选择器模块导入失败: {e}")
        return None, False

# 初始化angle_selector相关变量，但不立即执行导入
create_angle_visualization_component = None
ANGLE_SELECTOR_AVAILABLE = False

import shutil
import webbrowser

# ==================== 常量定义 ====================
# 获取当前脚本所在目录
current_dir = Path(__file__).parent
scripts_dir = current_dir
qwen_image_dir = current_dir.parent / "qwen-image"

# 添加当前脚本目录到系统路径
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))
# ==================== 常量定义 ====================
# 获取当前脚本所在目录
current_dir = Path(__file__).parent
scripts_dir = current_dir
qwen_image_dir = current_dir.parent / "qwen-image"

# 添加qwen-image目录到系统路径，以便导入预处理模块
sys.path.append(str(qwen_image_dir))
from qwen_image_controlnet import preprocess_for_qwen_image_controlnet

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

# ==================== 模型列表获取函数 ====================
# 获取模型文件列表
def get_model_choices(model_dir):
    """获取指定目录下的模型文件列表"""
    try:
        # 预先初始化choices变量
        choices = []
        
        if not model_dir.exists():
            print(f"警告: 模型目录不存在 {model_dir}")
            # 尝试创建目录
            model_dir.mkdir(parents=True, exist_ok=True)
            # 添加默认选项
            choices.append(("未找到模型文件", ""))
            return choices
        
        # 直接在指定目录查找模型文件，不深入子目录
        model_files = list(model_dir.glob("*.safetensors"))
        
        # 如果仍然没有找到任何模型文件，添加默认选项
        if not model_files:
            # 添加默认选项，即使没有找到模型文件
            choices.append(("未找到模型文件", ""))
        else:
            # 返回 (显示名称, 文件名) 的元组列表
            choices = [(f.name, f.name) for f in model_files]
        
        # 检查本地SDNQ模型是否存在（在标准模型目录下）
        local_sdnq_path = Path(shared.models_path) / "Qwen-Image-2512-SDNQ-4bit-dynamic"
        if local_sdnq_path.exists():
            # 如果本地SDNQ模型存在，则添加到选项中，不添加远程版本
            choices.append(("Qwen-Image-2512-SDNQ-4bit (本地)", str(local_sdnq_path)))
        else:
            # 如果本地模型不存在，则添加远程模型选项
            choices.append(("Qwen-Image-2512-SDNQ-4bit", "Disty0/Qwen-Image-2512-SDNQ-4bit-dynamic"))
        
        return choices
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        traceback.print_exc()
        # 返回默认选项而不是空列表
        return [("未找到模型文件", "")]

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
    # 添加"无"选项到模型选择列表
    qwenimage_model_choices = ["无"] + qwenimage_model_choices if qwenimage_model_choices else ["无"]
    qwenimage_edit_model_choices = get_model_choices(qwenimage_edit_models_dir)
    qwenimage_lora_choices = get_lora_choices(qwenimage_lora_dir)  # 获取LoRA模型列表
except Exception as e:
    print(f"加载模型列表时出错: {e}")
    traceback.print_exc()
    qwenimage_model_choices = ["无"]
    qwenimage_edit_model_choices = ["无"]
    qwenimage_lora_choices = [("无", "")]  # 默认LoRA选项

# ==================== 库导入和可用性检查 ====================
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

# ==================== ControlNet 预处理器相关 ====================
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
        
        # 定义Qwen-Image-ControlNet-Union模型支持的预处理器，按类别分组
        supported_preprocessors_by_category = {
            "Canny": [
                "canny"
            ],
            "Depth": [
                "depth_midas",
                "depth_leres", 
                "depth_leres++",
                "depth_anything",
                "depth_anything_v2",
                "depth_hand_refiner",
                "depth_marigold",
                "depth_zoe"
            ],
            "Pose": [
                "openpose_full",
                "openpose",
                "openpose_face",
                "openpose_faceonly",
                "openpose_hand",
                "dw_openpose_full",
                "animal_openpose",
                "densepose",
                "densepose_parula"
            ],
            "Lineart": [
                "lineart_standard",
                "lineart_realistic",
                "lineart_coarse",
                "lineart_anime",
                "lineart_anime_denoise"
            ],
            "Softedge": [
                "scribble_pidinet",
                "softedge_pidinet",
                "softedge_pidinet_safe",
                "softedge_pidinstruct",
                "softedge_hed",
                "softedge_hedsafe"
            ]
        }
        
        # 构建预处理器选项列表
        preprocessors = [("none", "None")]  # 确保"None"选项在列表的开头
        
        # 按类别添加预处理器
        for category, processors in supported_preprocessors_by_category.items():
            for name in processors:
                preprocessor = supported_preprocessors.get(name)
                if preprocessor is not None:
                    # 使用预处理器的标签作为显示名称，如果没有则使用名称本身
                    display_name = getattr(preprocessor, 'label', name)
                    # 在显示名称前加上类别前缀
                    full_display_name = f"[{category}] {display_name}"
                    preprocessors.append((name, full_display_name))
                else:
                    # 如果找不到预处理器，跳过此项
                    continue
        
        return preprocessors
    except Exception as e:
        print(f"获取预处理器列表时出错: {e}")
        # 出错时返回默认列表，确保"None"选项在列表的开头
        return [
            ("none", "None"),
            # Canny 类别
            ("canny", "[Canny] Canny"),
            # Depth 类别
            ("depth_midas", "[Depth] Depth Midas"),
            ("depth_leres", "[Depth] Depth Leres"),
            ("depth_leres++", "[Depth] Depth Leres++"),
            ("depth_anything", "[Depth] Depth Anything"),
            ("depth_anything_v2", "[Depth] Depth Anything V2"),
            ("depth_hand_refiner", "[Depth] Depth Hand Refiner"),
            ("depth_marigold", "[Depth] Depth Marigold"),
            ("depth_zoe", "[Depth] Depth Zoe"),
            # Pose 类别
            ("openpose_full", "[Pose] Openpose Full"),
            ("openpose", "[Pose] Openpose"),
            ("openpose_face", "[Pose] Openpose Face"),
            ("openpose_faceonly", "[Pose] Openpose Faceonly"),
            ("openpose_hand", "[Pose] Openpose Hand"),
            ("dw_openpose_full", "[Pose] DW Openpose Full"),
            ("animal_openpose", "[Pose] Animal Openpose"),
            ("densepose", "[Pose] Densepose (purple bg & purple torso)"),
            ("densepose_parula", "[Pose] Densepose Parula (black bg & blue torso)"),
            # Lineart 类别
            ("lineart_standard", "[Lineart] Lineart Standard (from white bg & black line)"),
            ("lineart_realistic", "[Lineart] Lineart Realistic"),
            ("lineart_coarse", "[Lineart] Lineart Coarse"),
            ("lineart_anime", "[Lineart] Lineart Anime"),
            ("lineart_anime_denoise", "[Lineart] Lineart Anime Denoise"),
            # Softedge 类别
            ("scribble_pidinet", "[Softedge] Scribble Pidinet"),
            ("softedge_pidinet", "[Softedge] Softedge Pidinet"),
            ("softedge_pidinet_safe", "[Softedge] Softedge Pidinet Safe"),
            ("softedge_pidinstruct", "[Softedge] Softedge Pidinstruct"),
            ("softedge_hed", "[Softedge] Softedge Hed"),
            ("softedge_hedsafe", "[Softedge] Softedge Hedsafe")
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
        # 处理带前缀的显示名称，如"[Pose] dw_openpose_full"
        clean_display_name = display_name
        if isinstance(display_name, str) and "]" in display_name:
            # 去除"[Pose] "这样的前缀
            clean_display_name = display_name.split("]", 1)[1].strip()
        
        mapping[display_name] = internal_name
        mapping[clean_display_name] = internal_name
    
    # 确保"None"映射到"none"
    mapping["None"] = "none"
    mapping["none"] = "none"
    return mapping

# 获取预处理器显示名称到内部标识符的映射
CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL = get_preprocessor_display_to_internal()

# ==================== 图像处理辅助函数 ====================
def parse_script_output(output):
    """解析脚本输出，提取图像路径和信息文件路径"""
    try:
        lines = output.strip().split('\n')
        result = {
            "image_paths": []  # 使用列表存储多个图像路径
        }
        
        for line in lines:
            if line.startswith("SUCCESS:"):
                # 添加图像路径到列表中
                image_path = line[8:].strip()
                result["image_paths"].append(image_path)
            elif line.startswith("INFO_FILE:"):
                result["info_file"] = line[10:].strip()
        
        return result
    except Exception as e:
        print(f"解析脚本输出时出错: {e}")
        traceback.print_exc()
        return {}

def preprocess_control_image(image_input, preprocessor_display_name):
    """预处理控制图像"""
    try:
        # 将UI显示名称转换为内部标识符
        mapped_preprocessor_type = CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_display_name, "none")
        
        # 直接使用qwen_image_controlnet.py中的预处理函数
        processed_image = preprocess_for_qwen_image_controlnet(image_input, mapped_preprocessor_type)
        
        return processed_image

    except Exception as e:
        print(f"预处理控制图像时出错: {e}")
        traceback.print_exc()
        return image_input


def save_processed_image(processed_image):
    """保存预处理后的图像到临时文件"""
    try:
        import uuid
        # 生成唯一文件名
        temp_filename = f"processed_{uuid.uuid4().hex[:8]}.png"
        temp_path = qwen_image_dir / "temp" / temp_filename
        temp_path.parent.mkdir(exist_ok=True)
        
        # 如果是numpy数组，转换为PIL图像
        if isinstance(processed_image, np.ndarray):
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                # RGB图像
                image = Image.fromarray(processed_image, mode='RGB')
            elif len(processed_image.shape) == 2:
                # 灰度图
                image = Image.fromarray(processed_image, mode='L')
            else:
                # 其他情况默认转RGB
                image = Image.fromarray(processed_image).convert('RGB')
        elif isinstance(processed_image, Image.Image):
            # 如果已经是PIL图像，直接使用
            image = processed_image
        else:
            print(f"不支持的图像类型: {type(processed_image)}")
            return None
            
        # 保存图像
        image.save(str(temp_path), 'PNG')
        return str(temp_path)
    except Exception as e:
        print(f"保存预处理图像时出错: {e}")
        traceback.print_exc()
        return None

# 添加函数来保存numpy数组为图像文件
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

# ==================== ControlNet 模型相关 ====================
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
        qwen_image_models = [("无", "无")]  # 添加"无"选项作为默认值
        for model in all_models:
            # 检查是否包含qwen（不区分大小写）
            if "qwen" in model.lower():
                qwen_image_models.append((model, model))
        
        # 如果没有找到Qwen Image模型，则添加默认列表
        if len(qwen_image_models) <= 1:  # 只有"无"选项
            # 手动添加已知的Qwen Image模型
            known_models = [
                "Qwen-Image-ControlNet-Union",
                "Qwen-Image-ControlNet-Inpainting"
            ]
            
            # 检查这些模型是否存在于模型目录中
            # 使用统一定义的控制网络模型目录
            for model_name in known_models:
                model_path = qwenimage_controlnet_dir / model_name
                if model_path.exists():
                    display_name = model_name
                    qwen_image_models.append((display_name, model_name))
        
        # 如果还是没有找到任何模型，使用默认列表
        if len(qwen_image_models) <= 1:  # 只有"无"选项
            qwen_image_models.extend([
                ("Qwen-Image-ControlNet-Union", "Qwen-Image-ControlNet-Union")
            ])
        
        return qwen_image_models
    except Exception as e:
        print(f"获取Qwen Image ControlNet模型列表时出错: {e}")
        import traceback
        traceback.print_exc()
        # 出错时返回默认列表，包含"无"选项
        return [
            ("无", "无"),
            ("Qwen-Image-ControlNet-Union", "Qwen-Image-ControlNet-Union"),
            ("Qwen-Image-ControlNet-Inpainting", "Qwen-Image-ControlNet-Inpainting")
        ]

# ==================== 核心功能函数 ====================
def run_text_to_image(prompt, negative_prompt, width, height, steps, cfg_scale, 
                      model_file, scheduler, scheduler_type, lora_model_1="", lora_model_2="", 
                      lora_weight_1=1.0, lora_weight_2=1.0, seed=-1, batch_size=1,
                      controlnet_model=None,
                      control_image=None, control_mask=None, controlnet_conditioning_scale=1.0,
                      controlnet_preprocessor="none", controlnet_start=0.0, controlnet_end=1.0,
                      sdnq_enable=False):  # 移除id_task参数，让wrap_gradio_gpu_call自动处理
    try:
        print("开始执行文生图功能...")
        # 删除了从outputs目录自动选择最新图像的功能代码
        
        # 处理control_image参数，如果它是numpy数组则保存为临时文件
        processed_control_image = control_image
        if isinstance(control_image, np.ndarray):
            # 为numpy数组创建临时文件
            temp_dir = qwen_image_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_image_path = temp_dir / f"control_image_{int(time.time() * 1000)}.png"
            save_result = save_numpy_image(control_image, temp_image_path)
            if save_result:
                processed_control_image = str(temp_image_path)
            else:
                processed_control_image = None
        elif hasattr(control_image, 'save'):  # 如果是PIL Image对象
            # 为PIL Image创建临时文件
            temp_dir = qwen_image_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_image_path = temp_dir / f"control_image_{int(time.time() * 1000)}.png"
            try:
                control_image.save(temp_image_path)
                processed_control_image = str(temp_image_path)
            except Exception as e:
                print(f"保存PIL Image对象时出错: {e}")
                processed_control_image = None
                
        
        # 处理control_mask参数，如果它是numpy数组则保存为临时文件
        processed_control_mask = control_mask
        if isinstance(control_mask, np.ndarray):
            # 为numpy数组创建临时文件
            temp_dir = qwen_image_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_mask_path = temp_dir / f"control_mask_{int(time.time() * 1000)}.png"
            save_result = save_numpy_image(control_mask, temp_mask_path)
            if save_result:
                processed_control_mask = str(temp_mask_path)
            else:
                processed_control_mask = None
        elif hasattr(control_mask, 'save'):  # 如果是PIL Image对象
            # 为PIL Image创建临时文件
            temp_dir = qwen_image_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_mask_path = temp_dir / f"control_mask_{int(time.time() * 1000)}.png"
            try:
                control_mask.save(temp_mask_path)
                processed_control_mask = str(temp_mask_path)
            except Exception as e:
                print(f"保存PIL Image对象时出错: {e}")
                processed_control_mask = None
        
        
        # 准备参数
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "model_file": model_file,  # 只传递模型文件名
            "model_dir": str(qwenimage_models_dir),  # 传递基础模型目录路径
            "scheduler": scheduler,
            "controlnet_enable": processed_control_image is not None,  # 基于是否提供了控制图像来启用ControlNet
            "controlnet_model": controlnet_model if processed_control_image is not None else None,
            "control_image": processed_control_image if processed_control_image is not None else None,
            "control_mask": processed_control_mask if processed_control_mask is not None else None,  # 修复：应使用 processed_control_mask 变量进行判断
            "controlnet_conditioning_scale": controlnet_conditioning_scale if processed_control_image is not None else 1.0,
            "controlnet_preprocessor": controlnet_preprocessor if processed_control_image is not None else "none",
            "controlnet_start": controlnet_start if processed_control_image is not None else 0.0,
            "controlnet_end": controlnet_end if processed_control_image is not None else 1.0,
            "lora_model_1": lora_model_1 if lora_model_1 else None,
            "lora_model_2": lora_model_2 if lora_model_2 else None,
            "lora_weight_1": lora_weight_1,
            "lora_weight_2": lora_weight_2,
            "seed": seed,
            "batch_size": batch_size,
            "output_dir": str(qwen_image_outputs_dir),
            "sdnq_enable": sdnq_enable,  # 添加SDNQ启用参数
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
        if "image_paths" in output_info and output_info["image_paths"]:
            # 过滤有效的图像路径
            valid_image_paths = []
            for path in output_info["image_paths"]:
                path_str = str(path).strip()
                # 更严格的路径验证，防止路径变成 'D:\\:' 这样的格式
                if (path_str and 
                    len(path_str) > 5 and  # 确保路径长度合理
                    ':' in path_str and 
                    '\\' in path_str and
                    path_str.count(':') == 1 and  # 确保只有一个冒号
                    not path_str.endswith(':') and  # 确保不以冒号结尾
                    not path_str.startswith(':')):  # 确保不以冒号开头
                    
                    try:
                        test_path = Path(path_str)
                        # 检查路径格式是否有效
                        if (test_path.is_absolute() and 
                            len(str(test_path.parent)) > 3 and 
                            test_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']):
                            
                            if os.path.exists(path_str) and os.path.isfile(path_str):
                                valid_image_paths.append(path_str)
                    except Exception:
                        continue  # 如果路径验证失败，跳过此路径
                    
            if valid_image_paths:
                image_paths = valid_image_paths
                print(f"文生图生成成功，共生成 {len(image_paths)} 张图像")
                return image_paths, "生成成功"  # 返回图像列表和状态，与outputs=[text_to_image_output, text_to_image_status]匹配
            else:
                error_msg = f"生成失败: 生成的路径无效或不存在"
                print(f"生成失败: {error_msg}")
                return None, error_msg
        else:
            error_msg = f"生成失败: {result.stdout}"
            print(f"生成失败: {error_msg}")
            return None, error_msg
            
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"生成过程中出现异常: {error_msg}")
        traceback.print_exc()
        return None, error_msg

# ==================== UI事件处理函数 ====================
def open_output_directory():
    """打开输出目录"""
    try:
        output_dir = Path(shared.data_path) / "outputs"
        output_dir_str = str(output_dir)
        
        if os.path.exists(output_dir):
            if sys.platform == "win32":
                os.startfile(output_dir_str)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_dir_str])
            else:
                subprocess.Popen(["xdg-open", output_dir_str])
            return "已打开输出目录"
        else:
            return "输出目录不存在"
    except Exception as e:
        error_msg = f"打开输出目录时出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

# 添加预处理器类别变化事件
def update_preprocessor_choices(category):
    """根据选择的类别更新预处理器选项"""
    if category == "All":
        # 返回所有预处理器
        return gr.update(choices=CONTROLNET_PREPROCESSORS)
    else:
        # 只返回选定类别的预处理器
        filtered_choices = [p for p in CONTROLNET_PREPROCESSORS if p[1].startswith(f"[{category}]")]
        # 确保"None"选项始终存在
        none_option = [("none", "None")]
        filtered_choices = none_option + [p for p in filtered_choices if p[0] != "none"]
        return gr.update(choices=filtered_choices)

# 添加LoRA模型选择变化事件处理函数
# 使用JavaScript实现点击跳转功能
lora_1_js = """
(lora_model) => {
    if (lora_model && lora_model !== "无") {
        // 直接从全局变量中获取URL映射（需要在页面加载时注入）
        if (window.qwen_lora_urls) {
            const modelName = lora_model.replace(/\.[^/.]+$/, ""); // 去掉扩展名
            if (window.qwen_lora_urls[modelName]) {
                window.open(window.qwen_lora_urls[modelName], '_blank');
            } else {
                console.log("未找到LoRA模型 " + modelName + " 的URL");
            }
        }
    }
}
"""

lora_2_js = """
(lora_model) => {
    if (lora_model && lora_model !== "无") {
        // 直接从全局变量中获取URL映射（需要在页面加载时注入）
        if (window.qwen_lora_urls) {
            const modelName = lora_model.replace(/\.[^/.]+$/, ""); // 去掉扩展名
            if (window.qwen_lora_urls[modelName]) {
                window.open(window.qwen_lora_urls[modelName], '_blank');
            } else {
                console.log("未找到LoRA模型 " + modelName + " 的URL");
            }
        }
    }
}
"""

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

def on_control_image_upload(image_input):
    """当控制图像上传时自动设置尺寸"""
    if image_input is not None:
        try:
            from PIL import Image
            # 处理不同类型的输入
            if isinstance(image_input, str) and os.path.exists(image_input):  # 文件路径
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):  # numpy数组
                image = Image.fromarray(image_input)
            elif hasattr(image_input, 'save'):  # PIL Image对象
                image = image_input
            else:
                return gr.update(), gr.update()
            
            width, height = image.size
            
            # 调整尺寸到64的倍数
            target_width = ((width + 31) // 64) * 64
            target_height = ((height + 31) // 64) * 64
            
            # 限制尺寸在合理范围内
            target_width = max(256, min(2048, target_width))
            target_height = max(256, min(2048, target_height))
            
            print(f"自动设置图像尺寸: {width}x{height} -> {target_width}x{target_height}")
            return gr.update(value=target_width), gr.update(value=target_height)
        except Exception as e:
            print(f"自动设置图像尺寸时出错: {e}")
            traceback.print_exc()
            return gr.update(), gr.update()
    return gr.update(), gr.update()

def on_preprocess_params_change(image_path, preprocessor_type, preprocess_refresh):
    """当预处理参数改变时触发"""
    try:
        print(f"预处理参数变更: image_path={image_path}, preprocessor_type={preprocessor_type}, preprocess_refresh={preprocess_refresh}")
        # 只有在提供了图像且预处理器不是"none"时才进行预处理
        if preprocess_refresh and image_path is not None and ((isinstance(image_path, str) and os.path.exists(image_path)) or isinstance(image_path, np.ndarray)) and preprocessor_type != "none":
            processed_image = preprocess_control_image(image_path, preprocessor_type)
            if processed_image is not None and hasattr(processed_image, 'size') and processed_image.size > 0:
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

def on_preprocess_button_click(image_input, preprocessor_type):
    """预处理按钮点击事件处理函数"""
    # 处理预览更新
    preview_update = gr.update(visible=False)
    
    # 检查输入是文件路径还是numpy数组
    image_path = None
    if isinstance(image_input, str):  # 文件路径
        image_path = image_input
    elif isinstance(image_input, np.ndarray):  # numpy数组
        # 为numpy数组创建临时文件
        temp_dir = qwen_image_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        image_path = temp_dir / f"control_image_temp_{int(time.time() * 1000)}.png"
        saved_path = save_numpy_image(image_input, image_path)
        if saved_path:
            image_path = saved_path
    
    # 只有在提供了图像且预处理器不是"none"时才进行预处理
    if image_path and os.path.exists(image_path) and preprocessor_type != "none":
        processed_image = preprocess_control_image(image_path, preprocessor_type)
        if processed_image is not None:
            temp_path = save_processed_image(processed_image)
            if temp_path and os.path.exists(temp_path):
                preview_update = gr.update(visible=True, value=temp_path)
            else:
                print("无法保存预览图像")
        else:
            print("预处理未返回有效图像")
    else:
        print("不满足预览条件")
    
    return preview_update

# ==================== 主UI创建函数 ====================
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
        
        # 添加自定义CSS样式来隐藏ForgeCanvas的background组件
        gr.HTML("""
        <style>
        .logical_image_background {
            display: none !important;
        }
        </style>
        """)
        
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
                            text_to_image_steps = gr.Slider(
                                minimum=1, maximum=50, step=1, value=8,
                                label="推理步数",
                                min_width=80
                            )
                            
                            text_to_image_cfg = gr.Slider(
                                minimum=1.0, maximum=20.0, step=0.1, value=4.0,
                                label="CFG Scale",
                                min_width=80
                            )
                        
                        # 添加CFG参数说明
                        gr.Markdown(
                            """
                            <div style="font-size: 0.85em; color: #ffffff; margin-top: -10px; margin-bottom: 10px;">
                            <strong>参数说明</strong>: cfg引导数Lightning模型为1，普通模型为4，推理步数Lightning模型为10，普通模型为20，LoRA权重为1.5
                            </div>
                            """
                        )
                        
                        with gr.Row():
                            # 添加采样方法选择组件
                            # 获取WebUI内置的采样器选项
                            if WEBUI_SAMPLERS_AVAILABLE:
                                try:
                                    sampler_choices = [(sampler.name, sampler.name) for sampler in sd_samplers.visible_samplers()]
                                except:
                                    sampler_choices = [
                                        ("Euler", "euler"),
                                        ("Euler Ancestral", "euler_ancestral"),
                                        ("Heun", "heun"),
                                        ("DPM++ 2M", "dpmpp_2m")
                                    ]
                            else:
                                sampler_choices = [
                                    ("Euler", "euler"),
                                    ("Euler Ancestral", "euler_ancestral"),
                                    ("Heun", "heun"),
                                    ("DPM++ 2M", "dpmpp_2m")
                                ]
                            
                            text_to_image_scheduler = gr.Dropdown(
                                choices=sampler_choices,
                                value=sampler_choices[0][1] if sampler_choices else "euler",
                                label="采样方法",
                                min_width=120
                            )
                            
                            # 添加调度器选项
                            # 获取WebUI内置的调度器选项
                            if WEBUI_SCHEDULERS_AVAILABLE:
                                try:
                                    scheduler_choices = [(scheduler.label, scheduler.name) for scheduler in sd_schedulers.schedulers]
                                except:
                                    scheduler_choices = [
                                        ("Automatic", "automatic"),
                                        ("Karras", "karras"),
                                        ("Exponential", "exponential"),
                                        ("SGM Uniform", "sgm_uniform"),
                                        ("Simple", "simple"),
                                        ("Normal", "normal"),
                                        ("DDIM", "ddim_uniform")
                                    ]
                            else:
                                scheduler_choices = [
                                    ("Automatic", "automatic"),
                                    ("Karras", "karras"),
                                    ("Exponential", "exponential"),
                                    ("SGM Uniform", "sgm_uniform"),
                                    ("Simple", "simple"),
                                    ("Normal", "normal"),
                                    ("DDIM", "ddim_uniform")
                                ]
                            
                            text_to_image_scheduler_type = gr.Dropdown(
                                choices=scheduler_choices,
                                value=scheduler_choices[0][1] if scheduler_choices else "automatic",
                                label="调度器",
                                min_width=120
                            )
                            
                            # Add base model selection dropdown
                            text_to_image_model = gr.Dropdown(
                                choices=qwenimage_model_choices,
                                label="nunchaku加速模型",
                                value="无",
                                interactive=True,
                                min_width=150
                            )
                        

                        # 添加LoRA模型选择组件
                        with gr.Accordion("LoRA模型设置", open=False):
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        text_to_image_lora_1 = gr.Dropdown(
                                            choices=qwenimage_lora_choices,
                                            label="LoRA模型 1",
                                            value="",  # 默认选择"无"
                                            interactive=True,
                                            min_width=150,
                                            elem_classes=["lora-model-dropdown"]
                                        )
                                    with gr.Column(scale=1):
                                        text_to_image_lora_weight_1 = gr.Slider(
                                            minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                                            label="LoRA 1 强度",
                                            min_width=120
                                        )
                                
                                gr.Markdown("---")  # 添加分隔线
                                
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        text_to_image_lora_2 = gr.Dropdown(
                                            choices=qwenimage_lora_choices,
                                            label="LoRA模型 2",
                                            value="",  # 默认选择"无"
                                            interactive=True,
                                            min_width=150,
                                            elem_classes=["lora-model-dropdown"]
                                        )
                                    with gr.Column(scale=1):
                                        text_to_image_lora_weight_2 = gr.Slider(
                                            minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                                            label="LoRA 2 强度",
                                            min_width=120
                                        )
                                
                                # 添加刷新按钮
                                with gr.Row():
                                    from modules.ui_components import ToolButton
                                    lora_refresh_button = ToolButton(value='\U0001f504', elem_classes=['tool'])
                                    gr.Markdown("<div style='font-size: 0.8em; color: #aaaaaa; margin-top: 2px; margin-bottom: 2px;'>点击刷新按钮更新LoRA模型列表</div>")               
                        
                        # 添加LoRA模型选择变化事件
                        # 使用JavaScript实现点击跳转功能
                        lora_1_js = """
                        (lora_model) => {
                            if (lora_model && lora_model !== "无") {
                                // 直接从全局变量中获取URL映射（需要在页面加载时注入）
                                if (window.qwen_lora_urls) {
                                    const modelName = lora_model.replace(/\.[^/.]+$/, ""); // 去掉扩展名
                                    if (window.qwen_lora_urls[modelName]) {
                                        window.open(window.qwen_lora_urls[modelName], '_blank');
                                    } else {
                                        console.log("未找到LoRA模型 " + modelName + " 的URL");
                                    }
                                }
                            }
                        }
                        """
                        
                        lora_2_js = """
                        (lora_model) => {
                            if (lora_model && lora_model !== "无") {
                                // 直接从全局变量中获取URL映射（需要在页面加载时注入）
                                if (window.qwen_lora_urls) {
                                    const modelName = lora_model.replace(/\.[^/.]+$/, ""); // 去掉扩展名
                                    if (window.qwen_lora_urls[modelName]) {
                                        window.open(window.qwen_lora_urls[modelName], '_blank');
                                    } else {
                                        console.log("未找到LoRA模型 " + modelName + " 的URL");
                                    }
                                }
                            }
                        }
                        """
                        
                        # 添加点击事件
                        # 移除未定义的JavaScript函数调用
                        # text_to_image_lora_1.select(
                        #     fn=None,
                        #     inputs=[text_to_image_lora_1],
                        #     outputs=[],
                        #     _js=lora_1_js
                        # )
                        # 
                        # text_to_image_lora_2.select(
                        #     fn=None,
                        #     inputs=[text_to_image_lora_2],
                        #     outputs=[],
                        #     _js=lora_2_js
                        # )
                        
                        # 添加随机种子和生成批次组件
                        with gr.Row():
                            text_to_image_seed = gr.Number(
                                label="随机种子 (-1为随机)",
                                value=-1,
                                precision=0,
                                min_width=120
                            )
                            
                            text_to_image_batch_size = gr.Slider(
                                minimum=1, maximum=8, step=1, value=1,
                                label="生成批次",
                                min_width=120
                            )
                        
                        # 添加多角度提示词可视化选择器（如果可用）
                        # 使用真正的延迟加载
                        global create_angle_visualization_component, ANGLE_SELECTOR_AVAILABLE
                        if not ANGLE_SELECTOR_AVAILABLE and create_angle_visualization_component is None:
                            # 首次使用时尝试加载
                            create_angle_visualization_component, ANGLE_SELECTOR_AVAILABLE = get_angle_selector()
                        
                        if ANGLE_SELECTOR_AVAILABLE and create_angle_visualization_component:
                            with gr.Accordion("多角度提示词可视化选择器", open=False):
                                angle_selector_component = create_angle_visualization_component(text_to_image_prompt)

                        # 添加ControlNet相关组件 (参考WebUI中ControlNet的设计)
                        with gr.Accordion("ControlNet 控制", open=False):
                            qwen_image_controlnet_models = get_qwen_image_controlnet_models()
                            
                            with gr.Tabs(visible=True):
                                with gr.Tab(label="Single Image"):
                                    with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                                        with gr.Group(elem_classes=["cnet-input-image-group"]):
                                            # 使用ForgeCanvas支持绘图功能，允许用户绘制蒙版
                                            # 注意：ForgeCanvas需要在modules_forge.forge_canvas.canvas中导入
                                            from modules_forge.forge_canvas.canvas import ForgeCanvas
                                            control_image = ForgeCanvas(
                                                elem_id="qwen_image_control_image",
                                                elem_classes=["cnet-image"],
                                                height=300,
                                                contrast_scribbles=True,
                                                numpy=True  # 设置为True以返回numpy数组而不是文件路径
                                            )
                                            
                                        with gr.Group(elem_classes=["cnet-generated-image-group"]):
                                            # 预处理效果图预览 (参考WebUI中ControlNet的设计)
                                            preprocess_preview = gr.Image(
                                                label="预处理效果图预览",
                                                interactive=False,
                                                elem_classes=["cnet-image"],
                                                visible=False,
                                                height=300
                                            )
                                    
                                    # 根据项目规范，对于Inpainting模型，我们需要使用background作为原始图像
                                    # foreground用于蒙版绘制，但隐藏其UI显示
                                    control_image.background.visible = True
                                    control_image.background.render = True
                                    # 隐藏foreground组件的UI显示，但保持功能可用
                                    control_image.foreground.visible = False
                                    control_image.foreground.render = False
                                    
                                    with gr.Row(elem_classes="controlnet_image_controls"):
                                        # 创建一个Radio按钮用于选择预处理器类别
                                        preprocessor_category = gr.Radio(
                                            choices=["All", "Canny", "Depth", "Pose", "Lineart", "Softedge"],
                                            value="All",
                                            label="预处理器类别",
                                            interactive=True,
                                            elem_classes=["cnet-preprocessor-category"]
                                        )
                                        
                                        controlnet_preprocessor = gr.Dropdown(
                                            choices=CONTROLNET_PREPROCESSORS,
                                            value="none",  # 保持默认值为"none"
                                            label="预处理器",
                                            interactive=True,
                                            elem_classes=["cnet-preprocessor-dropdown"]
                                        )
                                        
                                        # 添加预处理按钮，使用爆炸图标
                                        from modules.ui_components import ToolButton
                                        preprocess_button = ToolButton(
                                            value="\U0001F4A5",  # 💥爆炸图标
                                            elem_classes=["cnet-run-preprocessor", "cnet-toolbutton"],
                                            tooltip="运行预处理器"
                                        )
                                        
                                        controlnet_model = gr.Dropdown(
                                            choices=qwen_image_controlnet_models,
                                            value="无",  # 默认选择"无"
                                            label="ControlNet 模型",
                                            interactive=True
                                        )
                                        
                                        refresh_models_button = ToolButton(
                                            value="\U0001f504",  # 🔄刷新图标
                                            elem_classes=["cnet-toolbutton"],
                                            tooltip="刷新模型列表"
                                        )
                                    
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
                                   
                    # 将输出组件放在右侧列中（在按钮点击事件之前定义）
                    with gr.Column():
                        # 调整图像组件的显示尺寸，使用Gallery组件显示多个图像
                        text_to_image_output = gr.Gallery(label="生成结果", interactive=False, height=512, object_fit="contain", columns=3)
                        text_to_image_status = gr.Textbox(label="状态", interactive=False)
                        
                        # 添加打开输出目录按钮
                        open_output_dir_button = gr.Button("打开输出目录")
                        open_output_dir_button.click(
                            fn=open_output_directory,
                            inputs=[],
                            outputs=[text_to_image_status]
                        )
                        
                        # 添加队列功能区域
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
                        
                        # 生成按钮
                        text_to_image_button = gr.Button("生成图像")
                    
        # 设置事件处理
        preprocessor_category.change(
            fn=update_preprocessor_choices,
            inputs=[preprocessor_category],
            outputs=[controlnet_preprocessor]
        )
        
        # 保留控制图像变化事件处理
        control_image.background.change(
            fn=on_preprocess_params_change,
            inputs=[control_image.background, controlnet_preprocessor, preprocess_button],
            outputs=[preprocess_preview]
        )
        
        # 添加控制图像上传事件处理，自动设置尺寸
        control_image.background.upload(
            fn=on_control_image_upload,
            inputs=[control_image.background],
            outputs=[text_to_image_width, text_to_image_height]
        )
        
        # 预处理按钮点击事件（手动预览）
        preprocess_button.click(
            fn=on_preprocess_button_click,
            inputs=[control_image.background, controlnet_preprocessor],
            outputs=[preprocess_preview]
        )

        text_to_image_button.click(
            fn=wrap_gradio_gpu_call(run_text_to_image, extra_outputs=[None, '']),  # 使用wrap_gradio_gpu_call直接包装原函数
            inputs=[
                text_to_image_prompt,
                text_to_image_negative_prompt,
                text_to_image_width,
                text_to_image_height,
                text_to_image_steps,
                text_to_image_cfg,
                text_to_image_model,
                text_to_image_scheduler,
                text_to_image_scheduler_type,  # 添加调度器参数
                text_to_image_lora_1,
                text_to_image_lora_2,
                text_to_image_lora_weight_1,
                text_to_image_lora_weight_2,
                text_to_image_seed,
                text_to_image_batch_size,
                controlnet_model,
                control_image.background,
                control_image.foreground,
                controlnet_conditioning_scale,
                controlnet_preprocessor,
                controlnet_start,
                controlnet_end,
            ],
            outputs=[text_to_image_output, text_to_image_status]
        )
        
        # 为按钮添加事件
        add_to_queue_btn.click(
            fn=lambda prompt, negative_prompt, width, height, steps, cfg_scale, \
               model_file, scheduler, scheduler_type, lora_model_1, lora_model_2, \
               lora_weight_1, lora_weight_2, seed, batch_size, \
               controlnet_model, control_image_bg, control_image_fg, controlnet_conditioning_scale, \
               controlnet_preprocessor, controlnet_start, controlnet_end: \
               add_to_queue('qwen_image', 
                   prompt, negative_prompt, width, height, steps, cfg_scale, 
                   model_file, scheduler, scheduler_type, lora_model_1, lora_model_2, 
                   lora_weight_1, lora_weight_2, seed, batch_size,
                   controlnet_model, control_image_bg, control_image_fg, controlnet_conditioning_scale,
                   controlnet_preprocessor, controlnet_start, controlnet_end),
            inputs=[
                text_to_image_prompt,
                text_to_image_negative_prompt,
                text_to_image_width,
                text_to_image_height,
                text_to_image_steps,
                text_to_image_cfg,
                text_to_image_model,
                text_to_image_scheduler,
                text_to_image_scheduler_type,
                text_to_image_lora_1,
                text_to_image_lora_2,
                text_to_image_lora_weight_1,
                text_to_image_lora_weight_2,
                text_to_image_seed,
                text_to_image_batch_size,
                controlnet_model,
                control_image.background,
                control_image.foreground,
                controlnet_conditioning_scale,
                controlnet_preprocessor,
                controlnet_start,
                controlnet_end,
            ],
            outputs=[queue_operation_status]
        )
        
        # 更新队列状态
        def update_queue_status():
            return get_queue_status()
        
        def update_detailed_queue_status():
            return get_detailed_queue_status()

        # 队列控制功能
        def get_queue_status():
            from modules import progress
            pending_count = len(progress.pending_tasks)
            current = "无" if progress.current_task is None else (progress.current_task if len(progress.current_task) < 30 else progress.current_task[:27]+"...")
            finished_count = len(progress.finished_tasks)
            return f"待处理: {pending_count} | 当前任务: {current} | 已完成: {finished_count}"
        # 队列执行功能（模拟，实际上由WebUI系统自动处理）
        def execute_queue():
            from modules import progress
            if len(progress.pending_tasks) > 0:
                return f"开始处理 {len(progress.pending_tasks)} 个队列任务..."
            else:
                return "队列为空，无任务需要处理"

        # 清空队列功能
        def clear_queue():
            from modules import progress
            cleared_count = len(progress.pending_tasks)
            progress.pending_tasks.clear()
            return f"已清空队列，共清除 {cleared_count} 个待处理任务"


        # 添加按钮点击事件来更新队列状态
        add_to_queue_btn.click(
            fn=update_queue_status,
            inputs=[],
            outputs=[queue_status_text]
        )
        
        add_to_queue_btn.click(
            fn=update_detailed_queue_status,
            inputs=[],
            outputs=[detailed_queue_status]
        )
        
        process_queue_btn.click(
            fn=process_queue,
            inputs=[],
            outputs=[text_to_image_status, text_to_image_output]
        )
        
        process_queue_btn.click(
            fn=update_queue_status,
            inputs=[],
            outputs=[queue_status_text]
        )
        
        process_queue_btn.click(
            fn=update_detailed_queue_status,
            inputs=[],
            outputs=[detailed_queue_status]
        )
        
        # 清空队列按钮事件
        def clear_queue():
            global task_queue
            task_queue = queue.Queue()  # 重新创建空队列
            return "队列已清空"
        
        clear_queue_btn.click(
            fn=clear_queue,
            inputs=[],
            outputs=[queue_operation_status]
        )
        
        clear_queue_btn.click(
            fn=update_queue_status,
            inputs=[],
            outputs=[queue_status_text]
        )
        
        clear_queue_btn.click(
            fn=update_detailed_queue_status,
            inputs=[],
            outputs=[detailed_queue_status]
        )
        
        open_output_dir_button.click(
            fn=open_output_directory,
            inputs=[],
            outputs=[text_to_image_status]
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
            "text_to_image_scheduler_type": text_to_image_scheduler_type,  # 添加调度器组件
            "text_to_image_lora_1": text_to_image_lora_1,
            "text_to_image_lora_2": text_to_image_lora_2,
            "text_to_image_lora_weight_1": text_to_image_lora_weight_1,
            "text_to_image_lora_weight_2": text_to_image_lora_weight_2,
            "text_to_image_seed": text_to_image_seed,
            "text_to_image_batch_size": text_to_image_batch_size,
            "text_to_image_negative_prompt": text_to_image_negative_prompt,
            "text_to_image_button": text_to_image_button,
            "text_to_image_output": text_to_image_output,
            "text_to_image_status": text_to_image_status,
        }
        
        # 如果成功创建了angle_selector_component，也加入返回字典
        if 'angle_selector_component' in locals():
            result["angle_selector_component"] = angle_selector_component
            
        # 添加队列相关组件到返回字典
        result.update({
            "queue_status_text": queue_status_text,
            "add_to_queue_btn": add_to_queue_btn,
            "process_queue_btn": process_queue_btn,
            "clear_queue_btn": clear_queue_btn,
            "queue_operation_status": queue_operation_status,
            "detailed_queue_status": detailed_queue_status
        })
        
        # 添加刷新LoRA模型列表的函数
        def refresh_lora_models():
            """刷新LoRA模型列表"""
            try:
                lora_choices = get_lora_choices(qwenimage_lora_dir)
                
                # 返回更新后的选项
                return [
                    gr.update(choices=lora_choices),
                    gr.update(choices=lora_choices)
                ]
            except Exception as e:
                print(f"刷新LoRA模型列表时出错: {e}")
                import traceback
                traceback.print_exc()
                return [
                    gr.update(),
                    gr.update()
                ]
        
        # 为刷新按钮添加事件监听
        try:
            lora_refresh_button.click(
                fn=refresh_lora_models,
                inputs=[],
                outputs=[text_to_image_lora_1, text_to_image_lora_2]
            )
        except AttributeError:
            # 忽略在没有Gradio上下文时的错误
            pass
        
        
        print("Qwen Image UI 创建完成")
        return result
        
    except Exception as e:
        print(f"创建Qwen Image UI时出错: {e}")
        traceback.print_exc()
        # 返回空字典而不是None，避免破坏UI
        return {}

# ==================== 任务队列相关功能 ====================
# 创建全局任务队列
task_queue = queue.Queue()

def add_to_queue(task_type, *args):
    """将任务添加到队列"""
    # 根据任务类型解析参数
    if task_type == 'qwen_image':
        # qwen_image任务参数: prompt, negative_prompt, width, height, steps, cfg_scale, 
        # model_file, scheduler, scheduler_type, lora_model_1, lora_model_2, 
        # lora_weight_1, lora_weight_2, seed, batch_size,
        # controlnet_model, control_image, control_mask, controlnet_conditioning_scale,
        # controlnet_preprocessor, controlnet_start, controlnet_end
        task_info = {
            'type': task_type,
            'params': {
                'prompt': args[0],
                'negative_prompt': args[1],
                'width': args[2],
                'height': args[3],
                'steps': args[4],
                'cfg_scale': args[5],
                'model_file': args[6],
                'scheduler': args[7],
                'scheduler_type': args[8],
                'lora_model_1': args[9],
                'lora_model_2': args[10],
                'lora_weight_1': args[11],
                'lora_weight_2': args[12],
                'seed': args[13],
                'batch_size': args[14],
                'controlnet_model': args[15],
                'control_image': args[16],
                'control_mask': args[17],
                'controlnet_conditioning_scale': args[18],
                'controlnet_preprocessor': args[19],
                'controlnet_start': args[20],
                'controlnet_end': args[21]
            }
        }
    
    task = {
        'info': task_info,
        'args': args
    }
    task_queue.put(task)
    
    # 返回当前队列大小和任务信息摘要
    queue_size = task_queue.qsize()
    task_summary = f"生成任务: {args[2]}x{args[3]}, 步数: {args[4]}, 提示词: {args[0][:30]}{'...' if len(args[0]) > 30 else ''}"
    
    return f"任务已添加 - {task_summary}，当前队列大小: {queue_size}"


def process_queue():
    """处理队列中的所有任务"""
    results = []
    statuses = []
    task_num = 1
    
    while not task_queue.empty():
        task = task_queue.get()
        task_info = task['info']
        args = task['args']
        task_type = task_info['type']
        
        try:
            if task_type == 'qwen_image':
                result_images, status = run_text_to_image(*args)
                results.extend(result_images if result_images else [])
                statuses.append(f"任务{task_num}: {status}")
            else:
                status = f"未知的任务类型: {task_type}"
                statuses.append(status)
                
            task_num += 1
        except Exception as e:
            error_msg = f"任务{task_num}执行失败: {str(e)}"
            statuses.append(error_msg)
            print(error_msg)
            traceback.print_exc()
            task_num += 1
    
    if results:
        return "所有任务已完成: " + "; ".join(statuses), results
    else:
        return "队列为空，没有任务需要执行", []


def get_queue_status():
    """获取当前队列状态"""
    size = task_queue.qsize()
    return f"当前队列大小: {size}"


def get_detailed_queue_status():
    """获取详细的队列状态，包括任务参数"""
    import copy
    temp_queue = queue.Queue()
    details = []
    idx = 1
    
    # 临时取出所有任务，记录详情，并放回原队列
    while not task_queue.empty():
        task = task_queue.get()
        temp_queue.put(task)
        
        task_info = task['info']
        task_type = task_info['type']
        
        if task_type == 'qwen_image':
            detail = f"任务{idx}: 生成图像 - {task_info['params']['width']}x{task_info['params']['height']}"
            detail += f", 步数: {task_info['params']['steps']}, CFG: {task_info['params']['cfg_scale']}"
            detail += f", 模型: {task_info['params']['model_file']}"
            detail += f", 提示词: {task_info['params']['prompt'][:30]}{'...' if len(task_info['params']['prompt']) > 30 else ''}"
            if task_info['params']['lora_model_1']:
                detail += f", LoRA1: {task_info['params']['lora_model_1']}"
            if task_info['params']['lora_model_2']:
                detail += f", LoRA2: {task_info['params']['lora_model_2']}"
        
        details.append(detail)
        idx += 1
    
    # 将任务放回原队列
    while not temp_queue.empty():
        task_queue.put(temp_queue.get())
    
    if details:
        return "\n".join(details)
    else:
        return "队列为空"


# ==================== 模块可用性变量 ====================
QWEN_IMAGE_MODULE_AVAILABLE = QWEN_IMAGE_AVAILABLE
