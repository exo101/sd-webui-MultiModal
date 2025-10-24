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
import numpy as np
from PIL import Image
from modules import script_callbacks, shared
from modules.paths import script_path
import shutil

# 获取当前脚本所在目录
current_dir = Path(__file__).parent
scripts_dir = current_dir
qwen_image_dir = current_dir.parent / "qwen-image"
# 修改模型路径为指定目录
models_dir = Path(shared.models_path) / "qwen-image"
qwenimage_models_dir = models_dir / "qwenimage"
qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
qwenimage_lora_dir = qwen_image_dir / "loras"
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

# 定义需要隐藏的预处理器
HIDDEN_PREPROCESSORS = {
    'shuffle', 
    'invert (from white bg & black line)',
    'InsightFace+CLIP-H (IPAdapter)',
    'threshold',
    't2ia_sketch_pidi',
    't2ia_color_grid',
    'depth_marigold'
}

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
        
        # 获取所有预处理器并按优先级排序
        preprocessors = []
        sorted_preprocessors = []
        
        # 先添加None预处理器
        none_preprocessor = supported_preprocessors.get('None')
        if none_preprocessor:
            sorted_preprocessors.append(none_preprocessor)
        
        # 添加其他预处理器并按优先级排序
        other_preprocessors = [p for k, p in supported_preprocessors.items() if k != 'None' and k not in HIDDEN_PREPROCESSORS]
        other_preprocessors = sorted(other_preprocessors, key=lambda x: str(x.sorting_priority).zfill(8) + x.name, reverse=True)
        sorted_preprocessors.extend(other_preprocessors)
        
        # 构建预处理器选项列表，使用与WebUI ControlNet一致的显示名称
        for preprocessor in sorted_preprocessors:
            # 使用预处理器的name属性作为内部标识符，label属性作为显示名称
            internal_name = preprocessor.name
            display_name = getattr(preprocessor, 'label', internal_name)
            preprocessors.append((internal_name, display_name))
        
        return preprocessors
    except Exception as e:
        print(f"获取预处理器列表时出错: {e}")
        import traceback
        traceback.print_exc()
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
            ("lineart_standard", "Lineart Standard (from white bg & black line)"),
            ("lineart_realistic", "Lineart Realistic"), 
            ("lineart_anime", "Lineart Anime"),
            ("lineart_anime_denoise", "Lineart Anime Denoise")
        ]

def get_controlnet_preprocessor_categories():
    """获取预处理器分类"""
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
        
        # 获取所有预处理器分类标签
        categories = set()
        for k, p in supported_preprocessors.items():
            if k not in HIDDEN_PREPROCESSORS:
                categories.update(p.tags)
        
        # 转换为排序列表并添加"All"选项
        categories = sorted(list(categories))
        return ['All'] + categories
    except Exception as e:
        print(f"获取预处理器分类时出错: {e}")
        import traceback
        traceback.print_exc()
        # 出错时返回默认分类
        return ['All', 'Canny', 'Depth', 'OpenPose', 'Lineart', 'SoftEdge']

def get_filtered_preprocessors(category):
    """根据分类获取预处理器列表"""
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
        
        # 如果是"All"分类，返回所有预处理器
        if category == 'All':
            return get_controlnet_preprocessors()
        
        # 根据分类过滤预处理器
        filtered_preprocessors = []
        for k, p in supported_preprocessors.items():
            if k not in HIDDEN_PREPROCESSORS and (category in p.tags or k == 'None'):
                internal_name = p.name
                display_name = getattr(p, 'label', internal_name)
                filtered_preprocessors.append((internal_name, display_name))
        
        return filtered_preprocessors
    except Exception as e:
        print(f"获取分类预处理器时出错: {e}")
        import traceback
        traceback.print_exc()
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
            ("lineart_standard", "Lineart Standard (from white bg & black line)"),
            ("lineart_realistic", "Lineart Realistic"), 
            ("lineart_anime", "Lineart Anime"),
            ("lineart_anime_denoise", "Lineart Anime Denoise")
        ]


# 获取预处理器选项
CONTROLNET_PREPROCESSORS = get_controlnet_preprocessors()


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
    """解析脚本的标准输出，提取关键信息"""
    result = {}
    lines = output.strip().split('\n')
    
    for line in lines:
        if line.startswith("SUCCESS:"):
            result["image_path"] = line[8:].strip()  # 去掉"SUCCESS:"前缀
        elif line.startswith("INFO_FILE:"):
            result["info_file"] = line[10:].strip()  # 去掉"INFO_FILE:"前缀
        # 可以添加更多关键字的解析
    
    return result

# 预处理器类型映射（UI显示名称到内部标识符）
# 注意：现在我们直接使用WebUI的预处理器管理系统，不再需要手动维护映射表
# 但为了向后兼容，保留此变量，其值通过动态方式获取
def get_preprocessor_display_to_internal():
    """动态获取预处理器显示名称到内部标识符的映射"""
    mapping = {}
    for internal_name, display_name in CONTROLNET_PREPROCESSORS:
        mapping[display_name] = internal_name
    
    # 不再将"None"映射到"none"，因为None表示不使用预处理器
    return mapping

PREPROCESSOR_DISPLAY_TO_INTERNAL = get_preprocessor_display_to_internal()

# 全局缓存预处理器显示名称到内部标识符的映射，避免重复计算
PREPROCESSOR_CACHE = {}

def preprocess_control_image(image_path, preprocessor_display_name):
    """预处理控制图像"""
    try:
        print(f"组合处理: image_path={image_path}, preprocessor_type={preprocessor_display_name}")
        
        # 特殊处理"None"预处理器 - 直接返回原始图像路径，不执行任何预处理
        if preprocessor_display_name in [None, "None", "none"]:
            print("使用无预处理模式，直接返回原始图像路径")
            return image_path
            
        # 尝试从缓存获取映射，如果没有则计算并缓存
        if preprocessor_display_name in PREPROCESSOR_CACHE:
            mapped_preprocessor_type = PREPROCESSOR_CACHE[preprocessor_display_name]
        else:
            mapped_preprocessor_type = PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_display_name, "none")
            PREPROCESSOR_CACHE[preprocessor_display_name] = mapped_preprocessor_type
            
        print(f"开始使用预处理器 {preprocessor_display_name} ({mapped_preprocessor_type}) 处理图像: {image_path}")
        
        # 构造参数
        args = {
            "image_path": image_path,
            "preprocessor_type": mapped_preprocessor_type
        }
        
        # 创建临时参数文件
        timestamp = int(time.time() * 1000)
        args_file = qwen_image_dir / f"temp_args_preprocess_{timestamp}.json"
        with open(args_file, 'w', encoding='utf-8') as f:
            json.dump(args, f, ensure_ascii=False, indent=2)
        
        # 构造命令并执行
        cmd = [
            main_python,
            str(qwen_image_dir / "qwen_image_scripts.py"),
            str(args_file)
        ]
        
        print(f"执行预处理命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=str(qwen_image_dir))
        
        # 删除临时参数文件
        args_file.unlink(missing_ok=True)
        
        # 处理结果
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.startswith("SUCCESS:"):
                    processed_image_path = line[8:].strip()
                    print(f"预处理成功，结果路径: {processed_image_path}")
                    return processed_image_path
                elif "错误" in line or "失败" in line:
                    print(f"预处理过程中出现错误: {line}")
                    return None
            
            # 如果没有找到SUCCESS行，但返回码是0，可能是其他输出
            print(f"预处理完成，但未找到明确的成功标识。标准输出: {result.stdout}")
            return result.stdout.strip()
        else:
            error_msg = f"预处理脚本执行失败 (返回码: {result.returncode})\n标准输出: {result.stdout}\n错误输出: {result.stderr}"
            print(error_msg)
            return None
            
    except Exception as e:
        error_msg = f"预处理控制图像时发生异常: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

# 控制图像交互性更新函数
def update_control_image_interactivity(model_name):
    """根据选择的ControlNet模型更新控制图像组件的交互性"""
    # 检查是否为Inpainting模型
    is_inpainting_model = model_name and "inpaint" in model_name.lower()
    
    # 对于Inpainting模型，保持控制图像可见和可交互
    # 对于其他模型，也保持可见和可交互
    return gr.update(visible=True, interactive=True)

# 预处理器可见性更新函数
def update_preprocessor_visibility(model_name):
    """根据选择的ControlNet模型更新预处理器组件的可见性"""
    # 检查是否为Inpainting模型
    is_inpainting_model = model_name and "inpaint" in model_name.lower()
    
    # 对于所有模型类型，预处理器功能都应该可见
    # 即使是Inpainting模型，也应该可以预览预处理效果
    return [
        gr.update(visible=True),      # controlnet_preprocessor
        gr.update(visible=True),      # preprocess_button
        gr.update(visible=True)       # preprocess_preview
    ]


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

# 添加函数来保存处理后的图像
def save_processed_image(processed_image):
    """保存处理后的图像到临时文件"""
    try:
        if processed_image is None:
            return None
            
        # 创建临时目录
        temp_dir = qwen_image_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # 生成唯一文件名
        timestamp = int(time.time() * 1000)
        temp_path = temp_dir / f"preprocess_preview_{timestamp}.png"
        
        # 保存图像
        saved_path = save_numpy_image(processed_image, temp_path)
        if saved_path and os.path.exists(saved_path):
            return saved_path
        else:
            print("无法保存处理后的图像")
            return None
    except Exception as e:
        print(f"保存处理后图像时出错: {e}")
        traceback.print_exc()
        return None

def run_text_to_image(prompt, negative_prompt, width, height, steps, cfg_scale, 
                      model_file, scheduler, controlnet_enable=False, controlnet_model=None,
                      control_image=None, control_mask=None, controlnet_conditioning_scale=1.0,
                      controlnet_preprocessor="none", controlnet_start=0.0, controlnet_end=1.0,
                      seed=-1, batch_size=1):  # 添加种子和批次参数
    try:
        print("开始执行文生图功能...")
        # 如果没有提供控制图像，则从outputs目录中查找最新的图像
        if control_image is None and controlnet_enable:
            # 查找outputs目录中的最新图像文件
            outputs_dir = Path(shared.data_path) / "outputs"
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
            latest_image = None
            latest_time = 0
            
            # 遍历所有子目录查找图像文件
            for extension in image_extensions:
                for image_path in outputs_dir.rglob(extension):
                    if image_path.stat().st_mtime > latest_time:
                        latest_time = image_path.stat().st_mtime
                        latest_image = image_path
            
            if latest_image:
                control_image = str(latest_image)
                print(f"从outputs目录自动选择最新图像: {latest_image}")
        
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
        
        # 检查是否使用Inpainting模型
        is_inpainting_model = controlnet_model and "inpaint" in controlnet_model.lower()
        
        # 如果有控制图像，调整其尺寸以确保与模型兼容（包括Inpainting模型）
        if processed_control_image and controlnet_enable and CONTROLNET_AVAILABLE:
            try:
                from PIL import Image
                pil_image = Image.open(processed_control_image).convert("RGB")
                original_width, original_height = pil_image.size
                print(f"原始控制图像尺寸: {original_width}x{original_height}")
                
                # 调整图像尺寸为64的倍数以匹配模型要求
                target_width = ((original_width + 31) // 64) * 64
                target_height = ((original_height + 31) // 64) * 64
                
                # 确保尺寸在合理范围内
                target_width = max(256, min(2048, target_width))
                target_height = max(256, min(2048, target_height))
                
                # 如果尺寸发生了变化，则调整图像
                if target_width != original_width or target_height != original_height:
                    print(f"调整控制图像尺寸: {original_width}x{original_height} -> {target_width}x{target_height}")
                    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    # 保存调整后的图像到临时文件
                    temp_dir = qwen_image_dir / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    resized_image_path = temp_dir / f"resized_control_image_{int(time.time() * 1000)}.png"
                    pil_image.save(resized_image_path)
                    processed_control_image = str(resized_image_path)
                    print(f"调整后控制图像保存到: {resized_image_path}")
                
                # 更新width和height参数以匹配控制图像尺寸（仅对非Inpainting模型）
                if not is_inpainting_model:
                    width = target_width
                    height = target_height
                    print(f"更新生成尺寸为: {width}x{height}")
                    
            except Exception as e:
                print(f"处理控制图像尺寸时出错: {e}")
                traceback.print_exc()
        
        # 准备参数
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "model_file": model_file,
            "model_dir": str(qwenimage_models_dir),
            "scheduler": scheduler,
            "controlnet_enable": controlnet_enable,
            "controlnet_model": controlnet_model,
            "control_image": processed_control_image,
            "control_mask": processed_control_mask,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "controlnet_preprocessor": controlnet_preprocessor,
            "controlnet_start": controlnet_start,
            "controlnet_end": controlnet_end,
            "output_dir": str(qwen_image_outputs_dir),
            "seed": seed,              # 添加种子参数
            "batch_size": batch_size   # 添加批次参数
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
            print("图像生成成功")
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
            return None, "请至少上传一张图像或确保outputs目录中有图像文件", "暂无生成记录"
        
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
                            
                            # 添加随机种子输入框
                            text_to_image_seed = gr.Number(
                                label="随机种子 (-1为随机)",
                                value=-1,
                                precision=0,
                                min_width=120
                            )
                            
                        with gr.Row():
                            # 添加生成批次输入框
                            text_to_image_batch_size = gr.Slider(
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=1,
                                label="生成批次",
                                min_width=120
                            )
                            
                            # Add base model selection dropdown
                            text_to_image_model = gr.Dropdown(
                                choices=qwenimage_model_choices,
                                label="基础模型选择",
                                value=None,
                                interactive=True,
                                min_width=150
                            )
                        
                        # 添加ControlNet相关组件 (参考WebUI中ControlNet的设计)
                        with gr.Accordion("ControlNet 控制", open=False):
                            with gr.Row():
                                controlnet_enable = gr.Checkbox(
                                    label="启用ControlNet",
                                    value=False,
                                    interactive=True
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
                                            # 使用统一定义的控制网络模型目录
                                            for model_name in known_models:
                                                model_path = qwenimage_controlnet_dir / model_name
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
                                        # 添加预处理器分类选择器
                                        controlnet_preprocessor_category = gr.Dropdown(
                                            choices=get_controlnet_preprocessor_categories(),
                                            value="All",
                                            label="预处理器分类",
                                            interactive=True,
                                            elem_classes=["cnet-preprocessor-category-dropdown"]
                                        )
                                        
                                        controlnet_preprocessor = gr.Dropdown(
                                            choices=CONTROLNET_PREPROCESSORS,
                                            value="none",
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
                                            value=qwen_image_controlnet_models[0][0] if qwen_image_controlnet_models else "Qwen-Image-ControlNet-Union",
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
                        
                        # 当模型选择改变时，更新控制图像组件
                        controlnet_model.change(
                            fn=update_control_image_interactivity,
                            inputs=[controlnet_model],
                            outputs=[control_image.background]
                        )
                        
                        # 当模型选择改变时，更新预处理功能可见性
                        controlnet_model.change(
                            fn=update_preprocessor_visibility,
                            inputs=[controlnet_model],
                            outputs=[controlnet_preprocessor, preprocess_button, preprocess_preview]
                        )
                        
                        # 添加预处理器分类选择的事件处理
                        controlnet_preprocessor_category.change(
                            fn=lambda category: gr.update(choices=get_filtered_preprocessors(category)),
                            inputs=[controlnet_preprocessor_category],
                            outputs=[controlnet_preprocessor]
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
        
        def on_preprocess_params_change(image_path, preprocessor_type, preprocess_refresh):
            """当预处理参数改变时触发"""
            try:
                print(f"预处理参数变更: image_path={image_path}, preprocessor_type={preprocessor_type}, preprocess_refresh={preprocess_refresh}")
                if preprocess_refresh and image_path and os.path.exists(image_path) and preprocessor_type != "none":
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
            
            if image_path and os.path.exists(image_path) and preprocessor_type not in [None, "none", "None", "无", "none (default)"]:
                # 直接调用预处理函数而不是通过子进程
                processed_image = preprocess_control_image(image_path, preprocessor_type)
                if processed_image is not None:
                    # 检查返回的是否是有效的numpy数组
                    if isinstance(processed_image, np.ndarray) and processed_image.size > 0:
                        temp_path = save_processed_image(processed_image)
                        if temp_path and os.path.exists(temp_path):
                            preview_update = gr.update(visible=True, value=temp_path)
                        else:
                            print("无法保存预览图像")
                    # 检查返回的是否是有效的图像路径
                    elif isinstance(processed_image, str) and os.path.exists(processed_image):
                        print(f"直接返回图像路径: {processed_image}")
                        preview_update = gr.update(visible=True, value=processed_image)
                    else:
                        print("预处理返回了无效结果")
                else:
                    print("预处理未返回有效图像")
            elif image_path and os.path.exists(image_path) and preprocessor_type in [None, "none", "None", "无", "none (default)"]:
                # 当预处理器为None时，直接使用原始图像作为预览
                print("使用无预处理模式，直接使用原始图像作为预览")
                preview_update = gr.update(visible=True, value=image_path)
            else:
                print("不满足预览条件")
            
            return preview_update
        
        # 组合控制图像和预处理参数改变事件的处理函数
        def combined_control_image_handler(background_input, foreground_input, preprocessor_type, controlnet_model):
            """处理控制图像变化的组合函数"""
            image_path = None
            mask_path = None
            width = None
            height = None
            
            # 检查是否为Inpainting模型
            is_inpainting_model = controlnet_model and "inpaint" in controlnet_model.lower()
            
            # 处理背景图像（实际图像）
            if isinstance(background_input, str):  # 文件路径
                image_path = background_input
            elif isinstance(background_input, np.ndarray):  # numpy数组
                # 为numpy数组创建临时文件
                temp_dir = qwen_image_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                image_path = temp_dir / f"control_image_temp_{int(time.time() * 1000)}.png"
                saved_path = save_numpy_image(background_input, image_path)
                if saved_path:
                    image_path = saved_path
            
            # 处理前景图像（蒙版）
            if isinstance(foreground_input, str):  # 文件路径
                mask_path = foreground_input
            elif isinstance(foreground_input, np.ndarray):  # numpy数组
                # 为numpy数组创建临时文件
                temp_dir = qwen_image_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                mask_path = temp_dir / f"control_mask_temp_{int(time.time() * 1000)}.png"
                saved_path = save_numpy_image(foreground_input, mask_path)
                if saved_path:
                    mask_path = saved_path
            
            # 对于所有模型类型（包括Inpainting），都获取背景图像尺寸
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    width, height = image.size
                except Exception as e:
                    print(f"读取图像尺寸时出错: {e}")
            
            # 处理预览更新
            preview_update = gr.update(visible=False)
            # 注意：即使是Inpainting模型，也应该可以预览预处理效果
            if image_path and os.path.exists(image_path) and preprocessor_type not in [None, "none", "None", "无", "none (default)"]:
                print(f"组合处理: image_path={image_path}, preprocessor_type={preprocessor_type}")
                # 直接调用预处理函数而不是通过子进程
                processed_image = preprocess_control_image(image_path, preprocessor_type)
                if processed_image is not None:
                    # 检查返回的是否是有效的numpy数组
                    if isinstance(processed_image, np.ndarray) and processed_image.size > 0:
                        temp_path = save_processed_image(processed_image)
                        if temp_path and os.path.exists(temp_path):
                            print(f"组合处理预览图像已保存到: {temp_path}")
                            preview_update = gr.update(visible=True, value=temp_path)
                        else:
                            print("组合处理无法保存预览图像")
                    # 检查返回的是否是有效的图像路径
                    elif isinstance(processed_image, str) and os.path.exists(processed_image):
                        print(f"组合处理直接返回图像路径: {processed_image}")
                        preview_update = gr.update(visible=True, value=processed_image)
                    else:
                        print("组合处理预处理返回了无效结果")
                else:
                    print("组合处理预处理未返回有效图像")
            elif image_path and os.path.exists(image_path) and preprocessor_type in [None, "none", "None", "无", "none (default)"]:
                # 当预处理器为None时，直接使用原始图像作为预览
                print(f"组合处理: image_path={image_path}, preprocessor_type={preprocessor_type}")
                print("使用无预处理模式，直接使用原始图像作为预览")
                preview_update = gr.update(visible=True, value=image_path)
            else:
                print("不满足预览条件")
            
            # 如果获取到有效的宽度和高度，则更新宽度和高度滑块（适用于所有模型类型）
            if width is not None and height is not None:
                # 限制宽度和高度在滑块的有效范围内
                width = max(256, min(2048, width))
                height = max(256, min(2048, height))
                # 确保宽度和高度是64的倍数
                width = (width // 64) * 64
                height = (height // 64) * 64
                return preview_update, gr.update(value=width), gr.update(value=height)
            else:
                return preview_update, gr.update(), gr.update()
        
        # 绑定组合事件处理程序
        control_image.background.change(
            fn=combined_control_image_handler,
            inputs=[control_image.background, control_image.foreground, controlnet_preprocessor, controlnet_model],
            outputs=[preprocess_preview, text_to_image_width, text_to_image_height]
        )
        control_image.foreground.change(
            fn=combined_control_image_handler,
            inputs=[control_image.background, control_image.foreground, controlnet_preprocessor, controlnet_model],
            outputs=[preprocess_preview, text_to_image_width, text_to_image_height]
        )
        
        # 预处理按钮点击事件（手动预览）
        preprocess_button.click(
            fn=on_preprocess_button_click,
            inputs=[control_image.background, controlnet_preprocessor],
            outputs=[preprocess_preview]
        )

        # 更新text_to_image_button.click事件，添加新的参数
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
                control_image.background,  # 使用background组件作为控制图像
                control_image.foreground,  # 使用foreground组件作为控制蒙版
                controlnet_conditioning_scale,
                controlnet_preprocessor,
                controlnet_start,
                controlnet_end,
                text_to_image_seed,        # 添加随机种子参数
                text_to_image_batch_size   # 添加生成批次参数
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
            "control_image": control_image.background,  # 返回background组件
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
