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

import shutil
import webbrowser

# ==================== 常量定义 ====================
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

# ==================== 获取LoRA网址 ====================
def get_lora_urls():
    """获取LoRA模型URL映射"""
    try:
        # 读取LoRA网址配置文件
        config_path = Path(shared.cmd_opts.data_dir) / "models" / "Lora" / "configs" / "lora_urls.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                lora_urls = json.load(f)
            return lora_urls
        else:
            print(f"LoRA URLs配置文件不存在: {config_path}")
            return {}
    except Exception as e:
        print(f"读取LoRA URLs配置文件时出错: {e}")
        return {}

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
                    # 如果找不到预处理器，使用默认显示名称
                    display_name_map = {
                        "canny": "Canny",
                        "depth_midas": "Depth Midas",
                        "depth_leres": "Depth Leres",
                        "depth_leres++": "Depth Leres++",
                        "depth_anything": "Depth Anything",
                        "depth_anything_v2": "Depth Anything V2",
                        "depth_hand_refiner": "Depth Hand Refiner",
                        "depth_marigold": "Depth Marigold",
                        "depth_zoe": "Depth Zoe",
                        "openpose_full": "Openpose Full",
                        "openpose": "Openpose",
                        "openpose_face": "Openpose Face",
                        "openpose_faceonly": "Openpose Faceonly",
                        "openpose_hand": "Openpose Hand",
                        "dw_openpose_full": "DW Openpose Full",
                        "animal_openpose": "Animal Openpose",
                        "densepose": "Densepose (purple bg & purple torso)",
                        "densepose_parula": "Densepose Parula (black bg & blue torso)",
                        "lineart_standard": "Lineart Standard (from white bg & black line)",
                        "lineart_realistic": "Lineart Realistic",
                        "lineart_coarse": "Lineart Coarse",
                        "lineart_anime": "Lineart Anime",
                        "lineart_anime_denoise": "Lineart Anime Denoise",
                        "scribble_pidinet": "Scribble Pidinet",
                        "softedge_pidinet": "Softedge Pidinet",
                        "softedge_pidinet_safe": "Softedge Pidinet Safe",
                        "softedge_pidinstruct": "Softedge Pidinstruct",
                        "softedge_hed": "Softedge Hed",
                        "softedge_hedsafe": "Softedge Hedsafe"
                    }
                    display_name = display_name_map.get(name, name)
                    # 在显示名称前加上类别前缀
                    full_display_name = f"[{category}] {display_name}"
                    preprocessors.append((name, full_display_name))
        
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

def preprocess_control_image(image_input, preprocessor_display_name):
    """预处理控制图像"""
    try:
        image_path = None
        
        # 处理输入是numpy数组的情况
        if isinstance(image_input, np.ndarray):
            # 为numpy数组创建临时文件
            temp_dir = qwen_image_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            image_path = temp_dir / f"preprocess_input_{int(time.time() * 1000)}.png"
            save_result = save_numpy_image(image_input, image_path)
            if not save_result:
                print(f"无法保存numpy数组为图像文件")
                return None
            image_path = str(image_path)
        elif isinstance(image_input, str):
            image_path = image_input
        else:
            print(f"不支持的图像输入类型: {type(image_input)}")
            return None
            
        if not image_path or not os.path.exists(image_path):
            print(f"预处理图像路径无效: {image_path}")
            return None
        
        # 加载图像
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # 调整图像尺寸以匹配模型要求（确保是64的倍数）
        # 这可以解决"mat1 and mat2 shapes cannot be multiplied"错误
        original_width, original_height = image.size
        print(f"原始控制图像尺寸: {original_width}x{original_height}")
        
        # 将尺寸调整为64的倍数
        target_width = ((original_width + 31) // 64) * 64  # 向上取整到最接近的64倍数
        target_height = ((original_height + 31) // 64) * 64
        
        # 但也要确保不超过合理范围
        target_width = max(256, min(2048, target_width))
        target_height = max(256, min(2048, target_height))
        
        # 如果尺寸发生了变化，则调整图像
        if target_width != original_width or target_height != original_height:
            print(f"调整控制图像尺寸: {original_width}x{original_height} -> {target_width}x{target_height}")
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 将UI显示名称转换为内部标识符
        mapped_preprocessor_type = CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL.get(preprocessor_display_name, "none")
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
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(qwen_image_dir), timeout=1200)
        
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
                      controlnet_preprocessor="none", controlnet_start=0.0, controlnet_end=1.0):
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
            "control_mask": processed_control_mask if processed_control_image is not None else None,
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
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(qwen_image_dir), timeout=1200)
        
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
            image_paths = output_info["image_paths"]
            print(f"文生图生成成功，共生成 {len(image_paths)} 张图像")
            return image_paths, "生成成功", "生成完成"  # 返回图像路径列表
        else:
            error_msg = f"生成失败: {result.stdout}"
            print(f"生成失败: {error_msg}")
            return None, error_msg, "生成失败"
            
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"生成过程中出现异常: {error_msg}")
        traceback.print_exc()
        return None, error_msg, "生成失败"

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
def on_lora_model_change(lora_model_name):
    """当LoRA模型选择变化时触发"""
    try:
        if lora_model_name and lora_model_name != "无":
            # 获取LoRA URL映射
            lora_urls = get_lora_urls()
            
            # 从文件名中提取模型名称（去掉扩展名）
            model_name = Path(lora_model_name).stem
            
            # 查找对应的URL
            if model_name in lora_urls:
                url = lora_urls[model_name]
                # 返回JavaScript代码打开新窗口
                js_code = f"window.open('{url}', '_blank');"
                return gr.update()
            else:
                print(f"未找到LoRA模型 {model_name} 的URL")
        return gr.update()
    except Exception as e:
        print(f"处理LoRA模型变化时出错: {e}")
        return gr.update()

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
                                label="基础模型选择",
                                value=None,
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
                                
                                # 添加LoRA模型1的网址链接组件
                                with gr.Row():
                                    text_to_image_lora_url_1 = gr.HTML(
                                        value="<div style='font-size: 0.85em; color: #aaaaaa; margin-top: 2px; margin-bottom: 5px;'>模型网址: <a href='#' target='_blank' style='color: #4f94ef;'>未设置</a></div>",
                                        elem_classes=["lora-url-link"]
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
                                
                                # 添加LoRA模型2的网址链接组件
                                with gr.Row():
                                    text_to_image_lora_url_2 = gr.HTML(
                                        value="<div style='font-size: 0.85em; color: #aaaaaa; margin-top: 2px; margin-bottom: 5px;'>模型网址: <a href='#' target='_blank' style='color: #4f94ef;'>未设置</a></div>",
                                        elem_classes=["lora-url-link"]
                                    )
                                
                                # 添加刷新按钮
                                with gr.Row():
                                    from modules.ui_components import ToolButton
                                    lora_refresh_button = ToolButton(value='\U0001f504', elem_classes=['tool'])
                                    gr.Markdown("<div style='font-size: 0.8em; color: #aaaaaa; margin-top: 2px; margin-bottom: 2px;'>点击刷新按钮更新LoRA模型列表</div>")
                        
                        # 为LoRA组件添加点击事件
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
                        text_to_image_lora_1.select(
                            fn=None,
                            inputs=[text_to_image_lora_1],
                            outputs=[],
                            _js=lora_1_js
                        )
                        
                        text_to_image_lora_2.select(
                            fn=None,
                            inputs=[text_to_image_lora_2],
                            outputs=[],
                            _js=lora_2_js
                        )
                        
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
                        
                        # 删除与Inpainting模型相关的事件处理
                        
                        # 添加故障排除指南折叠模块
                        with gr.Accordion("Qwen模型报错排查指南", open=False):
                            gr.Markdown(
                                """
                                ### 常见问题及解决方案
                                
                                1. **显存要求**：推荐使用12GB及以上显存。低于12GB显存时生成时间会很漫长，且容易因显存不足而崩溃。

                                2.下载完整的模型组件，而不是仅下载模型文件

                                3. **非50系显卡模型选择**：对于非RTX 50系列显卡，推荐下载以下模型版本以获得更好兼容性：
                                   - `svdq-int4_r128-qwen-image.safetensors`
                                   - `svdq-int4_r128-qwen-image-lightningv1.1-8steps.safetensors`

                                4. 生成时间过长或爆显存可选择r32 lightningv1.1-4steps版本模型

                                5. **ControlNet图像尺寸限制**：在ControlNet控制模块中请勿上传超过1500像素的图像，否则可能会因显存不足而报错。  
                                                
                                6. **模型下载地址**：
                                   - [Qwen Image Models on ModelScope](https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image)
                                """
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
                controlnet_end
            ],
            outputs=[text_to_image_output, text_to_image_status]
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
        
        # 添加LoRA模型选择变化时更新网址链接的事件处理
        def update_lora_url_display(lora_name):
            """更新LoRA模型网址链接显示"""
            try:
                from pathlib import Path
                from modules import shared
                import json
                
                def normalize_lora_name(name):
                    """标准化LoRA模型名称，移除或添加常见的文件扩展名"""
                    if not name or name == "无":
                        return ""
                        
                    extensions = ['.safetensors', '.pt', '.bin', '.ckpt']
                    # 如果名称以任何扩展名结尾，则移除它
                    for ext in extensions:
                        if name.endswith(ext):
                            return name[:-len(ext)]
                    return name
                
                # 读取LoRA网址配置文件
                config_path = Path(shared.cmd_opts.data_dir) / "models" / "Lora" / "configs" / "lora_urls.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        lora_urls = json.load(f)
                    
                    # 标准化配置中的模型名称
                    normalized_urls = {}
                    for model_name, url in lora_urls.items():
                        normalized_name = normalize_lora_name(model_name)
                        normalized_urls[normalized_name] = url
                    
                    # 标准化输入的模型名称
                    normalized_lora_name = normalize_lora_name(lora_name)
                    
                    # 尝试精确匹配
                    if normalized_lora_name in normalized_urls:
                        url = normalized_urls[normalized_lora_name]
                        display_text = url.split('/')[-1][:30] + "..." if len(url.split('/')[-1]) > 30 else url.split('/')[-1]
                        return gr.update(value=f"<div style='font-size: 0.85em; color: #aaaaaa; margin-top: 2px; margin-bottom: 5px;'>模型网址: <a href='{url}' target='_blank' style='color: #4f94ef;' title='{url}'>{display_text}</a></div>")
                    
                    # 尝试模糊匹配
                    for model_name, url in normalized_urls.items():
                        if normalized_lora_name in model_name or model_name in normalized_lora_name:
                            display_text = url.split('/')[-1][:30] + "..." if len(url.split('/')[-1]) > 30 else url.split('/')[-1]
                            return gr.update(value=f"<div style='font-size: 0.85em; color: #aaaaaa; margin-top: 2px; margin-bottom: 5px;'>模型网址: <a href='{url}' target='_blank' style='color: #4f94ef;' title='{url}'>{display_text}</a></div>")
                
                # 如果没有找到匹配的网址
                return gr.update(value="<div style='font-size: 0.85em; color: #aaaaaa; margin-top: 2px; margin-bottom: 5px;'>模型网址: <a href='#' target='_blank' style='color: #4f94ef;'>未设置</a> <span style='color: #777777;'>(在/models/Lora/configs/lora_urls.json中添加)</span></div>")
            
            except Exception as e:
                print(f"更新LoRA网址显示时出错: {e}")
                import traceback
                traceback.print_exc()
                return gr.update(value="<div style='font-size: 0.85em; color: #aaaaaa; margin-top: 2px; margin-bottom: 5px;'>模型网址: <a href='#' target='_blank' style='color: #4f94ef;'>未设置</a></div>")
        
        # 为两个LoRA模型添加事件监听
        try:
            text_to_image_lora_1.change(
                fn=update_lora_url_display,
                inputs=[text_to_image_lora_1],
                outputs=[text_to_image_lora_url_1]
            )
            
            text_to_image_lora_2.change(
                fn=update_lora_url_display,
                inputs=[text_to_image_lora_2],
                outputs=[text_to_image_lora_url_2]
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

# ==================== 模块可用性变量 ====================
QWEN_IMAGE_MODULE_AVAILABLE = QWEN_IMAGE_AVAILABLE

# ==================== CSS 和 JS ====================
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

/* LoRA模型下拉框悬停提示 */
.lora-model-dropdown {
    cursor: pointer;
}

/* LoRA网址链接 */
.lora-url-link {
    font-size: 0.85em;
    color: #aaaaaa;
    margin-top: 2px;
    margin-bottom: 5px;
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


