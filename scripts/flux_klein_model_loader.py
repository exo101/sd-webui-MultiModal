import torch
import os
import gc
from pathlib import Path
from modules import shared

# 尝试导入diffusers相关模块
try:
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("请安装diffusers库: pip install diffusers")

# 尝试导入modelscope相关模块
try:
    from modelscope import Flux2KleinPipeline
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("请安装modelscope库: pip install modelscope")

# 尝试导入transformers相关模块
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("请安装transformers库: pip install transformers")

# 根据依赖库是否可用决定插件是否可用
FLUX_KLEIN_AVAILABLE = (DIFFUSERS_AVAILABLE or MODELSCOPE_AVAILABLE) and TRANSFORMERS_AVAILABLE

# 全局变量
pipe = None
FLUX_KLEIN_LOADED = False


def _is_fp8_model(model_identifier):
    """
    通用的FP8模型检测函数
    :param model_identifier: 模型标识符（可以是文件路径或模型名称）
    :return: 是否为FP8模型
    """
    import os
    from pathlib import Path
    
    # 检测模型名称中是否包含FP8标记
    is_fp8_by_name = "fp8" in model_identifier.lower()
    
    # 如果是目录路径，检查目录内容
    if os.path.isdir(model_identifier):
        is_fp8_by_content = (
            any(f.name.lower().endswith('_fp8.safetensors') for f in Path(model_identifier).iterdir() if f.is_file()) or
            "fp8" in os.path.basename(model_identifier).lower()
        )
    elif os.path.isfile(model_identifier):
        # 如果是文件路径，检查文件名是否包含FP8标记
        is_fp8_by_content = (
            any(model_identifier.lower().endswith(ending) for ending in ['_fp8.safetensors', '_fp8.bin', '.fp8']) or
            "fp8" in os.path.basename(model_identifier).lower()
        )
    else:
        # 如果只是一个名称字符串，只检查名称
        is_fp8_by_content = False
    
    return is_fp8_by_name or is_fp8_by_content


def _scan_model_directory(model_dir, model_type_filter):
    """
    扫描模型目录，获取模型列表
    :param model_dir: 模型目录路径
    :param model_type_filter: 模型类型过滤器 ('bf16' 或 'fp8')
    :return: 模型列表
    """
    import os
    from pathlib import Path
    
    model_choices = ["无"]  # 添加"无"选项
    
    if not os.path.exists(model_dir):
        return model_choices
    
    for item in os.listdir(model_dir):
        item_path = Path(model_dir) / item
        
        # 检查是否为FP8模型文件（单个文件）
        if item_path.is_file() and item_path.suffix in ['.safetensors', '.bin']:
            if model_type_filter == 'fp8' and _is_fp8_model(str(item_path)):
                model_choices.append(f"{item}")  # 直接显示文件名
        # 检查是否为模型目录
        elif item_path.is_dir():
            # 检查目录中是否包含模型文件
            has_model_files = (
                any(file.suffix in [".bin", ".safetensors", ".pt", ".ckpt"] for file in item_path.iterdir() if file.is_file()) or
                (item_path / "model_index.json").exists()
            )
            
            if not has_model_files:
                continue
                
            # 检查是否为FP8模型
            is_fp8 = _is_fp8_model(str(item_path))
            
            # 根据过滤器类型决定是否添加到列表
            if model_type_filter == 'fp8' and is_fp8:
                model_choices.append(f"{item} (FP8)")
            elif model_type_filter == 'bf16' and not is_fp8:
                model_type_suffix = "(BF16-9B)" if "9B" in item else "(BF16-4B)"
                model_choices.append(f"{item} {model_type_suffix}")
    
    return model_choices


def get_bf16_models():
    """获取BF16模型列表"""
    model_dir = os.path.join("models", "FLUX.2-klein")
    return _scan_model_directory(model_dir, 'bf16')


def get_fp8_models():
    """获取FP8模型列表"""
    model_dir = os.path.join("models", "FLUX.2-klein")
    return _scan_model_directory(model_dir, 'fp8')


def list_flux_klein_models():
    """列出FLUX.2-klein模型文件，BF16原版显示基本名称，FP8量化模型显示具体模型名"""
    import os
    from pathlib import Path
    
    model_dir = os.path.join("models", "FLUX.2-klein")
    model_choices = []
    
    # 获取BF16和FP8模型列表
    bf16_models = _scan_model_directory(model_dir, 'bf16')
    fp8_models = _scan_model_directory(model_dir, 'fp8')
    
    # 合并模型列表，排除"无"选项（因为每个列表都有"无"选项）
    model_choices = [model for model in bf16_models if model != "无"] + [model for model in fp8_models if model != "无"]
    
    # 如果需要添加总的"无"选项（例如用于刷新列表），可以在这里添加
    # 但根据项目规范，这个函数可能不直接用于UI组件
    if model_choices:
        model_choices = ["无"] + model_choices  # 在列表开头添加一个"无"
    
    # 如果没有找到模型，提供默认选项
    if not model_choices:
        model_choices = [
            "FLUX_2-klein-base-4B (BF16-4B)",
            "FLUX_2-klein-9B (BF16-9B)"
        ]
    
    return model_choices


def _load_fp8_model(full_model_path, model_type, dtype):
    """加载FP8模型文件"""
    try:
        # 获取基础模型路径 - 从文件名推断基础模型
        model_dir = os.path.dirname(full_model_path)
        model_filename = os.path.basename(full_model_path)
        
        # 从文件名猜测基础模型名称
        # 例如，从 "FLUX.2-klein-base-4b-fp8_V1.safetensors" 推断基础模型是 "FLUX_2-klein-base-4B"
        base_model_name = model_filename.replace("-fp8", "").replace("_V1", "").replace(".safetensors", "")
        if "4b" in base_model_name.lower():
            base_model_path_candidate = os.path.join("models", "FLUX.2-klein", base_model_name.replace("-4b", "-4B"))
        elif "9b" in base_model_name.lower():
            base_model_path_candidate = os.path.join("models", "FLUX.2-klein", base_model_name.replace("-9b", "-9B"))
        else:
            # 默认使用基础模型
            base_model_path_candidate = os.path.join("models", "FLUX.2-klein", "FLUX_2-klein-base-4B")
        
        # 检查推断的基础模型路径是否存在
        if os.path.exists(base_model_path_candidate):
            base_model_path = base_model_path_candidate
        else:
            # 如果推断的路径不存在，尝试几种可能的路径格式
            possible_paths = [
                os.path.join("models", "FLUX.2-klein", "FLUX_2-klein-base-4B"),
                os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-base-4B"),
                os.path.join("models", "FLUX.2-klein", "FLUX_2-klein-9B"),
                os.path.join("models", "FLUX.2-klein", "FLUX_2-klein-9B")
            ]
            
            base_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    base_model_path = path
                    break
            
            # 如果仍然找不到合适的模型路径，抛出错误
            if base_model_path is None:
                raise FileNotFoundError(f"无法找到合适的基础模型路径，尝试了: {possible_paths}")
        
        # 加载基础模型
        pipe = Flux2KleinPipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        return pipe
    except Exception as e:
        print(f"加载FP8模型失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None


def get_full_model_path(model_choice):
    """获取完整模型路径"""
    # 如果是None或空值，返回默认模型路径
    if not model_choice:
        return os.path.join("models", "FLUX.2-klein", "FLUX_2-klein-base-4B")
    
    # 移除可能的后缀标签，如(BF16-4B)等
    clean_name = model_choice.replace(" (BF16-4B)", "").replace(" (BF16-9B)", "").replace(" (FP8)", "")
    
    # 检查是否是相对路径格式 (目录/文件名)
    if "/" in clean_name or "\\" in clean_name:
        # 这是FP8模型的格式，直接构建路径
        parts = clean_name.replace("\\", "/").split("/")
        base_path = os.path.join("models", "FLUX.2-klein", *parts)
        return base_path
    else:
        # 这是BF16基础模型，直接构建路径
        return os.path.join("models", "FLUX.2-klein", clean_name)

def apply_lora(pipe, lora_model, lora_weight):
    """应用LoRA模型"""
    try:
        if not lora_model:
            return
        
        lora_path = os.path.join(shared.models_path, "Lora", lora_model)
        if os.path.exists(lora_path):
            # 应用LoRA模型
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)
            print(f"LoRA模型已应用: {lora_model}, 权重: {lora_weight}")
        else:
            print(f"LoRA模型不存在: {lora_path}")
    except Exception as e:
        print(f"应用LoRA模型失败: {e}")

def list_lora_models():
    """列出Lora模型文件，支持主目录和专属目录"""
    import os
    
    # 主Lora目录
    main_lora_dir = os.path.join("models", "Lora")
    # 专属Lora目录
    klein_lora_dir = os.path.join("models", "Lora", "FLUX.2-klein-lora")
    
    lora_files = []
    
    # 搜索主Lora目录
    if os.path.exists(main_lora_dir):
        for root, dirs, files in os.walk(main_lora_dir):
            for file in files:
                if file.endswith(('.safetensors', '.ckpt', '.pt')):
                    # 获取相对路径，以便在UI中显示
                    rel_path = os.path.relpath(os.path.join(root, file), main_lora_dir)
                    lora_files.append(rel_path)
    
    # 搜索专属Lora目录
    if os.path.exists(klein_lora_dir):
        for root, dirs, files in os.walk(klein_lora_dir):
            for file in files:
                if file.endswith(('.safetensors', '.ckpt', '.pt')):
                    # 获取相对路径，以便在UI中显示
                    rel_path = os.path.relpath(os.path.join(root, file), main_lora_dir)
                    if rel_path not in lora_files:  # 避免重复添加
                        lora_files.append(rel_path)
    
    return lora_files

def load_flux_klein_pipeline(model_type):
    """加载FLUX.2-klein模型管道"""
    global pipe, FLUX_KLEIN_LOADED
    
    try:
        # 获取完整的模型路径
        full_model_path = get_full_model_path(model_type)
        
        # 检查路径是否为文件还是目录
        if os.path.isfile(full_model_path):
            # 如果是文件，需要找到其所在的模型目录
            model_dir = os.path.dirname(full_model_path)
        else:
            # 如果是目录，直接使用
            model_dir = full_model_path
        
        # 使用通用函数检测模型是否为FP8
        is_fp8_detected = _is_fp8_model(model_type) or _is_fp8_model(full_model_path) or "fp8" in model_type.lower()
        
        # 根据检测到的模型类型选择数据类型
        if is_fp8_detected:
            # 总是使用BF16加载基础模型，避免直接处理FP8权重
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16
        
        # 加载模型管道
        if MODELSCOPE_AVAILABLE:
            from modelscope.hub.snapshot_download import snapshot_download
            
            # 确保使用正确的模型目录路径
            resolved_model_path = str(model_dir)
            
            # 检查模型目录是否包含model_index.json
            model_index_path = os.path.join(resolved_model_path, "model_index.json")
            if os.path.exists(model_index_path):
                # 如果存在model_index.json，使用常规方式加载
                pipe = Flux2KleinPipeline.from_pretrained(
                    resolved_model_path, 
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
            else:
                # 如果不存在model_index.json，检查是否为单独的FP8模型文件
                if os.path.isfile(full_model_path) and is_fp8_detected:
                    pipe = _load_fp8_model(full_model_path, model_type, dtype)
                    if pipe is None:
                        return None
                else:
                    print(f"错误：在目录 {resolved_model_path} 中未找到 model_index.json 文件")
                    print("此模型似乎不是一个完整的diffusion模型目录结构")
                    print("请确保您使用的模型是完整结构的diffusion模型")
                    return None
            
            # 启用模型CPU卸载以节省显存 - 这是关键优化
            pipe.enable_model_cpu_offload()
            
            # 启用切片注意力和VAE切片以进一步节省显存
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
        else:
            # 如果模型不可用，返回错误
            print(f"错误：无法加载模型，ModelScope可用性: {MODELSCOPE_AVAILABLE}")
            return None
        # 由于启用了CPU卸载，不需要手动移动到设备
        FLUX_KLEIN_LOADED = True
        print("FLUX.2-klein模型加载完成！")
        return pipe
    except Exception as e:
        print(f"加载FLUX.2-klein模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None
