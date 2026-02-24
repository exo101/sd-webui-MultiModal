import torch
import os
import gc
from pathlib import Path
from modules import shared
import logging

# 创建logger实例
logger = logging.getLogger('flux_klein')

# 尝试导入diffusers相关模块
try:
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# 尝试导入modelscope相关模块
try:
    from modelscope import Flux2KleinPipeline
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

# 尝试导入transformers相关模块
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 尝试导入SDNQ相关模块
SDNQ_AVAILABLE = False
try:
    import diffusers
    from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    SDNQ_AVAILABLE = True
except ImportError:
    pass

# 尝试导入SageAttention和Flash Attention
try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# Flash Attention检测
FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass

# 根据依赖库是否可用决定插件是否可用
FLUX_KLEIN_AVAILABLE = (DIFFUSERS_AVAILABLE or MODELSCOPE_AVAILABLE) and TRANSFORMERS_AVAILABLE

# 全局变量
pipe = None
FLUX_KLEIN_LOADED = False


def apply_attention_optimizations(pipe, model_type='original'):
    """应用注意力优化到模型，使用WebUI标准方式"""
    try:
        # 应用SageAttention或Flash Attention优化
        if SAGE_ATTENTION_AVAILABLE:
            replace_transformer_attention_with_sage(pipe)
        elif FLASH_ATTENTION_AVAILABLE:
            replace_transformer_attention_with_flash(pipe)
    except Exception:
        pass


def replace_transformer_attention_with_sage(pipe):
    """将pipeline中的transformer注意力机制替换为SageAttention"""
    try:
        from diffusers.models.attention_processor import Attention
        
        def sage_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # 确定使用哪个hidden_states来生成key和value
            if encoder_hidden_states is not None:
                kv_states = encoder_hidden_states
            else:
                kv_states = hidden_states
            
            # 原始的query/key/value投影
            query = self.to_q(hidden_states)
            key = self.to_k(kv_states)
            value = self.to_v(kv_states)

            # 确保维度正确
            batch_size, seq_len, dim = query.shape
            head_dim = dim // self.heads
            heads = self.heads

            # 重塑为多头形式
            query = query.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

            # 使用SageAttention进行计算
            out = sageattn(query, key, value, 
                         scale=head_dim**(-0.5), 
                         attention_dropout=0.0, 
                         causal=False)
            
            # 重塑回原始格式
            out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
            
            # 通过输出投影
            out = self.to_out[0](out) if isinstance(self.to_out, (list, tuple)) else self.to_out(out)
            
            return out

        # 检查pipeline是否有transformer组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 遍历transformer中的所有注意力层并替换forward方法
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention):
                    # 保存原始forward方法的引用（以防需要恢复）
                    module.original_forward = module.forward
                    # 替换forward方法
                    module.forward = sage_attention_forward.__get__(module, type(module))
        else:
            return
            
    except Exception:
        pass


def replace_transformer_attention_with_flash(pipe):
    """将pipeline中的transformer注意力机制替换为FlashAttention"""
    try:
        import torch.nn.functional as F
        from diffusers.models.attention_processor import Attention
        
        def flash_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # 确定使用哪个hidden_states来生成key和价值
            if encoder_hidden_states is not None:
                kv_states = encoder_hidden_states
            else:
                kv_states = hidden_states
            
            # 原始的query/key/value投影
            query = self.to_q(hidden_states)
            key = self.to_k(kv_states)
            value = self.to_v(kv_states)

            # 确保维度正确
            batch_size, seq_len, dim = query.shape
            head_dim = dim // self.heads
            heads = self.heads

            # 重塑为多头形式
            query = query.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

            # 尝试使用Flash Attention
            try:
                # Flash Attention 2 implementation
                from flash_attn import flash_attn_func
                out = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
            except Exception:
                # 回退到PyTorch的scaled_dot_product_attention
                out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

            # 重塑回原始格式
            out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
            
            # 通过输出投影
            out = self.to_out[0](out) if isinstance(self.to_out, (list, tuple)) else self.to_out(out)
            
            return out

        # 检查pipeline是否有transformer组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 遍历transformer中的所有注意力层并替换forward方法
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention):
                    # 保存原始forward方法的引用（以防需要恢复）
                    module.original_forward = module.forward
                    # 替换forward方法
                    module.forward = flash_attention_forward.__get__(module, type(module))
        else:
            return
    except Exception:
        pass


def _is_fp8_model(model_identifier):
    """
    通用的FP8模型检测函数
    :param model_identifier: 模型标识符（可以是文件路径或模型名称）
    :return: 是否为FP8模型
    """
    import os
    from pathlib import Path
    
    # FP8相关关键词列表（添加下划线避免与模型大小混淆）
    fp8_keywords = ['fp8', 'q4_', 'q5_', 'q8_', '_int8', '_quant']  # 使用下划线作为前缀/后缀分隔符
    
    # 明确的FP8标识（独立检查）
    explicit_fp8_indicators = ['_fp8', '-fp8', '.fp8']
    
    # 检测模型名称中是否包含明确的FP8标识
    model_name_lower = model_identifier.lower()
    has_explicit_fp8 = any(indicator in model_name_lower for indicator in explicit_fp8_indicators)
    
    # 如果有明确的FP8标识，则直接判定为FP8模型
    if has_explicit_fp8:
        return True
    
    # 检查是否包含FP8关键词（使用更精确的匹配）
    is_fp8_by_name = any(keyword in model_name_lower for keyword in fp8_keywords)
    
    # 如果是目录路径，检查目录内容
    if os.path.isdir(model_identifier):
        dir_path = Path(model_identifier)
        
        # 检查文件名是否包含FP8标记（更精确的匹配）
        has_fp8_files = any(
            f.name.lower().endswith('_fp8.safetensors') or 
            f.name.lower().endswith('_fp8.bin')
            for f in dir_path.iterdir() if f.is_file()
        )
        
        # 检查目录名是否包含FP8标记（更严格的匹配）
        dir_basename = os.path.basename(model_identifier).lower()
        dir_name_contains_fp8 = (
            '_fp8' in dir_basename or 
            '-fp8' in dir_basename or
            any(keyword in dir_basename for keyword in ['q4_', 'q5_', 'q8_', '_int8'])
        )
        
        is_fp8_by_content = has_fp8_files or dir_name_contains_fp8
    elif os.path.isfile(model_identifier):
        # 如果是文件路径，检查文件名是否包含FP8标记
        file_basename = os.path.basename(model_identifier).lower()
        is_fp8_by_content = (
            file_basename.endswith('_fp8.safetensors') or
            file_basename.endswith('_fp8.bin') or
            file_basename.endswith('.fp8') or
            any(keyword in file_basename for keyword in fp8_keywords)
        )
    else:
        # 如果只是一个名称字符串，只检查名称
        is_fp8_by_content = False
    
    result = is_fp8_by_name or is_fp8_by_content
    return result


def _is_sdnq_model(model_identifier):
    """
    检测是否为SDNQ 4bit量化模型
    :param model_identifier: 模型标识符
    :return: 是否为SDNQ模型
    """
    import os
    from pathlib import Path
    
    # SDNQ相关关键词
    sdnq_keywords = ['sdnq', '4bit', 'dynamic-svd', 'r32']
    
    model_name_lower = model_identifier.lower()
    
    # 检查是否包含SDNQ关键词
    has_sdnq_keywords = any(keyword in model_name_lower for keyword in sdnq_keywords)
    
    # 检查是否为特定的SDNQ模型名称
    is_specific_sdnq = 'flux.2-klein-9b-sdnq-4bit-dynamic-svd-r32' in model_name_lower
    
    # 如果是目录路径，检查目录内容
    if os.path.isdir(model_identifier):
        dir_path = Path(model_identifier)
        dir_basename = os.path.basename(model_identifier).lower()
        
        # 检查目录名是否包含SDNQ标识
        dir_has_sdnq = (
            'sdnq' in dir_basename or 
            '4bit' in dir_basename or
            'dynamic-svd' in dir_basename or
            'r32' in dir_basename
        )
        
        # 检查目录中是否包含SDNQ特有的配置文件
        has_sdnq_config = (dir_path / "config.json").exists() and "sdnq" in dir_basename
        
        return has_sdnq_keywords or is_specific_sdnq or dir_has_sdnq or has_sdnq_config
    elif os.path.isfile(model_identifier):
        # 如果是文件路径，检查文件名
        file_basename = os.path.basename(model_identifier).lower()
        return (
            'sdnq' in file_basename or 
            '4bit' in file_basename or
            has_sdnq_keywords
        )
    else:
        # 如果只是一个名称字符串
        return has_sdnq_keywords or is_specific_sdnq


def _identify_model_type(model_name):
    """
    动态识别模型类型和版本
    :param model_name: 模型名称
    :return: 模型类型标识字符串
    """
    model_name_lower = model_name.lower()
    
    # 首先检查是否为SDNQ模型
    if _is_sdnq_model(model_name):
        return "(SDNQ-4bit-9B-Dynamic-SVD-R32)"
    
    # 检测模型大小（参数量）
    if '9b' in model_name_lower:
        size_info = "9B"
    elif '4b' in model_name_lower:
        size_info = "4B"
    elif 'klein-4' in model_name_lower:  # 修正：识别 FLUX.2-klein-4B 格式
        size_info = "4B"
    elif 'base' in model_name_lower:
        # 对于base模型，默认认为是4B
        size_info = "4B"
    else:
        # 无法确定大小时，使用通用标识
        size_info = "Unknown"
    
    # 检测模型变体
    if 'base' in model_name_lower:
        variant_info = "Base"
    elif 'dev' in model_name_lower:
        variant_info = "Dev"
    elif 'schnell' in model_name_lower:
        variant_info = "Schnell"
    elif 'klein-4' in model_name_lower:  # 修正：识别标准变体
        variant_info = "Standard"
    else:
        variant_info = "Standard"
    
    # 检测量化类型（虽然这里是BF16，但保持一致性）
    quant_info = "BF16"
    
    return f"({quant_info}-{size_info}-{variant_info})"


def _scan_model_directory(model_dir, model_type_filter):
    """
    扫描模型目录，获取模型列表
    :param model_dir: 模型目录路径
    :param model_type_filter: 模型类型过滤器 ('bf16', 'fp8' 或 'sdnq')
    :return: 模型列表
    """
    import os
    from pathlib import Path
    
    model_choices = ["无"]  # 添加"无"选项
    
    if not os.path.exists(model_dir):
        return model_choices
    
    for item in os.listdir(model_dir):
        item_path = Path(model_dir) / item
        
        # 检查是否为单独的模型文件
        if item_path.is_file() and item_path.suffix in ['.safetensors', '.bin']:
            if model_type_filter == 'fp8' and _is_fp8_model(str(item_path)):
                model_choices.append(f"{item}")  # 直接显示文件名
            elif model_type_filter == 'sdnq' and _is_sdnq_model(str(item_path)):
                model_choices.append(f"{item} (SDNQ-4bit)")
        # 检查是否为模型目录
        elif item_path.is_dir():
            # 检查目录中是否包含模型文件
            has_model_files = (
                any(file.suffix in [".bin", ".safetensors", ".pt", ".ckpt"] for file in item_path.iterdir() if file.is_file()) or
                (item_path / "model_index.json").exists()
            )
            
            if not has_model_files:
                continue
                
            # 检查模型类型
            is_fp8 = _is_fp8_model(str(item_path))
            is_sdnq = _is_sdnq_model(str(item_path))
            
            # 根据过滤器类型决定是否添加到列表
            if model_type_filter == 'fp8' and is_fp8:
                model_choices.append(f"{item} (FP8)")
            elif model_type_filter == 'sdnq' and is_sdnq:
                model_choices.append(f"{item} (SDNQ-4bit-Dynamic-SVD-R32)")
            elif model_type_filter == 'bf16' and not is_fp8 and not is_sdnq:
                # 动态识别模型类型，不再硬编码4B/9B
                model_type_info = _identify_model_type(item)
                model_choices.append(f"{item} {model_type_info}")
    
    return model_choices


def get_bf16_models():
    """获取BF16模型列表 - 支持动态识别各种模型版本"""
    model_dir = os.path.join("models", "FLUX.2-klein")
    return _scan_model_directory(model_dir, 'bf16')


def get_fp8_models():
    """获取FP8模型列表 - 支持动态识别各种模型版本"""
    model_dir = os.path.join("models", "FLUX.2-klein")
    return _scan_model_directory(model_dir, 'fp8')


def get_sdnq_models():
    """获取SDNQ模型列表 - 支持动态识别SDNQ 4bit模型"""
    model_dir = os.path.join("models", "FLUX.2-klein")
    return _scan_model_directory(model_dir, 'sdnq')


def list_flux_klein_models():
    """列出FLUX.2-klein模型文件，支持动态识别各种模型版本"""
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
    
    # 如果没有找到模型，提供更灵活的默认选项
    if not model_choices:
        model_choices = [
            "FLUX.2-klein-base-4B (BF16-4B-Base)",
            "FLUX.2-klein-9B (BF16-9B-Standard)",
            "FLUX.2-klein-dev-4B (BF16-4B-Dev)",
            "FLUX.2-klein-schnell-4B (BF16-4B-Schnell)"
        ]
    
    return model_choices


def _load_fp8_model(full_model_path, model_type, dtype):
    """加载FP8模型文件"""
    try:
        # 获取基础模型路径 - 从文件名推断基础模型
        model_dir = os.path.dirname(full_model_path)
        model_filename = os.path.basename(full_model_path)
        
        # 从文件名猜测基础模型名称
        # 例如，从 "FLUX.2-klein-base-4b-fp8_V1.safetensors" 推断基础模型是 "FLUX.2-klein-4B"
        base_model_name = model_filename.replace("-fp8", "").replace("_V1", "").replace(".safetensors", "")
        if "4b" in base_model_name.lower():
            # 修正：使用正确的目录名格式 FLUX.2-klein-4B
            base_model_path_candidate = os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-4B")
        elif "9b" in base_model_name.lower():
            base_model_path_candidate = os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-9B")
        else:
            # 默认使用基础模型
            base_model_path_candidate = os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-4B")
        
        # 检查推断的基础模型路径是否存在
        if os.path.exists(base_model_path_candidate):
            base_model_path = base_model_path_candidate
        else:
            # 如果推断的路径不存在，尝试几种可能的路径格式
            possible_paths = [
                os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-4B"),  # 修正：使用实际存在的目录名
                os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-base-4B"),
                os.path.join("models", "FLUX.2-klein", "FLUX_2-klein-base-4B"),
                os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-9B"),
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
    """获取完整模型路径 - 支持动态识别的各种模型格式"""
    # 如果是None或空值，返回默认模型路径
    if not model_choice or model_choice == "无":
        return os.path.join("models", "FLUX.2-klein", "FLUX.2-klein-4B")  # 修正：使用正确的默认路径
    
    # 如果已经是完整路径，直接返回
    if os.path.isabs(model_choice) or model_choice.startswith("models"):
        return model_choice
    
    # 构造完整路径
    model_dir = os.path.join("models", "FLUX.2-klein")
    
    # 处理带括号的模型名称（去除括号及其内容）
    clean_model_name = model_choice
    if '(' in clean_model_name and ')' in clean_model_name:
        # 去除英文括号及其内容
        clean_model_name = clean_model_name[:clean_model_name.find('(')].strip()
    if '（' in clean_model_name and '）' in clean_model_name:
        # 去除中文括号及其内容
        clean_model_name = clean_model_name[:clean_model_name.find('（')].strip()
    
    # 检查是否为目录名
    dir_path = os.path.join(model_dir, clean_model_name)
    if os.path.isdir(dir_path):
        return dir_path
    
    # 检查是否为文件名
    file_path = os.path.join(model_dir, clean_model_name)
    if os.path.isfile(file_path):
        return file_path
    
    # 特殊处理SDNQ模型名称
    if "sdnq" in model_choice.lower() or "4bit" in model_choice.lower():
        # 尝试匹配SDNQ模型目录
        sdnq_dirs = [
            "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
            "FLUX.2-klein-9B-SDNQ-4bit",
            "FLUX.2-klein-SDNQ-4bit"
        ]
        
        for sdnq_dir in sdnq_dirs:
            sdnq_path = os.path.join(model_dir, sdnq_dir)
            if os.path.exists(sdnq_path):
                print(f"[INFO] 找到SDNQ模型目录: {sdnq_path}")
                return sdnq_path
    
    # 如果都不存在，返回默认路径并记录警告
    default_path = os.path.join(model_dir, "FLUX.2-klein-4B")
    print(f"[WARNING] 无法找到模型 '{model_choice}'，使用默认路径: {default_path}")
    return default_path


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


def validate_model_paths():
    """验证FLUX.2-klein模型路径配置"""
    model_base_dir = os.path.join("models", "FLUX.2-klein")
    
    print(f"[INFO] 检查模型目录: {model_base_dir}")
    
    if not os.path.exists(model_base_dir):
        print(f"[ERROR] 模型基础目录不存在: {model_base_dir}")
        return False
    
    # 检查预期的模型目录
    expected_dirs = [
        "FLUX.2-klein-4B",
        "FLUX.2-klein-9B", 
        "FLUX.2-klein-base-4B"
    ]
    
    found_dirs = []
    missing_dirs = []
    
    for dir_name in expected_dirs:
        dir_path = os.path.join(model_base_dir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            found_dirs.append(dir_name)
            print(f"[INFO] 找到模型目录: {dir_name}")
        else:
            missing_dirs.append(dir_name)
    
    # 检查FP8模型文件
    fp8_files = []
    for item in os.listdir(model_base_dir):
        item_path = os.path.join(model_base_dir, item)
        if os.path.isfile(item_path) and item.endswith(('.safetensors', '.bin')) and 'fp8' in item.lower():
            fp8_files.append(item)
            print(f"[INFO] 找到FP8模型文件: {item}")
    
    if found_dirs:
        print(f"[SUCCESS] 已找到 {len(found_dirs)} 个模型目录: {found_dirs}")
    else:
        print(f"[WARNING] 未找到任何预期的模型目录")
    
    if missing_dirs:
        print(f"[WARNING] 缺少模型目录: {missing_dirs}")
    
    if fp8_files:
        print(f"[INFO] 发现 {len(fp8_files)} 个FP8模型文件")
    else:
        print(f"[WARNING] 未发现FP8模型文件")
    
    return len(found_dirs) > 0 or len(fp8_files) > 0


def load_flux_klein_pipeline(model_type):
    """加载FLUX.2-klein模型管道"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 首先卸载现有模型释放显存
    if pipe is not None:
        try:
            # 尝试卸载模型到CPU
            pipe = pipe.to("cpu")
        except:
            pass  # 如果出错则忽略，继续清理
        pipe = None  # 删除引用
        # 强制垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        # 获取完整的模型路径
        full_model_path = get_full_model_path(model_type)
        
        # 检查是否为SDNQ模型
        is_sdnq_model = _is_sdnq_model(model_type) or _is_sdnq_model(full_model_path)
        
        # 检查路径是否为文件还是目录
        if os.path.isfile(full_model_path):
            # 如果是文件，说明是FP8模型文件，需要特殊处理
            model_dir = os.path.dirname(full_model_path)
            is_fp8_file = _is_fp8_model(full_model_path)
        else:
            # 如果是目录，直接使用
            model_dir = full_model_path
            is_fp8_file = _is_fp8_model(full_model_path)
        
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            print(f"错误：模型目录不存在: {model_dir}")
            return None
        
        # 特殊处理SDNQ模型
        if is_sdnq_model and SDNQ_AVAILABLE:
            print(f"[INFO] 检测到SDNQ 4bit模型: {model_type}")
            return _load_sdnq_model(model_type)
        
        # 如果是FP8模型文件，需要特殊处理
        if is_fp8_file and os.path.isfile(full_model_path):
            pipe = _load_fp8_model(full_model_path, model_type, torch.bfloat16)
            if pipe is None:
                print(f"加载FP8模型失败: {full_model_path}")
                return None
        else:
            # 检查model_index.json是否存在（对于完整模型目录）
            model_index_path = os.path.join(model_dir, "model_index.json")
            if not os.path.exists(model_index_path):
                print(f"错误：在目录 {model_dir} 中未找到 model_index.json 文件")
                print("请确保您使用的模型是完整结构的FLUX.2-klein模型")
                return None
        
            # 根据检测到的模型类型选择数据类型
            is_fp8_detected = _is_fp8_model(model_type) or _is_fp8_model(full_model_path) or "fp8" in model_type.lower()
            
            if is_fp8_detected:
                # 总是使用BF16加载基础模型，避免直接处理FP8权重
                dtype = torch.bfloat16
            else:
                dtype = torch.bfloat16
            
            # 加载模型管道
            if MODELSCOPE_AVAILABLE:
                # 确保使用正确的模型目录路径
                resolved_model_path = str(model_dir)
                
                # 检查模型目录是否包含model_index.json
                if os.path.exists(model_index_path):
                    # 如果存在model_index.json，使用常规方式加载
                    from diffusers import Flux2KleinPipeline
                    pipe = Flux2KleinPipeline.from_pretrained(
                        resolved_model_path, 
                        torch_dtype=dtype,
                        low_cpu_mem_usage=False  # 设为False以确保模型正确加载
                    )
                
            elif DIFFUSERS_AVAILABLE:
                # 如果ModelScope不可用，尝试使用diffusers
                resolved_model_path = str(model_dir)
                
                # 检查模型目录是否包含model_index.json
                if os.path.exists(model_index_path):
                    from diffusers import Flux2KleinPipeline
                    pipe = Flux2KleinPipeline.from_pretrained(
                        resolved_model_path, 
                        torch_dtype=dtype,
                        low_cpu_mem_usage=False  # 设为False以确保模型正确加载
                    )
                else:
                    print(f"错误：在目录 {resolved_model_path} 中未找到 model_index.json 文件")
                    print("此模型似乎不是一个完整的diffusion模型目录结构")
                    print("请确保您使用的模型是完整结构的diffusion模型")
                    return None
            else:
                # 如果模型不可用，返回错误
                print(f"错误：无法加载模型，ModelScope可用性: {MODELSCOPE_AVAILABLE}, Diffusers可用性: {DIFFUSERS_AVAILABLE}")
                return None
        
        # 应用注意力优化
        if FLASH_ATTENTION_AVAILABLE or SAGE_ATTENTION_AVAILABLE:
            print(f"[INFO] 应用注意力优化...")
            apply_attention_optimizations(pipe, model_type)
        
        # 使用模型自带的设备管理机制来优化内存使用
        # 启用模型CPU卸载以节省显存
        if hasattr(pipe, 'enable_model_cpu_offload'):
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                # 如果CPU卸载失败，尝试其他显存优化方法
                if torch.cuda.is_available():
                    try:
                        pipe = pipe.to("cuda")
                    except RuntimeError as e2:
                        if "out of memory" in str(e2).lower():
                            if hasattr(pipe, 'enable_sequential_cpu_offload'):
                                pipe.enable_sequential_cpu_offload()
                        else:
                            raise e2
        elif hasattr(pipe, 'enable_sequential_cpu_offload'):
            pipe.enable_sequential_cpu_offload()
        else:
            # 如果没有CPU卸载功能，则尝试将模型移动到GPU
            try:
                pipe = pipe.to("cuda")
            except Exception as move_error:
                # 即使移动到GPU失败，也继续执行，因为可以在CPU上运行
                pipe = pipe.to("cpu")
        
        # 启用切片注意力和VAE切片以进一步节省显存
        if hasattr(pipe, 'enable_attention_slicing'):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
        
        FLUX_KLEIN_LOADED = True
        return pipe
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _load_sdnq_model(model_type):
    """
    加载SDNQ 4bit量化模型
    """
    try:
        # 优先使用本地模型路径
        full_model_path = get_full_model_path(model_type)
        model_id = full_model_path
        
        # 验证本地模型路径是否存在
        if not os.path.exists(model_id):
            # 如果本地路径不存在，尝试使用HuggingFace仓库ID作为备选方案
            if "flux.2-klein-9b-sdnq-4bit-dynamic-svd-r32" in model_type.lower():
                model_id = "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
            else:
                return None
        
        # 加载SDNQ模型管道
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        )
        
        # 应用SDNQ优化选项
        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            # pipe.transformer = torch.compile(pipe.transformer) # 可选的加速选项
        
        # 启用模型CPU卸载
        pipe.enable_model_cpu_offload()
        
        return pipe
        
    except Exception:
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None
