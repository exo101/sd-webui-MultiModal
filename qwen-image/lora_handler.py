
"""
Qwen Image Extension - LoRA处理模块
用于处理LoRA模型加载和应用
"""

import sys
from pathlib import Path
import torch
from safetensors.torch import load_file as load_state_dict_in_safetensors


def load_lora_model(model_path_str, weight, model_name, pipeline, transformer=None):
    """加载LoRA模型到pipeline"""
    try:
        # 检查model_path_str是否为"无"、"None"或空字符串
        if not model_path_str or model_path_str == "无" or model_path_str == "None" or model_path_str == "":
            print(f"跳过LoRA模型加载（未选择模型或模型路径为空）: {model_name}")
            return False  # 返回False表示未加载
        
        model_path = Path(model_path_str)  # 使用外部导入的Path
        
        print(f"尝试加载LoRA模型: {model_path}")
        
        if not model_path.exists():
            print(f"LoRA模型文件不存在: {model_path}")
            return False
        
        # 确定模型类型：如果是SDNQ模型，使用diffusers方式；如果是nunchaku模型，使用nunchaku方式
        model_type = _detect_model_type(pipeline, transformer)
        
        print(f"检测到模型类型: {model_type}")
        print(f"加载LoRA模型 {model_name}: {model_path} (强度: {weight})")
        
        if model_type == "sdnq":
            return _load_lora_with_diffusers(model_path, weight, model_name, pipeline)
        elif model_type == "nunchaku":
            return _load_lora_with_nunchaku(model_path, weight, model_name, pipeline, transformer)
        else:
            # 默认尝试diffusers方式，失败后回退到nunchaku方式
            success = _load_lora_with_diffusers(model_path, weight, model_name, pipeline)
            if not success:
                print("diffusers LoRA加载失败，尝试使用nunchaku方式")
                return _load_lora_with_nunchaku(model_path, weight, model_name, pipeline, transformer)
                
    except Exception as e:
        print(f"加载LoRA模型时出错: {e}")
        print(f"错误详情: 可能是LoRA模型与基础模型不兼容，请确保LoRA模型与当前使用的QwenImage模型版本和rank匹配")
        import traceback
        traceback.print_exc()
        return False


def _detect_model_type(pipeline, transformer=None):
    """检测模型类型是SDNQ还是nunchaku"""
    # 检查transformer是否包含nunchaku特有的属性或类型
    if transformer is not None:
        # 检查transformer是否是nunchaku模型的类型
        transformer_str = str(type(transformer))
        if 'nunchaku' in transformer_str.lower() or 'svdq' in transformer_str.lower():
            return "nunchaku"
    
    # 检查pipeline是否包含SDNQ特有的属性
    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
        transformer_obj = pipeline.transformer
        # 检查是否包含SDNQ特有的量化层
        transformer_str = str(type(transformer_obj))
        if 'awqw4a16linear' in str(transformer_obj) or 'sdnq' in transformer_str.lower():
            return "sdnq"
        
        # 检查模型名称或路径中是否包含SDNQ标识
        if hasattr(transformer_obj, 'config') and hasattr(transformer_obj.config, '_name_or_path'):
            model_path = transformer_obj.config._name_or_path
            if 'sdnq' in model_path.lower():
                return "sdnq"
    
    # 如果pipeline是通过SDNQ方式加载的，通常会有SDNQ特有的特征
    if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, '__class__'):
        class_name = pipeline.transformer.__class__.__name__
        if 'sdnq' in class_name.lower() or 'quantized' in class_name.lower():
            return "sdnq"
    
    # 通过传入的transformer参数的类型判断
    if transformer is not None:
        transformer_class_name = transformer.__class__.__name__
        if 'nunchaku' in transformer_class_name.lower() or 'svdq' in transformer_class_name.lower():
            return "nunchaku"
    
    # 如果都没有检测到明确的类型，尝试从pipeline的类型来判断
    pipeline_str = str(type(pipeline)).lower()
    if 'sdnq' in pipeline_str:
        return "sdnq"
    
    # 默认情况下，先尝试diffusers方式，如果检测到量化相关错误再认为是nunchaku
    return "unknown"


def _load_lora_with_diffusers(model_path, weight, model_name, pipeline):
    """使用diffusers方式加载LoRA模型"""
    try:
        # 检查pipeline是否支持LoRA加载
        if not hasattr(pipeline, 'load_lora_weights'):
            print(f"当前pipeline不支持diffusers LoRA加载: {type(pipeline)}")
            return False
        
        # 使用diffusers的内置LoRA加载功能
        lora_state_dict = load_state_dict_in_safetensors(model_path)
        
        try:
            # 尝试加载LoRA权重
            pipeline.load_lora_weights(
                pretrained_model_name_or_path_or_dict=lora_state_dict,
                adapter_name=f"lora_{model_name}",
                lora_scale=weight
            )
            
            # 如果有激活adapter的需求，可以设置
            if hasattr(pipeline, 'set_adapters'):
                pipeline.set_adapters([f"lora_{model_name}"], [weight])
            
            print(f"LoRA模型 {model_name} 加载成功，使用diffusers标准方式")
            return True
        except Exception as e:
            print(f"使用diffusers标准LoRA加载失败: {e}")
            print("尝试使用QwenImage专用LoRA转换方法")
            
            # 尝试使用diffusers库的转换函数
            try:
                from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers
                
                # 转换LoRA权重为diffusers格式
                converted_state_dict = _convert_non_diffusers_qwen_lora_to_diffusers(lora_state_dict)
                
                # 加载转换后的权重
                pipeline.load_lora_weights(
                    pretrained_model_name_or_path_or_dict=converted_state_dict,
                    adapter_name=f"lora_{model_name}",
                    lora_scale=weight
                )
                
                if hasattr(pipeline, 'set_adapters'):
                    pipeline.set_adapters([f"lora_{model_name}"], [weight])
                
                print(f"LoRA模型 {model_name} 转换并加载成功")
                return True
            except Exception as conversion_error:
                print(f"LoRA模型 {model_name} 转换失败: {conversion_error}")
                return False
    except Exception as e:
        print(f"使用diffusers加载LoRA模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def _load_lora_with_nunchaku(model_path, weight, model_name, pipeline, transformer=None):
    """使用nunchaku方式加载LoRA模型"""
    try:
        # 使用importlib.util直接从文件导入，避免触发整个nunchaku包的加载
        import importlib.util
        import sys
        
        # 构建LoRA模块文件路径
        lora_module_path = Path(__file__).parent.parent / "nunchaku-2_lora_concat" / "nunchaku" / "lora" / "flux" / "v1" / "lora_flux_v2.py"
        
        # 检查模块文件是否存在
        if lora_module_path.exists():
            spec = importlib.util.spec_from_file_location("lora_flux_v2", str(lora_module_path))
            lora_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lora_module)
            
            # 获取所需函数
            update_lora_params = getattr(lora_module, 'update_lora_params_v2', None)
            set_lora_strength_v2 = getattr(lora_module, 'set_lora_strength_v2', None)
            
            if update_lora_params:
                # 加载LoRA权重
                lora_state_dict = load_state_dict_in_safetensors(model_path)
                if lora_state_dict:
                    try:
                        # 尝试应用LoRA权重
                        if transformer is not None:
                            # 对于Nunchaku模型，直接对transformer应用LoRA权重
                            update_lora_params(transformer, lora_state_dict, strength=weight)
                        else:
                            # 对于普通模型，应用到pipeline.transformer
                            update_lora_params(pipeline.transformer, lora_state_dict, strength=weight)
                        
                        print(f"LoRA模型 {model_name} (nunchaku方式) 加载成功")
                            
                        # 重新初始化CPU卸载管理器以适应LoRA加载后模型参数维度的变化
                        if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, 'offload') and pipeline.transformer.offload:
                            print("重新初始化CPU卸载管理器以适应LoRA模型参数")
                            try:
                                # 保存当前的卸载设置
                                use_pin_memory = getattr(pipeline.transformer.offload_manager, 'use_pin_memory', True)
                                num_blocks_on_gpu = getattr(pipeline.transformer.offload_manager, 'num_blocks_on_gpu', 1)
                                
                                # 重新设置卸载
                                pipeline.transformer.set_offload(False)  # 先关闭
                                pipeline.transformer.set_offload(True, use_pin_memory=use_pin_memory, num_blocks_on_gpu=num_blocks_on_gpu)  # 再开启
                            except Exception as e:
                                print(f"重新初始化CPU卸载管理器时出错: {e}")
                        
                        return True
                    except Exception as e:
                        print(f"应用LoRA模型时发生错误 (nunchaku方式): {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    print(f"LoRA模型 {model_name} (nunchaku方式) 加载失败: 无法加载权重")
                    return False
            else:
                print(f"LoRA模块缺少必要的函数: update_lora_params_v2")
                return False
        else:
            print(f"LoRA模块文件不存在: {lora_module_path}")
            return False
    except Exception as e:
        print(f"使用nunchaku加载LoRA模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return False