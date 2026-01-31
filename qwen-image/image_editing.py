
"""
Qwen Image Extension - 图像编辑处理模块
用于处理图像编辑功能
"""

import os
import sys
import json
import time as time_module  # 重命名time模块避免冲突
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import math
from safetensors.torch import load_file as load_state_dict_in_safetensors
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import FlowMatchHeunDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch.nn.functional as F


# 添加logging导入
import logging

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

# 添加项目路径到sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
webui_root = parent_dir.parent.parent.parent

# 添加必要的路径
paths_to_add = [
    str(parent_dir),  # qwen-image目录
    str(webui_root),  # 主目录
    str(webui_root / "extensions-builtin"),
    str(webui_root / "extensions-builtin" / "forge_legacy_preprocessors")
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入模块
from qwen_image_controlnet import preprocess_for_qwen_image_controlnet
from config import CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL
from preprocessor import preprocess_control_image
from lora_handler import load_lora_model


def run_image_editing(args_file):
    """运行图像编辑功能"""
    try:
        print(f"开始执行图像编辑功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time_module.time()
        
        # 检查参数文件是否存在
        print(f"检查参数文件是否存在: {args_file}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"参数文件绝对路径: {os.path.abspath(args_file)}")
        
        if not os.path.exists(args_file):
            print(f"错误: 参数文件不存在: {args_file}")
            # 列出当前目录下的文件
            current_dir = os.path.dirname(args_file) if os.path.dirname(args_file) else "."
            if os.path.exists(current_dir):
                print(f"目录 {current_dir} 中的文件:")
                for file in os.listdir(current_dir):
                    print(f"  {file}")
            return []  # 返回空列表而不是None
        
        if not os.path.isfile(args_file):
            print(f"错误: 参数文件不是一个有效的文件: {args_file}")
            return []
            
        # 检查文件是否可读
        if not os.access(args_file, os.R_OK):
            print(f"错误: 参数文件不可读: {args_file}")
            return []
            
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 添加详细的参数检查和日志
        print(f"接收到的完整参数: {args}")
        
        # 检查必需的参数是否存在
        required_args = ["prompt", "images", "steps", "cfg_scale", "scheduler"]
        for arg in required_args:
            if arg not in args:
                print(f"错误: 缺少必需参数 '{arg}'")
                return []
        
        # 获取参数
        prompt = args["prompt"]
        negative_prompt = args.get("negative_prompt", "")
        # 确保negative_prompt是字符串类型
        if not isinstance(negative_prompt, str):
            negative_prompt = str(negative_prompt)
        input_images = args["images"]  # 这是输入图像
        steps = args["steps"] if args["steps"] is not None else 8  # 设置默认步数为8
        cfg_scale = args["cfg_scale"]
        scheduler_type = args["scheduler"]
        
        # 添加LoRA模型参数
        lora_model_1 = args.get("lora_model_1")
        lora_model_2 = args.get("lora_model_2")
        lora_weight_1 = args.get("lora_weight_1", 1.0)
        lora_weight_2 = args.get("lora_weight_2", 1.0)
        
        # 添加参数详细信息日志
        print(f"输入图像路径: {input_images}")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"推理步数: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"调度器类型: {scheduler_type}")
        print(f"LoRA模型1: {lora_model_1}, 权重: {lora_weight_1}")
        print(f"LoRA模型2: {lora_model_2}, 权重: {lora_weight_2}")
        
        # 查找第一个非空的图像路径作为输入图像，第二个非空路径作为控制图像（如果存在）
        input_image_path = None
        control_image_path = None
        
        # 第一个非空路径作为输入图像，第二个非空路径作为控制图像（如果存在）
        non_empty_paths = [img_path.strip() for img_path in input_images 
                          if img_path and isinstance(img_path, str) and img_path.strip()]
        
        if len(non_empty_paths) > 0:
            input_image_path = non_empty_paths[0]
        if len(non_empty_paths) > 1:
            control_image_path = non_empty_paths[1]

        # 加载输入图像以获取尺寸
        init_image = None
        width = 1024
        height = 1024
        if input_image_path and os.path.exists(input_image_path):
            try:
                init_image = Image.open(input_image_path)
                orig_width, orig_height = init_image.size
                width, height = orig_width, orig_height
                print(f"从输入图像 {input_image_path} 获取尺寸: {width}x{height}")
            except Exception as e:
                print(f"无法加载输入图像以获取尺寸，使用默认尺寸 1024x1024: {e}")
        else:
            print(f"未提供有效输入图像，使用默认尺寸: {width}x{height}")
        
        # 处理ControlNet相关参数
        controlnet_conditioning_scale = args.get("controlnet_conditioning_scale", 1.0)
        controlnet_preprocessor = args.get("controlnet_preprocessor", "none")
        controlnet_start = args.get("controlnet_start", 0.0)
        controlnet_end = args.get("controlnet_end", 1.0)
        
        # 获取ControlNet相关参数
        controlnet_model_selected = args.get("controlnet_model", "无")
        
        # 获取蒙版图像参数
        mask_image_path = args.get("mask_image")
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        has_control_image = control_image_path is not None
        controlnet_enable = (has_control_image 
                           and controlnet_model_selected != "无")
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像存在: {has_control_image}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        print(f"用户选择步数: {args['steps']}")

        
        # 验证输入图像参数
        if not isinstance(input_images, list):
            print(f"错误: images参数应该是一个列表，但实际类型是 {type(input_images)}")
            return []
            
        if len(input_images) == 0:
            print("错误: images参数是空列表，未提供任何图像路径")
            return []
            

        # 导入必要的库
        from diffusers import QwenImageEditPlusPipeline
        # 使用稳健的方式导入Transformer模型，优先使用支持LoRA的版本
        EditTransformer = None
        try:
            # 首先尝试导入支持LoRA的nunchaku版本
            from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
            print("成功导入支持LoRA的NunchakuQwenImageTransformer2DModel (编辑版)")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"无法导入支持LoRA的nunchaku版本 (编辑版): {e}")
            try:
                # 回退到diffusers标准版本
                from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as EditTransformer
                print("回退到diffusers标准版本的QwenImageTransformer2DModel (编辑版)")
            except Exception as e2:
                print(f"无法导入diffusers标准版本 (编辑版): {e2}")
                EditTransformer = None
        
        if EditTransformer is None:
            print("错误: 无法导入任何可用的Transformer模型 (编辑版)")
            return []
            
        from nunchaku.utils import get_gpu_memory
        from diffusers.utils import load_image
        # from PIL import Image
        
        print("依赖库导入成功")
        
        # 获取用户选择的采样方法
        scheduler_type = args.get("scheduler", "euler")
        
        # Scheduler 配置
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        
        # 根据用户选择创建相应的调度器
        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "euler_ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "dpmpp_2m":
            # DPM++ 2M 调度器配置稍有不同
            dpm_config = scheduler_config.copy()
            dpm_config.update({
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
            })
            scheduler = DPMSolverMultistepScheduler.from_config(dpm_config)
        else:
            # 默认使用 Euler 调度器
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        
        # 获取模型路径
        # 修复：使用传递的model_dir参数而不是硬编码路径
        model_dir = args.get("model_dir")
        if model_dir:
            qwenimage_edit_models_dir = Path(model_dir)
        else:
            # 回退到默认路径
            models_dir = Path(__file__).parent.parent / "models"
            qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
        
        # 检查是否使用SDNQ量化模型
        sdnq_enable = args.get("sdnq_enable", False)
        
        # 初始化pipeline变量
        pipeline = None
        
        if sdnq_enable:
            # 自动检测SDNQ模型路径
            # 优先检查用户指定的路径
            sdnq_model_path = None
            
            # 首先尝试从参数中获取模型路径
            model_dir = args.get("model_dir")
            if model_dir:
                sdnq_model_path = Path(model_dir)
            else:
                # 尝试从标准模型目录查找
                from modules import shared
                models_dir = Path(shared.models_path) if hasattr(shared, 'models_path') else Path(__file__).parent.parent.parent.parent / "models"
                
                # 优先查找SDNQ模型
                sdnq_candidates = [
                    models_dir / "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
                    models_dir / "qwen-image" / "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
                    models_dir / "qwen-image-edit" / "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32"
                ]
                
                for candidate_path in sdnq_candidates:
                    if candidate_path.exists():
                        sdnq_model_path = candidate_path
                        break
            
            # 如果还是没找到，尝试扫描models目录下可能的SDNQ模型
            if sdnq_model_path is None or not sdnq_model_path.exists():
                models_dir = Path(shared.models_path) if hasattr(shared, 'models_path') else Path(__file__).parent.parent.parent.parent / "models"
                for item in models_dir.iterdir():
                    if item.is_dir() and "SDNQ" in item.name and "uint4" in item.name:
                        sdnq_model_path = models_dir / item.name
                        break
            
            # 如果仍然不存在，返回错误
            if sdnq_model_path is None or not sdnq_model_path.exists():
                print(f"SDNQ模型路径不存在: {sdnq_model_path}")
                return []
            
            # 使用SDNQ量化模型
            model_path = sdnq_model_path
            print(f"使用SDNQ量化模型: {model_path}")
            
            # 检查目录是否包含必要的文件
            try:
                if not (model_path / "transformer").exists():
                    print(f"SDNQ模型缺少transformer目录: {model_path}")
                    return []
                if not (model_path / "text_encoder").exists():
                    print(f"SDNQ模型缺少text_encoder目录: {model_path}")
                    return []
                if not (model_path / "vae").exists():
                    print(f"SDNQ模型缺少vae目录: {model_path}")
                    return []
            except Exception as e:
                print(f"检查SDNQ模型路径时出错: {model_path} - {str(e)}")
                return []

            # 使用sdnq_model.py模块加载SDNQ模型
            try:
                # 将当前目录添加到sys.path，以便正确导入sdnq_model
                current_dir = Path(__file__).parent
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                
                # 导入SDNQ模型加载函数
                from sdnq_model import load_sdnq_model
                
                # 确定torch_dtype - SDNQ模型应使用bfloat16
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                # 加载SDNQ模型
                pipeline = load_sdnq_model(model_path, torch_dtype)
                
                if pipeline is None:
                    print("SDNQ模型加载失败")
                    return []
            except Exception as e:
                print(f"SDNQ模型加载过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
                return []
            
            # 检查并修复Qwen2_5_VLConfig缺少的属性
            if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'config'):
                text_config = pipeline.text_encoder.config
                # 确保必要的视觉token id存在
                if not hasattr(text_config, 'vision_start_token_id'):
                    setattr(text_config, 'vision_start_token_id', 151652)
                if not hasattr(text_config, 'vision_end_token_id'):
                    setattr(text_config, 'vision_end_token_id', 151653)
                if not hasattr(text_config, 'vision_token_id'):
                    setattr(text_config, 'vision_token_id', 151654)
                if not hasattr(text_config, 'image_token_id'):
                    setattr(text_config, 'image_token_id', 151655)
                if not hasattr(text_config, 'video_token_id'):
                    setattr(text_config, 'video_token_id', 151656)
            
            # 设置SDNQ模型的内存管理
            try:
                pipeline.enable_model_cpu_offload()
                print("为SDNQ启用模型CPU卸载")
            except Exception as e:
                print(f"SDNQ CPU卸载设置失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("SDNQ模型加载完成")
        else:  # 检查是否未启用SDNQ，而是启用Nunchaku
            # 检查是否使用Nunchaku加速模型
            nunchaku_enable = args.get("nunchaku_enable", False)
            if nunchaku_enable:
                # 获取Nunchaku模型参数
                nunchaku_precision = args.get("nunchaku_precision", "fp4")
                nunchaku_rank = args.get("nunchaku_rank", 128)
                
                # 使用Nunchaku加速模型
                steps = args["steps"]
                
                # 获取模型路径 - 从参数中获取模型目录
                model_dir = args.get("model_dir")
                if model_dir:
                    qwenimage_edit_models_dir = Path(model_dir)
                else:
                    # 根据项目规范，使用正确的模型目录
                    # 模型应位于 shared.models_path / "qwen-image" / "qwen-image-edit" 目录下
                    import sys
                    from modules import shared
                    models_dir = Path(shared.models_path) / "qwen-image"
                    qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
                
                # 获取用户选择的模型文件
                model_file = args.get("model_file")
                
                # 检查model_file是否有效
                if not model_file or model_file == "无" or model_file == "None" or model_file == "":
                    print("错误: 未选择模型文件")
                    return []
                
                # 使用用户选择的模型文件
                model_path = qwenimage_edit_models_dir / model_file
                
                print(f"用户选择步数: {steps}")
                print(f"使用Nunchaku加速模型")
                print(f"模型路径: {model_path}")
                
                # 检查模型文件是否存在
                if not model_path.exists():
                    print(f"模型文件不存在: {model_path}")
                    return
                
                # 导入必要的库
                from diffusers import QwenImageEditPlusPipeline
                # 导入Transformer模型
                from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
                print("成功导入支持LoRA的NunchakuQwenImageTransformer2DModel (编辑版)")
                
                # 加载模型 - 按照官方示例方式
                print("开始加载模型...")
                transformer = EditTransformer.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16
                )
                
                # 使用基础模型路径，而不是模型文件所在子目录
                # 根据项目规范，文生图功能对应的Nunchaku模型必须存储在 models/qwen-image/qwenimage 子目录下
                # 图像编辑功能对应的Nunchaku模型必须存储在 models/qwen-image/qwen-image-edit 子目录下
                # 但基础模型路径应该是 models/qwen-image
                base_model_path = model_path.parent.parent  # 获取models/qwen-image目录
                base_model_path = base_model_path.resolve()  # 获取绝对路径
                
                print(f"模型根目录: {base_model_path}")
                
                # 确保基础路径存在
                if not base_model_path.exists():
                    print(f"模型根目录不存在: {base_model_path}")
                    return []
                
                # 使用本地组件创建pipeline - 按照官方示例方式
                # 首先加载基础pipeline
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    str(base_model_path),
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16
                )
                
                # 检查并修复Qwen2_5_VLConfig缺少的属性
                if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'config'):
                    text_config = pipeline.text_encoder.config
                    # 确保必要的视觉token id存在
                    if not hasattr(text_config, 'vision_start_token_id'):
                        setattr(text_config, 'vision_start_token_id', 151652)
                    if not hasattr(text_config, 'vision_end_token_id'):
                        setattr(text_config, 'vision_end_token_id', 151653)
                    if not hasattr(text_config, 'vision_token_id'):
                        setattr(text_config, 'vision_token_id', 151654)
                    if not hasattr(text_config, 'image_token_id'):
                        setattr(text_config, 'image_token_id', 151655)
                    if not hasattr(text_config, 'video_token_id'):
                        setattr(text_config, 'video_token_id', 151656)
                
                print("模型加载完成")
                
                # 移除text_encoder量化部分，因为会影响生成速度
                # 保留原始pipeline，不进行量化处理
                
                # 设置Nunchaku模型的内存管理 - 与官方示例保持一致
                try:
                    # 从nunchaku.utils导入get_gpu_memory函数
                    from nunchaku.utils import get_gpu_memory
                    
                    if get_gpu_memory() > 16:
                        pipeline.enable_model_cpu_offload()
                        print("为高内存GPU启用模型CPU卸载")
                    else:
                        # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
                        transformer.set_offload(
                            True, use_pin_memory=False, num_blocks_on_gpu=6
                        )  # increase num_blocks_on_gpu if you have more VRAM
                        pipeline._exclude_from_cpu_offload.append("transformer")
                        pipeline.enable_sequential_cpu_offload()
                        print("为低内存GPU启用逐层卸载")
                except Exception as e:
                    print(f"内存管理设置失败: {e}")
                    try:
                        # 尝试基本的CPU卸载
                        pipeline.enable_model_cpu_offload()
                        print("启用基础CPU卸载")
                    except Exception:
                        print("无法设置内存管理，继续执行...")
                        import traceback
                        traceback.print_exc()
                        # 不返回，继续执行后续逻辑
                finally:
                    pass  # 添加空的finally子句以解决语法错误
                
                # 为Nunchaku模型设置max_txt_seq_len属性，以避免运行时错误
                if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                    # 检查是否是Nunchaku模型，如果是则设置必要的属性
                    transformer_class_name = pipeline.transformer.__class__.__name__
                    if "Nunchaku" in transformer_class_name or "nunchaku" in transformer_class_name.lower():
                        print(f"检测到Nunchaku模型: {transformer_class_name}")
                        # 设置Nunchaku模型需要的参数
                        if not hasattr(pipeline.transformer, 'max_txt_seq_len'):
                            setattr(pipeline.transformer, 'max_txt_seq_len', 256)
                            print("已为Nunchaku模型设置max_txt_seq_len=256")
                        if hasattr(pipeline.transformer, 'config') and not hasattr(pipeline.transformer.config, 'max_txt_seq_len'):
                            setattr(pipeline.transformer.config, 'max_txt_seq_len', 256)
                            print("已为Nunchaku模型config设置max_txt_seq_len=256")
                        
                        # 重写forward方法，以提供默认的txt_seq_lens值
                        original_forward = pipeline.transformer.forward
                        
                        def patched_forward(
                            hidden_states,
                            encoder_hidden_states=None,
                            encoder_hidden_states_mask=None,
                            timestep=None,
                            img_shapes=None,
                            txt_seq_lens=None,
                            guidance=None,
                            attention_kwargs=None,
                            controlnet_block_samples=None,
                            return_dict=True,
                        ):
                            # 如果txt_seq_lens未提供，尝试从encoder_hidden_states推断或使用默认值
                            if txt_seq_lens is None:
                                if encoder_hidden_states is not None:
                                    # 使用encoder_hidden_states的实际序列长度
                                    seq_len = encoder_hidden_states.size(1)  # 获取序列长度维度
                                    batch_size = encoder_hidden_states.size(0) if encoder_hidden_states.dim() > 0 else 1
                                    txt_seq_lens = [seq_len] * batch_size
                                else:
                                    # 如果没有encoder_hidden_states，使用一个足够大的默认值
                                    txt_seq_lens = [256]  # 使用256作为默认值，与max_txt_seq_len一致
                            elif isinstance(txt_seq_lens, int):
                                # 如果txt_seq_lens是单个整数，将其转换为列表
                                txt_seq_lens = [txt_seq_lens]
                            
                            return original_forward(
                                hidden_states=hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask,
                                timestep=timestep,
                                img_shapes=img_shapes,
                                txt_seq_lens=txt_seq_lens,
                                guidance=guidance,
                                attention_kwargs=attention_kwargs,
                                controlnet_block_samples=controlnet_block_samples,
                                return_dict=return_dict,
                            )
                        
                        # 替换forward方法
                        pipeline.transformer.forward = patched_forward
                        print("已为Nunchaku模型patch forward方法以处理txt_seq_lens参数")
                    else:
                        # 即使不是Nunchaku模型，也要尝试设置属性，以防是QwenImageTransformer的变种
                        try:
                            if not hasattr(pipeline.transformer, 'max_txt_seq_len'):
                                setattr(pipeline.transformer, 'max_txt_seq_len', 256)
                                print("已为QwenImageTransformer设置max_txt_seq_len=256")
                            if hasattr(pipeline.transformer, 'config') and not hasattr(pipeline.transformer.config, 'max_txt_seq_len'):
                                setattr(pipeline.transformer.config, 'max_txt_seq_len', 256)
                                print("已为QwenImageTransformer config设置max_txt_seq_len=256")
                        except Exception as e:
                            print(f"设置max_txt_seq_len属性时出错: {e}")
                
                # 应用注意力优化
                apply_attention_optimizations(pipeline)

            else:
                # 标准diffusers模型加载路径 (既未启用SDNQ也未启用Nunchaku)
                print("正在加载标准diffusers模型...")
                
                # 获取模型路径
                model_dir = args.get("model_dir")
                if model_dir:
                    qwenimage_edit_models_dir = Path(model_dir)
                else:
                    models_dir = Path(__file__).parent.parent / "models"
                    qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
                
                # 获取用户选择的模型文件
                model_file = args.get("model_file")
                if not model_file or model_file in ["无", "None", ""]:
                    print("错误: 未选择模型文件")
                    return []
                
                model_path = qwenimage_edit_models_dir / model_file
                
                # 检查模型文件是否存在
                if not model_path.exists():
                    print(f"模型文件不存在: {model_path}")
                    return []
                
                # 加载Transformer模型
                transformer = EditTransformer.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16
                )
                
                # 基础模型路径
                base_model_path = model_path.parent.parent.resolve()
                print(f"标准模型根目录: {base_model_path}")
                
                # 创建pipeline
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    str(base_model_path),
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16
                )
                
                # 检查并修复Qwen2_5_VLConfig缺少的属性
                if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'config'):
                    text_config = pipeline.text_encoder.config
                    for attr, value in [
                        ('vision_start_token_id', 151652),
                        ('vision_end_token_id', 151653),
                        ('vision_token_id', 151654),
                        ('image_token_id', 151655),
                        ('video_token_id', 151656)
                    ]:
                        if not hasattr(text_config, attr):
                            setattr(text_config, attr, value)
                
                # 设置内存管理
                try:
                    pipeline.enable_model_cpu_offload()
                    print("为标准模型启用模型CPU卸载")
                except Exception as e:
                    print(f"标准模型CPU卸载设置失败: {e}")
                
                # 应用注意力优化
                apply_attention_optimizations(pipeline)
                print("标准模型加载完成")
        
        # 确保pipeline已定义后再加载LoRA模型
        try:
            if 'pipeline' in locals() and pipeline is not None:
                # 加载两个LoRA模型
                if lora_model_1 and lora_model_1 != "无" and lora_model_1 != "None":
                    load_lora_model(lora_model_1, lora_weight_1, "1", pipeline, transformer)
                    
                if lora_model_2 and lora_model_2 != "无" and lora_model_2 != "None":
                    load_lora_model(lora_model_2, lora_weight_2, "2", pipeline, transformer)
            else:
                print("警告: pipeline变量未定义，跳过LoRA模型加载")
        except Exception as e:
            print(f"LoRA加载过程中出现错误: {e}")
                
        # 处理输入图像和控制图像
        init_image = None
        control_image_path = None
        
        # 获取从单独参数传递的控制图像（编辑模型UI使用这种方式）
        control_image_path = args.get("control_image")
        
        # 查找第一个非空的图像路径作为输入图像，第二个非空路径作为控制图像（如果存在）
        non_empty_paths = [img_path.strip() for img_path in input_images 
                          if img_path and isinstance(img_path, str) and img_path.strip()]
        
        if len(non_empty_paths) > 0:
            input_image_path = non_empty_paths[0]
        # 如果没有从单独参数获取到控制图像，再尝试从images列表中获取
        if not control_image_path and len(non_empty_paths) > 1:
            control_image_path = non_empty_paths[1]
        
        # 使用UI传递的controlnet_enable参数来控制ControlNet启用状态
        controlnet_enable = args.get("controlnet_enable", False)
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        print(f"用户选择步数: {args['steps']}")
        
        # 预处理控制图像（如果启用）
        processed_control_image = None
        if controlnet_enable and control_image_path:
            # 处理预处理器名称，去除可能的前缀（如"[Pose] "）
            clean_preprocessor_type = controlnet_preprocessor
            if isinstance(clean_preprocessor_type, str) and "]" in clean_preprocessor_type:
                # 去除"[Pose] "这样的前缀
                clean_preprocessor_type = clean_preprocessor_type.split("]", 1)[1].strip()
                print(f"预处理器名称已清理: '{controlnet_preprocessor}' -> '{clean_preprocessor_type}'")
            
            # 特殊处理一些常见的预处理器名称映射
            preprocessor_mapping = {
                "dw_openpose_full": "dw_openpose_full",
                "openpose_full": "openpose_full",
                "canny": "canny",
                "depth_midas": "depth_midas",
                "depth_anything_v2": "depth_anything_v2",
                "softedge_hed": "softedge_hed",
                "lineart_standard": "lineart_standard",
                "lineart_realistic": "lineart_realistic",
                "lineart_anime_denoise": "lineart_anime_denoise"
            }
            
            # 如果clean_preprocessor_type在映射中，使用映射值
            if clean_preprocessor_type in preprocessor_mapping:
                clean_preprocessor_type = preprocessor_mapping[clean_preprocessor_type]
            
            # 特殊处理inpaint_only预处理器，需要蒙版图像
            mask_path = mask_image_path  # 使用从参数中获取的蒙版图像路径
            if clean_preprocessor_type == "inpaint_only":
                if mask_image_path:
                    print(f"为inpaint_only预处理器提供蒙版图像: {mask_path}")
                else:
                    # 如果没有单独提供蒙版，则使用control_image_path本身（假设它包含蒙版信息）
                    mask_path = control_image_path
                    print("使用control_image本身作为inpaint_only预处理器的蒙版")
            
            processed_control_image = preprocess_control_image(control_image_path, clean_preprocessor_type, mask_path)
            if processed_control_image is None:
                print("控制图像处理失败")
                has_control_image = False
            else:
                # 再次确保图像是RGB模式
                # 检查是numpy数组还是PIL图像
                if isinstance(processed_control_image, np.ndarray):
                    # 如果是numpy数组，先转换为PIL图像
                    processed_control_image = Image.fromarray(processed_control_image)
                
                if processed_control_image.mode != 'RGB':
                    processed_control_image = processed_control_image.convert('RGB')
                
                print(f"控制图像预处理完成，尺寸: {processed_control_image.size}")
        else:
            print("未启用ControlNet或控制图像不存在")
            processed_control_image = None

        # 准备生成参数
        print(f"开始处理输入图像，图像路径列表: {input_images}")
        
        if not input_image_path:
            print("错误: 未提供输入图像")
            return []
            
        try:
            print(f"尝试加载图像: {input_image_path}")
            init_image = load_image(input_image_path)
            print(f"图像加载结果: {init_image}")
            if init_image is None:
                print("错误: 无法加载输入图像")
                return
                
            # 严格按照官方示例方式处理图像
            # 确保图像是RGB模式
            init_image = init_image.convert("RGB")
            
            # 获取原始尺寸
            orig_width, orig_height = init_image.size
            width, height = orig_width, orig_height
            print(f"输入图像原始尺寸: {orig_width}x{orig_height}")
        except Exception as e:
            print(f"加载输入图像失败: {e}")
            import traceback
            traceback.print_exc()
            return []
                
        # 生成图像
        print("开始生成图像...")
        
        # 获取生成批次大小
        batch_size = args.get("batch_size", 1)
        
        # 创建生成器 - 修复：确保生成器与模型在相同设备上
        device = "cuda" if torch.cuda.is_available() and 'pipeline' in locals() and pipeline is not None else "cpu"
        generator = torch.Generator(device=device).manual_seed(args.get("seed", -1))
        
        # 准备生成参数 - 严格按照官方示例方式准备
        generation_params = {
            "image": init_image,
            "prompt": prompt,
            "true_cfg_scale": cfg_scale,
            "num_inference_steps": steps,
            "generator": generator,
            "num_images_per_prompt": batch_size,
        }
        
        # 只有当true_cfg_scale > 1时才传递negative_prompt，因为这时才启用classifier-free guidance
        if cfg_scale > 1.0:
            generation_params["negative_prompt"] = negative_prompt if negative_prompt else " "
        else:
            # 当cfg_scale <= 1.0时，不传递negative_prompt参数
            pass
        
        # 根据使用的Pipeline类型和是否启用ControlNet来处理图像输入
        if controlnet_enable and processed_control_image is not None:
            # 对于启用了ControlNet的情况，将参考图像和控制图像作为列表传递
            generation_params["image"] = [init_image, processed_control_image]
        else:
            # 对于普通编辑模式，只传递输入图像
            generation_params["image"] = init_image

        # 为SDNQ模型添加官方推荐的参数
        if sdnq_enable:
            generation_params["guidance_scale"] = 1.0  # 官方示例中的参数
            # SDNQ模型需要negative_prompt参数，即使为空格
            generation_params["negative_prompt"] = negative_prompt if negative_prompt else " "

        # 生成图像 - 使用torch.inference_mode优化性能
        try:
            # 直接执行生成，不使用线程
            with torch.inference_mode():
                images = pipeline(**generation_params).images
            
            # 保存图像，使用时间戳确保文件名唯一
            timestamp = int(time_module.time() * 1000)  # 毫秒级时间戳
            output_paths = []
            
            # 处理输出图像 - 严格按照官方示例方式处理
            for i, image in enumerate(images):
                # 确保图像是PIL Image对象
                if not isinstance(image, Image.Image):
                    # 如果是numpy数组，转换为PIL Image
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                
                # 严格按照官方示例方式处理图像
                # 确保图像是RGB模式
                image = image.convert("RGB")
                
                # 确保输出目录路径是有效的
                output_dir = args.get("output_dir", "./outputs")
                if not output_dir or not isinstance(output_dir, str):
                    output_dir = "./outputs"
                
                # 确保路径是合法的，不包含特殊字符
                output_dir = os.path.normpath(output_dir)
                
                # 直接保存图像，不做任何额外处理
                output_path = Path(output_dir) / f"qwen_image_edit_{timestamp}_{i}.png"
                
                # 确保输出目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                image.save(output_path)
                output_paths.append(str(output_path))  # 确保返回字符串路径
            
            
            return output_paths  # 成功时返回路径列表
        
        except Exception as e:
            print(f"图像生成过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return []  # 返回已有的路径，可能为空列表

    except Exception as e:
        print(f"执行图像编辑功能时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []  # 发生错误时返回空列表

    finally:
        # 清理资源（如有需要）
        pass

def apply_attention_optimizations(pipe, is_quantized_model=False):
    """应用注意力优化到模型"""
    try:
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            # 如果是量化模型，跳过优化以避免冲突
            if is_quantized_model:
                return
            # 应用SageAttention或Flash Attention优化
            if SAGE_ATTENTION_AVAILABLE:
                replace_transformer_attention_with_sage(pipe.transformer)
            elif FLASH_ATTENTION_AVAILABLE:
                replace_transformer_attention_with_flash(pipe.transformer)
        else:
            pass  # 不输出日志
    except Exception as e:
        pass  # 不输出日志


def replace_transformer_attention_with_sage(transformer):
    """将transformer中的注意力机制替换为SageAttention"""
    try:
        for name, module in transformer.named_modules():
            if 'attn' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # 替换forward方法以使用SageAttention
                original_forward = module.forward
                
                def sage_forward(hidden_states, *args, **kwargs):
                    # 检查是否有额外的位置参数
                    if len(args) > 0:
                        # 提取encoder_hidden_states（如果有）
                        encoder_hidden_states = args[0] if len(args) > 0 else hidden_states
                    else:
                        encoder_hidden_states = kwargs.get('encoder_hidden_states', hidden_states)

                    # 原始的query/key/value投影
                    query = module.to_q(hidden_states)
                    key = module.to_k(encoder_hidden_states)
                    value = module.to_v(encoder_hidden_states)

                    # 确保维度正确
                    batch_size, seq_len, dim = query.shape
                    head_dim = dim // module.heads
                    heads = module.heads

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
                    if hasattr(module, 'to_out'):
                        if not isinstance(module.to_out, (list, tuple)):
                            out = module.to_out(out)
                        else:
                            for layer in module.to_out:
                                out = layer(out)
                    
                    return out
                
                # 替换模块的forward方法
                module.forward = sage_forward
        print("[INFO] SageAttention优化应用成功")
    except Exception as e:
        print(f"[ERROR] 应用SageAttention优化失败: {str(e)}")


def replace_transformer_attention_with_flash(transformer):
    """将transformer中的注意力机制替换为FlashAttention"""
    try:
        import torch.nn.functional as F
        
        for name, module in transformer.named_modules():
            if 'attn' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # 替换forward方法以使用Flash Attention
                original_forward = module.forward
                
                def flash_forward(hidden_states, *args, **kwargs):
                    # 检查是否有额外的位置参数
                    if len(args) > 0:
                        # 提取encoder_hidden_states（如果有）
                        encoder_hidden_states = args[0] if len(args) > 0 else hidden_states
                    else:
                        encoder_hidden_states = kwargs.get('encoder_hidden_states', hidden_states)

                    # 原始的query/key/value投影
                    query = module.to_q(hidden_states)
                    key = module.to_k(encoder_hidden_states)
                    value = module.to_v(encoder_hidden_states)

                    # 确保维度正确
                    batch_size, seq_len, dim = query.shape
                    head_dim = dim // module.heads
                    heads = module.heads

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
                    if hasattr(module, 'to_out'):
                        if not isinstance(module.to_out, (list, tuple)):
                            out = module.to_out(out)
                        else:
                            for layer in module.to_out:
                                out = layer(out)
                                
                    return out
                
                # 替换模块的forward方法
                module.forward = flash_forward
        print("[INFO] Flash Attention优化应用成功")
    except Exception as e:
        print(f"[ERROR] 应用Flash Attention优化失败: {str(e)}")