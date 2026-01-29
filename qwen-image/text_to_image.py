#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen Image Extension - 文生图处理模块
用于处理文本到图像生成功能
"""

import json
import sys
import os
import math
from pathlib import Path
import torch
import sys
from pathlib import Path

# 获取当前文件目录并添加到sys.path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
import time
import gc
import cv2
import numpy as np
from PIL import Image
from safetensors.torch import load_file as load_state_dict_in_safetensors
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import FlowMatchHeunDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image

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


def run_text_to_image(args_file):
    """运行文生图功能"""
    try:
        print(f"开始执行文生图功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 获取参数
        prompt = args["prompt"]
        negative_prompt = args.get("negative_prompt", "")
        width = args["width"]
        height = args["height"]
        steps = args["steps"]
        cfg_scale = args["cfg_scale"]
        scheduler_type = args["scheduler"]
        
        # 确保sys模块已导入
        import sys
        # 导入必要的库
        from diffusers import QwenImagePipeline
        # 使用稳健的方式导入Transformer模型，优先使用支持LoRA的版本
        LightningTransformer = None
        try:
            # 首先尝试导入支持LoRA的nunchaku版本
            from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as LightningTransformer
            print("成功导入支持LoRA的NunchakuQwenImageTransformer2DModel")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"无法导入支持LoRA的nunchaku版本: {e}")
            try:
                # 回退到diffusers标准版本
                from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as LightningTransformer
                print("回退到diffusers标准版本的QwenImageTransformer2DModel")
            except Exception as e2:
                print(f"无法导入diffusers标准版本: {e2}")
                LightningTransformer = None
        
        if LightningTransformer is None:
            print("错误: 无法导入任何可用的Transformer模型")
            return
            
        from nunchaku.utils import get_gpu_memory, get_precision

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
            qwenimage_models_dir = Path(model_dir)
        else:
            # 回退到默认路径
            models_dir = Path(__file__).parent.parent / "models"
            qwenimage_models_dir = models_dir / "qwenimage"
        steps = args["steps"]
        
        # 定义torch_dtype
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # 获取用户选择的模型文件
        model_file = args.get("model_file")
        print(f"用户选择的模型文件: {model_file}")
        
        # 检查是否启用了SDNQ
        sdnq_enable = args.get("sdnq_enable", False)
        
        # 检查是否为SDNQ模型（如果启用了SDNQ或者模型名包含SDNQ相关关键词）
        # 注意：svdq是nunchaku模型，不是SDNQ模型
        is_sdnq_model = sdnq_enable or (model_file and "Qwen-Image-2512-SDNQ-4bit" in model_file)
        sdnq_model_path = None
        
        if is_sdnq_model and model_file and model_file != "无":
            if model_file.startswith("Disty0/") or len(model_file.split(os.sep)) == 1:
                # 远程模型或 safetensors 文件
                sdnq_model_path = model_file
            else:
                # 本地路径
                sdnq_model_path = Path(model_file)
        
        # 预先定义transformer变量，避免在后面的代码中出现未定义错误
        transformer = None
        pipeline = None
        
        # 获取ControlNet相关参数
        controlnet_model_selected = args.get("controlnet_model", "无")
        control_image_path = args.get("control_image")
        mask_path = args.get("control_mask", None)  # 从参数中获取mask路径
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        controlnet_enable = (control_image_path is not None 
                           and control_image_path != "" 
                           and controlnet_model_selected != "无")
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        
        # 加载ControlNet模型
        controlnet = None
        if controlnet_enable:
            try:
                # 加载ControlNet模型
                if controlnet_model_selected and controlnet_model_selected != "无":
                    # 修复：使用正确的ControlNet模型路径，应该在主models目录下，而不是插件目录
                    controlnet_base_path = Path(__file__).parent.parent.parent.parent / "models" / "ControlNet"
                    model_name = controlnet_model_selected.split('/')[-1] if '/' in controlnet_model_selected else controlnet_model_selected
                    controlnet_local_path = controlnet_base_path / model_name

                    # 确保ControlNet模型目录存在
                    controlnet_local_path.mkdir(parents=True, exist_ok=True)

                    if controlnet_local_path and (controlnet_local_path / "config.json").exists():
                        print(f"从本地路径加载ControlNet模型: {controlnet_local_path}")
                        from diffusers import QwenImageControlNetModel
                        controlnet = QwenImageControlNetModel.from_pretrained(
                            str(controlnet_local_path),
                            torch_dtype=torch_dtype,
                            trust_remote_code=True
                        )
                    else:
                        print(f"ControlNet模型不存在: {controlnet_local_path}")
                        controlnet = None
                        controlnet_enable = False
                        
                    if controlnet is not None:
                        print("ControlNet模型加载成功")
                        print(f"ControlNet模型类型: {type(controlnet)}")
                else:
                    controlnet = None
                    controlnet_enable = False
            except Exception as e:
                print(f"ControlNet模型加载失败: {e}")
                traceback.print_exc()
                controlnet = None
                controlnet_enable = False
        else:
            controlnet = None
            controlnet_enable = False
        
        # 预先定义ControlNet图像相关变量
        processed_control_image = None
        processed_control_mask = None
        
        # 如果是SDNQ模型，使用专门的加载方式
        if is_sdnq_model and sdnq_model_path:
            try:
                print("正在加载SDNQ模型...")
                # 将当前目录添加到sys.path，以便正确导入sdnq_model
                current_dir = Path(__file__).parent
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                
                # 尝试导入SDNQ模型
                import sdnq_model
                # 确定torch_dtype - SDNQ模型应使用bfloat16
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                # 检查是否启用了ControlNet，如果是，则需要使用QwenImageControlNetPipeline
                if controlnet_enable and controlnet is not None:
                    # 在这种情况下，我们需要先加载基础模型，然后包装成ControlNetPipeline
                    print("SDNQ模型检测到ControlNet启用，创建QwenImageControlNetPipeline")
                    from diffusers import QwenImageControlNetPipeline
                    
                    # 先加载SDNQ模型的基础组件
                    base_pipeline = sdnq_model.load_sdnq_model(sdnq_model_path, torch_dtype)
                    
                    if base_pipeline is None:
                        print("SDNQ模型加载失败")
                        return []
                    
                    # 用SDNQ模型的组件创建ControlNetPipeline
                    pipeline = QwenImageControlNetPipeline(
                        transformer=base_pipeline.transformer,
                        text_encoder=base_pipeline.text_encoder,
                        tokenizer=base_pipeline.tokenizer,
                        vae=base_pipeline.vae,
                        scheduler=base_pipeline.scheduler,
                        controlnet=controlnet
                    )
                    print("QwenImageControlNetPipeline创建成功")
                else:
                    # 没有启用ControlNet，直接使用基础SDNQ模型
                    base_pipeline = sdnq_model.load_sdnq_model(sdnq_model_path, torch_dtype)
                    
                    if base_pipeline is None:
                        print("SDNQ模型加载失败")
                        return []
                    
                    pipeline = base_pipeline
                
                # 为SDNQ模型启用内存管理 - 与图像编辑功能保持一致
                try:
                    pipeline.enable_model_cpu_offload()
                    print("为SDNQ启用基础CPU卸载（适用于SDNQ等模型）")
                except Exception as e:
                    print(f"SDNQ CPU卸载设置失败: {e}")
                    import traceback
                    traceback.print_exc()
                
            except ImportError as e:
                print(f"导入SDNQ模型模块失败: {e}")
                # 当triton不可用时跳过导入，不强制导入
                return []
            except Exception as e:
                print(f"加载SDNQ模型时发生错误: {e}")
                traceback.print_exc()
                return []
            
            # 为SDNQ模型定义一个虚拟的transformer变量，避免后续内存管理错误
            transformer = pipeline.transformer
        # 检查model_file是否有效，如果选择"无"则跳过模型加载
        # 修复：确保如果不是SDNQ模型才执行常规模型加载
        model_path = None
        if model_file and model_file != "无" and model_file != "None" and model_file != "" and not is_sdnq_model:
            model_path = qwenimage_models_dir / model_file
            
            print(f"用户选择步数: {steps}")
            print(f"模型路径: {model_path}")
            
            # 检查模型文件是否存在
            if not model_path.exists():
                print(f"模型文件不存在: {model_path}")
                return
        else:
            # 如果是SDNQ模型，跳过常规模型加载
            if is_sdnq_model and sdnq_model_path:
                print(f"使用SDNQ模型: {sdnq_model_path}")
            elif not is_sdnq_model and (not model_file or model_file == "无"):
                print("未选择模型文件，跳过模型加载")
                return

        # 如果不是SDNQ模型且模型路径存在，执行常规模型加载
        if not is_sdnq_model and model_path and model_path.exists():
            print(f"正在加载模型: {model_path}")
            
            # 尝试加载transformer
            try:
                # 显式导入Nunchaku模型类
                from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
                
                transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                    str(model_path),
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                print("Transformer加载成功")
            except ImportError as e:
                print(f"无法导入Nunchaku模型: {e}")
                print("尝试使用diffusers标准版本...")
                try:
                    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
                    transformer = QwenImageTransformer2DModel.from_pretrained(
                        str(model_path),
                        torch_dtype=torch_dtype,
                        trust_remote_code=True
                    )
                    print("使用diffusers标准Transformer加载成功")
                except Exception as e2:
                    print(f"标准Transformer加载也失败: {e2}")
                    import traceback
                    traceback.print_exc()
                    return []
            except Exception as e:
                print(f"Transformer加载失败: {e}")
                import traceback
                traceback.print_exc()
                return []
                
            # 添加ControlNet相关路径到系统路径
            controlnet_path = Path(__file__).parent.parent / "ControlNet" / "Qwen-Image-ControlNet-Union"
            controlnet_path_str = str(controlnet_path)
            if controlnet_path_str not in sys.path:
                sys.path.append(controlnet_path_str)
                print(f"已添加ControlNet路径到sys.path: {controlnet_path_str}")
            
            # 使用模型根目录作为基础路径，而不是模型文件所在子目录
            # 模型根目录包含model_index.json和其他必要组件
            base_model_path = str(model_path.parent.parent)  # models/qwen-image
            
            if controlnet_enable and controlnet is not None:
                print("尝试使用ControlNet管道")
                try:
                    from diffusers import QwenImageControlNetPipeline
                    print(f"ControlNet类类型: {type(controlnet)}")
                    print(f"ControlNet设备: {next(controlnet.parameters()).device if hasattr(controlnet, 'parameters') else 'unknown'}")
                    
                    # 创建ControlNet Pipeline，使用模型根目录作为基础路径
                    # 注意：from_pretrained 方法可能不直接接受 torch_dtype 参数
                    pipeline = QwenImageControlNetPipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        controlnet=controlnet,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
                    # 将整个pipeline移动到指定的数据类型
                    if torch_dtype != pipeline.transformer.dtype:
                        pipeline.to(torch_dtype)
                                
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
                                    
                    print("ControlNet管道创建成功")
                except Exception as e:
                    print(f"ControlNet管道创建失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("回退到标准QwenImagePipeline管道")
                    from diffusers import QwenImagePipeline
                    pipeline = QwenImagePipeline.from_pretrained(
                        base_model_path,
                        transformer=transformer,
                        scheduler=scheduler,
                        torch_dtype=torch_dtype
                    )
                    # 将整个pipeline移动到指定的数据类型
                    if torch_dtype != pipeline.transformer.dtype:
                        pipeline.to(torch_dtype)
                    controlnet_enable = False
            else:
                print("使用标准QwenImagePipeline管道")
                from diffusers import QwenImagePipeline
                pipeline = QwenImagePipeline.from_pretrained(
                    base_model_path,
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=torch_dtype
                )
                # 将整个pipeline移动到指定的数据类型
                if torch_dtype != pipeline.transformer.dtype:
                    pipeline.to(torch_dtype)

        print("Pipeline已构建")
        
        # 移除text_encoder量化部分，因为会影响生成速度
        # 保留原始pipeline，不进行量化处理
        
        print("模型加载完成")
        
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
        
        # ControlNet模型已经在前面的代码块中处理过了，这里不再重复处理

        # 继续常规模型加载流程（已在上面的else块中处理）
        
        # 设置Nunchaku模型的内存管理 - 与官方示例保持一致
        try:
            # 从nunchaku.utils导入get_gpu_memory函数
            from nunchaku.utils import get_gpu_memory
            
            if pipeline is None:
                print("Pipeline未定义，无法设置内存管理")
                return []
                
            if get_gpu_memory() > 16:
                pipeline.enable_model_cpu_offload()
                print("为高内存GPU启用模型CPU卸载")
            else:
                # 检查transformer是否有set_offload方法（仅Nunchaku模型有此方法）
                if hasattr(transformer, 'set_offload'):
                    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
                    transformer.set_offload(
                        True, use_pin_memory=False, num_blocks_on_gpu=6
                    )  # increase num_blocks_on_gpu if you have more VRAM
                    pipeline._exclude_from_cpu_offload.append("transformer")
                    pipeline.enable_sequential_cpu_offload()
                    print("为低内存GPU启用逐层卸载")
                else:
                    # 如果transformer没有set_offload方法（如SDNQ模型），则使用基础CPU卸载
                    pipeline.enable_model_cpu_offload()
                    print("启用基础CPU卸载（适用于SDNQ等模型）")
        except Exception as e:
            print(f"内存管理设置失败: {e}")
            try:
                # 尝试基本的CPU卸载
                pipeline.enable_model_cpu_offload()
                print("启用基础CPU卸载")
            except Exception:
                print("无法设置内存管理，继续执行...")
        
        # 获取随机种子
        seed = args.get("seed", -1)
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # 创建生成器 - 修复：确保生成器与模型在相同设备上
        device = "cuda" if torch.cuda.is_available() and pipeline is not None and hasattr(pipeline, 'device') else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # 处理ControlNet相关参数
        control_image_path = args.get("control_image")
        controlnet_conditioning_scale = args.get("controlnet_conditioning_scale", 1.0)
        controlnet_preprocessor = args.get("controlnet_preprocessor", "none")
        controlnet_start = args.get("controlnet_start", 0.0)
        controlnet_end = args.get("controlnet_end", 1.0)
        mask_path = args.get("mask_path", None)  # 添加mask_path参数获取
        
        # 统一ControlNet启用条件：必须提供控制图像且模型不是"无"
        controlnet_model_selected = args.get("controlnet_model", "无")
        controlnet_enable = (control_image_path is not None 
                           and control_image_path != "" 
                           and controlnet_model_selected != "无")
        
        print(f"ControlNet启用状态: {controlnet_enable}")
        print(f"控制图像路径: {control_image_path}")
        print(f"选择的ControlNet模型: {controlnet_model_selected}")
        
        if controlnet_enable and control_image_path:
            # 使用qwen_image_controlnet.py中的预处理函数
            processed_control_image = preprocess_for_qwen_image_controlnet(control_image_path, controlnet_preprocessor, mask_path)
            if processed_control_image is None:
                print("控制图像处理失败")
                controlnet_enable = False
            else:
                # 再次确保图像是RGB模式
                # 检查是numpy数组还是PIL图像
                if isinstance(processed_control_image, np.ndarray):
                    # 如果是numpy数组，先转换为PIL图像
                    processed_control_image = Image.fromarray(processed_control_image)
                
                # 现在确保是PIL图像并转换为RGB模式
                if processed_control_image.mode != 'RGB':
                    processed_control_image = processed_control_image.convert('RGB')
                
                # 设置processed_control_mask为None，因为我们目前不处理单独的mask
                processed_control_mask = None
        else:
            processed_control_image = None
            controlnet_enable = False

        # 准备生成参数
        generation_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg_scale,
            "generator": generator,
        }
        
        # 只有当true_cfg_scale > 1时才传递negative_prompt，因为这时才启用classifier-free guidance
        if cfg_scale > 1.0:
            generation_params["negative_prompt"] = negative_prompt
        
        # 获取生成批次大小并添加到生成参数中
        batch_size = args.get("batch_size", 1)
        generation_params["num_images_per_prompt"] = batch_size
        
        # 如果启用了ControlNet，添加ControlNet相关参数
        if controlnet_enable and controlnet is not None and processed_control_image is not None:
            generation_params.update({
                "control_image": processed_control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": controlnet_start,
                "control_guidance_end": controlnet_end,
            })
            print(f"ControlNet已启用，参数: 强度={controlnet_conditioning_scale}, 开始={controlnet_start}, 结束={controlnet_end}")
        else:
            print("ControlNet未启用或条件不满足")

        # 添加LoRA模型参数
        lora_model_1 = args.get("lora_model_1")
        lora_model_2 = args.get("lora_model_2")
        lora_weight_1 = args.get("lora_weight_1", 1.0)
        lora_weight_2 = args.get("lora_weight_2", 1.0)
        
        # 添加参数详细信息日志
        print(f"宽度: {width}, 高度: {height}")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"推理步数: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"调度器类型: {scheduler_type}")
        print(f"LoRA模型1: {lora_model_1}, 权重: {lora_weight_1}")
        print(f"LoRA模型2: {lora_model_2}, 权重: {lora_weight_2}")
        
        # 添加nunchaku目录到sys.path
        nunchaku_path = Path(__file__).parent.parent / "nunchaku-2_lora_concat"
        if str(nunchaku_path) not in sys.path:
            sys.path.insert(0, str(nunchaku_path))  # 插入到路径开头确保优先级
        
        # 确保pipeline已定义后再加载LoRA模型
        try:
            if pipeline is not None:
                # 加载两个LoRA模型
                if lora_model_1 and lora_model_1 != "无" and lora_model_1 != "None":
                    load_lora_model(lora_model_1, lora_weight_1, "1", pipeline, transformer)
                    
                if lora_model_2 and lora_model_2 != "无" and lora_model_2 != "None":
                    load_lora_model(lora_model_2, lora_weight_2, "2", pipeline, transformer)
            else:
                print("警告: pipeline变量未定义，跳过LoRA模型加载")
        except Exception as e:
            print(f"LoRA加载过程中出现错误: {e}")

        # 生成图像
        try:
            print("开始生成图像...")
            
            # 检查是否为SDNQ模型，如果是则使用专门的生成函数
            if is_sdnq_model and sdnq_model_path:
                # 使用SDNQ专用的生成函数
                import sdnq_model  # 使用绝对导入
                images = sdnq_model.run_sdnq_generation(
                    pipe=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=cfg_scale,
                    seed=seed,
                    num_images_per_prompt=batch_size,  # 传递批量生成参数
                    control_image=processed_control_image if controlnet_enable else None,
                    controlnet_conditioning_scale=controlnet_conditioning_scale if controlnet_enable else 1.0,
                    control_guidance_start=controlnet_start if controlnet_enable else 0.0,
                    control_guidance_end=controlnet_end if controlnet_enable else 1.0
                )
                
                if images is not None and len(images) > 0:
                    # 确保images是一个列表
                    if not isinstance(images, list):
                        images = [images]
                else:
                    print("SDNQ图像生成失败")
                    return []
            else:
                # 使用官方推荐的参数
                # 不再显式传递 txt_seq_lens 参数，由 transformer 的 forward 方法内部逻辑自动处理
                images = pipeline(**generation_params).images
            
            print("图像生成完成")
            
            # 保存图像，使用时间戳确保文件名唯一
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳

            output_paths = []
            
            # 获取生成批次大小
            batch_size = args.get("batch_size", 1)
            
            for i, image in enumerate(images):
                # 确保输出目录路径是有效的
                output_dir = args.get("output_dir", "./outputs")
                if not output_dir or not isinstance(output_dir, str):
                    output_dir = "./outputs"
                
                # 确保路径是合法的，不包含特殊字符
                output_dir = os.path.normpath(output_dir)
                
                output_path = Path(output_dir) / f"qwen_image_{timestamp}_{i}.png"
                
                # 确保输出目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                image.save(output_path)
                output_paths.append(str(output_path))  # 确保返回字符串路径
                print(f"图像保存完成: {output_path}")
            
            # 输出成功信息，输出所有图像路径
            for output_path in output_paths:
                print(f"SUCCESS: {output_path}")
        except Exception as e:
            print(f"图像生成过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
    except Exception as e:
        print(f"运行文生图功能时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return