#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import os
from pathlib import Path
import torch
import math
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import FlowMatchHeunDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
import time
import psutil
import gc

def get_system_info():
    """获取系统配置信息"""
    try:
        # 获取GPU信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
        else:
            gpu_info = "CPU Only"
        
        # 获取系统内存信息
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024**3)  # GB
        available_memory = memory.available / (1024**3)  # GB
        
        return {
            "gpu": gpu_info,
            "system_memory": f"{total_memory:.1f}GB",
            "available_memory": f"{available_memory:.1f}GB"
        }
    except Exception as e:
        return {
            "gpu": "NVIDIA RTX 4070 Ti",
            "system_memory": "64GB",
            "available_memory": "Unknown"
        }

def run_text_to_image(args_file):
    """运行文生图功能"""
    try:
        print(f"开始执行文生图功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        print(f"参数加载成功: {args}")
        
        # 导入必要的库
        from diffusers import QwenImagePipeline
        from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as LightningTransformer
        from nunchaku.utils import get_gpu_memory, get_precision
        from PIL import Image
        
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
        
        print(f"Scheduler配置完成: {scheduler_type}")
        
        # 获取模型路径
        models_dir = Path(__file__).parent / "models"
        qwenimage_models_dir = models_dir / "qwenimage"
        steps = args["steps"]
        
        # 获取用户选择的模型文件
        model_file = args.get("model_file")
        print(f"用户选择的模型文件: {model_file}")
        if model_file:
            # 使用用户选择的模型文件
            model_path = qwenimage_models_dir / model_file
        else:
            # 如果没有指定模型文件，则使用默认模型
            model_path = None
            # 查找默认模型文件
            print(f"在目录 {qwenimage_models_dir} 中查找模型文件")
            for file_path in qwenimage_models_dir.glob("*.safetensors"):
                print(f"找到模型文件: {file_path}")
                model_path = file_path
                break
            
            if model_path is None:
                print("未找到任何模型文件")
                return
        
        print(f"用户选择步数: {steps}")
        print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not model_path or not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
        
        # 加载模型
        print("开始加载模型...")
        transformer = None
        pipe = None
        
        # 尝试不同的模型加载方式
        load_methods = [
            {
                "name": "量化模型加载方式",
                "import_path": "nunchaku.models.transformers.transformer_qwenimage",
                "class_name": "NunchakuQwenImageTransformer2DModel",
                "pipeline_path": str(models_dir),
                "is_quantized": True
            },
            {
                "name": "FP8量化模型加载方式",
                "import_path": "nunchaku.models.transformers.transformer_qwenimage",
                "class_name": "NunchakuQwenImageTransformer2DModel",
                "pipeline_path": str(models_dir),
                "is_quantized": True
            },
            {
                "name": "非量化模型加载方式",
                "import_path": "nunchaku",
                "class_name": "NunchakuQwenImageTransformer2DModel",
                "pipeline_path": "Qwen/Qwen-Image",
                "is_quantized": False
            }
        ]
        
        transformer = None
        pipe = None
        
        # 首先尝试使用模型文件直接创建pipeline（适用于合并的模型文件）
        try:
            print("尝试直接从模型文件创建pipeline...")
            from nunchaku.pipelines import QwenImagePipeline
            # 尝试直接从模型文件加载完整pipeline
            pipe = QwenImagePipeline.from_pretrained(
                str(model_path),
                scheduler=scheduler,
                torch_dtype=torch.bfloat16
            )
            print("直接从模型文件创建pipeline成功")
        except Exception as e:
            print(f"直接从模型文件创建pipeline失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果直接加载失败，则使用原来的加载方法
            for method in load_methods:
                try:
                    print(f"尝试使用{method['name']}...")
                    # 导入相应的类
                    module = __import__(method["import_path"], fromlist=[method["class_name"]])
                    TransformerClass = getattr(module, method["class_name"])
                    
                    # 检查模型路径
                    print(f"正在从 {model_path} 加载transformer...")
                    if model_path is None:
                        raise ValueError("模型路径为None")
                    
                    # 检查模型文件是否存在且可读
                    if not model_path.exists():
                        raise FileNotFoundError(f"模型文件不存在: {model_path}")
                    
                    # 尝试加载transformer
                    transformer = TransformerClass.from_pretrained(str(model_path))
                    print("Transformer加载成功")
                    
                    # 创建pipeline
                    print(f"正在从 {method['pipeline_path']} 创建pipeline...")
                    pipe = QwenImagePipeline.from_pretrained(
                        method["pipeline_path"],
                        transformer=transformer,
                        scheduler=scheduler,
                        torch_dtype=torch.bfloat16
                    )
                    print(f"{method['name']}成功")
                    break  # 成功则退出循环
                    
                except Exception as e:
                    print(f"{method['name']}失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 确保在下一次尝试前清理可能损坏的对象
                    transformer = None
                    pipe = None
                    continue
        
        # 检查transformer和pipe是否成功创建
        if transformer is None and pipe is None:
            print("模型加载失败：无法创建transformer或pipeline")
            return
            
        if pipe is None:
            print("模型加载失败：无法创建pipeline")
            return
        
        # 检查模型文件是否存在（已在前面检查过，这里不需要重复检查）
        # if not model_path.exists():
        #     print(f"模型文件不存在: {model_path}")
        #     return
        
        print("模型加载完成")
        
        # 设置模型卸载
        if get_gpu_memory() > 18:
            pipe.enable_model_cpu_offload()
            print("启用CPU卸载")
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
            print("启用顺序CPU卸载")
        
        # 生成图像
        print("开始生成图像...")
        # 使用官方推荐的参数
        image = pipe(
            prompt=args["prompt"],
            negative_prompt=args.get("negative_prompt", ""),  # 添加负面提示词参数
            width=args["width"],
            height=args["height"],
            num_inference_steps=steps,
            true_cfg_scale=args["cfg_scale"],
            generator=torch.manual_seed(0),  # 添加随机种子以确保结果可重现
        ).images[0]
        
        print("图像生成完成")
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_path = Path(args["output_dir"]) / f"qwen_image_{timestamp}.png"
        image.save(output_path)
        
        # 计算生成时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"图像保存完成: {output_path}")
        print(f"图像生成耗时: {generation_time:.2f}秒")
        
        # 准备生成信息
        generation_info = {
            "推理步数": steps,
            "提示词引导系数 (CFG Scale)": args["cfg_scale"],
            "宽度": args["width"],
            "高度": args["height"],
            "模型类型": "Qwen文生图模型",
            "模型文件": model_path.name if model_path else "未知",
            "采样方法": scheduler_type,
            "生成时间": f"{generation_time:.2f}秒",
            "GPU配置": system_info["gpu"],
            "系统内存": system_info["system_memory"]
        }
        
        
        # 将生成信息保存到文件，供UI读取
        info_file = Path(args["output_dir"]) / f"qwen_image_info_{timestamp}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, ensure_ascii=False, indent=2)
        
        # 输出成功信息，分别输出图像路径和信息文件路径
        print(f"SUCCESS: {output_path}")
        print(f"INFO_FILE: {info_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_image_editing(args_file):
    """运行图像编辑功能"""
    try:
        print(f"开始执行图像编辑功能，参数文件: {args_file}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 读取参数
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        print(f"参数加载成功: {args}")
        
        # 导入必要的库
        from diffusers import QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel as EditTransformer
        from nunchaku.utils import get_gpu_memory, get_precision
        from diffusers.utils import load_image
        from PIL import Image
        
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
        models_dir = Path(__file__).parent / "models"
        qwenimage_edit_models_dir = models_dir / "qwen-image-edit"
        steps = args["steps"]
        
        # 获取用户选择的模型文件
        model_file = args.get("model_file")
        if model_file:
            # 使用用户选择的模型文件
            model_path = qwenimage_edit_models_dir / model_file
        else:
            # 如果没有指定模型文件，则使用默认模型
            model_path = None
            # 查找默认模型文件
            for file_path in qwenimage_edit_models_dir.glob("*.safetensors"):
                model_path = file_path
                break
            
            if model_path is None:
                print("未找到任何编辑模型文件")
                return
        
        print(f"用户选择步数: {steps}")
        print(f"模型路径: {model_path}")
        
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
        
        # 加载模型
        print("开始加载模型...")
        transformer = EditTransformer.from_pretrained(str(model_path))
        
        # 使用本地组件创建pipeline
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            models_dir,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        )
        
        print("模型加载完成")
        
        print("未启用LoRA功能")
        
        # 设置模型卸载
        if get_gpu_memory() > 18:
            pipeline.enable_model_cpu_offload()
            print("启用CPU卸载")
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipeline._exclude_from_cpu_offload.append("transformer")
            pipeline.enable_sequential_cpu_offload()
            print("启用顺序CPU卸载")
        
        # 加载图像
        print("开始加载图像...")
        images = []
        for image_path in args["images"]:
            if image_path is not None:
                images.append(load_image(image_path).convert("RGB"))
        
        print("图像加载完成")
        
        # 准备输入
        inputs = {
            "image": images[0] if len(images) == 1 else images,  # 单张图像直接传递，多张图像传递列表
            "prompt": args["prompt"],
            "true_cfg_scale": args["cfg_scale"],
            "negative_prompt": args["negative_prompt"],
            "num_inference_steps": args["steps"],
            "generator": torch.manual_seed(0),  # 添加随机种子以确保结果可重现
            "guidance_scale": 1.0,  # 按照官方推荐设置
            "num_images_per_prompt": 1,
        }
        
        print("开始生成编辑后的图像...")
        # 生成图像
        output = pipeline(**inputs)
        output_image = output.images[0]
        
        print("图像生成完成")
        
        # 保存图像，使用时间戳确保文件名唯一
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_path = Path(args["output_dir"]) / f"qwen_image_edit_{timestamp}.png"
        output_image.save(output_path)
        
        # 计算生成时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"图像保存完成: {output_path}")
        print(f"图像生成耗时: {generation_time:.2f}秒")
        
        # 准备生成信息
        generation_info = {
            "推理步数": steps,
            "提示词引导系数 (CFG Scale)": args["cfg_scale"],
            "模型类型": "Qwen图像编辑模型",
            "模型文件": model_path.name if model_path else "未知",
            "采样方法": scheduler_type,
            "生成时间": f"{generation_time:.2f}秒",
            "GPU配置": system_info["gpu"],
            "系统内存": system_info["system_memory"]
        }
        
        
        # 将生成信息保存到文件，供UI读取
        info_file = Path(args["output_dir"]) / f"qwen_image_edit_info_{timestamp}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, ensure_ascii=False, indent=2)
        
        # 输出成功信息，分别输出图像路径和信息文件路径
        print(f"SUCCESS: {output_path}")
        print(f"INFO_FILE: {info_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python qwen_image_scripts.py <args_file>")
        sys.exit(1)
    
    args_file = sys.argv[1]
    
    # 检查参数文件是否存在
    if not os.path.exists(args_file):
        print(f"参数文件不存在: {args_file}")
        sys.exit(1)
    
    # 读取参数确定运行哪个功能
    with open(args_file, 'r', encoding='utf-8') as f:
        args = json.load(f)
    
    # 根据参数判断运行哪个功能
    if "images" in args:
        run_image_editing(args_file)
    else:
        run_text_to_image(args_file)
