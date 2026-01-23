import gradio as gr
import torch
import os
import gc
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from modules import shared
from modules import sd_samplers
from modules.ui_components import ToolButton
from modules import ui_common  # 导入ui_common模块
from modules import util  # 导入util模块
import requests
from PIL import Image
import asyncio
import queue

# 添加当前脚本目录到模块搜索路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# 导入flux_klein_angle_selector模块
from flux_klein_angle_selector import create_flux_klein_angle_visualization_component


# 初始化全局变量
pipe = None
FLUX_KLEIN_LOADED = False
task_queue = queue.Queue()


def add_to_queue(task_type, *args):
    """将任务添加到队列"""
    # 根据任务类型解析参数
    if task_type == 'multi':
        # multi任务参数: img1, img2, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
        task_info = {
            'type': task_type,
            'params': {
                'image_count': '双图像' if args[1] is not None else '单图像',
                'prompt': args[2],
                'steps': args[3],
                'guidance_scale': args[4],
                'seed': args[5],
                'model_type': args[6],
                'batch_size': args[7],
                'lora_enabled': args[8],
                'lora_model': args[9] if args[8] else 'N/A',
                'lora_weight': args[10] if args[8] else 0.0
            }
        }
    elif task_type == 'sprite':
        # sprite任务参数: img1, prompt, steps, guidance_scale, seed, model_type, sprite_rows, sprite_cols, batch_size, lora_enable, lora_model, lora_weight
        task_info = {
            'type': task_type,
            'params': {
                'image_count': '单图像',
                'prompt': args[1],
                'steps': args[2],
                'guidance_scale': args[3],
                'seed': args[4],
                'model_type': args[5],
                'sprite_rows': args[6],
                'sprite_cols': args[7],
                'batch_size': args[8],
                'lora_enabled': args[9],
                'lora_model': args[10] if args[9] else 'N/A',
                'lora_weight': args[11] if args[9] else 0.0
            }
        }
    elif task_type == 'inpaint':
        # inpaint任务参数: image_with_mask, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
        task_info = {
            'type': task_type,
            'params': {
                'prompt': args[1],
                'steps': args[2],
                'guidance_scale': args[3],
                'seed': args[4],
                'model_type': args[5],
                'batch_size': args[6],
                'lora_enabled': args[7],
                'lora_model': args[8] if args[7] else 'N/A',
                'lora_weight': args[9] if args[7] else 0.0
            }
        }
    elif task_type == 'outpaint':
        # outpaint任务参数: image, prompt, steps, guidance_scale, left, right, top, bottom, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
        task_info = {
            'type': task_type,
            'params': {
                'prompt': args[1],
                'steps': args[2],
                'guidance_scale': args[3],
                'left': args[4],
                'right': args[5],
                'top': args[6],
                'bottom': args[7],
                'seed': args[8],
                'model_type': args[9],
                'batch_size': args[10],
                'lora_enabled': args[11],
                'lora_model': args[12] if args[11] else 'N/A',
                'lora_weight': args[13] if args[11] else 0.0
            }
        }
    
    task = {
        'info': task_info,
        'args': args
    }
    task_queue.put(task)
    
    # 返回当前队列大小和任务信息摘要
    queue_size = task_queue.qsize()
    if task_type == 'sprite':
        task_summary = f"生成Spritesheet: {args[6]}x{args[7]}, 提示词: {args[1][:30]}{'...' if len(args[1]) > 30 else ''}"
    elif task_type == 'outpaint':
        task_summary = f"图像扩展: {args[4]}x{args[5]}x{args[6]}x{args[7]}, 提示词: {args[1][:30]}{'...' if len(args[1]) > 30 else ''}"
    elif task_type == 'inpaint':
        task_summary = f"局部重绘: 提示词: {args[1][:30]}{'...' if len(args[1]) > 30 else ''}"
    else:
        task_summary = f"双图结合: {'双图像' if args[1] is not None else '单图像'}, 提示词: {args[2][:30]}{'...' if len(args[2]) > 30 else ''}"
    
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
            if task_type == 'multi':
                result, status = multi_img_flux_klein(*args)
            elif task_type == 'sprite':
                result, status = generate_spritesheet_from_image(*args)
            elif task_type == 'inpaint':
                result, status = inpaint_flux_klein(*args)
            elif task_type == 'outpaint':
                result, status = outpaint_flux_klein(*args)
            else:
                result, status = None, f"未知的任务类型: {task_type}"
                
            results.extend(result if result else [])
            statuses.append(f"任务{task_num}: {status}")
            task_num += 1
        except Exception as e:
            results.append(None)
            statuses.append(f"任务{task_num}执行失败: {str(e)}")
            task_num += 1
    
    if results:
        return results, "所有任务已完成: " + "; ".join(statuses)
    else:
        return [], "队列为空，没有任务需要执行"


def get_queue_status():
    """获取当前队列状态"""
    size = task_queue.qsize()
    return f"当前队列大小: {size}"


# 存储任务详情的函数
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
        
        if task_type == 'sprite':
            detail = f"任务{idx}: Spritesheet生成 - {task_info['params']['sprite_rows']}x{task_info['params']['sprite_cols']}网格"
            detail += f", 提示词: {task_info['params']['prompt'][:30]}{'...' if len(task_info['params']['prompt']) > 30 else ''}"
            detail += f", 步数: {task_info['params']['steps']}, 种子: {task_info['params']['seed']}"
            if task_info['params']['lora_enabled']:
                detail += f", LoRA: {task_info['params']['lora_model']}(权重:{task_info['params']['lora_weight']})"
        elif task_type == 'outpaint':
            detail = f"任务{idx}: 图像扩展 - L:{task_info['params']['left']}, R:{task_info['params']['right']}, T:{task_info['params']['top']}, B:{task_info['params']['bottom']}"
            detail += f", 提示词: {task_info['params']['prompt'][:30]}{'...' if len(task_info['params']['prompt']) > 30 else ''}"
            detail += f", 步数: {task_info['params']['steps']}, 种子: {task_info['params']['seed']}"
            if task_info['params']['lora_enabled']:
                detail += f", LoRA: {task_info['params']['lora_model']}(权重:{task_info['params']['lora_weight']})"
        elif task_type == 'inpaint':
            detail = f"任务{idx}: 局部重绘"
            detail += f", 提示词: {task_info['params']['prompt'][:30]}{'...' if len(task_info['params']['prompt']) > 30 else ''}"
            detail += f", 步数: {task_info['params']['steps']}, 种子: {task_info['params']['seed']}"
            if task_info['params']['lora_enabled']:
                detail += f", LoRA: {task_info['params']['lora_model']}(权重:{task_info['params']['lora_weight']})"
        else:
            detail = f"任务{idx}: 双图结合生成 - {task_info['params']['image_count']}"
            detail += f", 提示词: {task_info['params']['prompt'][:30]}{'...' if len(task_info['params']['prompt']) > 30 else ''}"
            detail += f", 步数: {task_info['params']['steps']}, 种子: {task_info['params']['seed']}"
            if task_info['params']['lora_enabled']:
                detail += f", LoRA: {task_info['params']['lora_model']}(权重:{task_info['params']['lora_weight']})"
        
        details.append(detail)
        idx += 1
    
    # 将任务放回原队列
    while not temp_queue.empty():
        task_queue.put(temp_queue.get())
    
    if details:
        return "\n".join(details)
    else:
        return "队列为空"


# 尝试导入diffusers相关模块
try:
    from diffusers import FluxPipeline
    from diffusers.schedulers import (
        FlowMatchEulerDiscreteScheduler,
        EulerDiscreteScheduler,
        DDIMScheduler,
        DDPMScheduler
    )
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

def download_flux_klein_model(model_path, model_type):
    """从魔搭社区下载FLUX.2-klein模型"""
    try:
        from modelscope import snapshot_download
        from pathlib import Path
        
        # 根据模型类型选择模型ID
        if model_type == "FLUX.2-klein-9B":
            model_id = "black-forest-labs/FLUX.2-klein-9B"
        else:  # 包括 "FLUX.2-klein-base-4B" 和其他情况
            model_id = "black-forest-labs/FLUX.2-klein-base-4B"  # 默认模型
            
        print(f"正在使用魔搭社区下载FLUX.2-klein模型，模型ID: {model_id} 到 {model_path}...")
        
        # 创建模型目录
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # 使用魔搭社区下载模型文件
        snapshot_download(
            model_id=model_id,
            cache_dir=model_path,
            revision='master'
        )
        print(f"FLUX.2-klein模型已成功从魔搭社区下载到: {model_path}")
        return True
    except Exception as e:
        print(f"使用魔搭社区下载FLUX.2-klein模型失败: {e}")
        # 如果魔搭社区下载失败，尝试Hugging Face方式作为备选
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            from pathlib import Path
            
            # 根据模型类型选择模型ID
            if model_type == "FLUX.2-klein-9B":
                model_id = "black-forest-labs/FLUX.2-klein-9B"
            else:  # 包括 "FLUX.2-klein-base-4B" 和其他情况
                model_id = "black-forest-labs/FLUX.2-klein-base-4B"  # 默认模型
                
            print(f"正在使用Hugging Face Hub下载FLUX.2-klein模型，模型ID: {model_id} 到 {model_path}...")
            
            # 创建模型目录
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            # 使用Hugging Face下载模型文件
            hf_snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"FLUX.2-klein模型已成功使用Hugging Face Hub下载到: {model_path}")
            return True
        except Exception as e2:
            print(f"下载FLUX.2-klein模型失败: {e2}")
            return False

def load_flux_klein_pipeline(model_path, model_type, device="cuda"):
    """加载FLUX.2-klein模型管道"""
    global pipe, FLUX_KLEIN_LOADED
    try:
        print(f"正在加载FLUX.2-klein模型，路径: {model_path}，类型: {model_type}...")
        
        # 确保model_type是字符串类型
        if isinstance(model_type, int):
            # 将索引转换为实际的模型类型字符串
            model_types = ["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"]
            if 0 <= model_type < len(model_types):
                model_type = model_types[model_type]
            else:
                model_type = "FLUX.2-klein-base-4B"  # 默认模型类型
        elif not isinstance(model_type, str):
            model_type = "FLUX.2-klein-base-4B"  # 默认模型类型
        
        # 检查模型路径是否存在，如果不存在则尝试构造完整路径
        base_model_path = Path(model_path)
        
        # 检查基础路径是否直接包含模型文件（检查是否有model_index.json）
        if (base_model_path / "model_index.json").exists():
            full_model_path = base_model_path
        else:
            # 基础路径不包含模型文件，尝试构造完整路径
            print(f"基础模型路径不包含模型文件，正在尝试查找实际模型目录...")
            if model_type == "FLUX.2-klein-9B":
                full_model_path = base_model_path / "FLUX_2-klein-9B"
            else:  # 包括 "FLUX.2-klein-base-4B" 和其他情况
                full_model_path = base_model_path / "FLUX_2-klein-base-4B"
        
        print(f"实际使用的模型路径: {full_model_path}")
        
        if not full_model_path.exists():
            print(f"模型路径不存在: {full_model_path}")
            return None

        # 使用本地Flux2KleinPipeline
        if MODELSCOPE_AVAILABLE and str(model_type).startswith("FLUX.2-klein-"):
            dtype = torch.bfloat16
            pipe = Flux2KleinPipeline.from_pretrained(
                str(full_model_path), 
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
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
            # 使用diffusers加载
            from diffusers import DiffusionPipeline
            
            pipe = DiffusionPipeline.from_pretrained(
                str(full_model_path),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # 启用显存优化
            pipe.enable_model_cpu_offload()  # 启用CPU卸载
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
        
        # 由于启用了CPU卸载，不需要手动移动到设备
        print(f"模型已配置CPU卸载以节省显存")
        
        FLUX_KLEIN_LOADED = True
        print("FLUX.2-klein模型加载完成！")
        return pipe
    except Exception as e:
        print(f"加载FLUX.2-klein模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

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

def generate_flux_klein_image(prompt, steps, guidance_scale, height, width, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein生成图像"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe
        
    try:
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 生成图像 (FLUX模型不支持negative_prompt参数)
        images = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_klein_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"图像生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"生成FLUX.2-klein图像失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存并提示用户
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"图像生成失败: {e}"


def multi_img_flux_klein(img1, img2, prompt, steps, guidance_scale, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行双图像结合"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if img1 is None:
            return None, "第一张图像不能为空"
            
        # 处理图像尺寸 - 保持原始尺寸，不强制调整
        img1 = Image.fromarray(img1).convert("RGB")
        # 不再调整为固定尺寸，保持原始宽高比
        
        # 如果提供了第二张图像，将其与第一张图像结合
        if img2 is not None:
            img2 = Image.fromarray(img2).convert("RGB")
            # 不再调整为固定尺寸，保持原始宽高比
            # 将两张图像作为列表传递给管道
            image_input = [img1, img2]
        else:
            # 只有一张图像，直接使用
            image_input = img1
        
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 使用图像作为条件进行生成，FLUX模型不使用negative_prompt
        images = pipe(
            prompt=prompt,
            image=image_input,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_multi_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"双图像结合生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"双图像结合生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"双图像结合生成失败: {e}"

def inpaint_flux_klein(image_with_mask, prompt, steps, guidance_scale, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行局部编辑 - 使用蒙版区域作为编辑位置的指导"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if image_with_mask is None:
            return None, "输入图像不能为空"
        
        # 确保处理的是正确的数据类型
        image = None
        mask = None
        
        # 检查是否是ImageMask返回的字典格式
        if isinstance(image_with_mask, dict):
            # 根据webui实现，ImageMask返回包含'background'和'composite'键的字典
            if 'background' in image_with_mask and 'composite' in image_with_mask:
                # 获取图像和蒙版数据
                image_data = image_with_mask['background']
                mask_data = image_with_mask['composite']
                
                # 从复合图像中提取蒙版部分
                if isinstance(mask_data, np.ndarray):
                    # 如果是RGBA数组，提取alpha通道作为蒙版
                    if mask_data.ndim == 3 and mask_data.shape[2] == 4:
                        # Alpha channel is the 4th channel
                        mask_raw = mask_data[:, :, 3]
                        # 转换为L模式的PIL图像
                        mask = Image.fromarray(mask_raw.astype('uint8'), mode='L')
                    else:
                        # 如果不是RGBA，则使用灰度转换
                        mask_gray = np.dot(mask_data[...,:3], [0.2989, 0.5870, 0.1140])
                        mask = Image.fromarray(mask_gray.astype('uint8'), mode='L')
                elif isinstance(mask_data, Image.Image):
                    # 如果已经是PIL图像，提取alpha通道作为蒙版
                    mask = mask_data.split()[-1] if mask_data.mode in ('RGBA', 'LA') else mask_data.convert("L")
                else:
                    return None, f"蒙版数据格式不支持: {type(mask_data)}"
                
                # 确保图像是PIL RGB格式
                if isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data.astype('uint8'))
                elif isinstance(image_data, Image.Image):
                    image = image_data
                else:
                    # 如果是其他类型，尝试转换
                    try:
                        image = Image.fromarray(np.asarray(image_data).astype('uint8'))
                    except:
                        return None, f"图像数据格式不支持: {type(image_data)}"
                image = image.convert("RGB")
            else:
                return None, f"字典格式不正确，缺少必要键: {image_with_mask.keys() if hasattr(image_with_mask, 'keys') else type(image_with_mask)}"
        else:
            # 如果不是字典格式，说明没有蒙版，创建一个全黑的蒙版
            if isinstance(image_with_mask, np.ndarray):
                image = Image.fromarray(image_with_mask.astype('uint8'))
            elif isinstance(image_with_mask, Image.Image):
                image = image_with_mask
            else:
                # 尝试其他转换方式
                try:
                    image = Image.fromarray(np.asarray(image_with_mask).astype('uint8'))
                except:
                    return None, f"图像数据格式不支持: {type(image_with_mask)}"
            image = image.convert("RGB")
            # 创建一个全黑的蒙版（表示没有要编辑的区域）
            mask = Image.new("L", image.size, 0)
        
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 由于Flux2KleinPipeline不支持mask_image参数，我们只能使用图像作为条件进行生成
        # 我们可以尝试使用蒙版信息预处理图像，或者简单地依赖文本提示来指导编辑
        images = pipe(
            prompt=prompt,
            image=image,  # 使用图像作为条件
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_inpaint_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"局部编辑生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"局部编辑生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"局部编辑生成失败: {e}"

def expand_canvas(image, left=0, right=0, top=0, bottom=0):
    """
    扩展图像画布，返回扩展后的图像和蒙版
    """
    original_width, original_height = image.size
    
    # 计算新尺寸
    new_width = original_width + left + right
    new_height = original_height + top + bottom
    
    # 创建新画布，使用透明背景
    expanded_image = Image.new("RGB", (new_width, new_height), color="black")
    
    # 将原始图像粘贴到新画布的中心位置
    expanded_image.paste(image, (left, top))
    
    # 创建蒙版，黑色区域表示保持不变，白色区域表示需要生成
    mask = Image.new("L", (new_width, new_height), 0)  # 全黑蒙版
    mask_data = Image.new("L", (original_width, original_height), 255)  # 白色矩形表示原始图像区域
    
    # 在扩展区域上粘贴白色蒙版，表示这些区域需要生成
    mask.paste(0, (0, 0, new_width, top))  # 顶部
    mask.paste(0, (0, original_height + top, new_width, new_height))  # 底部
    mask.paste(0, (0, top, left, original_height + top))  # 左侧
    mask.paste(0, (original_width + left, top, new_width, original_height + top))  # 右侧
    
    return expanded_image, mask


def outpaint_flux_klein(image, prompt, steps, guidance_scale, left=0, right=0, top=0, bottom=0, seed=None, model_type="FLUX.2-klein-base-4B", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行图像扩展(outpainting)"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if image is None:
            return None, "输入图像不能为空"
        
        # 确保处理的是正确的数据类型
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        image = image.convert("RGB")
        
        # 扩展画布
        expanded_image, mask = expand_canvas(image, left, right, top, bottom)
        
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA（如果启用）
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 由于Flux2KleinPipeline不支持mask_image参数，我们只能使用图像作为条件进行生成
        # 尝试使用提示词引导模型扩展图像
        images = pipe(
            prompt=prompt,
            image=expanded_image,  # 使用扩展后的图像作为输入
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=int(batch_size),  # 添加批次大小
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存图像到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for idx, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_outpaint_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath)
            image_paths.append(filepath)
        
        return image_paths, f"图像扩展生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"图像扩展生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"图像扩展生成失败: {e}"


def generate_spritesheet_from_image(image, prompt, steps, guidance_scale, seed=None, model_type="FLUX.2-klein-base-4B", rows=2, cols=2, batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein基于上传图像生成Spritesheet，支持多视角"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if image is None:
            return None, "输入图像不能为空"
        
        # 确保处理的是正确的数据类型
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        image = image.convert("RGB")
        
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA（如果启用）
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 清理现有缓存以释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 准备输出图像
        total_images = rows * cols
        spritesheet_images = []
        
        # 解析模板关键词，根据不同的视角模板生成不同的图像
        template_mapping = {
            "ISOMETRIC ↘ Front-right view": "isometric view, front-right perspective",
            "ISOMETRIC ↙ Front-left view": "isometric view, front-left perspective", 
            "SIDE VIEW ← Profile from left": "side profile view from left",
            "TOP-DOWN ↑ Bird's eye view": "top-down view, bird's eye perspective"
        }
        
        # 检查prompt中是否包含预设模板
        templates_in_prompt = []
        for template_key, template_desc in template_mapping.items():
            if template_key in prompt:
                templates_in_prompt.append((template_key, template_desc))
        
        # 如果找到了预设模板，按模板数量生成图像
        if templates_in_prompt:
            # 循环使用找到的模板，直到达到所需图像总数
            for i in range(total_images):
                template_key, template_desc = templates_in_prompt[i % len(templates_in_prompt)]
                # 组合原始提示词和视角描述
                combined_prompt = f"{prompt.replace(template_key, '').strip()}, {template_desc}"
                
                generated_images = pipe(
                    prompt=combined_prompt,
                    image=image,  # 使用上传的图像作为条件
                    num_inference_steps=int(steps),
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                    output_type="pil"  # 明确指定输出类型
                ).images
                
                spritesheet_images.extend(generated_images)
        else:
            # 如果没有找到预设模板，则使用原始提示词重复生成
            for i in range(total_images):
                generated_images = pipe(
                    prompt=prompt,
                    image=image,  # 使用上传的图像作为条件
                    num_inference_steps=int(steps),
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                    output_type="pil"  # 明确指定输出类型
                ).images
                
                spritesheet_images.extend(generated_images)
        
        # 将生成的图像拼接成Spritesheet
        sprite_width, sprite_height = spritesheet_images[0].size if spritesheet_images else (512, 512)
        spritesheet = Image.new('RGB', (cols * sprite_width, rows * sprite_height))
        
        for idx, img in enumerate(spritesheet_images[:total_images]):
            row = idx // cols
            col = idx % cols
            spritesheet.paste(img, (col * sprite_width, row * sprite_height))
        
        # 保存Spritesheet到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flux_spritesheet_{timestamp}_{seed}.png"
        filepath = os.path.join(output_dir, filename)
        
        spritesheet.save(filepath)
        
        # 生成完成后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return [filepath], f"Spritesheet生成成功，尺寸: {cols}x{rows}，种子: {seed}，视角数: {len(templates_in_prompt) if templates_in_prompt else 1}种"
    except Exception as e:
        print(f"Spritesheet生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"Spritesheet生成失败: {e}"


def open_folder(folder_path):
    """打开指定文件夹"""
    folder_abs_path = os.path.abspath(folder_path)
    if os.path.exists(folder_abs_path):
        if sys.platform == 'win32':
            os.startfile(folder_abs_path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', folder_abs_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_abs_path])
    return gr.update()


def create_flux_klein_ui():
    """创建FLUX.2-klein的UI界面"""
    if not FLUX_KLEIN_AVAILABLE:
        with gr.Column():
            gr.Markdown("FLUX.2-klein模块当前不可用，可能是因为缺少依赖项。")
            gr.Markdown("- 需要安装 `diffusers` 库")
            gr.Markdown("- 需要安装 `modelscope` 库")
            gr.Markdown("- 需要安装 `transformers` 库")
        return

    with gr.Tabs():
        with gr.TabItem("文生图"):
            # 文生图界面组件
            with gr.Row():
                with gr.Column():
                    # 提示词输入区域
                    prompt = gr.Textbox(label="正面提示词 (Prompt)", lines=3, value="一只可爱的小猫")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 生成参数
                        steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 尺寸参数
                        height = gr.Slider(label="高度", minimum=256, maximum=1536, value=1024, step=64)
                        width = gr.Slider(label="宽度", minimum=256, maximum=1536, value=768, step=64)
                    
                    with gr.Row():
                        # 随机种子
                        seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        batch_count = gr.Slider(label="批次数量", minimum=1, maximum=8, value=1, step=1)
                        batch_size = gr.Slider(label="每批数量", minimum=1, maximum=8, value=1, step=1)
                    
                    with gr.Row():
                        # 模型类型选择
                        model_type = gr.Dropdown(
                            choices=["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"],
                            value="FLUX.2-klein-base-4B",
                            label="模型类型"
                        )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                refresh_lora_button = gr.Button("刷新LoRA模型列表")
                    
                    with gr.Row():
                        # 生成按钮和打开目录按钮
                        gen_btn = gr.Button("生成图像", variant="primary")
                        open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                
                with gr.Column():
                    # 结果展示画廊
                    result_gallery = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="flux_klein_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 任务队列
                    with gr.Accordion("任务队列", open=False):
                        queue_status = gr.Textbox(label="队列状态", interactive=False)
                        with gr.Row():
                            add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                            process_queue_btn = gr.Button("处理队列任务", variant="primary")
                            clear_queue_btn = gr.Button("清空队列", variant="stop")
                        queue_result = gr.Textbox(label="队列操作结果", interactive=False)
                        detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, max_lines=10)
            
            # 事件绑定 - 文生图部分
            def update_lora_interactive(enable_lora):
                return gr.update(interactive=not (enable_lora is None or enable_lora is False))
            
            lora_enable.change(
                fn=update_lora_interactive,
                inputs=[lora_enable],
                outputs=[lora_model]
            )
            
            # 刷新LoRA模型列表的函数
            def refresh_lora_models():
                updated_choices = list_lora_models()
                default_value = updated_choices[0] if updated_choices else ""
                return gr.update(choices=updated_choices, value=default_value)
            
            # 绑定刷新按钮事件
            refresh_lora_button.click(
                fn=refresh_lora_models,
                inputs=[],
                outputs=[lora_model]
            )
            
            gen_btn.click(
                fn=generate_flux_klein_image,
                inputs=[prompt, steps, guidance_scale, height, width, seed, model_type, batch_size, lora_enable, lora_model, lora_weight],
                outputs=[result_gallery, result_status]
            )
            
            # 打开输出目录事件
            open_outputs_btn.click(
                fn=lambda: util.open_folder("outputs"),
                inputs=[],
                outputs=[]
            )
            
            # 任务队列相关事件
            add_to_queue_btn.click(
                fn=add_to_queue,
                inputs=[
                    prompt, width, height, 
                    steps, guidance_scale, seed,
                    model_type, batch_size,
                    lora_enable, lora_model, lora_weight
                ],
                outputs=[queue_status]
            )
            process_queue_btn.click(
                fn=process_queue,
                inputs=[],
                outputs=[queue_result]
            )

            clear_queue_btn.click(
                fn=lambda: setattr(task_queue, 'queue', []) or "Queue cleared",
                inputs=[],
                outputs=[queue_status, detailed_queue_status]
            )

            # 更新详细队列状态
            queue_status.change(
                fn=get_detailed_queue_status,
                inputs=[],
                outputs=[detailed_queue_status]
            )

        with gr.TabItem("双图像结合"):
            # 双图像结合界面组件
            with gr.Row():
                with gr.Column():
                    # 双图像结合输入区域
                    with gr.Row():  # 新增行容器，使两个图像组件并排
                        with gr.Column():
                            multi_img1 = gr.Image(label="第一张图像", type="numpy", height=300)
                        with gr.Column():
                            multi_img2 = gr.Image(label="第二张图像 (可选，留空则仅处理第一张图像)", type="numpy", height=300)
                    
                    # 提示词输入区域
                    multi_prompt = gr.Textbox(label="提示词", lines=3, value="结合两张图像特征生成")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    # 3D角度可视化选择器折叠模块
                    with gr.Accordion("3D角度可视化选择器", open=False):
                        create_flux_klein_angle_visualization_component(multi_prompt)
                    
                    with gr.Row():
                        # 双图像结合参数
                        multi_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        multi_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 随机种子
                        multi_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        multi_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    with gr.Row():
                        # 模型类型选择
                        multi_model_type = gr.Dropdown(
                            choices=["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"],
                            value="FLUX.2-klein-base-4B",
                            label="模型类型"
                        )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                multi_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                multi_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                multi_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                multi_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_multi_lora_interactive(enable_lora):
                                return gr.update(interactive=enable_lora)
                            
                            multi_lora_enable.change(
                                fn=update_multi_lora_interactive,
                                inputs=multi_lora_enable,
                                outputs=multi_lora_model
                            )
                            
                            # 刷新LoRA模型列表的函数
                            def refresh_multi_lora_models():
                                updated_choices = list_lora_models()
                                default_value = updated_choices[0] if updated_choices else ""
                                return gr.update(choices=updated_choices, value=default_value)
                            
                            # 绑定刷新按钮事件
                            multi_refresh_lora_button.click(
                                fn=refresh_multi_lora_models,
                                inputs=[],
                                outputs=multi_lora_model
                            )
                    
                    # Spritesheet功能折叠模块（结合图像编辑功能）
                    with gr.Accordion("Spritesheet功能", open=False):
                        with gr.Group():
                            # 预设提示词模板
                            sprite_template_prompts = gr.Dropdown(
                                label="选择预设模板",
                                choices=[
                                    "ISOMETRIC ↘ Front-right view (等距视图 → 右前视角)",
                                    "ISOMETRIC ↙ Front-left view (等距视图 ← 左前视角)", 
                                    "SIDE VIEW ← Profile from left (侧视图 ← 从左侧看侧面)",
                                    "TOP-DOWN ↑ Bird's eye view (俯视图 ↑ 鸟瞰视角)",
                                    "ISOMETRIC ↘ Front-right view (等距视图 → 右前视角) | ISOMETRIC ↙ Front-left view (等距视图 ← 左前视角) | SIDE VIEW ← Profile from left (侧视图 ← 从左侧看侧面) | TOP-DOWN ↑ Bird's eye view (俯视图 ↑ 鸟瞰视角)"
                                ],
                                value=None,
                                info="选择一个预设视角模板，或者组合多个视角"
                            )
                            
                            def apply_template_selection(selection):
                                # 提取英文部分用于后续处理，去掉中文注释
                                # 如果是带中文注释的选项，提取原始英文关键词
                                import re
                                
                                # 处理单个模板的中文注释
                                cleaned_selection = re.sub(r'\s*\([^)]+\)', '', selection).strip()
                                
                                return cleaned_selection
                            
                            sprite_template_prompts.change(
                                fn=apply_template_selection,
                                inputs=sprite_template_prompts,
                                outputs=multi_prompt
                            )
                            
                            # Spritesheet特定参数
                            sprite_rows = gr.Slider(
                                label="行数",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=2,
                                info="精灵表的行数"
                            )
                            
                            sprite_cols = gr.Slider(
                                label="列数",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=2,
                                info="精灵表的列数"
                            )
                            
                            with gr.Row():
                                # 启用Spritesheet功能复选框
                                sprite_enabled = gr.Checkbox(
                                    label="启用Spritesheet功能",
                                    value=False,
                                    info="启用后将基于上传图像生成精灵表"
                                )
                    
                    with gr.Row():
                        # 生成按钮和打开目录按钮
                        multi_btn = gr.Button("双图像结合生成", variant="primary")
                        multi_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                
                with gr.Column():
                    # 结果展示画廊
                    multi_result_gallery = gr.Gallery(
                        label="生成结果",
                        show_label=True,
                        elem_id="flux_multi_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    multi_result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 队列功能区域
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
            
            # 事件绑定 - 双图像结合部分
            multi_btn.click(
                fn=lambda img1, img2, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight, sprite_enable, sprite_rows, sprite_cols: (
                    generate_spritesheet_from_image(img1, prompt, steps, guidance_scale, seed, model_type, sprite_rows, sprite_cols, batch_size, lora_enable, lora_model, lora_weight)
                    if sprite_enable
                    else multi_img_flux_klein(img1, img2, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight)
                ),
                inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_model_type, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight, sprite_enabled, sprite_rows, sprite_cols],
                outputs=[multi_result_gallery, multi_result_status]
            )
            
            # 更新队列状态函数
            def update_queue_status():
                return get_queue_status()
            
            def update_detailed_queue_status():
                return get_detailed_queue_status()
            
            # 统一处理所有事件绑定
            def setup_events():
                # 添加到队列的事件绑定
                add_to_queue_btn.click(
                    fn=lambda img1, img2, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight, sprite_enable, sprite_rows, sprite_cols: (
                        add_to_queue('sprite', img1, prompt, steps, guidance_scale, seed, model_type, sprite_rows, sprite_cols, batch_size, lora_enable, lora_model, lora_weight)
                        if sprite_enable
                        else add_to_queue('multi', img1, img2, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight)
                    ),
                    inputs=[multi_img1, multi_img2, multi_prompt, multi_steps, multi_guidance_scale, multi_seed, multi_model_type, multi_batch_size, multi_lora_enable, multi_lora_model, multi_lora_weight, sprite_enabled, sprite_rows, sprite_cols],
                    outputs=[queue_operation_status]
                ).then(
                    fn=update_queue_status,
                    inputs=[],
                    outputs=[queue_status_text]
                ).then(
                    fn=update_detailed_queue_status,
                    inputs=[],
                    outputs=[detailed_queue_status]
                )
                
                # 处理队列任务事件
                process_queue_btn.click(
                    fn=process_queue,
                    inputs=[],
                    outputs=[multi_result_gallery, multi_result_status]
                ).then(
                    fn=update_queue_status,
                    inputs=[],
                    outputs=[queue_status_text]
                ).then(
                    fn=update_detailed_queue_status,
                    inputs=[],
                    outputs=[detailed_queue_status]
                )
                
                # 清空队列按钮事件
                def clear_queue():
                    global task_queue
                    import queue
                    task_queue = queue.Queue()  # 重新创建空队列
                    return "队列已清空"
                
                clear_queue_btn.click(
                    fn=clear_queue,
                    inputs=[],
                    outputs=[queue_operation_status]
                ).then(
                    fn=update_queue_status,
                    inputs=[],
                    outputs=[queue_status_text]
                ).then(
                    fn=update_detailed_queue_status,
                    inputs=[],
                    outputs=[detailed_queue_status]
                )
            
            # 执行事件绑定设置
            setup_events()
            
            # 打开输出目录事件
            multi_open_outputs_btn.click(
                fn=lambda: util.open_folder("outputs"),
                inputs=[],
                outputs=[]
            )
            

        with gr.TabItem("局部重绘"):
            # 局部重绘界面组件
            with gr.Row():
                with gr.Column():
                    # 图像+蒙版输入区域（使用Gradio的ImageMask组件）
                    inpaint_image = gr.ImageMask(
                        label="上传图像并绘制蒙版区域",
                        sources=['upload'],
                        interactive=True,
                        type="pil"  # 使用pil类型以更好地兼容处理流程
                    )
                    
                    # 提示词输入区域
                    inpaint_prompt = gr.Textbox(label="提示词", lines=3, value="修复这个区域，画一只小狗")
                    # 移除负向提示词输入框，因为FLUX模型不支持
                    
                    with gr.Row():
                        # 局部重绘参数
                        inpaint_steps = gr.Slider(label="步数", minimum=1, maximum=50, value=4, step=1)
                        inpaint_guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                    
                    with gr.Row():
                        # 随机种子
                        inpaint_seed = gr.Number(label="种子 (Seed)", value=-1, precision=0)
                    
                    with gr.Row():
                        # 生成批次
                        inpaint_batch_size = gr.Slider(label="生成批次", minimum=1, maximum=8, value=1, step=1)
                    
                    with gr.Row():
                        # 模型类型选择
                        inpaint_model_type = gr.Dropdown(
                            choices=["FLUX.2-klein-base-4B", "FLUX.2-klein-9B"],
                            value="FLUX.2-klein-base-4B",
                            label="模型类型"
                        )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                inpaint_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                inpaint_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                inpaint_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,  # 设置最大值为5.0
                                    step=0.01,    # 设置步长为0.01，允许精确输入
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                inpaint_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_inpaint_lora_interactive(enable_lora):
                                return gr.update(interactive=enable_lora)
                            
                            inpaint_lora_enable.change(
                                fn=update_inpaint_lora_interactive,
                                inputs=inpaint_lora_enable,
                                outputs=inpaint_lora_model
                            )
                            
                            # 刷新LoRA模型列表的函数
                            def refresh_inpaint_lora_models():
                                updated_choices = list_lora_models()
                                default_value = updated_choices[0] if updated_choices else ""
                                return gr.update(choices=updated_choices, value=default_value)
                            
                            # 绑定刷新按钮事件
                            inpaint_refresh_lora_button.click(
                                fn=refresh_inpaint_lora_models,
                                inputs=[],
                                outputs=inpaint_lora_model
                            )
                    
                    # 队列功能区域
                    with gr.Accordion("任务队列", open=False):
                        with gr.Group():
                            queue_status_text = gr.Textbox(label="队列状态", value="当前队列大小: 0", interactive=False)
                            
                            with gr.Row():
                                # 添加到队列按钮
                                inpaint_add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                                inpaint_process_queue_btn = gr.Button("执行队列任务", variant="primary")
                                inpaint_clear_queue_btn = gr.Button("清空队列", variant="stop")
                            
                            # 队列操作状态
                            inpaint_queue_operation_status = gr.Textbox(label="队列操作状态", interactive=False)
                            
                            # 详细队列状态显示
                            inpaint_detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, lines=5, max_lines=10)
                    
                    # 生成按钮
                    inpaint_btn = gr.Button("局部重绘", variant="primary")
                
                with gr.Column():
                    # 结果展示画廊
                    inpaint_result_gallery = gr.Gallery(
                        label="局部重绘结果",
                        show_label=True,
                        elem_id="flux_inpaint_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    inpaint_result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 打开输出目录按钮
                    inpaint_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
            
            # 事件绑定 - 局部重绘部分
            inpaint_btn.click(
                fn=inpaint_flux_klein,
                inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_model_type, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
                outputs=[inpaint_result_gallery, inpaint_result_status]
            )
            
            # 更新队列状态函数
            def update_queue_status():
                return get_queue_status()
            
            def update_detailed_queue_status():
                return get_detailed_queue_status()
            
            # 统一处理所有事件绑定
            def setup_inpaint_events():
                # 添加到队列的事件绑定
                inpaint_add_to_queue_btn.click(
                    fn=lambda image_with_mask, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight: (
                        add_to_queue('inpaint', image_with_mask, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight)
                    ),
                    inputs=[inpaint_image, inpaint_prompt, inpaint_steps, inpaint_guidance_scale, inpaint_seed, inpaint_model_type, inpaint_batch_size, inpaint_lora_enable, inpaint_lora_model, inpaint_lora_weight],
                    outputs=[inpaint_queue_operation_status]
                ).then(
                    fn=update_queue_status,
                    inputs=[],
                    outputs=[queue_status_text]
                ).then(
                    fn=update_detailed_queue_status,
                    inputs=[],
                    outputs=[inpaint_detailed_queue_status]
                )
                
                # 处理队列任务事件
                inpaint_process_queue_btn.click(
                    fn=process_queue,
                    inputs=[],
                    outputs=[inpaint_result_gallery, inpaint_result_status]
                ).then(
                    fn=update_queue_status,
                    inputs=[],
                    outputs=[queue_status_text]
                ).then(
                    fn=update_detailed_queue_status,
                    inputs=[],
                    outputs=[inpaint_detailed_queue_status]
                )
                
                # 清空队列按钮事件
                def clear_inpaint_queue():
                    global task_queue
                    import queue
                    task_queue = queue.Queue()  # 重新创建空队列
                    return "队列已清空"
                
                inpaint_clear_queue_btn.click(
                    fn=clear_inpaint_queue,
                    inputs=[],
                    outputs=[inpaint_queue_operation_status]
                ).then(
                    fn=update_queue_status,
                    inputs=[],
                    outputs=[queue_status_text]
                ).then(
                    fn=update_detailed_queue_status,
                    inputs=[],
                    outputs=[inpaint_detailed_queue_status]
                )
            
            # 执行事件绑定设置
            setup_inpaint_events()
            
            # 打开输出目录事件
            inpaint_open_outputs_btn.click(
                fn=lambda: ui_common.open_folder("outputs"),
                inputs=[],
                outputs=[]
            )

        # 添加图像扩展(outpainting)标签页
        with gr.TabItem("图像扩展"):
            with gr.Row():
                with gr.Column():
                    # 图像上传
                    outpaint_image = gr.Image(
                        label="上传图像",
                        type="pil",
                        height=400,
                        interactive=True
                    )
                    
                    # 扩展尺寸设置
                    with gr.Row():
                        left_expand = gr.Number(label="左侧扩展像素", value=0, minimum=0, maximum=1024, step=8)
                        right_expand = gr.Number(label="右侧扩展像素", value=0, minimum=0, maximum=1024, step=8)
                    
                    with gr.Row():
                        top_expand = gr.Number(label="上方扩展像素", value=0, minimum=0, maximum=1024, step=8)
                        bottom_expand = gr.Number(label="下方扩展像素", value=0, minimum=0, maximum=1024, step=8)
                    
                    # 提示词输入
                    outpaint_prompt = gr.Textbox(
                        label="提示词",
                        placeholder="请输入扩展图像的描述...",
                        lines=3,
                        info="描述需要扩展的图像内容，例如：'Outpaint the image. Extend the landscape with mountains and trees'"
                    )
                    
                    # 生成参数设置
                    with gr.Group():
                        with gr.Accordion("高级参数", open=False):
                            outpaint_steps = gr.Slider(
                                label="步数",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=20,
                                info="采样步数"
                            )
                            
                            outpaint_guidance_scale = gr.Slider(
                                label="引导系数",
                                minimum=0.0,
                                maximum=20.0,
                                step=0.1,
                                value=3.5,
                                info="CFG引导强度"
                            )
                            
                            outpaint_seed = gr.Number(
                                label="种子",
                                minimum=-1,
                                maximum=2**31,
                                step=1,
                                value=-1,
                                info="随机种子(-1表示随机)"
                            )
                            
                            outpaint_model_type = gr.Dropdown(
                                label="模型类型",
                                choices=["FLUX.2-klein-base-4B"],
                                value="FLUX.2-klein-base-4B",
                                info="选择使用的模型变体"
                            )
                            
                            outpaint_batch_size = gr.Slider(
                                label="批次大小",
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=1,
                                info="一次生成图片的数量"
                            )
                    
                    # LoRA模型设置（放入Accordion折叠）
                    with gr.Accordion("LoRA模型设置", open=False):
                        with gr.Group():
                            with gr.Row():
                                outpaint_lora_enable = gr.Checkbox(
                                    label="启用LoRA",
                                    value=False,
                                    info="启用LoRA模型以修改生成风格"
                                )
                                outpaint_lora_model = gr.Dropdown(
                                    label="LoRA模型选择",
                                    choices=list_lora_models(),
                                    value=list_lora_models()[0] if list_lora_models() else "",
                                    interactive=False  # 默认不可交互
                                )
                                
                            with gr.Row():
                                outpaint_lora_weight = gr.Number(
                                    label="LoRA权重",
                                    minimum=0.0,
                                    maximum=5.0,
                                    step=0.01,
                                    value=1.0,    # 默认权重改为1.0
                                    info="控制LoRA模型的影响强度"
                                )
                                
                            # 添加刷新LoRA模型列表按钮
                            with gr.Row():
                                outpaint_refresh_lora_button = gr.Button("刷新LoRA模型列表")
                            
                            # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                            def update_outpaint_lora_interactive(enable_lora):
                                return gr.update(interactive=enable_lora)
                            
                            outpaint_lora_enable.change(
                                fn=update_outpaint_lora_interactive,
                                inputs=outpaint_lora_enable,
                                outputs=outpaint_lora_model
                            )
                            
                            # 刷新LoRA模型列表的函数
                            def refresh_outpaint_lora_models():
                                updated_choices = list_lora_models()
                                default_value = updated_choices[0] if updated_choices else ""
                                return gr.update(choices=updated_choices, value=default_value)
                            
                            # 绑定刷新按钮事件
                            outpaint_refresh_lora_button.click(
                                fn=refresh_outpaint_lora_models,
                                inputs=[],
                                outputs=outpaint_lora_model
                            )
                 
                
                with gr.Column():
                    # 结果展示画廊
                    outpaint_result_gallery = gr.Gallery(
                        label="图像扩展结果",
                        show_label=True,
                        elem_id="flux_outpaint_gallery",
                        columns=2,
                        object_fit="contain",
                        height="auto",
                    )
                    outpaint_result_status = gr.Textbox(label="状态信息", interactive=False)
                    
                    # 打开输出目录按钮
                    outpaint_open_outputs_btn = gr.Button("打开输出目录", variant="secondary")
                
                   
                # 队列功能区域
                    with gr.Accordion("任务队列", open=False):
                        with gr.Group():
                            outpaint_queue_status_text = gr.Textbox(label="队列状态", value="当前队列大小: 0", interactive=False)
                            
                            with gr.Row():
                                # 添加到队列按钮
                                outpaint_add_to_queue_btn = gr.Button("添加到队列", variant="secondary")
                                outpaint_process_queue_btn = gr.Button("执行队列任务", variant="primary")
                                outpaint_clear_queue_btn = gr.Button("清空队列", variant="stop")
                            
                            # 队列操作状态
                            outpaint_queue_operation_status = gr.Textbox(label="队列操作状态", interactive=False)
                            
                            # 详细队列状态显示
                            outpaint_detailed_queue_status = gr.Textbox(label="详细任务列表", interactive=False, lines=5, max_lines=10)
                    
                    # 生成按钮
                    outpaint_btn = gr.Button("图像扩展", variant="primary")
                    
                    # 事件绑定 - 图像扩展部分
                    outpaint_btn.click(
                        fn=outpaint_flux_klein,
                        inputs=[outpaint_image, outpaint_prompt, outpaint_steps, outpaint_guidance_scale, 
                               left_expand, right_expand, top_expand, bottom_expand,
                               outpaint_seed, outpaint_model_type, outpaint_batch_size, 
                               outpaint_lora_enable, outpaint_lora_model, outpaint_lora_weight],
                        outputs=[outpaint_result_gallery, outpaint_result_status]
                    )
                    
                    # 更新队列状态函数
                    def update_queue_status():
                        return get_queue_status()
                    
                    def update_detailed_queue_status():
                        return get_detailed_queue_status()
                    
                    # 统一处理所有事件绑定
                    def setup_outpaint_events():
                        # 添加到队列的事件绑定
                        outpaint_add_to_queue_btn.click(
                            fn=lambda image, prompt, steps, guidance_scale, left, right, top, bottom, seed, model_type, batch_size, lora_enable, lora_model, lora_weight: (
                                add_to_queue('outpaint', image, prompt, steps, guidance_scale, left, right, top, bottom, seed, model_type, batch_size, lora_enable, lora_model, lora_weight)
                            ),
                            inputs=[outpaint_image, outpaint_prompt, outpaint_steps, outpaint_guidance_scale, 
                                   left_expand, right_expand, top_expand, bottom_expand,
                                   outpaint_seed, outpaint_model_type, outpaint_batch_size, 
                                   outpaint_lora_enable, outpaint_lora_model, outpaint_lora_weight],
                            outputs=[outpaint_queue_operation_status]
                        ).then(
                            fn=update_queue_status,
                            inputs=[],
                            outputs=[outpaint_queue_status_text]
                        ).then(
                            fn=update_detailed_queue_status,
                            inputs=[],
                            outputs=[outpaint_detailed_queue_status]
                        )
                        
                        # 处理队列任务事件
                        outpaint_process_queue_btn.click(
                            fn=process_queue,
                            inputs=[],
                            outputs=[outpaint_result_gallery, outpaint_result_status]
                        ).then(
                            fn=update_queue_status,
                            inputs=[],
                            outputs=[outpaint_queue_status_text]
                        ).then(
                            fn=update_detailed_queue_status,
                            inputs=[],
                            outputs=[outpaint_detailed_queue_status]
                        )
                        
                        # 清空队列按钮事件
                        def clear_outpaint_queue():
                            global task_queue
                            import queue
                            task_queue = queue.Queue()  # 重新创建空队列
                            return "队列已清空"
                        
                        outpaint_clear_queue_btn.click(
                            fn=clear_outpaint_queue,
                            inputs=[],
                            outputs=[outpaint_queue_operation_status]
                        ).then(
                            fn=update_queue_status,
                            inputs=[],
                            outputs=[outpaint_queue_status_text]
                        ).then(
                            fn=update_detailed_queue_status,
                            inputs=[],
                            outputs=[outpaint_detailed_queue_status]
                        )
                    
                    # 执行事件绑定设置
                    setup_outpaint_events()
                    
                    # 打开输出目录按钮事件
                    outpaint_open_outputs_btn.click(
                        fn=lambda: ui_common.open_folder("outputs"),
                        inputs=[],
                        outputs=[]
                    )

    # 返回组件列表以便在其他地方使用（如果需要）
    return {
        "prompt": prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "seed": seed,
        "model_type": model_type,
        "gen_btn": gen_btn,
        "result_gallery": result_gallery,
        "result_status": result_status,
        "multi_img1": multi_img1,
        "multi_img2": multi_img2,
        "multi_prompt": multi_prompt,
        "multi_steps": multi_steps,
        "multi_guidance_scale": multi_guidance_scale,
        "multi_seed": multi_seed,
        "multi_model_type": multi_model_type,
        "multi_btn": multi_btn,
        "multi_result_gallery": multi_result_gallery,
        "multi_result_status": multi_result_status,
        "inpaint_image": inpaint_image,
        "inpaint_prompt": inpaint_prompt,
        "inpaint_steps": inpaint_steps,
        "inpaint_guidance_scale": inpaint_guidance_scale,
        "inpaint_seed": inpaint_seed,
        "inpaint_model_type": inpaint_model_type,
        "inpaint_btn": inpaint_btn,
        "inpaint_result_gallery": inpaint_result_gallery,
        "inpaint_result_status": inpaint_result_status
    }


def generate_spritesheet_flux_klein(prompt, steps, guidance_scale, seed=None, model_type="FLUX.2-klein-base-4B", rows=2, cols=2, batch_size=1, lora_enable=False, lora_model="", lora_weight=1.0):
    """使用FLUX.2-klein生成Spritesheet（精灵表）"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 获取默认模型路径
        model_path = os.path.join("models", "FLUX.2-klein")
        loaded_pipe = load_flux_klein_pipeline(model_path, model_type)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        # 设置随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31)
        generator = torch.Generator().manual_seed(seed)
        
        # 应用LoRA（如果启用）
        if lora_enable and lora_model:
            apply_lora(pipe, lora_model, lora_weight)
        
        # 计算总共需要生成的图像数量
        total_images = rows * cols
        
        # 生成多张图像
        sprites = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=total_images,  # 生成指定数量的图像
            output_type="pil"  # 明确指定输出类型
        ).images
        
        # 将生成的图像拼接成Spritesheet
        sprite_sheet = create_spritesheet(sprites, rows, cols)
        
        # 保存Spritesheet到outputs目录
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flux_spritesheet_{timestamp}_{seed}.png"
        filepath = os.path.join(output_dir, filename)
        
        sprite_sheet.save(filepath)
        
        return [filepath], f"Spritesheet生成成功，尺寸：{cols}x{rows}，共{total_images}张图像，种子: {seed}"
    except Exception as e:
        print(f"Spritesheet生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"Spritesheet生成失败: {e}"


def create_spritesheet(images, rows, cols):
    """
    将多张图像拼接成Spritesheet
    """
    if not images:
        return None
    
    # 获取单张图像的尺寸
    img_width, img_height = images[0].size
    
    # 创建一个新的画布，用于拼接图像
    sheet_width = img_width * cols
    sheet_height = img_height * rows
    spritesheet = Image.new("RGB", (sheet_width, sheet_height))
    
    # 按行列拼接图像
    for i, img in enumerate(images):
        if i >= rows * cols:
            break  # 防止单超出画布
            
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        
        # 确保图像格式一致
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        spritesheet.paste(img, (x, y))
    
    return spritesheet
