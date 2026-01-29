
import torch
import os
import gc
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import sys

from scripts.flux_klein_model_loader import pipe, FLUX_KLEIN_LOADED, load_flux_klein_pipeline, apply_lora, get_full_model_path

def generate_flux_klein_image(prompt, steps, guidance_scale, height, width, seed=None, model_choice="FLUX.2-klein-base-4B (BF16)", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein生成图像"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 确保model_choice是字符串
    if isinstance(model_choice, list):
        model_choice = model_choice[0] if len(model_choice) > 0 else "FLUX_2-klein-base-4B (BF16-4B)"
    
    # 解析模型选择，获取实际模型类型 - 检查是否包含"9B"来确定模型类型
    actual_model_type = "FLUX.2-klein-9B" if "9B" in model_choice else "FLUX.2-klein-base-4B"
    
    # 检测是否包含FP8模型文件
    model_path_for_check = get_full_model_path(model_choice)
    has_fp8 = any(Path(model_path_for_check).glob("*fp8*")) or any(Path(model_path_for_check).glob("*FP8*"))
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 直接使用model_choice参数，不再拆分为model_path和actual_model_type
        loaded_pipe = load_flux_klein_pipeline(model_choice)
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


def multi_img_flux_klein(img1, img2, prompt, steps, guidance_scale, seed=None, model_choice="FLUX.2-klein-base-4B (BF16)", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行图像编辑"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 确保model_choice是字符串
    if isinstance(model_choice, list):
        model_choice = model_choice[0] if len(model_choice) > 0 else "FLUX_2-klein-base-4B (BF16-4B)"
    
    # 解析模型选择，获取实际模型类型 - 检查是否包含"9B"来确定模型类型
    actual_model_type = "FLUX.2-klein-9B" if "9B" in model_choice else "FLUX.2-klein-base-4B"
    
    # 检测是否包含FP8模型文件
    model_path_for_check = get_full_model_path(model_choice)
    has_fp8 = any(Path(model_path_for_check).glob("*fp8*")) or any(Path(model_path_for_check).glob("*FP8*"))
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 直接使用model_choice参数，不再拆分为model_path和actual_model_type
        loaded_pipe = load_flux_klein_pipeline(model_choice)
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
        
        return image_paths, f"图像编辑生成成功，共生成{len(image_paths)}张，种子: {seed}"
    except Exception as e:
        print(f"图像编辑生成失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果遇到显存不足错误，尝试清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None, f"图像编辑生成失败: {e}"

def inpaint_flux_klein(image_with_mask, prompt, steps, guidance_scale, seed=None, model_choice="FLUX.2-klein-base-4B (BF16)", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行局部编辑 - 使用蒙版区域作为编辑位置的指导"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 确保model_choice是字符串
    if isinstance(model_choice, list):
        model_choice = model_choice[0] if len(model_choice) > 0 else "FLUX_2-klein-base-4B (BF16-4B)"
    
    # 解析模型选择，获取实际模型类型 - 检查是否包含"9B"来确定模型类型
    actual_model_type = "FLUX.2-klein-9B" if "9B" in model_choice else "FLUX.2-klein-base-4B"
    
    # 检测是否包含FP8模型文件
    model_path_for_check = get_full_model_path(model_choice)
    has_fp8 = any(Path(model_path_for_check).glob("*fp8*")) or any(Path(model_path_for_check).glob("*FP8*"))
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 直接使用model_choice参数，不再拆分为model_path和actual_model_type
        loaded_pipe = load_flux_klein_pipeline(model_choice)
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
                        
                        # 转换蒙版为PIL图像并调整到与原图相同的尺寸
                        image = Image.fromarray(image_data).convert("RGB")
                        mask = Image.fromarray(mask_raw).convert("L")
                        mask = mask.resize(image.size)
                    elif mask_data.ndim == 2:
                        # 如果已经是二维数组（灰度蒙版）
                        image = Image.fromarray(image_data).convert("RGB")
                        mask = Image.fromarray(mask_data).convert("L")
                        mask = mask.resize(image.size)
                    else:
                        # 其他情况，直接使用图像数据
                        image = Image.fromarray(image_data).convert("RGB")
                        # 默认蒙版 - 全白表示全部区域
                        mask = Image.new("L", image.size, 255)
                else:
                    # 如果mask_data不是numpy数组，尝试直接处理
                    image = Image.fromarray(image_data).convert("RGB")
                    # 默认蒙版 - 全白表示全部区域
                    mask = Image.new("L", image.size, 255)
            else:
                # 如果没有预期的键，尝试直接使用整个image_with_mask
                image = Image.fromarray(image_with_mask).convert("RGB")
                # 默认蒙版 - 全白表示全部区域
                mask = Image.new("L", image.size, 255)
        else:
            # 如果不是字典格式，尝试直接处理
            image = Image.fromarray(image_with_mask).convert("RGB")
            # 默认蒙版 - 全白表示全部区域
            mask = Image.new("L", image.size, 255)
        
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
        
        # 使用图像和蒙版进行局部编辑生成，FLUX模型不使用negative_prompt
        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,  # 使用蒙版指定编辑区域
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
            filename = f"flux_inpaint_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
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

def extend_flux_klein(image, prompt, steps, guidance_scale, seed=None, model_choice="FLUX.2-klein-base-4B (BF16)", batch_size=1, lora_enable=False, lora_model="", lora_weight=0.5):
    """使用FLUX.2-klein进行图像扩展"""
    global pipe, FLUX_KLEIN_LOADED
    
    # 确保model_choice是字符串
    if isinstance(model_choice, list):
        model_choice = model_choice[0] if len(model_choice) > 0 else "FLUX_2-klein-base-4B (BF16-4B)"
    
    # 解析模型选择，获取实际模型类型 - 检查是否包含"9B"来确定模型类型
    actual_model_type = "FLUX.2-klein-9B" if "9B" in model_choice else "FLUX.2-klein-base-4B"
    
    # 检测是否包含FP8模型文件
    model_path_for_check = get_full_model_path(model_choice)
    has_fp8 = any(Path(model_path_for_check).glob("*fp8*")) or any(Path(model_path_for_check).glob("*FP8*"))
    
    # 如果模型未加载，则尝试加载
    if not FLUX_KLEIN_LOADED or pipe is None:
        # 直接使用model_choice参数，不再拆分为model_path和actual_model_type
        loaded_pipe = load_flux_klein_pipeline(model_choice)
        if loaded_pipe is None:
            return None, f"无法加载模型，请确保模型已下载"
        pipe = loaded_pipe

    try:
        if image is None:
            return None, "输入图像不能为空"
        
        # 处理图像尺寸 - 保持原始尺寸，不强制调整
        image = Image.fromarray(image).convert("RGB")
        
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
        
        # 使用图像进行扩展生成，FLUX模型不使用negative_prompt
        # 注意：这里我们暂时使用普通生成，因为FLUX官方可能没有专门的扩展功能
        # 在实际应用中，可能需要创建一个更大的画布并在其中放置原始图像，然后生成周围区域
        images = pipe(
            prompt=prompt,
            image=image,
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
            filename = f"flux_extend_{timestamp}_{seed}_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
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