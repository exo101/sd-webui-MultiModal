import queue
import threading
import time
import tempfile
import os
import json
from scripts.flux_klein_generators import generate_flux_klein_image, multi_img_flux_klein, inpaint_flux_klein, extend_flux_klein
from PIL import Image
import numpy as np

# 全局任务队列
task_queue = queue.Queue()

def save_image_with_mask_to_temp(image_with_mask):
    """
    将图像和蒙版数据保存到临时文件，并返回临时文件路径信息
    """
    temp_files_info = {}
    
    if isinstance(image_with_mask, dict):
        # 如果是字典格式，分别处理图像和蒙版
        if 'image' in image_with_mask:
            # 保存图像
            img_data = image_with_mask['image']
            if isinstance(img_data, np.ndarray):
                img_pil = Image.fromarray(img_data.astype('uint8'), 'RGB')
            else:
                img_pil = img_data
            
            temp_img_path = tempfile.mktemp(suffix='.png')
            img_pil.save(temp_img_path)
            temp_files_info['image_path'] = temp_img_path
        
        if 'mask' in image_with_mask:
            # 保存蒙版
            mask_data = image_with_mask['mask']
            if isinstance(mask_data, np.ndarray):
                mask_pil = Image.fromarray(mask_data.astype('uint8'), 'L')
            else:
                mask_pil = mask_data
            
            temp_mask_path = tempfile.mktemp(suffix='_mask.png')
            mask_pil.save(temp_mask_path)
            temp_files_info['mask_path'] = temp_mask_path
    elif hasattr(image_with_mask, 'convert'):  # PIL Image
        # 如果只是图像，保存为临时文件
        temp_img_path = tempfile.mktemp(suffix='.png')
        image_with_mask.save(temp_img_path)
        temp_files_info['image_path'] = temp_img_path
    elif isinstance(image_with_mask, np.ndarray):
        # 如果是numpy数组，转换为PIL然后保存
        img_pil = Image.fromarray(image_with_mask.astype('uint8'))
        temp_img_path = tempfile.mktemp(suffix='.png')
        img_pil.save(temp_img_path)
        temp_files_info['image_path'] = temp_img_path
    
    return temp_files_info

def load_image_with_mask_from_temp(temp_info):
    """
    从临时文件加载图像和蒙版数据
    """
    if not temp_info or not isinstance(temp_info, dict):
        return None
    
    image_with_mask = {}
    
    if 'image_path' in temp_info and os.path.exists(temp_info['image_path']):
        image_with_mask['image'] = Image.open(temp_info['image_path']).convert('RGB')
    
    if 'mask_path' in temp_info and os.path.exists(temp_info['mask_path']):
        image_with_mask['mask'] = Image.open(temp_info['mask_path']).convert('L')
    
    # 如果只有一个图像路径
    if 'image_path' in temp_info and 'mask_path' not in temp_info:
        return Image.open(temp_info['image_path']).convert('RGB')
    
    return image_with_mask

def cleanup_temp_files(temp_info):
    """
    清理临时文件
    """
    if not temp_info or not isinstance(temp_info, dict):
        return
    
    for key, path in temp_info.items():
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass  # 忽略删除失败的错误

def add_to_queue(task_type, *args):
    """将任务添加到队列"""
    # 清理 task_type 参数，去除可能的空白字符（包括换行符）
    task_type = str(task_type).strip() if task_type is not None else ''
    
    # 验证必需的参数存在
    if not args:
        return f"错误: 任务缺少必要参数"
    
    # 初始化 task_info
    task_info = None
    
    # 如果是inpaint任务，需要处理图像和蒙版数据
    processed_args = list(args)
    temp_files_info = None
    
    if task_type == 'inpaint':
        # 对于inpaint任务，将图像数据保存到临时文件
        original_image_data = args[0]  # image_with_mask
        temp_files_info = save_image_with_mask_to_temp(original_image_data)
        # 将临时文件路径信息替换原始图像数据
        processed_args[0] = temp_files_info
    
    try:
        # 根据任务类型解析参数
        if task_type == 'multi':
            # multi任务参数: img1, img2, prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
            task_info = {
                'type': task_type,
                'params': {
                    'image_count': '双图像' if args[1] is not None else '单图像',
                    'prompt': args[2] or '',
                    'steps': args[3],
                    'guidance_scale': args[4],
                    'seed': args[5],
                    'model_type': args[6] or 'default',
                    'batch_size': args[7],
                    'lora_enabled': bool(args[8]),
                    'lora_model': args[9] if args[8] and args[9] else 'N/A',
                    'lora_weight': float(args[10]) if args[8] else 0.0
                }
            }
        elif task_type == 'inpaint':
            # inpaint任务参数: image_with_mask (临时文件路径信息), prompt, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
            task_info = {
                'type': task_type,
                'params': {
                    'prompt': args[1] or '',
                    'steps': args[2],
                    'guidance_scale': args[3],
                    'seed': args[4],
                    'model_type': args[5] or 'default',
                    'batch_size': args[6],
                    'lora_enabled': bool(args[7]),
                    'lora_model': args[8] if args[7] and args[8] else 'N/A',
                    'lora_weight': float(args[9]) if args[7] else 0.0
                }
            }
        elif task_type == 'outpaint':
            # outpaint任务参数: image, prompt, steps, guidance_scale, left, right, top, bottom, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
            task_info = {
                'type': task_type,
                'params': {
                    'prompt': args[1] or '',
                    'steps': args[2],
                    'guidance_scale': args[3],
                    'left': args[4],
                    'right': args[5],
                    'top': args[6],
                    'bottom': args[7],
                    'seed': args[8],
                    'model_type': args[9] or 'default',
                    'batch_size': args[10],
                    'lora_enabled': bool(args[11]),
                    'lora_model': args[12] if args[11] and args[12] else 'N/A',
                    'lora_weight': float(args[13]) if args[11] else 0.0
                }
            }
        elif task_type == 'extend':
            # extend任务参数: image, prompt, steps, guidance_scale, seed, model_choice, batch_size, lora_enable, lora_model, lora_weight, extend_left, extend_right, extend_top, extend_bottom
            if len(args) >= 14:
                # 包含扩展参数的情况
                task_info = {
                    'type': task_type,
                    'params': {
                        'prompt': args[1] or '',
                        'steps': args[2],
                        'guidance_scale': args[3],
                        'seed': args[4],
                        'model_type': args[5] or 'default',
                        'batch_size': args[6],
                        'lora_enabled': bool(args[7]),
                        'lora_model': args[8] if args[7] and args[8] else 'N/A',
                        'lora_weight': float(args[9]) if args[7] else 0.0,
                        'extend_left': args[10],
                        'extend_right': args[11],
                        'extend_top': args[12],
                        'extend_bottom': args[13]
                    }
                }
            else:
                # 旧的参数格式（不包含扩展参数）
                task_info = {
                    'type': task_type,
                    'params': {
                        'prompt': args[1] or '',
                        'steps': args[2],
                        'guidance_scale': args[3],
                        'seed': args[4],
                        'model_type': args[5] or 'default',
                        'batch_size': args[6],
                        'lora_enabled': bool(args[7]),
                        'lora_model': args[8] if args[7] and args[8] else 'N/A',
                        'lora_weight': float(args[9]) if args[7] else 0.0
                    }
                }
        elif task_type == 'txt2img':
            # txt2img任务参数: prompt, width, height, steps, guidance_scale, seed, model_type, batch_size, lora_enable, lora_model, lora_weight
            task_info = {
                'type': task_type,
                'params': {
                    'prompt': args[0] or '',
                    'width': args[1],
                    'height': args[2],
                    'steps': args[3],
                    'guidance_scale': args[4],
                    'seed': args[5],
                    'model_type': args[6] or 'default',
                    'batch_size': args[7],
                    'lora_enabled': bool(args[8]),
                    'lora_model': args[9] if args[8] and args[9] else 'N/A',
                    'lora_weight': float(args[10]) if args[8] else 0.0
                }
            }
        else:
            return f"错误: 未知的任务类型 '{task_type}'"
            
    except IndexError as e:
        return f"错误: 任务参数不完整 - 缺少索引 {str(e).split()[-1]}"
    except Exception as e:
        return f"错误: 处理任务参数时发生异常 - {str(e)}"
    
    # 检查 task_info 是否已成功初始化
    if task_info is None:
        return "错误: 无法创建任务信息"

    task = {
        'info': task_info,
        'args': processed_args,  # 使用处理后的参数（图像数据已被替换为临时文件路径）
        'temp_files_info': temp_files_info  # 保存临时文件信息以便清理
    }
    task_queue.put(task)
    
    # 返回当前队列大小和任务信息摘要
    queue_size = task_queue.qsize()
    if task_type == 'outpaint':
        task_summary = f"图像扩展: {args[4]}x{args[5]}x{args[6]}x{args[7]}, 提示词: {args[1][:30]}{'...' if len(str(args[1])) > 30 else ''}"
    elif task_type == 'extend':
        if len(args) >= 14:
            task_summary = f"图像扩展: L{args[10]} R{args[11]} T{args[12]} B{args[13]}, 提示词: {args[1][:30]}{'...' if len(str(args[1])) > 30 else ''}"
        else:
            task_summary = f"图像扩展: 提示词: {args[1][:30]}{'...' if len(str(args[1])) > 30 else ''}"
    elif task_type == 'inpaint':
        task_summary = f"局部重绘: 提示词: {args[1][:30]}{'...' if len(str(args[1])) > 30 else ''}"
    elif task_type == 'txt2img':
        task_summary = f"文生图: {args[1]}x{args[2]}, 提示词: {args[0][:30]}{'...' if len(str(args[0])) > 30 else ''}"
    else:
        task_summary = f"双图结合: {'双图像' if args[1] is not None else '单图像'}, 提示词: {args[2][:30]}{'...' if len(str(args[2])) > 30 else ''}"
    
    return f"任务已添加 - {task_summary}，当前队列大小: {queue_size}"


def process_queue():
    """处理队列中的所有任务"""
    results = []
    statuses = []
    task_num = 1
    
    while not task_queue.empty():
        try:
            task = task_queue.get(timeout=1)
            task_info = task['info']
            args = task['args']
            temp_files_info = task.get('temp_files_info', None)
            task_type = task_info['type']
            
            # 如果是inpaint任务，需要从临时文件加载图像数据
            original_args = list(args)
            if task_type == 'inpaint' and temp_files_info:
                # 从临时文件加载图像和蒙版
                image_with_mask = load_image_with_mask_from_temp(temp_files_info)
                # 替换参数中的临时文件路径为实际图像数据
                args = list(args)
                args[0] = image_with_mask
                original_args = args  # 更新original_args以用于后续的清理
            
            try:
                if task_type == 'multi':
                    result, status = multi_img_flux_klein(*args)
                elif task_type == 'inpaint':
                    result, status = inpaint_flux_klein(*args)
                elif task_type == 'extend':
                    if len(args) >= 14:
                        # 包含扩展参数的情况
                        result, status = extend_flux_klein(
                            args[0],   # image
                            args[1],   # prompt
                            args[2],   # steps
                            args[3],   # guidance_scale
                            args[4],   # seed
                            args[5],   # model_type
                            args[6],   # batch_size
                            args[7],   # lora_enable
                            args[8],   # lora_model
                            args[9],   # lora_weight
                            args[10],  # extend_left
                            args[11],  # extend_right
                            args[12],  # extend_top
                            args[13]   # extend_bottom
                        )
                    else:
                        # 旧的参数格式（不包含扩展参数）
                        result, status = extend_flux_klein(*args)
                elif task_type == 'txt2img':
                    # 处理文生图任务
                    result, status = generate_flux_klein_image(
                        args[0],  # prompt
                        args[3],  # steps
                        args[4],  # guidance_scale
                        args[2],  # height
                        args[1],  # width
                        args[5],  # seed
                        args[6],  # model_type
                        args[7],  # batch_size
                        args[8],  # lora_enable
                        args[9],  # lora_model
                        args[10]  # lora_weight
                    )
                else:
                    result, status = None, f"未知的任务类型: {task_type}"
                    
                results.extend(result if result else [])
                statuses.append(f"任务{task_num}: {status}")
                
            except Exception as e:
                error_msg = f"任务{task_num}执行失败: {str(e)}"
                results.append(None)
                statuses.append(error_msg)
                print(error_msg)  # 输出到控制台便于调试
                
            finally:
                # 清理临时文件（如果有）
                if temp_files_info:
                    cleanup_temp_files(temp_files_info)
                
        except queue.Empty:
            break
        except Exception as e:
            error_msg = f"获取任务时发生异常: {str(e)}"
            statuses.append(error_msg)
            print(error_msg)
            break
            
        finally:
            task_num += 1
    
    if results:
        return results, "所有任务已完成: " + "; ".join(statuses)
    else:
        return [], "队列为空或所有任务均已处理"


def get_queue_status():
    """获取当前队列状态"""
    size = task_queue.qsize()
    return f"当前队列大小: {size}"

def clear_queue():
    """清空队列"""
    global task_queue
    old_size = task_queue.qsize()
    task_queue = queue.Queue()  # 重新创建空队列
    return f"队列已清空 (原队列大小: {old_size})"

def get_detailed_queue_status():
    """获取详细的队列状态，包括任务参数"""
    if task_queue.empty():
        return "队列为空"