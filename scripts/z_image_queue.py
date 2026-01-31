import queue
import threading
import time
from pathlib import Path
from modules import shared

# 全局队列实例
task_queue = queue.Queue()

def add_to_queue(task_type, *args):
    """将任务添加到队列"""
    # 根据任务类型解析参数
    if task_type == 'zimage_txt2img':
        # zimage文生图任务参数
        task_info = {
            'type': task_type,
            'params': {
                'prompt': args[0],
                'negative_prompt': args[1],
                'width': args[2],
                'height': args[3],
                'steps': args[4],
                'cfg_scale': args[5],
                'seed': args[6],
                'sampler': args[7],
                'batch_size': args[8],
                'lora_enable': args[9],
                'lora_model_1': args[10] if args[9] else 'N/A',
                'lora_weight_1': args[11] if args[9] else 0.0,
                'lora_model_2': args[12] if args[9] else 'N/A',
                'lora_weight_2': args[13] if args[9] else 0.0,
                'selected_model': args[14] if len(args) > 14 else None
            }
        }
    elif task_type == 'zimage_img2img':
        # zimage图生图任务参数
        task_info = {
            'type': task_type,
            'params': {
                'init_image': args[0],
                'prompt': args[1],
                'negative_prompt': args[2],
                'width': args[3],
                'height': args[4],
                'steps': args[5],
                'cfg_scale': args[6],
                'seed': args[7],
                'strength': args[8],
                'sampler': args[9],
                'batch_size': args[10],
                'lora_enable': args[11],
                'lora_model_1': args[12] if args[11] else 'N/A',
                'lora_weight_1': args[13] if args[11] else 0.0,
                'lora_model_2': args[14] if args[11] else 'N/A',
                'lora_weight_2': args[15] if args[11] else 0.0,
                'selected_model': args[16] if len(args) > 16 else None
            }
        }
    
    task = {
        'info': task_info,
        'args': args
    }
    task_queue.put(task)
    
    # 返回当前队列大小和任务信息摘要
    queue_size = task_queue.qsize()
    if task_type == 'zimage_txt2img':
        task_summary = f"文生图任务: {args[2]}x{args[3]}, 步数: {args[4]}, 批次: {args[8]}, 提示词: {args[0][:30]}{'...' if len(args[0]) > 30 else ''}"
    else:
        task_summary = f"图生图任务: {args[3]}x{args[4]}, 步数: {args[5]}, 批次: {args[10]}, 提示词: {args[1][:30]}{'...' if len(args[1]) > 30 else ''}"
    
    return f"任务已添加 - {task_summary}，当前队列大小: {queue_size}"


def process_queue(generate_txt2img_func, generate_img2img_func):
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
            if task_type == 'zimage_txt2img':
                # 文生图任务
                result, images = generate_txt2img_func(*args)
                results.extend(images if images else [])
                statuses.append(f"任务{task_num}: {result}")
            elif task_type == 'zimage_img2img':
                # 图生图任务
                result, images = generate_img2img_func(*args)
                results.extend(images if images else [])
                statuses.append(f"任务{task_num}: {result}")
            else:
                result = f"未知的任务类型: {task_type}"
                statuses.append(f"任务{task_num}: {result}")
                
            task_num += 1
        except Exception as e:
            import traceback
            results.append(None)
            statuses.append(f"任务{task_num}执行失败: {str(e)}\n{traceback.format_exc()}")
            task_num += 1
    
    if results:
        return "所有任务已完成: " + "; ".join(statuses), results
    else:
        return "队列为空，没有任务需要执行", []


def get_queue_status():
    """获取队列状态"""
    size = task_queue.qsize()
    return f"当前队列大小: {size}"


def get_detailed_queue_status():
    """获取详细的队列状态，包括任务参数"""
    temp_queue = queue.Queue()
    details = []
    idx = 1
    
    # 临时取出所有任务，记录详情，并放回原队列
    while not task_queue.empty():
        task = task_queue.get()
        temp_queue.put(task)
        
        task_info = task['info']
        task_type = task_info['type']
        params = task_info['params']
        
        if task_type == 'zimage_txt2img':
            detail = f"任务{idx}: 文生图 - {params['width']}x{params['height']}"
            detail += f", 步数: {params['steps']}, CFG: {params['cfg_scale']}"
            detail += f", 提示词: {params['prompt'][:30]}{'...' if len(params['prompt']) > 30 else ''}"
            if params['selected_model']:
                detail += f", 模型: {params['selected_model']}"
            if params['lora_enable']:
                detail += f", LoRA: {params['lora_model_1']}/{params['lora_model_2']}"
        elif task_type == 'zimage_img2img':
            detail = f"任务{idx}: 图生图 - {params['width']}x{params['height']}"
            detail += f", 步数: {params['steps']}, 强度: {params['strength']}"
            detail += f", 提示词: {params['prompt'][:30]}{'...' if len(params['prompt']) > 30 else ''}"
            if params['selected_model']:
                detail += f", 模型: {params['selected_model']}"
            if params['lora_enable']:
                detail += f", LoRA: {params['lora_model_1']}/{params['lora_model_2']}"
        
        details.append(detail)
        idx += 1
    
    # 将任务放回原队列
    while not temp_queue.empty():
        task_queue.put(temp_queue.get())
    
    if details:
        return "\n".join(details)
    else:
        return "队列为空"


def clear_queue():
    """清空队列"""
    count = 0
    while not task_queue.empty():
        task_queue.get()
        count += 1
    return f"已清空 {count} 个待处理任务"