"""
Qwen3-VL-4B-Instruct 模型处理模块
专门用于处理 Qwen3-VL-4B-Instruct 模型的调用和处理逻辑
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到系统路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    # 尝试导入必要的库
    from modelscope import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch
    from PIL import Image
    
    # 检查bitsandbytes是否可用
    try:
        import bitsandbytes as bnb
        BNB_AVAILABLE = True
    except ImportError:
        BNB_AVAILABLE = False
        print("警告: bitsandbytes不可用，将无法使用4位量化加载模型")
    
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: 无法导入 transformers 或 modelscope 库，将使用模拟实现")

# 定义模型相关常量，使用相对路径
MODEL_DIR = Path("models/Qwen3-VL-4B-Instruct")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 添加一个标志来控制是否已经显示过跳过下载的消息
download_message_shown = False

# 全局模型和processor变量
model = None
processor = None

def get_model_info():
    """
    获取 Qwen3-VL-4B-Instruct 模型的信息
    """
    return {
        "model_name": "Qwen3-VL-4B-Instruct",
        "model_id": "qwen/qwen3-vl-4b-instruct",
        "description": "阿里通义千问推出的40亿参数视觉语言模型，具备强大的多模态理解能力，显存占用更少",
        "capabilities": [
            "视觉问答",
            "图像描述生成", 
            "OCR文本识别",
            "视觉推理",
            "GUI操作理解",
            "长文本理解"
        ]
    }

def is_model_downloaded():
    """
    检查模型是否已经下载
    """
    # 检查模型目录是否存在关键文件
    required_files = [
        "config.json",
        "pytorch_model.bin",  # 简化检查，实际模型可能有多个文件
        "tokenizer_config.json"
    ]
    
    for file_name in required_files:
        file_path = MODEL_DIR / file_name
        if not file_path.exists():
            return False
    
    return True

def download_model():
    """
    下载 Qwen3-VL-4B-Instruct 模型
    """
    global download_message_shown
    
    if is_model_downloaded():
        if not download_message_shown:
            print("模型已存在，跳过下载")
            download_message_shown = True
        return True
    
    if not TRANSFORMERS_AVAILABLE:
        print("错误: 缺少必要的依赖项（transformers 或 modelscope），无法下载模型")
        return False
    
    try:
        print("开始下载 Qwen3-VL-4B-Instruct 模型...")
        # 重置标志，以便下次重新检查时可以显示消息
        download_message_shown = False
        
        # 使用 ModelScope 下载模型
        model_dir = snapshot_download(
            "qwen/Qwen3-VL-4B-Instruct",
            local_dir=str(MODEL_DIR)
        )
        
        print("模型下载完成")
        return True
    except Exception as e:
        print(f"模型下载失败: {str(e)}")
        return False

def is_model_available():
    """
    检查 Qwen3-VL-4B-Instruct 模型是否可用
    """
    try:
        # 检查是否安装了必要的依赖
        import transformers
        import torch
        import modelscope
        return True
    except ImportError:
        return False

def load_model():
    """
    加载 Qwen3-VL-4B-Instruct 模型和processor
    """
    global model, processor
    
    if model is not None and processor is not None:
        return True
    
    if not is_model_available():
        print("错误：缺少必要的依赖项（transformers、modelscope 或 torch），请安装后再试。")
        return False
    
    try:
        # 确保模型已下载
        if not download_model():
            return False
            
        print("正在加载 Qwen3-VL-4B-Instruct 模型...")
        # 加载模型和processor
        model = AutoModelForImageTextToText.from_pretrained(
            str(MODEL_DIR),
            device_map="auto",  # 自动选择设备
            torch_dtype=torch.float16,  # 使用半精度减少显存占用
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 减少CPU内存使用
        )
        processor = AutoProcessor.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True
        )
        print("模型加载完成")
        return True
    except Exception as e:
        # 检查是否是显存不足的问题
        error_message = str(e)
        if "out of memory" in error_message.lower() or "allocation" in error_message.lower():
            print(f"显存不足错误: {error_message}")
            print("尝试使用更多内存优化设置重新加载模型...")
            
            # 清理可能已分配的内存
            if 'model' in locals():
                del model
            if 'processor' in locals():
                del processor
            torch.cuda.empty_cache()
            
            try:
                # 检查是否可以使用bitsandbytes进行4位量化
                load_in_4bit = False
                if BNB_AVAILABLE:
                    try:
                        # 尝试使用4位量化
                        model = AutoModelForImageTextToText.from_pretrained(
                            str(MODEL_DIR),
                            device_map="auto",
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            load_in_4bit=True,
                        )
                        load_in_4bit = True
                        print("使用4位量化模式加载模型成功")
                    except Exception as quant_error:
                        print(f"4位量化加载失败: {str(quant_error)}")
                        load_in_4bit = False
                
                # 如果4位量化不可用或失败，则使用基本优化设置
                if not load_in_4bit:
                    print("尝试使用基本优化设置加载模型...")
                    model = AutoModelForImageTextToText.from_pretrained(
                        str(MODEL_DIR),
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                
                processor = AutoProcessor.from_pretrained(
                    str(MODEL_DIR),
                    trust_remote_code=True
                )
                return True
            except Exception as e2:
                print(f"使用优化设置加载模型也失败了: {str(e2)}")
        elif "qwen3_vl" in error_message and "model type" in error_message:
            print(f"模型加载失败: {error_message}")
            print("请尝试更新Transformers库:")
            print("  pip install --upgrade transformers")
            print("或者从源码安装最新版本:")
            print("  pip install git+https://github.com/huggingface/transformers.git")
        else:
            print(f"模型加载失败: {error_message}")
        return False

def process_with_model(image_path=None, prompt="", **kwargs):
    """
    使用 Qwen3-VL-4B-Instruct 模型处理输入
    
    Args:
        image_path (str): 图像文件路径（可选）
        prompt (str): 输入提示词
        **kwargs: 其他参数
    
    Returns:
        str: 模型处理结果
    """
    try:
        # 确保模型已加载
        if not load_model():
            return "错误：模型加载失败，请检查依赖库版本"
        
        # 根据是否有图像路径决定处理方式
        if image_path and os.path.exists(image_path):
            # 处理图像和文本输入（视觉问答）
            result = _process_vision_task(image_path, prompt)
        else:
            # 仅处理文本输入（语言模型）
            result = _process_text_task(prompt)
            
        return result
    except Exception as e:
        return f"使用 Qwen3-VL-4B-Instruct 模型处理时出现错误: {str(e)}"

def _process_vision_task(image_path, prompt):
    """
    处理视觉任务（图像识别和问答）
    
    Args:
        image_path (str): 图像文件路径
        prompt (str): 用户提问
        
    Returns:
        str: 模型的响应
    """
    try:
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            return f"错误：图像文件不存在: {image_path}"
        
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"错误：无法加载图像文件: {str(e)}"
        
        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # 传递实际的图像对象而不是路径
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 准备推理输入
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理图像和文本
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        
        # 将输入移动到模型所在的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成响应 - 修复参数传递方式
        output = model.generate(
            **inputs,
            max_new_tokens=512,  # 减少max_new_tokens以降低显存需求
            use_cache=True,  # 启用KV缓存
        )
        
        # 解码输出，跳过输入部分
        response = processor.decode(output[0], skip_special_tokens=True)
        response = response[len(text):]  # 移除输入提示部分，只保留模型生成的部分
        
        return response
    except Exception as e:
        return f"处理视觉任务时出现错误: {str(e)}"

def _process_text_task(prompt):
    """
    处理纯文本任务（语言模型）
    
    Args:
        prompt (str): 用户输入
        
    Returns:
        str: 模型的响应
    """
    try:
        # 构建消息格式（仅文本）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 准备推理输入
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理文本
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        
        # 将输入移动到模型所在的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成响应 - 使用input_ids和attention_mask，并添加显存优化参数
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,  # 启用KV缓存以减少重复计算
        )
        
        # 解码输出，跳过输入部分
        response = processor.decode(output[0], skip_special_tokens=True)
        response = response[len(text):]  # 移除输入提示部分，只保留模型生成的部分
        
        return response
    except Exception as e:
        return f"处理文本任务时出现错误: {str(e)}"

def register_model():
    """
    注册模型到系统中
    """
    model_info = get_model_info()
    print(f"已注册模型: {model_info['model_name']}")
    return model_info

# 模型配置信息
MODEL_CONFIG = {
    "name": "Qwen3-VL-4B-Instruct",
    "version": "4B",
    "type": "vision-language",
    "max_context_length": 256000,  # 256K tokens
    "supports": [
        "image_understanding",
        "visual_question_answering", 
        "ocr",
        "visual_reasoning",
        "gui_understanding"
    ]
}

if __name__ == "__main__":
    # 测试模型注册
    register_model()