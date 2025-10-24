"""
DeepSeek-OCR 模型处理模块
专门用于处理 DeepSeek-OCR 模型的调用和处理逻辑
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
    from transformers import AutoModel, AutoTokenizer
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
MODEL_DIR = Path("models/DeepSeek-OCR")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 添加一个标志来控制是否已经显示过跳过下载的消息
download_message_shown = False

# 全局模型和tokenizer变量
model = None
tokenizer = None

def get_model_info():
    """
    获取 DeepSeek-OCR 模型的信息
    """
    return {
        "model_name": "DeepSeek-OCR",
        "model_id": "deepseek-ai/DeepSeek-OCR",
        "description": "DeepSeek推出的OCR模型，具备强大的文档理解和文本识别能力",
        "capabilities": [
            "光学字符识别",
            "文档理解", 
            "表格识别",
            "手写体识别",
            "多语言支持"
        ]
    }

def is_model_downloaded():
    """
    检查模型是否已经下载
    """
    # 检查模型目录是否存在基本配置文件
    required_files = [
        "config.json",
        "tokenizer_config.json"
    ]
    
    # 首先检查必需的配置文件
    for file_name in required_files:
        file_path = MODEL_DIR / file_name
        if not file_path.exists():
            return False
    
    # 检查是否存在模型文件（可能有不同的命名方式）
    model_files_patterns = [
        "model.safetensors",
        "model-00001-of-00001.safetensors",  # DeepSeek-OCR的实际文件名
        "pytorch_model.bin"
    ]
    
    # 检查是否存在任意一种模型文件
    model_file_exists = False
    for pattern in model_files_patterns:
        file_path = MODEL_DIR / pattern
        if file_path.exists():
            model_file_exists = True
            break
    
    # 如果没有找到模型文件，也检查是否有索引文件（表示分片模型）
    if not model_file_exists:
        index_file = MODEL_DIR / "model.safetensors.index.json"
        if index_file.exists():
            model_file_exists = True
    
    return model_file_exists

def download_model():
    """
    下载 DeepSeek-OCR 模型
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
        print("开始下载 DeepSeek-OCR 模型...")
        # 重置标志，以便下次重新检查时可以显示消息
        download_message_shown = False
        
        # 使用 ModelScope 下载模型
        model_dir = snapshot_download(
            "deepseek-ai/DeepSeek-OCR",
            local_dir=str(MODEL_DIR)
        )
        
        print("模型下载完成")
        return True
    except Exception as e:
        print(f"模型下载失败: {str(e)}")
        return False

def is_model_available():
    """
    检查 DeepSeek-OCR 模型是否可用
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
    加载 DeepSeek-OCR 模型和tokenizer
    """
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return True
    
    if not is_model_available():
        print("错误：缺少必要的依赖项（transformers、modelscope 或 torch），请安装后再试。")
        return False
    
    try:
        # 确保模型已下载
        if not download_model():
            return False
            
        print("正在加载 DeepSeek-OCR 模型...")
        # 加载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True
        )
        
        # 修复加载问题：不使用flash_attention_2，使用默认注意力实现
        model = AutoModel.from_pretrained(
            str(MODEL_DIR),
            device_map="auto",  # 自动选择设备
            torch_dtype=torch.bfloat16,  # 使用bfloat16精度
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 减少CPU内存使用
            # 移除 _attn_implementation 参数以避免导入错误
        )
        model = model.eval()
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
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
            
            try:
                # 检查是否可以使用bitsandbytes进行4位量化
                load_in_4bit = False
                if BNB_AVAILABLE:
                    try:
                        # 尝试使用4位量化
                        model = AutoModel.from_pretrained(
                            str(MODEL_DIR),
                            device_map="auto",
                            torch_dtype=torch.bfloat16,
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
                    model = AutoModel.from_pretrained(
                        str(MODEL_DIR),
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    str(MODEL_DIR),
                    trust_remote_code=True
                )
                model = model.eval()
                return True
            except Exception as e2:
                print(f"使用优化设置加载模型也失败了: {str(e2)}")
        else:
            print(f"模型加载失败: {error_message}")
        return False

def process_with_model(image_path=None, prompt="", **kwargs):
    """
    使用 DeepSeek-OCR 模型处理输入
    
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
            # 处理图像和文本输入（OCR任务）
            result = _process_ocr_task(image_path, prompt)
        else:
            # 仅处理文本输入（语言模型）
            result = _process_text_task(prompt)
            
        return result
    except Exception as e:
        return f"使用 DeepSeek-OCR 模型处理时出现错误: {str(e)}"

def _process_ocr_task(image_path, prompt):
    """
    处理OCR任务（图像识别和文本提取）
    
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
        
        # 如果没有提供提示词，使用默认提示词
        if not prompt:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        # 构建消息内容
        content = f"{prompt}"
        
        # 使用模型的infer方法进行OCR处理
        try:
            # 使用模型的infer方法
            result = model.infer(
                tokenizer=tokenizer,
                prompt=content,
                image_file=image_path,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True
            )
            return str(result)
        except Exception as infer_error:
            # 如果infer方法不可用，尝试使用生成方法
            print(f"infer方法调用失败: {str(infer_error)}，尝试使用generate方法")
            
            # 准备输入
            inputs = tokenizer(
                content, 
                image=image, 
                return_tensors="pt"
            ).to(model.device)
            
            # 生成响应
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
            
            # 解码输出
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            return response
    except Exception as e:
        return f"处理OCR任务时出现错误: {str(e)}"

def _process_text_task(prompt):
    """
    处理纯文本任务（语言模型）
    
    Args:
        prompt (str): 用户输入
        
    Returns:
        str: 模型的响应
    """
    try:
        if not prompt:
            return "请提供输入文本"
        
        # 对于纯文本任务，使用模型的文本生成能力
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成响应
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
        
        # 解码输出
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # 移除输入部分，只返回生成的部分
        response = response[len(prompt):].strip()
        
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
    "name": "DeepSeek-OCR",
    "version": "1.0",
    "type": "vision-language",
    "max_context_length": 32768,  # 32K tokens
    "supports": [
        "ocr",
        "document_understanding",
        "text_extraction",
        "table_recognition"
    ]
}

if __name__ == "__main__":
    # 测试模型注册
    register_model()