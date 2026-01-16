import gradio as gr
import json
import os
import base64
import mimetypes
from modules import shared
import time
import urllib.request
import subprocess
import platform
from typing import List, Dict, Any

# 尝试导入dashscope库
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("DashScope库未安装，请运行 pip install dashscope 安装")

# 设置API基础URL（中国站）
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


def set_api_key(api_key: str) -> str:
    """
    设置API Key到环境变量
    """
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
        return "API Key已设置成功！"
    else:
        return "请输入有效的API Key"


def encode_file(file_path: str) -> str:
    """
    将图像文件编码为Base64格式，用于API调用
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")


def call_qwen_api(model: str, messages: List[Dict[str, Any]], 
                  negative_prompt: str = "", 
                  prompt_extend: bool = True, 
                  watermark: bool = False,
                  size: str = "1024*1024",
                  n: int = 1) -> Dict[str, Any]:
    """
    调用Qwen API生成图像
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return {"error": "未找到DASHSCOPE_API_KEY环境变量，请先设置API密钥"}

    try:
        # 准备参数
        params = {
            'api_key': api_key,
            'model': model,
            'messages': messages,
            'result_format': 'message',
            'stream': False,
            'watermark': watermark,
            'prompt_extend': prompt_extend,
            'negative_prompt': negative_prompt,
            'n': n
        }
        
        # 仅当输出图像数量为1时才添加size参数
        if n == 1:
            size_parts = size.split("*")
            if len(size_parts) == 2:
                try:
                    width, height = int(size_parts[0]), int(size_parts[1])
                    # 确保尺寸在有效范围内
                    if 64 <= width <= 2048 and 64 <= height <= 2048:
                        params['size'] = size
                except ValueError:
                    # 如果解析失败，不添加size参数
                    pass
        
        print(f"调用Qwen API，模型: {model}, 参数: {params}")
        response = MultiModalConversation.call(**params)
        
        print(f"API响应状态码: {response.status_code}")
        print(f"API响应内容: {response}")
        
        # 检查响应状态
        if response.status_code == 200:
            # 由于dashscope响应对象不直接暴露__dict__，我们需要逐个访问属性
            response_dict = {
                'status_code': response.status_code,
                'request_id': getattr(response, 'request_id', ''),
                'code': getattr(response, 'code', ''),
                'message': getattr(response, 'message', ''),
                'output': getattr(response, 'output', {}),
                'usage': getattr(response, 'usage', {})
            }
            
            print(f"解析后的响应数据: {response_dict}")
            return response_dict
        else:
            error_msg = f"API调用失败: {getattr(response, 'message', 'Unknown error')}"
            print(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"API调用异常: {str(e)}"
        print(error_msg)
        return {"error": error_msg}


def generate_with_qwen_image_max(prompt: str, negative_prompt: str, size: str, seed: int) -> List[str]:
    """
    使用qwen-image-max模型生成图像（文生图）
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]
    
    size_parts = size.split("*")
    if len(size_parts) != 2:
        size = "1024*1024"  # 默认尺寸
    else:
        # 验证尺寸是否有效
        try:
            width, height = int(size_parts[0]), int(size_parts[1])
            # 确保尺寸在有效范围内
            if width < 64 or width > 2048 or height < 64 or height > 2048:
                size = "1024*1024"
        except ValueError:
            size = "1024*1024"
    
    result = call_qwen_api(
        model="qwen-image-max",
        messages=messages,
        negative_prompt=negative_prompt,
        size=size
    )
    
    if "error" in result:
        raise gr.Error(result["error"])
    
    # 提取图像URL并下载图像
    image_urls = []
    print(f"API响应数据结构: {result.keys() if isinstance(result, dict) else type(result)}")
    
    if "output" in result and "choices" in result["output"]:
        print(f"找到choices: {result['output']['choices']}")
        for choice in result["output"]["choices"]:
            if "message" in choice and "content" in choice["message"]:
                print(f"处理content: {choice['message']['content']}")
                for content_item in choice["message"]["content"]:
                    if "image" in content_item:
                        image_url = content_item["image"]
                        print(f"找到图像URL: {image_url}")
                        image_urls.append(image_url)
    else:
        print("未找到期望的响应结构，输出整个响应:")
        print(result)
    
    # 检查是否找到图像URL
    if not image_urls:
        print("警告: 未找到任何图像URL，请检查API响应格式")
        print(f"完整响应: {result}")
        raise gr.Error("API未返回图像，请检查提示词和参数设置")
    
    # 下载图像并保存到本地
    local_image_paths = []
    save_dir = os.path.join(shared.data_path, "outputs", "qwen-api")
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, url in enumerate(image_urls):
        try:
            timestamp = int(time.time())
            filename = f"qwen_image_max_{timestamp}_{idx}.png"
            save_path = os.path.join(save_dir, filename)
            
            urllib.request.urlretrieve(url, save_path)
            local_image_paths.append(save_path)
            print(f"已下载图像: {save_path}")
        except Exception as e:
            print(f"下载图像失败 {url}: {e}")
            raise gr.Error(f"下载图像失败: {e}")
    
    return local_image_paths


def generate_with_qwen_image_edit_plus(image1, image2, image3, prompt: str, negative_prompt: str, n: int) -> List[str]:
    """
    使用qwen-image-edit-plus模型编辑图像（支持多图融合）
    """
    # 收集所有上传的图像
    images = []
    if image1 is not None:
        images.append(image1)
    if image2 is not None:
        images.append(image2)
    if image3 is not None:
        images.append(image3)
    
    if not images:
        raise gr.Error("请至少上传一张图像")
    
    # 准备消息内容
    messages_content = []
    
    # 添加所有图像到消息内容
    for img_path in images:
        if isinstance(img_path, dict) and 'name' in img_path:
            # 如果是Gradio返回的字典格式
            img_path = img_path['name']
        encoded_image = encode_file(img_path)
        messages_content.append({"image": encoded_image})
    
    # 添加文本描述
    messages_content.append({"text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": messages_content
        }
    ]
    
    result = call_qwen_api(
        model="qwen-image-edit-plus",
        messages=messages,
        negative_prompt=negative_prompt,
        n=n
    )
    
    if "error" in result:
        raise gr.Error(result["error"])
    
    # 提取图像URL并下载图像
    image_urls = []
    print(f"API响应数据结构: {result.keys() if isinstance(result, dict) else type(result)}")
    
    if "output" in result and "choices" in result["output"]:
        print(f"找到choices: {result['output']['choices']}")
        for choice in result["output"]["choices"]:
            if "message" in choice and "content" in choice["message"]:
                print(f"处理content: {choice['message']['content']}")
                for content_item in choice["message"]["content"]:
                    if "image" in content_item:
                        image_url = content_item["image"]
                        print(f"找到图像URL: {image_url}")
                        image_urls.append(image_url)
    else:
        print("未找到期望的响应结构，输出整个响应:")
        print(result)
    
    # 检查是否找到图像URL
    if not image_urls:
        print("警告: 未找到任何图像URL，请检查API响应格式")
        print(f"完整响应: {result}")
        raise gr.Error("API未返回图像，请检查提示词和参数设置")
    
    # 下载图像并保存到本地
    local_image_paths = []
    save_dir = os.path.join(shared.data_path, "outputs", "qwen-api")
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, url in enumerate(image_urls):
        try:
            timestamp = int(time.time())
            filename = f"qwen_image_edit_plus_{timestamp}_{idx}.png"
            save_path = os.path.join(save_dir, filename)
            
            urllib.request.urlretrieve(url, save_path)
            local_image_paths.append(save_path)
            print(f"已下载图像: {save_path}")
        except Exception as e:
            print(f"下载图像失败 {url}: {e}")
            raise gr.Error(f"下载图像失败: {e}")
    
    return local_image_paths


def open_qwen_output_dir():
    """打开Qwen API输出目录"""
    output_dir = os.path.join(shared.data_path, "outputs", "qwen-api")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.run(["explorer", output_dir])
        elif system == "Darwin":  # macOS
            subprocess.run(["open", output_dir])
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", output_dir])
    except Exception as e:
        print(f"打开目录失败: {e}")


def create_qwen_api_ui():
    """
    创建Qwen API UI界面
    """
    with gr.Blocks() as qwen_api_interface:
        gr.Markdown("# Qwen图像生成与编辑 API")
        gr.Markdown("使用阿里云百炼平台的Qwen模型进行图像生成和编辑")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="请输入您的百炼API Key",
                    info="输入API Key后点击下方按钮设置"
                )
                set_api_key_btn = gr.Button("设置API Key", variant="secondary")
                api_key_status = gr.Textbox(label="状态", interactive=False)
                
                set_api_key_btn.click(
                    fn=set_api_key,
                    inputs=api_key_input,
                    outputs=api_key_status
                )
        
        with gr.Tabs():
            with gr.TabItem("文生图 (qwen-image-max)"):
                with gr.Row():
                    with gr.Column():
                        qwen_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入图像描述...",
                            lines=3,
                            max_lines=5
                        )
                        
                        qwen_negative_prompt = gr.Textbox(
                            label="反向提示词",
                            placeholder="请输入不希望出现的内容...",
                            lines=2,
                            max_lines=3
                        )
                        
                        with gr.Row():
                            qwen_size = gr.Dropdown(
                                label="图像尺寸",
                                choices=["1024*1024", "768*1024", "1024*768", "1280*720", "720*1280", "1664*928", "928*1664"],
                                value="1024*1024"
                            )
                            
                            qwen_seed = gr.Number(
                                label="随机种子",
                                value=0,
                                precision=0
                            )
                        
                        qwen_gen_btn = gr.Button("生成图像", variant="primary")
                    
                    with gr.Column():
                        qwen_gen_result = gr.Gallery(
                            label="生成结果",
                            show_label=True,
                            elem_id="qwen_gallery",
                            columns=2,
                            object_fit="contain",
                            height="auto"
                        )
                
                qwen_gen_btn.click(
                    fn=generate_with_qwen_image_max,
                    inputs=[qwen_prompt, qwen_negative_prompt, qwen_size, qwen_seed],
                    outputs=qwen_gen_result
                )
            
            with gr.TabItem("图像编辑 (qwen-image-edit-plus)"):
                with gr.Row():
                    with gr.Column():
                        # 并排显示三个图像上传组件
                        with gr.Row():
                            qwen_edit_image1 = gr.Image(
                                label="参考图像 1",
                                type="filepath",
                                height=200
                            )
                            qwen_edit_image2 = gr.Image(
                                label="参考图像 2",
                                type="filepath",
                                height=200
                            )
                            qwen_edit_image3 = gr.Image(
                                label="参考图像 3",
                                type="filepath",
                                height=200
                            )
                        
                        qwen_edit_prompt = gr.Textbox(
                            label="编辑提示词",
                            placeholder="请输入编辑描述（例如：图1中的女生穿着图2中的黑色裙子按图3的姿势坐下）",
                            lines=2,
                            max_lines=3
                        )
                        
                        qwen_edit_negative_prompt = gr.Textbox(
                            label="反向提示词",
                            placeholder="请输入不希望出现的内容...",
                            lines=2,
                            max_lines=3
                        )
                        
                        qwen_edit_count = gr.Slider(
                            label="生成数量",
                            minimum=1,
                            maximum=6,
                            step=1,
                            value=1
                        )
                        
                        qwen_edit_btn = gr.Button("编辑图像", variant="primary")
                    
                    with gr.Column():
                        qwen_edit_result = gr.Gallery(
                            label="编辑结果",
                            show_label=True,
                            elem_id="qwen_edit_gallery",
                            columns=2,
                            object_fit="contain",
                            height="auto"
                        )
                
                qwen_edit_btn.click(
                    fn=generate_with_qwen_image_edit_plus,
                    inputs=[qwen_edit_image1, qwen_edit_image2, qwen_edit_image3, qwen_edit_prompt, qwen_edit_negative_prompt, qwen_edit_count],
                    outputs=qwen_edit_result
                )
        
        # 添加打开输出目录按钮
        open_output_dir_btn = gr.Button("打开输出目录", variant="secondary")
        open_output_dir_btn.click(
            fn=open_qwen_output_dir,
            inputs=[],
            outputs=[]
        )

    return qwen_api_interface

# 定义模块可用性标志
QWEN_API_AVAILABLE = DASHSCOPE_AVAILABLE