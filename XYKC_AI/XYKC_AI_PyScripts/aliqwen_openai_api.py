# coding=utf-8

import sys
from openai import OpenAI
import mimetypes
import base64

def image_path_to_base64(image_file):
    img_str = base64.b64encode(image_file.read())
    return img_str.decode('utf-8')

def get_response_lvm_openai_api(
        input_api_key, 
        input_model_name, 
        input_content, 
        input_image_path,
        input_temperature,
        input_top_p,
        input_presence_penalty):
    client = OpenAI(api_key=input_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    mime_type, _ = mimetypes.guess_type(input_image_path)

    if mime_type and mime_type.startswith('image'):
        with open(input_image_path, 'rb') as image_file:
            # 拼接前缀和Base64编码的图像数据
            data_uri_prefix = f'data:{mime_type};base64,'
            encoded_image_str = data_uri_prefix + image_path_to_base64(image_file)

            completion = client.chat.completions.create(
                model=input_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这里有一张图"},
                        {"type": "image_url", "image_url": {"url": encoded_image_str}},
                        {"type": "text", "text": input_content}]
                    }],
                temperature=input_temperature,
                top_p=input_top_p,
                presence_penalty=input_presence_penalty
                )
            print(completion.choices[0].message.content)
            return
    print("不支持的文件格式")

def get_response_llm_openai_api(
        input_api_key, 
        input_model_name, 
        input_content, 
        input_file_content, 
        input_temperature,
        input_top_p,
        input_presence_penalty):
    client = OpenAI(api_key=input_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    completion = client.chat.completions.create(
        model=input_model_name,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': [
                {"type": "text", "text": "这里有一段信息"},
                {"type": "text", "text": input_file_content},
                {"type": "text", "text": input_content}]
            }],
        temperature=input_temperature,
        top_p=input_top_p,
        presence_penalty=input_presence_penalty
    )
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    arguments = sys.argv

    model_type = arguments[1]
    model_name = arguments[2]
    user_input = arguments[3]
    file = arguments[4]
    api_key = arguments[5]
    if api_key == "":
        print("API密钥不能为空  请检查")
    else:
        temperature = float(arguments[6])
        top_p = float(arguments[7])
        presence_penalty = float(arguments[8])

        if model_type == "vision":
            get_response_lvm_openai_api(api_key, model_name, user_input, file, temperature, top_p, presence_penalty)
        elif model_type == "language":
            get_response_llm_openai_api(api_key, model_name, user_input, file, temperature, top_p, presence_penalty)