# coding=utf-8

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(1234)

current_dir = os.path.abspath(os.getcwd())

model_paths = {"qwen-vl-chat": os.path.normpath(os.path.join(current_dir, "extensions\\sd-webui-XYKC\\models\\Qwen_Qwen-VL-Chat")),
               "qwen-vl-chat-int4": os.path.normpath(os.path.join(current_dir, "extensions\\sd-webui-XYKC\\models\\Qwen-VL-Chat-Int4")),
               "qwen-VL-Chat-Finetuned-Dense-Captioner": os.path.normpath(os.path.join(current_dir, "extensions\\sd-webui-XYKC\\models\\Qwen-VL-Chat-Finetuned-Dense-Captioner"))}

def lvm(model_path, image_path, user_input):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()

    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': user_input},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)

def llm(model_path, image_path, user_input):
    print("no transformers api llm")

if __name__ == "__main__":
    arguments = sys.argv

    model_type = arguments[1]
    model_name = arguments[2]
    user_input = arguments[3]
    file = arguments[4]
    model_path = model_paths[model_name]

    if model_type == "vision":
        lvm(model_path, file, user_input)
    elif model_type == "language":
        llm(model_path, file, user_input)