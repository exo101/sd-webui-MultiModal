#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen-Image LoRA编辑功能演示脚本
展示如何使用LoRA编辑功能调整已加载的LoRA参数
"""

import sys
import os
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def create_lora_edit_args():
    """
    创建用于LoRA编辑的参数文件示例
    """
    args = {
        # 基本参数
        "prompt": "A beautiful landscape with mountains and lake",
        "negative_prompt": "",
        "width": 768,
        "height": 1024,
        "steps": 20,
        "cfg_scale": 4,
        "model_file": "svdq-fp4_r128-qwen-image-lightningv1.1-8steps.safetensors",
        "model_dir": str(Path(__file__).parent.parent.parent.parent / "models" / "qwen-image" / "qwenimage"),
        "scheduler": "euler",
        
        # ControlNet参数
        "controlnet_enable": False,
        "controlnet_model": "Qwen-Image-ControlNet-Union",
        "control_image": None,
        "control_mask": None,
        "controlnet_conditioning_scale": 1,
        "controlnet_preprocessor": "None",
        "controlnet_start": 0,
        "controlnet_end": 1,
        
        # 输出参数
        "output_dir": str(Path(__file__).parent.parent / "outputs"),
        "seed": -1,
        "batch_size": 1,
        
        # LoRA参数
        "lora_enable": True,
        "lora_model_1": "example_lora.safetensors",
        "lora_strength_1": 1.0,
        "lora_model_2": "",
        "lora_strength_2": 1.0,
        
        # LoRA编辑参数
        "lora_edit_enable": True,
        "lora_edit_operation": "strength",  # 可选: strength, reset, info
        "lora_edit_strength": 1.5  # 当operation为strength时使用
    }
    
    # 保存参数到文件
    args_file = Path(__file__).parent / "lora_edit_args.json"
    with open(args_file, 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=2)
    
    print(f"LoRA编辑参数示例已保存到: {args_file}")
    print("参数说明:")
    print("  lora_edit_operation:")
    print("    'strength' - 调整LoRA强度")
    print("    'reset'    - 重置LoRA参数")
    print("    'info'     - 显示LoRA信息")
    print("  lora_edit_strength: 当operation为strength时的新LoRA强度值")


def main():
    print("Qwen-Image LoRA编辑功能演示")
    print("=" * 50)
    
    create_lora_edit_args()
    
    print("\n使用方法:")
    print("1. 确保已加载LoRA的Qwen-Image模型")
    print("2. 运行主脚本时添加lora_edit相关参数")
    print("3. 脚本会自动执行相应的LoRA编辑操作")
    
    print("\n支持的编辑操作:")
    print("- 调整LoRA强度: 在不重新加载模型的情况下调整LoRA的影响程度")
    print("- 重置LoRA: 移除所有已加载的LoRA权重，恢复模型到原始状态")
    print("- 查看LoRA信息: 显示当前加载的LoRA信息")


if __name__ == "__main__":
    main()