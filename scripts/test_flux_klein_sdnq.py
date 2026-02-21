#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLUX.2-klein SDNQ 4bit模型测试脚本
用于验证SDNQ 4bit动态SVD量化模型的加载和推理功能
"""

import torch
import os
import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_sdnq_imports():
    """测试SDNQ相关库的导入"""
    print("=" * 50)
    print("测试SDNQ库导入...")
    print("=" * 50)
    
    try:
        import diffusers
        print("✓ diffusers库导入成功")
    except ImportError as e:
        print(f"✗ diffusers库导入失败: {e}")
        return False
    
    try:
        from sdnq import SDNQConfig
        print("✓ sdnq.SDNQConfig导入成功")
    except ImportError as e:
        print(f"✗ sdnq.SDNQConfig导入失败: {e}")
        return False
    
    try:
        from sdnq.common import use_torch_compile as triton_is_available
        print("✓ sdnq.common.use_torch_compile导入成功")
    except ImportError as e:
        print(f"✗ sdnq.common.use_torch_compile导入失败: {e}")
        return False
    
    try:
        from sdnq.loader import apply_sdnq_options_to_model
        print("✓ sdnq.loader.apply_sdnq_options_to_model导入成功")
    except ImportError as e:
        print(f"✗ sdnq.loader.apply_sdnq_options_to_model导入失败: {e}")
        return False
    
    print("\n✓ 所有SDNQ相关库导入成功!")
    return True

def test_sdnq_model_loading():
    """测试SDNQ模型加载"""
    print("\n" + "=" * 50)
    print("测试SDNQ模型加载...")
    print("=" * 50)
    
    try:
        import diffusers
        from sdnq import SDNQConfig
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model
        
        print("开始加载FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32模型...")
        start_time = time.time()
        
        # 加载模型管道
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", 
            torch_dtype=torch.bfloat16
        )
        
        load_time = time.time() - start_time
        print(f"✓ 模型加载成功! 耗时: {load_time:.2f}秒")
        
        # 应用SDNQ优化选项
        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            print("应用SDNQ量化矩阵乘法优化...")
            start_time = time.time()
            
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            
            optimize_time = time.time() - start_time
            print(f"✓ SDNQ优化应用成功! 耗时: {optimize_time:.2f}秒")
        
        # 启用模型CPU卸载
        print("启用模型CPU卸载...")
        pipe.enable_model_cpu_offload()
        print("✓ 模型CPU卸载启用成功!")
        
        return pipe
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_sdnq_inference(pipe):
    """测试SDNQ模型推理"""
    if pipe is None:
        print("✗ 无法进行推理测试：模型未加载")
        return False
    
    print("\n" + "=" * 50)
    print("测试SDNQ模型推理...")
    print("=" * 50)
    
    try:
        prompt = "A cat holding a sign that says hello world"
        print(f"生成提示词: {prompt}")
        
        # 设置推理参数
        height = 1024
        width = 1024
        guidance_scale = 1.0
        num_inference_steps = 4
        seed = 0
        
        print(f"推理参数: {height}x{width}, CFG={guidance_scale}, Steps={num_inference_steps}, Seed={seed}")
        
        # 生成图像
        start_time = time.time()
        generator = torch.manual_seed(seed)
        
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        
        inference_time = time.time() - start_time
        print(f"✓ 推理完成! 耗时: {inference_time:.2f}秒")
        
        # 保存生成的图像
        if result.images and len(result.images) > 0:
            output_dir = Path("outputs") / "sdnq_test"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = output_dir / "flux-klein-sdnq-4bit-dynamic-svd-r32.png"
            result.images[0].save(str(image_path))
            print(f"✓ 图像已保存到: {image_path}")
            
            # 显示图像信息
            img = result.images[0]
            print(f"生成图像尺寸: {img.size}")
            print(f"图像模式: {img.mode}")
            
            return True
        else:
            print("✗ 未生成任何图像")
            return False
            
    except Exception as e:
        print(f"✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_sdnq_model():
    """测试本地SDNQ模型加载"""
    print("\n" + "=" * 50)
    print("测试本地SDNQ模型加载...")
    print("=" * 50)
    
    # 检查本地模型目录
    local_model_path = Path("models") / "FLUX.2-klein" / "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
    
    if not local_model_path.exists():
        print(f"✗ 本地SDNQ模型目录不存在: {local_model_path}")
        return None
    
    print(f"找到本地SDNQ模型目录: {local_model_path}")
    
    try:
        import diffusers
        from sdnq import SDNQConfig
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model
        
        print("开始加载本地SDNQ模型...")
        start_time = time.time()
        
        # 加载本地模型管道
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            str(local_model_path), 
            torch_dtype=torch.bfloat16
        )
        
        load_time = time.time() - start_time
        print(f"✓ 本地模型加载成功! 耗时: {load_time:.2f}秒")
        
        # 应用SDNQ优化选项
        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            print("应用SDNQ量化矩阵乘法优化...")
            start_time = time.time()
            
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            
            optimize_time = time.time() - start_time
            print(f"✓ SDNQ优化应用成功! 耗时: {optimize_time:.2f}秒")
        
        # 启用模型CPU卸载
        print("启用模型CPU卸载...")
        pipe.enable_model_cpu_offload()
        print("✓ 模型CPU卸载启用成功!")
        
        return pipe
        
    except Exception as e:
        print(f"✗ 本地模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("FLUX.2-klein SDNQ 4bit模型测试")
    print("=" * 60)
    
    # 测试1: 检查依赖库
    if not test_sdnq_imports():
        print("\n✗ 依赖库检查失败，请确保已安装所有必要库")
        return
    
    # 测试2: 在线模型加载和推理
    print("\n" + "=" * 60)
    print("测试在线SDNQ模型 (从HuggingFace下载)...")
    print("=" * 60)
    
    online_pipe = test_sdnq_model_loading()
    if online_pipe:
        test_sdnq_inference(online_pipe)
    else:
        print("✗ 在线模型测试失败")
    
    # 测试3: 本地模型加载和推理
    print("\n" + "=" * 60)
    print("测试本地SDNQ模型...")
    print("=" * 60)
    
    local_pipe = test_local_sdnq_model()
    if local_pipe:
        # 使用不同的提示词进行本地测试
        if hasattr(local_pipe, 'enable_model_cpu_offload'):
            local_pipe.enable_model_cpu_offload()
        
        print("开始本地模型推理测试...")
        try:
            prompt = "A beautiful landscape with mountains and sunset"
            print(f"生成提示词: {prompt}")
            
            start_time = time.time()
            generator = torch.manual_seed(42)
            
            result = local_pipe(
                prompt=prompt,
                height=768,
                width=768,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=generator
            )
            
            inference_time = time.time() - start_time
            print(f"✓ 本地模型推理完成! 耗时: {inference_time:.2f}秒")
            
            if result.images and len(result.images) > 0:
                output_dir = Path("outputs") / "sdnq_test"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                image_path = output_dir / "flux-klein-sdnq-local-test.png"
                result.images[0].save(str(image_path))
                print(f"✓ 本地测试图像已保存到: {image_path}")
            
        except Exception as e:
            print(f"✗ 本地模型推理失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()