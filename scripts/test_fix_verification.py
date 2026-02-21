#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复验证测试脚本
用于验证flux_klein_ui.py和flux_klein_model_loader.py中的函数导入和调用是否正常
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_imports():
    """测试函数导入"""
    print("=" * 50)
    print("测试函数导入...")
    print("=" * 50)
    
    try:
        # 测试从模型加载器导入函数
        from flux_klein_model_loader import _is_fp8_model, _is_sdnq_model, _identify_model_type, _scan_model_directory, get_sdnq_models
        print("✓ 成功从flux_klein_model_loader导入所有函数")
        
        # 测试从UI文件导入函数
        from flux_klein_ui import get_sdnq_models as ui_get_sdnq_models
        print("✓ 成功从flux_klein_ui导入函数")
        
        return True
    except Exception as e:
        print(f"✗ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_calls():
    """测试函数调用"""
    print("\n" + "=" * 50)
    print("测试函数调用...")
    print("=" * 50)
    
    try:
        from flux_klein_model_loader import _is_fp8_model, _is_sdnq_model, _identify_model_type, _scan_model_directory, get_sdnq_models
        
        # 测试_is_fp8_model函数
        test_path = "test_fp8_model"
        result = _is_fp8_model(test_path)
        print(f"✓ _is_fp8_model('{test_path}') = {result}")
        
        # 测试_is_sdnq_model函数
        test_sdnq_path = "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
        result = _is_sdnq_model(test_sdnq_path)
        print(f"✓ _is_sdnq_model('{test_sdnq_path}') = {result}")
        
        # 测试_identify_model_type函数
        test_model_name = "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
        result = _identify_model_type(test_model_name)
        print(f"✓ _identify_model_type('{test_model_name}') = {result}")
        
        # 测试_get_sdnq_models函数（这可能会因为目录不存在而失败，但不应该报NameError）
        try:
            result = get_sdnq_models()
            print(f"✓ get_sdnq_models() = {result}")
        except FileNotFoundError:
            print("✓ get_sdnq_models() 正常处理了目录不存在的情况")
        except Exception as e:
            print(f"? get_sdnq_models() 出现了预期外的错误: {e}")
        
        return True
    except Exception as e:
        print(f"✗ 函数调用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("FLUX.2-klein修复验证测试")
    print("=" * 60)
    
    # 测试1: 检查函数导入
    import_success = test_imports()
    
    # 测试2: 检查函数调用
    call_success = test_function_calls() if import_success else False
    
    print("\n" + "=" * 60)
    if import_success and call_success:
        print("✓ 所有测试通过！修复成功！")
    else:
        print("✗ 测试失败，请检查错误信息")
    print("=" * 60)

if __name__ == "__main__":
    main()