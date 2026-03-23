"""
简化SDNQ模型支持测试脚本
验证基本功能是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

def test_basic_imports():
    """测试基本导入功能"""
    print("=== 测试基本导入 ===")
    
    try:
        # 测试模型加载器导入
        import scripts.flux_klein_model_loader as model_loader
        print("✓ 模型加载器导入成功")
        
        # 测试SDNQ检测功能
        is_sdnq_func = getattr(model_loader, '_is_sdnq_model', None)
        if is_sdnq_func:
            test_result = is_sdnq_func("FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32")
            print(f"✓ SDNQ检测功能: {test_result}")
        else:
            print("✗ 未找到_is_sdnq_model函数")
            
        # 测试模型列表获取
        sdnq_models_func = getattr(model_loader, 'get_sdnq_models', None)
        if sdnq_models_func:
            try:
                sdnq_models = sdnq_models_func()
                print(f"✓ SDNQ模型列表获取成功: {len(sdnq_models)} 个模型")
                for model in sdnq_models:
                    print(f"  - {model}")
            except Exception as e:
                print(f"✗ 获取SDNQ模型列表失败: {e}")
        else:
            print("✗ 未找到get_sdnq_models函数")
            
        return True
        
    except Exception as e:
        print(f"✗ 导入测试失败: {e}")
        return False

def test_model_path_resolution():
    """测试模型路径解析功能"""
    print("\n=== 测试模型路径解析 ===")
    
    try:
        import scripts.flux_klein_model_loader as model_loader
        
        # 测试路径解析函数
        get_path_func = getattr(model_loader, 'get_full_model_path', None)
        if get_path_func:
            test_model = "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
            resolved_path = get_path_func(test_model)
            print(f"✓ 模型路径解析:")
            print(f"  输入: {test_model}")
            print(f"  输出: {resolved_path}")
            
            # 检查路径是否存在
            if os.path.exists(resolved_path):
                print("  ✓ 路径存在")
            else:
                print("  ⚠ 路径不存在（可能是正常的，如果模型未安装）")
        else:
            print("✗ 未找到get_full_model_path函数")
            
        return True
        
    except Exception as e:
        print(f"✗ 路径解析测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始SDNQ模型支持简化测试")
    print("=" * 40)
    
    tests = [
        ("基本导入功能", test_basic_imports),
        ("模型路径解析", test_model_path_resolution)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 40)
    print("测试总结:")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 基本功能测试通过！")
    else:
        print("⚠️  部分功能需要修复。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
