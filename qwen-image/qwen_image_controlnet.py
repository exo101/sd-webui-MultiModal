import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from annotator.util import HWC3

def preprocess_for_qwen_image_controlnet(image_path, preprocessor_type):
    """使用WebUI的预处理器系统处理图像"""
    try:
        if not image_path or not os.path.exists(image_path):
            print(f"预处理图像路径无效: {image_path}")
            return None
        
        # 特殊处理"none"预处理器 - 直接返回原始图像路径
        if preprocessor_type is None or preprocessor_type.lower() in ["none", "无", "none (default)"]:
            print("使用无预处理模式，直接返回原始图像路径")
            return image_path
            
        print(f"开始使用预处理器 {preprocessor_type} 处理图像: {image_path}")
        
        # 添加WebUI根目录到系统路径
        webui_root = Path(__file__).parent.parent.parent.parent
        extensions_builtin = webui_root / "extensions-builtin"
        
        paths_to_add = [
            str(webui_root),
            str(extensions_builtin),
            str(extensions_builtin / "forge_preprocessor_inpaint")
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.append(path)
        
        # 导入WebUI的预处理器管理模块
        from modules_forge.shared import supported_preprocessors
        from modules_forge.initialization import initialize_forge
        
        # 初始化Forge系统
        initialize_forge()
        
        # 手动导入legacy_preprocessors和reference预处理器以确保预处理器被正确加载
        try:
            import forge_legacy_preprocessors.scripts.legacy_preprocessors
        except Exception:
            pass
            
        try:
            import forge_preprocessor_reference.scripts.forge_reference
        except Exception:
            pass
            
        try:
            import sd_forge_ipadapter.scripts.ipadapter
        except Exception:
            pass
        
        # 获取预处理器对象
        preprocessor = supported_preprocessors.get(preprocessor_type)
        if preprocessor is None:
            # 尝试不同的命名变体
            variants = [
                preprocessor_type.lower(),
                preprocessor_type.lower().replace(" ", "_"),
                preprocessor_type.lower().replace("-", "_"),
                preprocessor_type.replace("-", "_"),
                preprocessor_type.replace(" ", "_")
            ]
            
            # 特别处理depth_anything_v2预处理器
            if preprocessor_type == "depth_anything_v2":
                variants.append("depth_anything_v2")
                
            # 特别处理IPAdapter预处理器
            if preprocessor_type == "CLIP-ViT-H (IPAdapter)":
                variants.append("ip-adapter_clip_h")
            
            for variant in variants:
                if variant in supported_preprocessors:
                    preprocessor = supported_preprocessors[variant]
                    preprocessor_type = variant  # 更新预处理器类型
                    break
        
        # 如果还是找不到，直接报错而不是回退到canny
        if preprocessor is None:
            available_preprocessors = list(supported_preprocessors.keys())
            print(f"错误：未找到预处理器 {preprocessor_type}，可用的预处理器: {available_preprocessors}")
            raise ValueError(f"未找到预处理器: {preprocessor_type}，请检查预处理器名称是否正确")
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        # 调用预处理器处理图像，使用标准参数
        try:
            # 使用标准的预处理器调用方式，传递标准参数
            # 对于特定预处理器，需要传递特定参数
            if preprocessor_type in ["depth_leres", "depth_leres++"]:
                # depth_leres 需要 thr_a 和 thr_b 参数
                processed_image = preprocessor(
                    input_image=image_array,
                    resolution=512,
                    slider_1=0,
                    slider_2=0
                )
            elif preprocessor_type == "canny":
                # canny 需要 low threshold 和 high threshold 参数
                processed_image = preprocessor(
                    input_image=image_array,
                    resolution=512,
                    slider_1=100,
                    slider_2=200
                )
            elif preprocessor_type in ["reference_only", "reference_adain", "reference_adain+attn"]:
                # reference预处理器需要特殊的处理方式
                # 对于reference预处理器，我们直接返回原始图像，因为它们在ControlNet中用于特殊处理
                print(f"Reference预处理器 {preprocessor_type} 检测到，直接返回原始图像")
                return image_path
            elif preprocessor_type == "CLIP-ViT-H (IPAdapter)":
                # IPAdapter预处理器需要特殊的处理方式
                # 对于IPAdapter预处理器，我们直接返回原始图像，因为它们在ControlNet中用于特殊处理
                print(f"IPAdapter预处理器 {preprocessor_type} 检测到，直接返回原始图像")
                return image_path
            else:
                # 对于大多数预处理器，使用更安全的调用方式
                # 检查预处理器是否是用 functools.partial 创建的，如果是，则避免参数冲突
                import functools
                if isinstance(preprocessor, functools.partial):
                    # 对于使用 functools.partial 包装的预处理器，使用位置参数而不是关键字参数
                    processed_image = preprocessor(image_array, 512, None, None, None)
                else:
                    # 正常调用预处理器
                    processed_image = preprocessor(
                        input_image=image_array, 
                        resolution=512, 
                        slider_1=None, 
                        slider_2=None, 
                        slider_3=None
                    )
        except Exception as e:
            print(f"预处理器执行出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 确保输出是numpy数组
        if processed_image is not None and hasattr(processed_image, 'size') and processed_image.size > 0:
            # 如果返回的是元组，取第一个元素
            if isinstance(processed_image, tuple):
                processed_image = processed_image[0]
            
            return processed_image
        else:
            print("预处理器返回了空结果")
            return None
            
    except Exception as e:
        print(f"使用WebUI预处理器时出错: {e}")
        import traceback
        traceback.print_exc()
        return None