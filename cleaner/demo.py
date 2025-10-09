#!/usr/bin/env python3
"""
LiteLama图像清理演示脚本

此脚本演示了如何使用LiteLama模型进行图像清理（去除不需要的对象）。
"""

import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # 尝试导入必要的库
    from PIL import Image
    import numpy as np
    
    # 尝试导入LiteLama
    try:
        from litelama import LiteLama
        from litelama.model import download_file
        LITELAMA_AVAILABLE = True
        print("✓ 成功导入litelama库")
    except ImportError as e:
        print(f"✗ 导入litelama库失败: {e}")
        print("请安装litelama库: pip install litelama")
        LITELAMA_AVAILABLE = False

    class LiteLamaDemo:
        """
        LiteLama演示类
        """
        
        def __init__(self, model_path=None):
            """
            初始化LiteLama模型
            
            Args:
                model_path (str): 模型文件路径，如果为None则自动下载
            """
            if not LITELAMA_AVAILABLE:
                return
                
            self.model = LiteLama()
            
            if model_path is None:
                # 设置模型路径
                model_dir = os.path.join(current_dir, "models")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                model_path = os.path.join(model_dir, "big-lama.safetensors")
                
                # 如果模型文件不存在，则下载
                if not os.path.exists(model_path) or not os.path.isfile(model_path):
                    print("正在下载模型文件...")
                    try:
                        download_file(
                            "https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors", 
                            model_path
                        )
                        print("模型文件下载完成")
                    except Exception as e:
                        print(f"模型文件下载失败: {e}")
                        return
            
            # 加载模型
            try:
                self.model.load(model_path, location="cpu")
                print("模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {e}")
        
        def create_mask(self, image_size, mask_area):
            """
            创建遮罩图像
            
            Args:
                image_size (tuple): 图像尺寸 (width, height)
                mask_area (tuple): 遮罩区域 (x, y, width, height)
                
            Returns:
                PIL.Image: 遮罩图像
            """
            mask = Image.new("RGB", image_size, (0, 0, 0))  # 黑色背景
            # 在指定区域绘制白色遮罩
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            x, y, w, h = mask_area
            draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 255))  # 白色遮罩区域
            return mask
        
        def clean_image(self, image_path, mask_area=None):
            """
            清理图像
            
            Args:
                image_path (str): 输入图像路径
                mask_area (tuple): 遮罩区域 (x, y, width, height)，如果为None则需要提供遮罩图像
                
            Returns:
                PIL.Image: 清理后的图像
            """
            if not LITELAMA_AVAILABLE:
                return None
                
            try:
                # 加载图像
                image = Image.open(image_path).convert("RGB")
                print(f"已加载图像: {image_path}, 尺寸: {image.size}")
                
                # 创建遮罩
                if mask_area:
                    mask = self.create_mask(image.size, mask_area)
                    print(f"已创建遮罩，区域: {mask_area}")
                else:
                    # 如果没有指定遮罩区域，创建一个居中的遮罩
                    width, height = image.size
                    mask_area = (width//4, height//4, width//2, height//2)  # 居中区域
                    mask = self.create_mask(image.size, mask_area)
                    print(f"已创建默认遮罩，区域: {mask_area}")
                
                # 执行清理
                print("正在执行图像清理...")
                result = self.model.predict(image, mask)
                
                if result is not None:
                    print("图像清理完成")
                    return result
                else:
                    print("图像清理失败")
                    return None
                    
            except Exception as e:
                print(f"处理图像时出错: {e}")
                return None

    def main():
        """
        主函数
        """
        print("LiteLama图像清理演示")
        print("=" * 30)
        
        if not LITELAMA_AVAILABLE:
            print("LiteLama库不可用，无法进行演示")
            return
            
        # 创建演示实例
        demo = LiteLamaDemo()
        
        # 检查命令行参数
        if len(sys.argv) < 2:
            print("使用方法: python demo.py <image_path> [x y width height]")
            print("示例: python demo.py input.jpg 100 100 200 200")
            return
            
        image_path = sys.argv[1]
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return
            
        # 解析遮罩区域参数
        mask_area = None
        if len(sys.argv) >= 6:
            try:
                x = int(sys.argv[2])
                y = int(sys.argv[3])
                width = int(sys.argv[4])
                height = int(sys.argv[5])
                mask_area = (x, y, width, height)
            except ValueError:
                print("遮罩区域参数无效，使用默认区域")
        
        # 执行图像清理
        result = demo.clean_image(image_path, mask_area)
        
        if result is not None:
            # 保存结果
            output_path = "cleaned_output.png"
            result.save(output_path)
            print(f"清理后的图像已保存到: {output_path}")
        else:
            print("图像清理失败")

    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(f"脚本执行出错: {e}")