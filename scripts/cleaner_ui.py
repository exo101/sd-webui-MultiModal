import os
import sys
import gradio as gr

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


# 尝试导入WebUI模块
try:
    from modules.shared import opts
    from modules.ui_components import ToolButton, ResizeHandleRow
    MODULES_AVAILABLE = True
except ImportError:
    # 如果在WebUI环境外运行，创建模拟对象
    class MockOpts:
        def __init__(self):
            self.data = {"cleaner_use_gpu": True}
    
    class MockToolButton:
        def __init__(self, *args, **kwargs):
            pass
    
    class MockResizeHandleRow:
        def __init__(self, *args, **kwargs):
            pass
        
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
    
    opts = MockOpts()
    ToolButton = MockToolButton
    ResizeHandleRow = MockResizeHandleRow
    MODULES_AVAILABLE = False
    print("Warning: Running outside WebUI environment")

# 尝试多种方式导入parameters_copypaste
parameters_copypaste = None
try:
    import modules.generation_parameters_copypaste as parameters_copypaste
except ImportError:
    try:
        from modules import generation_parameters_copypaste as parameters_copypaste
    except ImportError:
        try:
            # 查找正确的导入路径
            import modules
            parameters_copypaste = modules.generation_parameters_copypaste
        except:
            print("Warning: Could not import generation_parameters_copypaste")
            parameters_copypaste = None

# 直接导入依赖库
try:
    from litelama import LiteLama
    from litelama.model import download_file
    CLEANER_AVAILABLE = True
except Exception as e:
    CLEANER_AVAILABLE = False

class LiteLama2(LiteLama):
    _instance = None
    
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
        
    def __init__(self, checkpoint_path=None, config_path=None):
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
            
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._model = None
        
        if self._checkpoint_path is None:
            EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            MODEL_PATH = os.path.join(EXTENSION_PATH, "cleaner", "models")
            checkpoint_path = os.path.join(MODEL_PATH, "big-lama.safetensors")
            if not os.path.exists(checkpoint_path) or not os.path.isfile(checkpoint_path):
                try:
                    download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors", checkpoint_path)
                except Exception as e:
                    raise e
            self._checkpoint_path = checkpoint_path
        
        try:
            self.load(location="cpu")
            self._initialized = True
        except Exception as e:
            raise e

    def to(self, device):
        """重写to方法以添加调试信息"""
        try:
            super().to(device)
        except Exception as e:
            raise e

    def predict(self, image, mask):
        """重写predict方法以添加调试信息"""
        try:
            result = super().predict(image, mask)
            return result
        except Exception as e:
            raise e


def convert_to_white_mask(mask_image):
    """
    将任意颜色的遮罩转换为白色遮罩
    """
    try:
        # 确保输入是PIL图像
        if not hasattr(mask_image, 'convert'):
            return mask_image
            
        # 转换为RGBA模式以处理透明度
        rgba_mask = mask_image.convert('RGBA')
        
        # 创建新的遮罩图像
        from PIL import Image
        white_mask = Image.new('RGB', rgba_mask.size, (0, 0, 0))  # 默认黑色背景
        
        # 获取像素数据
        pixels = rgba_mask.load()
        white_pixels = white_mask.load()
        
        # 遍历所有像素，将非透明和非黑色像素转换为白色
        for x in range(rgba_mask.size[0]):
            for y in range(rgba_mask.size[1]):
                r, g, b, a = pixels[x, y]
                # 如果像素不透明（alpha > 0）且不是黑色，则在遮罩中设为白色
                if a > 0 and (r > 0 or g > 0 or b > 0):  # 非透明且非黑色像素
                    white_pixels[x, y] = (255, 255, 255)  # 白色
        
        return white_mask
    except Exception as e:
        return mask_image  # 出错时返回原始遮罩


def clean_object_init_img_with_mask(init_img_with_mask):
    if not CLEANER_AVAILABLE or init_img_with_mask is None:
        return [None]
    try:
        # 检查数据格式并正确提取图像和遮罩
        if isinstance(init_img_with_mask, dict):
            # 兼容旧格式
            if 'image' in init_img_with_mask and 'mask' in init_img_with_mask:
                return clean_object(init_img_with_mask['image'], init_img_with_mask['mask'])
            # Gradio 4.x Sketchpad格式
            elif 'background' in init_img_with_mask and 'layers' in init_img_with_mask:
                image = init_img_with_mask['background']
                # 合并所有图层作为遮罩
                if init_img_with_mask['layers']:
                    mask = init_img_with_mask['layers'][0]  # 使用第一个图层作为遮罩
                    # 将任意颜色的遮罩转换为白色遮罩
                    if mask is not None:
                        mask = convert_to_white_mask(mask)
                else:
                    mask = None
                    
                if image is not None and mask is not None:
                    return clean_object(image, mask)
                else:
                    pass
        else:
            # 如果不是字典格式，尝试直接处理
            # 这可能是Gradio 4.x直接返回的Sketchpad对象
            if hasattr(init_img_with_mask, 'background') and hasattr(init_img_with_mask, 'layers'):
                image = init_img_with_mask.background
                # 合并所有图层作为遮罩
                if init_img_with_mask.layers:
                    mask = init_img_with_mask.layers[0]  # 使用第一个图层作为遮罩
                    # 将任意颜色的遮罩转换为白色遮罩
                    if mask is not None:
                        mask = convert_to_white_mask(mask)
                else:
                    mask = None
                    
                if image is not None and mask is not None:
                    return clean_object(image, mask)
                else:
                    pass
            else:
                pass
        
        # 如果以上都不匹配，返回空结果
        return [None]
    except Exception as e:
        return [None]

def clean_object(image, mask):
    if not CLEANER_AVAILABLE or image is None or mask is None:
        return [None]
        
    try:
        # 确保我们有PIL图像对象，而不是路径字符串
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)
            
        if isinstance(mask, str):
            from PIL import Image
            mask = Image.open(mask)
        
        # 确保我们有PIL图像对象
        if not hasattr(image, 'convert'):
            return [None]
            
        if not hasattr(mask, 'convert'):
            return [None]
        
        # 转换图像格式
        init_image = image.convert("RGB")
        mask_image = mask.convert("RGB")

        # 获取设备设置
        device_used = opts.data.get("cleaner_use_gpu", True)
        device = "cuda:0" if device_used else "cpu"

        # 创建LiteLama2实例
        Lama = LiteLama2()
        
        result = None
        try:
            Lama.to(device)
            result = Lama.predict(init_image, mask_image)
        except Exception as e:
            pass
        finally:
            try:
                Lama.to("cpu")
            except Exception as e:
                pass
        
        return [result]
    except Exception as e:
        return [None]

def send_to_cleaner(result):
    if not result or len(result) == 0 or result[0] is None:
        return None
    try:
        return result[0]
    except Exception as e:
        return None

# 添加一个新的函数来处理Gallery组件的返回值
def process_gallery_output(result):
    """
    处理返回给Gallery组件的结果，确保格式正确
    """
    if result is None or (isinstance(result, list) and len(result) == 0):
        # 返回空列表而不是包含None的列表
        return []
    elif isinstance(result, list):
        # 过滤掉None值
        filtered_result = [img for img in result if img is not None]
        return filtered_result
    else:
        # 如果是单个图像对象，包装成列表
        output = [result] if result is not None else []
        return output


# 添加一个新的函数来处理ImageEditor组件的返回值
def process_image_editor_output(result):
    """
    处理返回给ImageEditor组件的结果，确保格式正确
    """
    if result is None or (isinstance(result, list) and len(result) == 0):
        # 返回None而不是空列表
        return None
    elif isinstance(result, list) and len(result) > 0:
        # 返回第一个非None元素
        for img in result:
            if img is not None:
                # 如果是元组，提取第一个元素
                if isinstance(img, tuple):
                    if len(img) > 0:
                        extracted_img = img[0]
                        return extracted_img
                    else:
                        continue
                else:
                    return img
        # 如果所有元素都是None
        return None
    else:
        # 如果是单个图像对象
        if result is not None:
            # 如果是元组，提取第一个元素
            if isinstance(result, tuple):
                if len(result) > 0:
                    extracted_img = result[0]
                    return extracted_img
                else:
                    return None
            else:
                return result
        else:
            return None

def create_cleaner_ui():
    """
    创建图像清理UI模块
    """
    if not CLEANER_AVAILABLE:
        with gr.Group():

            gr.Markdown("LiteLama库未安装，功能不可用。请手动安装依赖：\n\n"
                       "1. 关闭WebUI\n"
                       "2. 打开命令行并运行: `pip install litelama`\n"
                       "3. 重新启动WebUI\n\n"
                       "如果仍有问题，请检查Python环境或尝试降级numpy版本:\n"
                       "`pip install numpy==1.24.4`")
        return None

    with gr.Group():
        gr.Markdown("## 图像清理器 (Cleaner)")
        
        with ResizeHandleRow(equal_height=False):
            init_img_with_mask = gr.Sketchpad(
                label="带遮罩的清理图像", 
                show_label=False, 
                elem_id="xykc_cleanup_img2maskimg", 
                sources=["upload"],
                interactive=True, 
                type="pil", 
                image_mode="RGBA", 
                height=480,  # 调整高度以避免超出边界
                brush=gr.Brush(default_color="#FFFFFF", color_mode="picker"),  # 允许颜色选择
                eraser=gr.Eraser(),  # 添加橡皮擦工具
                container=True,
                scale=3,  # 增加缩放比例以更好地适应容器
                min_width=300,  # 设置最小宽度
                canvas_size=(480, 480),  # 修改为与组件高度一致，解决图像显示过大的问题
                # fixed_canvas=True,  # 移除此参数以避免加载错误
            )
            
            with gr.Column(elem_id="xykc_cleanup_gallery_container"):
                clean_button = gr.Button("清理图像", variant="primary", elem_id="xykc_clean_button")
                result_gallery = gr.Gallery(
                    label='输出结果', 
                    show_label=False, 
                    elem_id="xykc_cleanup_gallery", 
                    preview=True, 
                    height=400,  # 调整高度以匹配Sketchpad组件
                    object_fit="contain"  # 确保图像完整显示在容器内
                )

                with gr.Row(elem_id="xykc_image_buttons", elem_classes="image-buttons"):
                    buttons = {
                        'img2img': ToolButton('🖼️', elem_id='xykc_send_to_img2img', tooltip="将图像和生成参数发送到图生图"),
                        'inpaint': ToolButton('🎨️', elem_id='xykc_send_to_inpaint', tooltip="将图像和生成参数发送到局部重绘"),
                        'extras': ToolButton('📐', elem_id='xykc_send_to_extras', tooltip="将图像和生成参数发送到后处理")
                    }

                    # 只有在parameters_copypaste可用时才注册粘贴参数按钮
                    if parameters_copypaste is not None:
                        for paste_tabname, paste_button in buttons.items():
                            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                                paste_button=paste_button, tabname=paste_tabname, source_tabname=None, source_image_component=result_gallery,
                                paste_field_names=[]
                            ))

                send_to_cleaner_button = gr.Button("发送回清理器", elem_id="xykc_send_to_cleaner")

        # 设置事件处理
        clean_button.click(
            fn=lambda x: process_gallery_output(clean_object_init_img_with_mask(x)),
            inputs=[init_img_with_mask],
            outputs=[result_gallery],
        )

        send_to_cleaner_button.click(
            fn=process_image_editor_output,
            inputs=[result_gallery],
            outputs=[init_img_with_mask]
        )

    # 返回UI组件字典，以便在主程序中引用
    return {
        "init_img_with_mask": init_img_with_mask,
        "result_gallery": result_gallery,
    }