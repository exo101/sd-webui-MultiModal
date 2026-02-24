import os
import sys
import torch
import gradio as gr
from diffusers import ZImageControlNetModel, ZImagePipeline, ZImageControlNetPipeline
from PIL import Image
import numpy as np
import cv2
import inspect
from pathlib import Path

# 添加当前脚本目录到Python路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

def get_cn_models():
    """获取ControlNet模型列表"""
    # 尝试不同的可能路径
    possible_paths = []
    
    # 首先尝试从shared获取路径
    try:
        from modules import shared
        possible_paths.append(os.path.join(shared.models_path, "ControlNet"))
    except:
        pass
    
    # 添加相对路径
    possible_paths.extend([
        os.path.join("models", "ControlNet"),
        os.path.join("..", "..", "models", "ControlNet"),
        os.path.join("..", "models", "ControlNet"),
        os.path.join(".", "models", "ControlNet")
    ])
    
    # 遍历所有可能的路径，直到找到存在的路径
    for path in possible_paths:
        if os.path.exists(path):
            cn_models_dir = path
            break
    else:
        # 如果以上路径都不存在，使用默认路径
        cn_models_dir = os.path.join("models", "ControlNet")
        os.makedirs(cn_models_dir, exist_ok=True)  # 创建目录（如果不存在）

    models = []
    if cn_models_dir and os.path.exists(cn_models_dir):
        for filename in os.listdir(cn_models_dir):
            # 检查是否为safetensors文件或目录
            item_path = os.path.join(cn_models_dir, filename)
            if os.path.isfile(item_path) and filename.endswith(".safetensors"):
                # 是一个safetensors文件
                models.append(filename)
            elif os.path.isdir(item_path):
                # 是一个目录，检查其中是否包含ControlNet相关的文件
                # 简单地将目录作为一个模型选项添加
                models.append(filename)
    
    # 过滤掉非ControlNet模型（如VAE、tokenizer等）
    filtered_models = []
    exclude_keywords = ['vae', 'tokenizer', 'text_encoder', 'feature_extractor', 'scheduler']
    
    for model in models:
        is_excluded = False
        model_lower = model.lower()
        for keyword in exclude_keywords:
            if keyword in model_lower:
                is_excluded = True
                break
        if not is_excluded:
            filtered_models.append(model)
    
    return sorted(filtered_models, key=lambda x: x.lower())

# 确保CONTROLNET_AVAILABLE被正确初始化
CONTROLNET_AVAILABLE = True

# 尝试导入预处理器相关模块
def get_preprocessor_list():
    """获取可用的预处理器列表"""
    try:
        from modules_forge.shared import supported_preprocessors
        preprocessor_names = list(supported_preprocessors.keys())
        
        # 添加分类前缀到预处理器名称
        categorized_preprocessors = []
        for name in preprocessor_names:
            if any(pose_keyword in name.lower() for pose_keyword in ['openpose', 'pose', 'keypoint']):
                categorized_preprocessors.append(f"[Pose] {name}")
            elif any(depth_keyword in name.lower() for depth_keyword in ['depth', 'normal', 'zoe']):
                categorized_preprocessors.append(f"[Depth] {name}")
            elif any(edge_keyword in name.lower() for edge_keyword in ['canny', 'lineart', 'hed', 'mlsd', 'softedge']):
                categorized_preprocessors.append(f"[Edge] {name}")
            elif any(seg_keyword in name.lower() for seg_keyword in ['seg', 'segmentation', 'anime', 'clip']):
                categorized_preprocessors.append(f"[Seg] {name}")
            elif 'inpaint' in name.lower():
                categorized_preprocessors.append(f"[Inpaint] {name}")
            elif 'scribble' in name.lower():
                categorized_preprocessors.append(f"[Scribble] {name}")
            elif 'tile' in name.lower():
                categorized_preprocessors.append(f"[Tile] {name}")
            else:
                categorized_preprocessors.append(f"[Misc] {name}")
        
        return categorized_preprocessors
    except ImportError:
        # 如果无法导入forge预处理器，则返回默认列表
        return [
            "[Edge] canny", "[Edge] softedge_hed", "[Edge] lineart", "[Edge] lineart_anime",
            "[Depth] depth_midas", "[Depth] depth_anything_v2", "[Depth] normal_bae",
            "[Pose] openpose_full", "[Pose] dw_openpose_full", "[Pose] animal_openpose",
            "[Seg] seg_ofade20k", "[Seg] seg_ufade20k", "[Seg] clipseg",
            "[Scribble] scribble_thr", "[Scribble] scribble_xdog",
            "[Inpaint] inpaint_only", "[Inpaint] inpaint_global_harmonious",
            "[Tile] tile_resample"
        ]


def get_controlnet_list():
    """获取可用的ControlNet模型列表"""
    try:
        from modules import shared
        controlnet_path = Path(shared.models_path) / "ControlNet"
        if controlnet_path.exists():
            # 获取所有目录和safetensors文件
            controlnet_items = []
            for item in controlnet_path.iterdir():
                if item.is_dir():
                    # 检查目录中是否包含ControlNet模型文件
                    if any(file.suffix in ['.safetensors', '.bin'] for file in item.iterdir()):
                        controlnet_items.append(item.name)
                elif item.is_file() and item.suffix == '.safetensors':
                    controlnet_items.append(item.name)
            
            # 过滤掉非ControlNet模型
            valid_controlnets = []
            exclude_keywords = ['vae', 'tokenizer', 'text_encoder', 'feature_extractor', 'scheduler']
            
            for item in controlnet_items:
                is_valid = True
                for keyword in exclude_keywords:
                    if keyword in item.lower():
                        is_valid = False
                        break
                if is_valid:
                    valid_controlnets.append(item)
            
            return valid_controlnets if valid_controlnets else ["None"]
        else:
            # 如果路径不存在，返回通用列表
            return get_cn_models() if get_cn_models() else ["None"]
    except Exception:
        # 如果出错，返回通用列表
        return get_cn_models() if get_cn_models() else ["None"]


def preprocess_controlnet_image(image, preprocessor_name, width=1024, height=1024):
    """对ControlNet输入图像进行预处理"""
    if image is None:
        print("错误: 输入图像为 None")
        return None

    print(f"开始ControlNet预处理，预处理器名称: {preprocessor_name}")

    # 清理预处理器名称，去除UI显示前缀（如"[Pose] "）
    clean_preprocessor_name = preprocessor_name
    if "]" in preprocessor_name:
        # 去除类似"[Pose] "这样的前缀
        clean_preprocessor_name = preprocessor_name.split("]", 1)[1].strip()
        print(f"预处理器名称已清理: '{preprocessor_name}' -> '{clean_preprocessor_name}'")

    try:
        # 尝试从WebUI获取预处理器
        try:
            from modules import shared
            from modules_forge.shared import supported_preprocessors
            from modules_forge.initialization import initialize_forge

            # 初始化Forge系统
            initialize_forge()

            # 检查预处理器是否支持
            if clean_preprocessor_name.lower() in ["none", "无", ""]:
                print("预处理器设置为'none'，直接返回原图")
                if isinstance(image, Image.Image):
                    return image.resize((width, height))
                else:
                    return Image.fromarray(image).resize((width, height))

            # 获取预处理器对象
            preprocessor = supported_preprocessors.get(clean_preprocessor_name)
            if preprocessor is None:
                # 尝试不同的命名变体
                variants = [
                    clean_preprocessor_name,
                    clean_preprocessor_name.lower(),
                    clean_preprocessor_name.lower().replace(" ", "_"),
                    clean_preprocessor_name.lower().replace("-", "_"),
                    clean_preprocessor_name.replace("-", "_"),
                    clean_preprocessor_name.replace(" ", "_")
                ]

                for variant in variants:
                    if variant in supported_preprocessors:
                        preprocessor = supported_preprocessors[variant]
                        print(f"找到预处理器变体: {variant}")
                        break

            # 如果还是找不到，返回错误
            if preprocessor is None:
                print(f"错误：未找到预处理器 {clean_preprocessor_name}")
                if isinstance(image, Image.Image):
                    return image.resize((width, height))
                else:
                    return Image.fromarray(image).resize((width, height))

            # 确保输入图像是numpy数组格式
            if isinstance(image, Image.Image):
                input_image = np.array(image)
            elif isinstance(image, np.ndarray):
                input_image = image
            else:
                input_image = np.array(Image.fromarray(image))

            print(f"输入图像形状: {input_image.shape if isinstance(input_image, np.ndarray) else 'N/A'}")

            # 确保图像是RGB格式
            if len(input_image.shape) == 2:
                # 灰度图转RGB
                input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            elif input_image.shape[2] == 4:
                # RGBA转RGB
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
            elif input_image.shape[2] == 3:
                # 已经是RGB格式
                pass
            else:
                # 其他情况，假设是BGR格式转RGB
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # 使用预处理器处理图像
            try:
                # 检查预处理器需要的参数并提供默认值
                sig = inspect.signature(preprocessor.__call__)
                call_kwargs = {}

                # 为常见参数提供默认值
                if 'resolution' in sig.parameters:
                    call_kwargs['resolution'] = min(width, height)
                if 'slider_1' in sig.parameters:
                    # 对于Canny预处理器，slider_1是低阈值
                    call_kwargs['slider_1'] = 100 if 'canny' in clean_preprocessor_name.lower() else 64
                if 'slider_2' in sig.parameters:
                    # 对于Canny预处理器，slider_2是高阈值
                    call_kwargs['slider_2'] = 200 if 'canny' in clean_preprocessor_name.lower() else 128
                if 'slider_3' in sig.parameters:
                    call_kwargs['slider_3'] = 64

                # 特殊处理inpaint_only预处理器，它需要input_mask参数
                if 'inpaint_only' in clean_preprocessor_name.lower():
                    print("inpaint_only预处理器直接返回原始图像，真正的处理将在扩散过程中进行")
                    return Image.fromarray(input_image).resize((width, height))

                # 确保所有数字参数都不是None
                for param in ['slider_1', 'slider_2', 'slider_3']:
                    if param in call_kwargs and call_kwargs[param] is None:
                        call_kwargs[param] = 64

                # 调用预处理器
                processed_result = preprocessor(input_image, **call_kwargs)

                # 处理不同的返回类型
                if processed_result is None:
                    print(f"预处理器 '{clean_preprocessor_name}' 返回 None。这可能表示图像中未检测到特征。")
                    return Image.fromarray(input_image).resize((width, height))

                if isinstance(processed_result, (list, tuple)):
                    processed_image = processed_result[0] if len(processed_result) > 0 else input_image
                    if processed_image is None:
                        print(f"预处理器 '{clean_preprocessor_name}' 返回 None 作为第一个元素。")
                        return Image.fromarray(input_image).resize((width, height))
                elif isinstance(processed_result, np.ndarray):
                    processed_image = processed_result
                else:
                    processed_image = processed_result

                print(f"ControlNet预处理完成，返回图像尺寸: {processed_image.shape if isinstance(processed_image, np.ndarray) else 'N/A'}")
                return Image.fromarray(processed_image).resize((width, height))

            except Exception as e:
                error_msg = f"运行预处理器 {clean_preprocessor_name} 时出错: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return Image.fromarray(input_image).resize((width, height))

        except ImportError:
            # 如果无法导入Forge预处理器，则返回原始图像
            print("未找到Forge预处理器，返回调整尺寸后的原始图像")
            if isinstance(image, Image.Image):
                return image.resize((width, height))
            else:
                return Image.fromarray(image).resize((width, height))

    except Exception as e:
        print(f"ControlNet预处理图像时发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 返回调整尺寸后的原始图像
        if isinstance(image, Image.Image):
            return image.resize((width, height))
        else:
            return Image.fromarray(image).resize((width, height))


def create_controlnet_ui():
    """创建ControlNet UI组件"""
    with gr.Accordion("ControlNet", open=False):
        # 启用ControlNet的复选框
        enabled = gr.Checkbox(label="启用 ControlNet", value=False)
        
        with gr.Group(visible=False) as controlnet_panel:
            # ControlNet模型选择
            model = gr.Dropdown(
                choices=get_controlnet_list(),
                label="ControlNet 模型",
                value=get_controlnet_list()[0] if get_controlnet_list() else "None"
            )
            
            # ControlNet输入图像
            input_image = gr.Image(label="ControlNet 输入图像", type="pil", height=400)
            
            # 预处理器选择
            preprocessor = gr.Dropdown(
                choices=get_preprocessor_list(),
                label="预处理器",
                value=get_preprocessor_list()[0] if get_preprocessor_list() else "None"
            )
            
            # 预处理后的图像展示
            preview = gr.Image(label="预处理结果预览", interactive=False, height=300)
            
            # ControlNet参数
            with gr.Row():
                weight = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.05, label="权重")
                guidance_start = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="引导开始")
                guidance_end = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="引导结束")
            
            # 预处理按钮
            run_preprocessor_btn = gr.Button(value="运行预处理器", variant="secondary")
            
            # 更新预处理结果
            run_preprocessor_btn.click(
                fn=preprocess_controlnet_image,
                inputs=[input_image, preprocessor],
                outputs=[preview]
            )
            
            # 当启用ControlNet时显示面板
            enabled.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[enabled],
                outputs=[controlnet_panel]
            )
    
    # 返回所有组件，方便在其他地方使用
    return enabled, model, input_image, preprocessor, weight, guidance_start, guidance_end, preview, run_preprocessor_btn


def apply_controlnet_to_pipeline(pipeline, controlnet_model_path):
    """
    将ControlNet应用到现有的pipeline中
    """
    if not CONTROLNET_AVAILABLE:
        print("ControlNet功能不可用，请安装相关依赖")
        return pipeline
    
    try:
        # 加载ControlNet模型
        controlnet = ZImageControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.bfloat16
        )
        
        # 创建新的ControlNet pipeline
        controlnet_pipeline = ZImageControlNetPipeline(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            transformer=pipeline.transformer,
            scheduler=pipeline.scheduler,
            controlnet=controlnet
        )
        
        return controlnet_pipeline
    except Exception as e:
        print(f"应用ControlNet到pipeline失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return pipeline


# 为了与其他UI组件兼容，定义可用性标志
Z_IMAGE_CONTROLNET_AVAILABLE = CONTROLNET_AVAILABLE