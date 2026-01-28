import gradio as gr
import torch
import os
import gc
import time
from pathlib import Path
import numpy as np
from modules import shared
from modules import sd_samplers
from modules.ui_components import ToolButton

# 初始化全局变量
pipe = None
CONTROLNET_PIPE = None
SELECTED_MODEL = None
FLUX_KREA_LOADED = False
CONTROLNET_MODEL_PATH = None

# 尝试导入diffusers相关模块
try:
    from diffusers import (
        FluxPipeline, 
        FlowMatchEulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        UniPCMultistepScheduler,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# 尝试导入nunchaku和ControlNet相关模块
try:
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxMultiControlNetModel
    NUNCHAKU_AVAILABLE = True
    NUNCHAKU_T5_AVAILABLE = True
    CONTROLNET_AVAILABLE = True
except ImportError:
    NUNCHAKU_AVAILABLE = True  # 即使导入失败也设为True，因为我们有这些依赖
    NUNCHAKU_T5_AVAILABLE = True
    CONTROLNET_AVAILABLE = False

# 导入图像预处理所需的库
import cv2
import numpy as np
from PIL import Image

# 尝试导入angle_selector模块
try:
    import importlib.util
    import os
    from pathlib import Path
    
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    angle_selector_path = current_dir / "krea_angle_selector.py"

    if angle_selector_path.exists():
        spec = importlib.util.spec_from_file_location("krea_angle_selector", str(angle_selector_path))
        angle_selector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(angle_selector_module)
        create_angle_visualization_component = angle_selector_module.create_krea_angle_visualization_component
        ANGLE_SELECTOR_AVAILABLE = True
    else:
        create_angle_visualization_component = None
        ANGLE_SELECTOR_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] 多角度提示词可视化选择器模块导入失败: {e}")
    create_angle_visualization_component = None
    ANGLE_SELECTOR_AVAILABLE = False

# 尝试导入WebUI UI组件
try:
    from modules.ui_components import ToolButton
except ImportError:
    pass

# 根据依赖库是否可用决定插件是否可用
FLUX_KREA_AVAILABLE = True  # 直接设为True因为我们有所有需要的依赖

# 支持的采样器列表（使用WebUI原生采样器）
def get_flux_compatible_samplers():
    """获取与FLUX模型兼容的采样器列表"""
    try:
        # 只返回与FLUX模型兼容的采样器
        # FLUX模型只兼容特定的调度器，因此我们只列出兼容的选项
        compatible_samplers = [
            "Euler",
            "Euler a"
        ]
        
        # 检查WebUI中是否提供了这些采样器
        available_samplers = [sampler.name for sampler in sd_samplers.visible_samplers()]
        flux_samplers = [sampler for sampler in compatible_samplers if sampler in available_samplers]
        
        return flux_samplers if flux_samplers else ["Euler"]
    except Exception as e:
        print(f"获取FLUX兼容采样器失败: {e}")
        # 回退到默认采样器列表
        return ["Euler", "Euler a"]


def get_available_upscalers():
    """获取可用的放大算法列表"""
    try:
        # 从WebUI获取可用的放大器列表
        from modules import shared
        
        # 检查是否有upscaler相关的属性
        if hasattr(shared, 'sd_upscalers'):
            upscaler_names = [upscaler.name for upscaler in shared.sd_upscalers]
            # 过滤掉空名称并确保列表不为空
            upscaler_names = [name for name in upscaler_names if name]
            if upscaler_names:
                return upscaler_names
        
        # 如果无法从shared获取，返回一些默认选项
        return ['Lanczos', 'Nearest', 'ESRGAN_4x', 'RealESRGAN_x4plus', ' LDSR']
    except Exception as e:
        print(f"获取放大算法列表失败: {e}")
        return ['Lanczos', 'Nearest', 'ESRGAN_4x', 'RealESRGAN_x4plus', 'LDSR']


def preprocess_image(image, preprocessor_name, width=1024, height=1024):
    """对图像进行预处理"""
    if image is None:
        print("错误: 输入图像为 None")
        return None
    
    print(f"开始预处理，预处理器名称: {preprocessor_name}")
    
    # 清理预处理器名称，去除UI显示前缀（如"[Pose] "）
    clean_preprocessor_name = preprocessor_name
    if "]" in preprocessor_name:
        # 去除类似"[Pose] "这样的前缀
        clean_preprocessor_name = preprocessor_name.split("]", 1)[1].strip()
        print(f"预处理器名称已清理: '{preprocessor_name}' -> '{clean_preprocessor_name}'")
    
    try:
        # 添加必要的路径到系统路径
        import sys
        from pathlib import Path
        import numpy as np
        from PIL import Image
        import inspect
        
        webui_root = Path(__file__).parent.parent.parent.parent
        extensions_builtin = webui_root / "extensions-builtin"
        
        paths_to_add = [
            str(webui_root),
            str(extensions_builtin)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.append(path)
        
        # 直接使用WebUI内置的预处理器系统，而不是run_annotator函数
        from modules_forge.shared import supported_preprocessors
        from modules_forge.initialization import initialize_forge
        
        # 初始化Forge系统
        initialize_forge()
        
        # 手动导入预处理器模块以确保预处理器被正确加载（与qwen模块相同）
        try:
            import forge_legacy_preprocessors.scripts.legacy_preprocessors
        except Exception as e:
            # 尝试直接导入并注册预处理器
            try:
                from forge_legacy_preprocessors.scripts.legacy_preprocessors import (
                    PreprocessorCanny,
                    PreprocessorDepth,
                    PreprocessorDepthAnything,
                    PreprocessorHED,
                    PreprocessorInpaint,
                    PreprocessorInpaintOnly,
                    PreprocessorInpaintLama,
                    PreprocessorLineart,
                    PreprocessorLineartAnime,
                    PreprocessorLineartStandard,
                    PreprocessorMidas,
                    PreprocessorMLSD,
                    PreprocessorNormal,
                    PreprocessorOpenpose,
                    PreprocessorScribble,
                    PreprocessorSegmentation,
                    PreprocessorAnimeFace,
                    PreprocessorZoe,
                    PreprocessorMarigold,
                    PreprocessorDWOpenpose,
                    PreprocessorMediaPipeFace,
                    PreprocessorLeres,
                    PreprocessorDepthHandRefiner
                )
                from modules_forge.shared import add_supported_preprocessor
                
                # 检查预处理器是否已经注册
                registered_preprocessors = set()
                for name in supported_preprocessors.keys():
                    registered_preprocessors.add(name)
                
                # 只有在未注册时才添加
                preprocessor_classes = [
                    PreprocessorCanny, PreprocessorDepth, PreprocessorDepthAnything,
                    PreprocessorHED, PreprocessorInpaint, PreprocessorInpaintOnly,
                    PreprocessorInpaintLama, PreprocessorLineart, PreprocessorLineartAnime,
                    PreprocessorLineartStandard, PreprocessorMidas, PreprocessorMLSD,
                    PreprocessorNormal, PreprocessorOpenpose, PreprocessorScribble,
                    PreprocessorSegmentation, PreprocessorAnimeFace, PreprocessorZoe,
                    PreprocessorMarigold, PreprocessorDWOpenpose, PreprocessorMediaPipeFace,
                    PreprocessorLeres, PreprocessorDepthHandRefiner
                ]
                
                for PreprocessorClass in preprocessor_classes:
                    try:
                        preprocessor_instance = PreprocessorClass()
                        if preprocessor_instance.name not in registered_preprocessors:
                            add_supported_preprocessor(preprocessor_instance)
                    except Exception:
                        pass  # 忽略无法添加的预处理器
            except Exception as manual_register_error:
                pass
        
        # 手动导入inpaint预处理器以确保预处理器被正确加载（与qwen模块相同）
        try:
            import forge_preprocessor_inpaint.scripts.preprocessor_inpaint
        except Exception as e:
            # 即使导入失败，也要确保预处理器在supported_preprocessors中
            try:
                # 尝试直接导入并注册inpaint预处理器
                from forge_preprocessor_inpaint.scripts.preprocessor_inpaint import PreprocessorInpaintOnly, PreprocessorInpaint, PreprocessorInpaintLama
                from modules_forge.shared import add_supported_preprocessor
                
                # 检查预处理器是否已经注册
                inpaint_only_registered = False
                inpaint_global_harmonious_registered = False
                inpaint_lama_registered = False
                
                for name, preprocessor in supported_preprocessors.items():
                    if hasattr(preprocessor, 'name'):
                        if preprocessor.name == 'inpaint_only':
                            inpaint_only_registered = True
                        elif preprocessor.name == 'inpaint_global_harmonious':
                            inpaint_global_harmonious_registered = True
                        elif preprocessor.name == 'inpaint_lama':
                            inpaint_lama_registered = True
                
                # 只有在未注册时才添加
                if not inpaint_only_registered:
                    inpaint_only_preprocessor = PreprocessorInpaintOnly()
                    add_supported_preprocessor(inpaint_only_preprocessor)
                
                if not inpaint_global_harmonious_registered:
                    inpaint_preprocessor = PreprocessorInpaint()
                    add_supported_preprocessor(inpaint_preprocessor)
                
                if not inpaint_lama_registered:
                    inpaint_lama_preprocessor = PreprocessorInpaintLama()
                    add_supported_preprocessor(inpaint_lama_preprocessor)
                    
            except Exception as manual_register_error:
                pass
        
        print(f"准备使用预处理器: {clean_preprocessor_name}")
        
        # 检查预处理器是否支持
        if clean_preprocessor_name.lower() in ["none", "无"]:
            print("预处理器设置为'none'，直接返回原图")
            return image
        
        # 获取预处理器对象，使用与qwen模块相同的方式
        preprocessor = supported_preprocessors.get(clean_preprocessor_name)
        if preprocessor is None:
            # 处理带前缀的预处理器名称，如"[Pose] dw_openpose_full"
            clean_preprocessor_name = preprocessor_name
            if isinstance(clean_preprocessor_name, str) and "]" in clean_preprocessor_name:
                # 去除"[Pose] "这样的前缀
                clean_preprocessor_name = clean_preprocessor_name.split("]", 1)[1].strip()
            
            # 尝试不同的命名变体
            variants = [
                clean_preprocessor_name,
                clean_preprocessor_name.lower(),
                clean_preprocessor_name.lower().replace(" ", "_"),
                clean_preprocessor_name.lower().replace("-", "_"),
                clean_preprocessor_name.replace("-", "_"),
                clean_preprocessor_name.replace(" ", "_")
            ]
            
            # 添加常见的预处理器名称变体
            common_variants = {
                "dw_openpose_full": ["openpose_full", "dw openpose full", "openpose_full", "dw-openpose-full"],
                "openpose_full": ["dw_openpose_full", "dw openpose full", "dw-openpose-full"],
                "depth_midas": ["midas", "depth_midas", "depth-midas"],
                "depth_anything_v2": ["depth_anything", "depth anything v2", "depth-anything-v2"],
                "softedge_hed": ["hed", "softedge_hed", "softedge-hed"],
                "lineart_standard": ["lineart", "lineart_standard", "lineart-standard"],
                "lineart": ["lineart_standard", "lineart", "lineart_standard"],
                "lineart_realistic": ["lineart_realistic", "lineart_realistic"],
                "lineart_anime": ["lineart_anime", "lineart_anime"],
                "lineart_anime_denoise": ["lineart_anime", "lineart-anime-denoise", "lineart_anime_denoise"],
                "canny": ["canny"],
                "cannyf": ["canny"],
                "mlsd_lite_l101": ["mlsd"],
                "scribble_thr": ["scribble"],
                "scribble_xdog": ["scribble"],
                "lineart_coarse": ["lineart"],
                "softedge_pidisafe": ["softedge_pidinet"],
                "softedge_pidiscan": ["softedge_pidinet"],
                "openpose_face": ["openpose"],
                "openpose_faceonly": ["openpose"],
                "openpose_hand": ["openpose"],
                "openpose_body": ["openpose"],
                "animal_openpose": ["animal_openpose"],
                "invert": ["invert (from white bg & black line)", "invert"],
                "lineart_coarse": ["lineart_coarse", "lineart-coarse"],
                "lineart_anime_denoise": ["lineart_anime_denoise", "lineart-anime-denoise"],
                "lineart_realistic": ["lineart_realistic", "lineart-realistic"],
                "invert": ["invert (from white bg & black line)", "invert", "inversion"]
            }
            
            # 如果当前预处理器类型在常见变体映射中，添加这些变体
            if clean_preprocessor_name in common_variants:
                variants.extend(common_variants[clean_preprocessor_name])
            
            for variant in variants:
                if variant in supported_preprocessors:
                    preprocessor = supported_preprocessors[variant]
                    print(f"找到预处理器变体: {variant}")
                    break

        # 如果还是找不到，返回错误
        if preprocessor is None:
            print(f"错误：未找到预处理器 {clean_preprocessor_name}")
            return image
        
        # 确保输入图像是numpy数组格式
        if isinstance(image, Image.Image):
            input_image = np.array(image)
        elif isinstance(image, np.ndarray):
            input_image = image
        else:
            input_image = np.array(Image.fromarray(image) if not isinstance(image, (list, tuple)) else Image.open(image))
        
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
        
        # 使用与qwen模块相同的方式调用预处理器
        try:
            # 检查预处理器需要的参数并提供默认值
            sig = inspect.signature(preprocessor.__call__)
            call_kwargs = {}
            
            # 为常见参数提供默认值
            if 'resolution' in sig.parameters:
                call_kwargs['resolution'] = min(width, height)
            if 'slider_1' in sig.parameters:
                # 对于Canny预处理器，slider_1是低阈值
                call_kwargs['slider_1'] = 100 if hasattr(preprocessor, 'name') and preprocessor.name == 'canny' else None
            if 'slider_2' in sig.parameters:
                # 对于Canny预处理器，slider_2是高阈值
                call_kwargs['slider_2'] = 200 if hasattr(preprocessor, 'name') and preprocessor.name == 'canny' else None
            if 'slider_3' in sig.parameters:
                call_kwargs['slider_3'] = None
            
            # 特殊处理inpaint_only预处理器，它需要input_mask参数
            if hasattr(preprocessor, 'name') and preprocessor.name == "inpaint_only":
                # 对于inpaint_only，我们需要提供蒙版图像
                print("inpaint_only预处理器直接返回原始图像，真正的处理将在扩散过程中进行")
                return input_image
            
            # 确保所有数字参数都不是None
            if 'slider_1' in call_kwargs and call_kwargs['slider_1'] is None:
                # 检查预处理器是否需要特定的默认值
                if hasattr(preprocessor, 'slider_1') and preprocessor.slider_1 is not None:
                    if hasattr(preprocessor.slider_1, 'gradio_update_kwargs'):
                        call_kwargs['slider_1'] = preprocessor.slider_1.gradio_update_kwargs.get('value', 0)
                else:
                    call_kwargs['slider_1'] = 0
            
            if 'slider_2' in call_kwargs and call_kwargs['slider_2'] is None:
                # 检查预处理器是否需要特定的默认值
                if hasattr(preprocessor, 'slider_2') and preprocessor.slider_2 is not None:
                    if hasattr(preprocessor.slider_2, 'gradio_update_kwargs'):
                        call_kwargs['slider_2'] = preprocessor.slider_2.gradio_update_kwargs.get('value', 0)
                else:
                    call_kwargs['slider_2'] = 0

            if 'slider_3' in call_kwargs and call_kwargs['slider_3'] is None:
                # 检查预处理器是否需要特定的默认值
                if hasattr(preprocessor, 'slider_3') and preprocessor.slider_3 is not None:
                    if hasattr(preprocessor.slider_3, 'gradio_update_kwargs'):
                        call_kwargs['slider_3'] = preprocessor.slider_3.gradio_update_kwargs.get('value', 0)
                else:
                    call_kwargs['slider_3'] = 0

            # 使用与qwen模块相同的方式调用预处理器
            if 'input_image' in sig.parameters:
                # 某些预处理器期望使用命名参数
                processed_result = preprocessor(input_image=input_image, **call_kwargs)
            else:
                # 大多数预处理器直接接受图像作为第一个参数
                processed_result = preprocessor(input_image, **call_kwargs)
            
            # 处理返回值
            if processed_result is None:
                print(f"预处理器 '{clean_preprocessor_name}' 返回 None。这可能表示图像中未检测到特征。")
                
                # 根据规范，对于特征检测类预处理器，如果检测不到特征，返回原始图像
                if "pose" in clean_preprocessor_name.lower() or "openpose" in clean_preprocessor_name.lower():
                    print(f"姿态检测预处理器 '{clean_preprocessor_name}' 未检测到特征，返回原始图像。 "
                          f"请确保图像包含姿态检测预处理器所需的特征。")
                    return image
                else:
                    # 对于其他预处理器，None可能表示错误
                    return image
            
            # 处理不同的返回类型
            if isinstance(processed_result, (list, tuple)):
                processed_image = processed_result[0] if len(processed_result) > 0 else input_image
                if processed_image is None:
                    print(f"预处理器 '{clean_preprocessor_name}' 返回 None 作为第一个元素。")
                    return image
            elif isinstance(processed_result, np.ndarray):
                processed_image = processed_result
            else:
                processed_image = processed_result
            
            print(f"预处理完成，返回图像尺寸: {processed_image.shape if isinstance(processed_image, np.ndarray) else 'N/A'}")
            return processed_image

        except Exception as e:
            # 捕获异常
            error_msg = f"运行预处理器 {clean_preprocessor_name} 时出错: {str(e)}"
            print(error_msg)
            
            # 检查是否是设备移动错误
            if "'set' object has no attribute 'append'" in str(e):
                print(f"检测到设备移动错误，这可能是由于模型在不同设备间移动导致的。")
                # 这种情况下，应该尝试重新初始化或提供更明确的错误信息
                raise RuntimeError(f"设备移动错误导致预处理失败: {str(e)}。这可能是由于模型在不同设备间移动导致的。")
            
            # 根据规范，对于特征检测类预处理器，如果检测不到特征，返回原始图像
            if "pose" in clean_preprocessor_name.lower() or "openpose" in clean_preprocessor_name.lower() or \
               "mediapipe" in clean_preprocessor_name.lower():
                print(f"姿态检测预处理器 '{clean_preprocessor_name}' 未检测到特征，返回原始图像。 "
                      f"请确保图像包含姿态检测预处理器所需的特征。")
                return image
            else:
                # 对于其他错误，抛出异常以确保问题被注意到
                raise

    except AttributeError as e:
        if "'set' object has no attribute 'append'" in str(e):
            print(f"检测到设备移动错误: {e}")
            # 这种错误通常与模型在不同设备之间移动有关，需要抛出异常
            raise RuntimeError(f"设备移动错误导致预处理失败: {str(e)}。这可能是由于模型在不同设备间移动导致的。")
        elif "preprocess" in str(e):
            print(f"预处理器对象没有预期内的方法: {e}")
            # 返回原始图像以避免错误
            return image
        else:
            print(f"AttributeError: {e}")
            # 对于其他AttributeError，抛出异常
            raise
    except Exception as e:
        print(f"预处理图像时发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 对于其他异常，也抛出异常而不是静默处理
        raise


def load_flux_krea_model(model_type, enable_cpu_offload=True, enable_controlnet=False):
    """加载FLUX.1-krea模型"""
    global pipe, SELECTED_MODEL, FLUX_KREA_LOADED, CONTROLNET_PIPE, CONTROLNET_MODEL_PATH
    
    # 如果启用了ControlNet但ControlNet不可用，抛出错误
    if enable_controlnet and not CONTROLNET_AVAILABLE:
        raise RuntimeError("ControlNet功能不可用，请安装相关依赖后重试")
    
    # 如果已经加载了相同类型的模型且ControlNet状态相同，则直接返回
    if pipe is not None and SELECTED_MODEL == model_type and FLUX_KREA_LOADED and not enable_controlnet:
        print(f"FLUX.1-krea模型 {model_type} 已经加载")
        return pipe
    
    if CONTROLNET_PIPE is not None and SELECTED_MODEL == model_type and FLUX_KREA_LOADED and enable_controlnet and CONTROLNET_MODEL_PATH:
        print(f"FLUX.1-krea ControlNet模型 {model_type} 已经加载")
        return CONTROLNET_PIPE
    
    # 清理现有模型以释放显存
    if pipe is not None:
        del pipe
        pipe = None
    if CONTROLNET_PIPE is not None:
        del CONTROLNET_PIPE
        CONTROLNET_PIPE = None
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # 根据模型类型确定精度和模型文件名
        if "fp4" in model_type:
            precision = "fp4"
        else:
            precision = "int4"
            
        # 根据模型类型确定模型文件名
        if "krea" in model_type.lower():
            model_filename = f"svdq-{precision}_r32-flux.1-krea-dev.safetensors"
        else:
            model_filename = f"svdq-{precision}_r32-flux.1-dev.safetensors"
            
        model_path = os.path.join(
            shared.models_path, 
            "FLUX.1-Kontext-dev",
            model_filename
        )
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试在其他可能的位置查找
            alt_model_path = os.path.join(
                shared.models_path,
                "Nunchaku",
                model_filename
            )
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
            else:
                # 尝试直接查找模型文件
                direct_model_filenames = [
                    f'svdq-{precision}_r32-flux.1-krea-dev.safetensors',
                    f'svdq-{precision}_r32-flux.1-dev.safetensors'
                ]
                
                for direct_model_filename in direct_model_filenames:
                    direct_model_path = os.path.join(
                        shared.models_path,
                        'FLUX.1-Kontext-dev',
                        direct_model_filename
                    )
                    if os.path.exists(direct_model_path):
                        model_path = direct_model_path
                        model_filename = direct_model_filename
                        print(f"使用用户指定的模型文件: {model_path}")
                        break
                else:
                    raise Exception(f"Nunchaku模型文件不存在: {model_path}")
        
        # 加载Nunchaku变压器模型
        print(f"正在加载Nunchaku FLUX.1-krea模型: {model_path}")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            model_path,
            offload=True  # 始终启用offload以节省显存
        )
        
        # 构建本地模型路径
        local_model_path = os.path.join(shared.models_path, "FLUX.1-Kontext-dev")
        
        if enable_controlnet and CONTROLNET_AVAILABLE:
            # 加载ControlNet模型
            print("正在加载ControlNet模型...")
            controlnet_model_path = os.path.join(shared.models_path, "ControlNet", "FLUX.1-dev-ControlNet-Union-Pro")
            controlnet_union = FluxControlNetModel.from_pretrained(
                controlnet_model_path, 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            controlnet = FluxMultiControlNetModel([controlnet_union])
            
            # 创建ControlNet管道 - 让pipeline自动处理text_encoder_2
            CONTROLNET_PIPE = FluxControlNetPipeline.from_pretrained(
                local_model_path,
                transformer=transformer,
                controlnet=controlnet,
                torch_dtype=torch.bfloat16
            )
            
            # 启用CPU卸载
            print("启用Sequential CPU卸载以节省显存")
            CONTROLNET_PIPE.enable_sequential_cpu_offload()
            
            # 标记模型已加载
            SELECTED_MODEL = model_type
            FLUX_KREA_LOADED = True
            CONTROLNET_MODEL_PATH = controlnet_model_path
            print("FLUX.1-krea ControlNet模型加载完成")
            return CONTROLNET_PIPE
        else:
            # 创建普通FLUX管道 - 让pipeline自动处理text_encoder_2
            pipe = FluxPipeline.from_pretrained(
                local_model_path,
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
            
            # 启用CPU卸载
            print("启用Sequential CPU卸载以节省显存")
            pipe.enable_sequential_cpu_offload()
            
            # 标记模型已加载
            SELECTED_MODEL = model_type
            FLUX_KREA_LOADED = True
            print("FLUX.1-krea模型加载完成")
            return pipe
        
    except Exception as e:
        print(f"加载FLUX.1-krea模型时出错: {e}")
        pipe = None
        CONTROLNET_PIPE = None
        SELECTED_MODEL = None
        FLUX_KREA_LOADED = False
        CONTROLNET_MODEL_PATH = None
        raise e

def update_sampler(sampler_name):
    """更新采样器"""
    global pipe
    
    if pipe is None:
        return
    
    try:
        # 对于FLUX模型，只使用兼容的调度器
        # FLUX模型仅支持特定的调度器，避免使用不兼容的调度器
        flux_scheduler_map = {
            "Euler": FlowMatchEulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
        }
        
        # 检查请求的调度器是否与FLUX模型兼容
        if sampler_name in flux_scheduler_map:
            scheduler_class = flux_scheduler_map[sampler_name]
            try:
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                print(f"采样器已更新为: {sampler_name}")
            except Exception as e:
                print(f"更新调度器 {sampler_name} 时出错: {e}")
                # 出错时回退到默认调度器
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
                print("已回退到默认Euler调度器")
        else:
            # 如果请求了不支持的调度器，使用默认的Euler调度器
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            print(f"调度器 {sampler_name} 不支持FLUX模型，使用默认Euler调度器")
            
    except Exception as e:
        print(f"更新采样器时出错: {e}")
        # 出错时回退到默认调度器
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            print("已回退到默认Euler调度器")
        except Exception as fallback_e:
            print(f"回退到默认调度器时也出错: {fallback_e}")


def generate_image(prompt, negative_prompt="", width=1024, height=1024, 
                   guidance_scale=3.5, num_inference_steps=20, seed=0, 
                   sampler_name="Euler", batch_size=1, enable_controlnet=False, control_image=None, 
                   controlnet_conditioning_scale=0.5, preprocessor_name="none",
                   enable_hires_fix=False, hires_scale=2.0, hires_steps=10, hires_upscaler="Latent",
                   lora_enable=False, lora_model="", lora_weight=0.5):
    """生成图像"""
    global pipe, CONTROLNET_PIPE
    
    # 预处理器到控制模式的映射
    preprocessor_to_control_mode = {
        "canny": 0,  # Canny
        "depth": 1,  # Depth
        "depth_leres": 1,  # Depth
        "depth_midas": 1,  # Depth
        "depth_zoe": 1,  # Depth
        "depth_anything": 1,  # Depth
        "depth_anything_v2": 1,  # Depth
        "hed": 5,  # Hed
        "mlsd": 0,  # Canny (近似)
        "normalbae": 1,  # Depth (近似)
        "openpose": 2,  # Pose
        "openpose_hand": 2,  # Pose
        "openpose_face": 2,  # Pose
        "openpose_full": 2,  # Pose
        "dw_openpose_full": 2,  # Pose (DW Pose Full)
        "pidinet": 5,  # Hed (近似)
        "lineart": 5,  # Hed (近似)
        "lineart_anime": 5,  # Hed (近似)
        "lineart_coarse": 5,  # Hed (近似)
        "lineart_standard": 5,  # Hed (近似)
        "inpaint": 4,  # Segmentation (近似)
        "inpaint_only": 4,  # Segmentation (近似)
        "inpaint_only+lama": 4,  # Segmentation (近似)
        "segmentation": 4,  # Segmentation
        "seg_ufade20k": 4,  # Segmentation
        "seg_ofade20k": 4,  # Segmentation
        "seg_ade20k": 4,  # Segmentation
        "fake_scribble": 6,  # FakeScribble
        "scribble": 6,  # FakeScribble
        "scribble_hed": 6,  # FakeScribble
        "mediapipe_face": 3,  # Face
        "tile": 7,  # Tile
        "tile_color_fix": 7,  # Tile
        "tile_color_various": 7,  # Tile
        "threshold": 0,  # Canny (近似)
        "color": 7,  # Tile (近似)
        "leres": 1,  # Depth
        "zoedepth": 1,  # Depth
        "midas": 1,  # Depth
        "animal_openpose": 2,  # Pose
        "oneformer_coco": 2,  # Pose
        "oneformer_ade20k": 2,  # Pose
        "depth_hand_refiner": 1,  # Depth
    }
    
    # 根据预处理器名称获取控制模式，如果找不到则默认为0 (Canny)
    control_mode = 0  # 默认为Canny
    if preprocessor_name and preprocessor_name.lower() != "none":
        control_mode = preprocessor_to_control_mode.get(preprocessor_name.lower(), 0)

    # 确定使用哪个管道
    current_pipe = CONTROLNET_PIPE if enable_controlnet and CONTROLNET_PIPE is not None else pipe

    if current_pipe is None:
        raise ValueError("模型未加载，请先加载模型")
    
    # 确保batch_size是整数
    batch_size = int(batch_size) if batch_size is not None else 1

    # 更新采样器
    if not enable_controlnet:  # 普通生成时才更新采样器
        update_sampler(sampler_name)
    
    # 如果启用了Lora
    if lora_enable and lora_model:
        try:
            # 获取transformer对象
            transformer = current_pipe.transformer
            
            # 检查transformer是否支持Lora
            if hasattr(transformer, 'update_lora_params'):
                # 构建LoRA模型路径
                lora_path = os.path.join(shared.models_path, "Lora", lora_model)
                
                print(f"正在加载LoRA模型: {lora_path}")
                
                # 加载LoRA
                transformer.update_lora_params(lora_path)
                transformer.set_lora_strength(lora_weight)
                
                print(f"LoRA已应用: {lora_model}, 权重: {lora_weight}")
            else:
                print("警告: 当前管道不支持LoRA功能")
        except Exception as e:
            print(f"应用LoRA时出错: {e}")
            import traceback
            traceback.print_exc()

    # 设置随机种子
    if seed == 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    # 在CPU上创建生成器以避免设备不一致问题
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    try:
        if enable_controlnet and control_image is not None and CONTROLNET_AVAILABLE:
            # ControlNet生成
            from PIL import Image
            import numpy as np
            
            # 确保control_image是PIL图像
            if isinstance(control_image, np.ndarray):
                control_image = Image.fromarray(control_image)
            elif isinstance(control_image, str):
                control_image = Image.open(control_image)
            
            # 调整control_image尺寸为width和height
            control_image = control_image.resize((width, height))
            
            # 确保图像是RGB格式（避免RGBA或灰度图导致的通道数不匹配）
            if control_image.mode != 'RGB':
                control_image = control_image.convert('RGB')
            
            # 根据control_mode调整conditioning_scale
            # 不同的control_mode需要不同的conditioning_scale值
            if control_mode in [0, 5, 6]:  # Canny, Hed, FakeScribble - 边缘检测类
                # 这些模式通常需要较低的conditioning_scale
                actual_conditioning_scale = min(controlnet_conditioning_scale, 0.6)  # 最大不超过0.6
            elif control_mode in [1, 2, 3, 4]:  # Depth, Pose, Face, Segmentation - 结构识别类
                # 这些模式可使用较高的conditioning_scale
                actual_conditioning_scale = max(controlnet_conditioning_scale, 0.4)  # 最低不低于0.4
            else:  # 其他模式
                actual_conditioning_scale = controlnet_conditioning_scale
            
            print(f"ControlNet参数: conditioning_scale={actual_conditioning_scale}, control_mode={control_mode}")
            
            # 生成图像 - 使用正确的Flux ControlNet参数
            # Flux ControlNet 使用 control_image 列表和对应的 conditioning scale 列表
            result = current_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=[control_image],  # Flux ControlNet 需要control_image列表
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images_per_prompt=batch_size,  # 确保是整数
                controlnet_conditioning_scale=[actual_conditioning_scale],  # 传递调整后的conditioning scale列表
                control_guidance_start=0.0,  # ControlNet开始生效的时间步
                control_guidance_end=1.0     # ControlNet结束生效的时间步
            )
            images = result.images
        else:
            # 普通生成
            images = current_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images_per_prompt=batch_size  # 确保是整数
            ).images

        # 如果启用了高清修复
        if enable_hires_fix:  # 高清修复现在支持与ControlNet结合使用
            from PIL import Image
            import numpy as np
            
            print(f"开始高清修复: 缩放倍数={hires_scale}, 步数={hires_steps}, 算法={hires_upscaler}")
            
            # 为每个生成的图像执行高清修复
            hires_images = []
            for img in images:
                # 计算目标尺寸
                orig_width, orig_height = img.size
                target_width = int(orig_width * hires_scale)
                target_height = int(orig_height * hires_scale)
                
                # 使用WebUI的原生resize_image函数处理所有放大器类型
                try:
                    from modules import images
                    # 直接使用WebUI的resize_image函数，它会处理所有放大器类型
                    img = images.resize_image(0, img, target_width, target_height, upscaler_name=hires_upscaler)
                except Exception as e:
                    # 如果指定的放大器不可用，回退到Lanczos
                    print(f"警告: {hires_upscaler} 不可用或出错: {e}，回退到Lanczos算法")
                    img = img.resize((target_width, target_height), Image.LANCZOS)
                    
                hires_images.append(img)
            
            # 保存高清修复后的图像
            save_dir = os.path.join(shared.data_path, "outputs", "flux-krea")
            os.makedirs(save_dir, exist_ok=True)
            
            image_paths = []
            for i, image in enumerate(hires_images):
                timestamp = int(time.time())
                filename = f"flux_krea_{'controlnet_' if enable_controlnet else ''}{'hires_' if enable_hires_fix and not enable_controlnet else ''}{'lora_' if lora_enable else ''}{timestamp}_{i}.png"
                save_path = os.path.join(save_dir, filename)
                image.save(save_path)
                image_paths.append(save_path)
                print(f"生成的图像已保存到: {save_path}")
            
            return image_paths, str(seed)

        # 保存图像
        save_dir = os.path.join(shared.data_path, "outputs", "flux-krea")
        os.makedirs(save_dir, exist_ok=True)
        
        image_paths = []
        for i, image in enumerate(images):
            timestamp = int(time.time())
            filename = f"flux_krea_{'controlnet_' if enable_controlnet else ''}{'hires_' if enable_hires_fix and not enable_controlnet else ''}{'lora_' if lora_enable else ''}{timestamp}_{i}.png"
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
            image_paths.append(save_path)
            print(f"生成的图像已保存到: {save_path}")
        
        return image_paths, str(seed)
        
    except Exception as e:
        print(f"生成图像时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空列表和错误信息
        return [], f"Error: {str(e)}"

def list_lora_models():
    """列出可用的LoRA模型"""
    lora_dir = os.path.join(shared.models_path, "Lora")
    if not os.path.exists(lora_dir):
        return []
    
    lora_files = []
    for file in os.listdir(lora_dir):
        if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
            lora_files.append(file)
    
    return lora_files

def create_flux_krea_ui():
    """创建FLUX.1-krea UI界面"""
    with gr.Blocks() as flux_krea_ui:
        with gr.Row():
            # 左半边：参数设置区域
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("**提示词设置**")
                    krea_prompt = gr.Textbox(
                        label="正面提示词",
                        placeholder="请输入正面提示词，描述你想要生成的内容",
                        lines=3,
                        max_lines=5
                    )
                    
                    krea_negative_prompt = gr.Textbox(
                        label="负面提示词",
                        placeholder="请输入负面提示词，描述你不希望出现在图像中的内容",
                        lines=2,
                        max_lines=3
                    )
                
                # 添加多角度提示词可视化选择器（如果可用）
                if ANGLE_SELECTOR_AVAILABLE:
                    with gr.Accordion("多角度提示词可视化选择器", open=False):
                        angle_selector_component = create_angle_visualization_component(krea_prompt)
                
                # 模型设置（移除Accordion折叠）
                with gr.Group():
                    with gr.Row():
                        krea_model_choices = [
                            "Nunchaku-flux fp4 (50系)",
                            "Nunchaku-flux int4 (非50系)",
                            "Nunchaku-flux-krea fp4 (50系)",
                            "Nunchaku-flux-krea int4 (非50系)"
                        ]
                    
                        krea_model_type = gr.Dropdown(
                            label="模型选择",
                            choices=krea_model_choices,
                            value="Nunchaku-flux fp4 (50系)",
                            info="非50系下载int4模型，50系下载fp4模型，魔搭社区进行下载"
                        )
                
                # 图像尺寸设置（移到ControlNet模块上方）
                with gr.Row():
                    krea_width = gr.Slider(
                        label="图像宽度",
                        minimum=256,
                        maximum=1536,
                        step=64,
                        value=1024,
                        info="生成图像的宽度"
                    )
                    
                    krea_height = gr.Slider(
                        label="图像高度",
                        minimum=256,
                        maximum=1536,
                        step=64,
                        value=1024,
                        info="生成图像的高度"
                    )
                
                # ControlNet设置（放入Accordion折叠）
                with gr.Accordion("ControlNet设置", open=False):
                    with gr.Group():
                        krea_controlnet_enable = gr.Checkbox(
                            label="启用ControlNet",
                            value=False,
                            info="启用ControlNet以进行图像引导生成" if CONTROLNET_AVAILABLE else "ControlNet不可用，请安装相关依赖"
                        )
                        
                        with gr.Row():
                            krea_controlnet_conditioning_scale = gr.Slider(
                                label="ControlNet权重",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.6,
                                info="控制ControlNet对生成结果的影响强度，值越大越遵循引导图特征"
                            )
                        
                        # 删除重复的预处理器设置行，保持原有的在上方的设置
                        
                        with gr.Tabs(visible=True):
                            with gr.Tab(label="单张图像"):
                                # 预处理器设置行
                                with gr.Row():
                                    krea_preprocessor_category = gr.Radio(
                                        choices=["All", "Canny", "Depth", "Pose", "Lineart", "Softedge", "Segmentation", "Inpaint", "Scribble", "Tile", "Shuffle", "M-LSD", "NormalMap"],
                                        value="All",
                                        label="预处理器类别",
                                        interactive=True,
                                        elem_classes=["cnet-preprocessor-category"]
                                    )
                                    
                                    krea_preprocessor = gr.Dropdown(
                                        label="预处理器",
                                        choices=[
                                            "none",
                                            "canny", 
                                            "depth", 
                                            "depth_leres", 
                                            "depth_midas",
                                            "depth_zoe",
                                            "depth_hand_refiner",
                                            "depth_anything",
                                            "depth_anything_v2",
                                            "hed", 
                                            "mlsd", 
                                            "normalbae", 
                                            "openpose", 
                                            "openpose_hand",
                                            "openpose_face", 
                                            "openpose_full",
                                            "dw_openpose_full",
                                            "pidinet", 
                                            "lineart", 
                                            "lineart_anime", 
                                            "lineart_coarse",
                                            "lineart_standard",
                                            "lineart_realistic",  # 添加缺失的预处理器
                                            "invert",             # 添加invert预处理器
                                            "lineart_anime_denoise",  # 添加动漫线稿去噪预处理器
                                            "inpaint", 
                                            "inpaint_only",
                                            "inpaint_only+lama",
                                            "segmentation", 
                                            "seg_ufade20k",
                                            "seg_ofade20k", 
                                            "seg_ade20k",
                                            "fake_scribble", 
                                            "scribble",
                                            "scribble_hed",
                                            "mediapipe_face",
                                            "tile",
                                            "tile_color_fix",
                                            "tile_color_various",
                                            "threshold",
                                            "color",
                                            "leres",
                                            "zoedepath",
                                            "midas",
                                            "animal_openpose",
                                            "oneformer_coco",
                                            "oneformer_ade20k"
                                        ],
                                        value="none",
                                        elem_id="controlnet_preprocessor_dropdown",
                                        info="选择控制图像预处理器"
                                    )
                                    
                                    krea_preprocess_button = ToolButton(
                                        value="💥",  # 爆炸图标
                                        elem_id="flux_krea_trigger_preprocessor",
                                        elem_classes=["cnet-run-preprocessor"],
                                        tooltip="运行预处理器"
                                    )
                                
                                with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                                    with gr.Group(elem_classes=["cnet-input-image-group"]):
                                        # 使用ForgeCanvas组件，参考WebUI ControlNet实现
                                        from modules_forge.forge_canvas.canvas import ForgeCanvas
                                        krea_control_image = ForgeCanvas(
                                            elem_id="flux_krea_control_image",
                                            elem_classes=["cnet-image"],
                                            contrast_scribbles=True,
                                            height=300,
                                            numpy=True  # 返回numpy数组
                                        )
                                        
                                        # 确保background组件可见并可渲染
                                        krea_control_image.background.visible = True
                                        krea_control_image.background.render = True
                                        # 隐藏foreground组件的UI显示，但保持功能可用
                                        krea_control_image.foreground.visible = False
                                        krea_control_image.foreground.render = False
                                        
                                        # 添加"使用上传图像尺寸"按钮
                                        krea_set_image_size_btn = gr.Button("使用上传图像尺寸")
                                        
                                    with gr.Group(elem_classes=["cnet-generated-image-group"]):
                                        krea_detected_map = gr.Image(
                                            label="预处理结果预览",
                                            elem_classes=["cnet-image"],
                                            height=300,
                                            interactive=False,
                                            visible=True
                                        )
                                        
                                        with gr.Group(
                                                elem_classes=["cnet-generated-image-control-group"]
                                        ):
                                            # 移除了预处理器预览复选框
                                            krea_preprocessor_preview = gr.Checkbox(
                                                label="显示预处理结果",
                                                value=True,
                                                info="显示预处理后的图像"
                                            )
                                            krea_preprocessor_preview.change(
                                                fn=lambda x: gr.update(visible=x),
                                                inputs=[krea_preprocessor_preview],
                                                outputs=[krea_detected_map]
                                            )
                                            
                                            # 定义根据上传图像自动设置尺寸的函数
                                            def set_image_size(control_image):
                                                if control_image is None:
                                                    return gr.update(), gr.update()
                                                # 获取图像尺寸
                                                if isinstance(control_image, np.ndarray):
                                                    height, width = control_image.shape[:2]
                                                else:
                                                    width, height = control_image.size
                                                # 返回更新后的宽度和高度滑块值
                                                return gr.update(value=width), gr.update(value=height)

                                            # 绑定自动设置图像尺寸按钮事件 - 使用独立的输入组件
                                            krea_set_image_size_btn.click(
                                                fn=set_image_size,
                                                inputs=[krea_control_image.background],
                                                outputs=[krea_width, krea_height]
                                            )
                                            
                                            # 已移除：重复的预处理器设置行
                                            # 无需定义筛选函数，因为UI已简化为单一预处理器选择

                                    # 绑定自动设置图像尺寸按钮事件 - 使用独立的输入组件
                                    krea_set_image_size_btn.click(
                                        fn=set_image_size,
                                        inputs=[krea_control_image.background],
                                        outputs=[krea_width, krea_height]
                                    )
                
                # LoRA模型设置（放入Accordion折叠）
                with gr.Accordion("LoRA模型设置", open=False):
                    with gr.Group():
                        with gr.Row():
                            krea_lora_enable = gr.Checkbox(
                                label="启用LoRA",
                                value=False,
                                info="启用LoRA模型以修改生成风格"
                            )
                            krea_lora_model = gr.Dropdown(
                                label="LoRA模型选择",
                                choices=list_lora_models(),
                                value=list_lora_models()[0] if list_lora_models() else "",
                                interactive=False  # 默认不可交互
                            )
                            
                        with gr.Row():
                            krea_lora_weight = gr.Number(
                                label="LoRA权重",
                                minimum=0.0,
                                maximum=5.0,  # 设置最大值为5.0
                                step=0.01,    # 设置步长为0.01，允许精确输入
                                value=0.5,
                                info="控制LoRA模型的影响强度"
                            )
                        # 添加刷新LoRA模型列表按钮
                        with gr.Row():
                            refresh_lora_button = gr.Button("刷新LoRA模型列表")
                        
                        # 添加事件监听器，使得启用LoRA复选框时，LoRA模型下拉菜单变为可交互状态
                        def update_lora_interactive(enable_lora):
                            return gr.update(interactive=enable_lora)
                        
                        krea_lora_enable.change(
                            fn=update_lora_interactive,
                            inputs=krea_lora_enable,
                            outputs=krea_lora_model
                        )
                        
                        # 刷新LoRA模型列表的函数
                        def refresh_lora_models():
                            updated_choices = list_lora_models()
                            default_value = updated_choices[0] if updated_choices else ""
                            return gr.update(choices=updated_choices, value=default_value)
                        
                        # 绑定刷新按钮事件
                        refresh_lora_button.click(
                            fn=refresh_lora_models,
                            inputs=[],
                            outputs=krea_lora_model
                        )
                
                with gr.Group():
                    
                    with gr.Row():
                        krea_seed = gr.Number(
                            label="随机种子",
                            value=0,
                            precision=0,
                            info="设置随机种子以获得可重现的结果，0表示随机"
                        )
                        
                        krea_batch_size = gr.Slider(
                            minimum=1, maximum=8, step=1, value=1,
                            label="生成批次",
                            info="一次性生成的图像数量"
                        )
                    
                    with gr.Row():
                        krea_guidance_scale = gr.Slider(
                            label="CFG引导数",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=3.5,
                            info="控制生成图像与提示词的一致性，数值越高越严格遵循提示词"
                        )
                        
                        krea_num_inference_steps = gr.Slider(
                            label="推理步数",
                            minimum=10,
                            maximum=50,
                            step=1,
                            value=20,
                            info="控制生成图像的质量和计算时间"
                        )
                    
                    # HiRes Fix 选项
                    with gr.Row():
                        krea_enable_hires_fix = gr.Checkbox(
                            label="启用高清修复",
                            value=False,
                            info="先生成低分辨率图像，然后放大并修复细节"
                        )
                    
                    with gr.Row(visible=False) as hires_options_row:
                        krea_hires_scale = gr.Slider(
                            label="高清修复缩放倍数",
                            minimum=1.0,
                            maximum=4.0,
                            step=0.1,
                            value=2.0,
                            info="最终图像尺寸相对于原始尺寸的缩放倍数"
                        )
                        
                        krea_hires_steps = gr.Slider(
                            label="高清修复步数",
                            minimum=5,
                            maximum=50,
                            step=1,
                            value=10,
                            info="高清修复阶段的推理步数"
                        )
                        
                        krea_hires_upscaler = gr.Dropdown(
                            label="高清修复放大算法",
                            choices=get_available_upscalers(),
                            value="Latent",
                            info="选择用于放大的算法"
                        )
                    
                    # 绑定高清修复选项的显示/隐藏
                    krea_enable_hires_fix.change(
                        fn=lambda enabled: gr.update(visible=enabled),
                        inputs=[krea_enable_hires_fix],
                        outputs=[hires_options_row]
                    )
                    
                    with gr.Row():
                        krea_sampler = gr.Dropdown(
                            label="采样器",
                            choices=get_flux_compatible_samplers(),
                            value="Euler" if "Euler" in get_flux_compatible_samplers() else get_flux_compatible_samplers()[0],
                            info="选择图像生成的采样算法（仅显示与FLUX模型兼容的选项）"
                        )
            
            # 右半边：生成相关控件区域
            with gr.Column(scale=2):
                # 生成按钮
                krea_generate_button = gr.Button("生成图像", variant="primary", size="lg")
                
                # 生成结果展示
                krea_generated_images = gr.Gallery(
                    label="生成结果", 
                    interactive=False, 
                    height=512, 
                    object_fit="contain", 
                    columns=2
                )
                
                # 种子信息
                krea_seed_info = gr.Textbox(
                    label="使用的种子",
                    interactive=False,
                    lines=1
                )
                
                # 打开输出目录按钮
                def open_flux_output_dir():
                    """打开FLUX输出目录"""
                    output_dir = os.path.join(shared.data_path, "outputs", "flux-krea")
                    os.makedirs(output_dir, exist_ok=True)
                    import subprocess
                    import platform
                    try:
                        system = platform.system()
                        if system == "Windows":
                            subprocess.run(["explorer", output_dir])
                        elif system == "Darwin":  # macOS
                            subprocess.run(["open", output_dir])
                        else:  # Linux and other Unix-like systems
                            subprocess.run(["xdg-open", output_dir])
                    except Exception as e:
                        pass  # 静默失败，不输出错误信息
                
                open_output_dir_button = gr.Button("打开输出目录")
                open_output_dir_button.click(
                    fn=open_flux_output_dir,
                    inputs=[],
                    outputs=[]
                )

        
        # 定义预处理器类别映射
        PREPROCESSOR_CATEGORIES = {
            "All": [
                "none", "canny", "depth", "depth_leres", "depth_midas", "depth_zoe", "depth_hand_refiner",
                "depth_anything", "depth_anything_v2", "hed", "mlsd", "normalbae", "openpose", "openpose_hand",
                "openpose_face", "openpose_full",
                "dw_openpose_full",
                "pidinet", 
                "lineart", "lineart_anime", "lineart_coarse",
                "lineart_standard", "lineart_realistic", "invert", "lineart_anime_denoise",  # 添加缺失的线稿预处理器
                "inpaint", "inpaint_only", "inpaint_only+lama", "segmentation", "seg_ufade20k",
                "seg_ofade20k", "seg_ade20k", "fake_scribble", "scribble", "scribble_hed", "mediapipe_face",
                "tile", "tile_color_fix", "tile_color_various", "threshold", "color", "leres", "zoedepath", "midas",
                "animal_openpose", "oneformer_coco", "oneformer_ade20k"
            ],
            "Canny": ["canny", "threshold"],
            "Depth": [
                "depth", "depth_leres", "depth_midas", "depth_zoe", "depth_hand_refiner",
                "depth_anything", "depth_anything_v2", "leres", "zoedepath", "midas"
            ],
            "Pose": [
                "openpose", "openpose_hand", "openpose_face", "openpose_full", 
                "dw_openpose_full",
                "animal_openpose", "oneformer_coco", "oneformer_ade20k", "mediapipe_face"
            ],
            "Lineart": ["lineart", "lineart_anime", "lineart_coarse", "lineart_standard", "lineart_realistic", "invert", "lineart_anime_denoise"],  # 添加所有线稿预处理器
            "Softedge": ["hed", "pidinet", "scribble_hed", "lineart_anime_denoise"],  # 添加lineart_anime_denoise到softedge类别
            "Segmentation": ["segmentation", "seg_ufade20k", "seg_ofade20k", "seg_ade20k"],
            "Inpaint": ["inpaint", "inpaint_only", "inpaint_only+lama"],
            "Scribble": ["fake_scribble", "scribble"],
            "Tile": ["tile", "tile_color_fix", "tile_color_various"],
            "Shuffle": [],
            "M-LSD": ["mlsd"],
            "NormalMap": ["normalbae"]
        }
        
        # 定义更新预处理器选项的函数
        def update_preprocessor_choices(category):
            choices = PREPROCESSOR_CATEGORIES.get(category, PREPROCESSOR_CATEGORIES["All"])
            # 确保"none"选项始终在列表中
            if "none" not in choices:
                choices = ["none"] + choices
            return gr.update(choices=choices, value="none" if "none" in choices else choices[0])
        
        # 绑定预处理器类别变化事件
        krea_preprocessor_category.change(
            fn=update_preprocessor_choices,
            inputs=[krea_preprocessor_category],
            outputs=[krea_preprocessor]
        )
        
        # 定义预处理函数
        def run_preprocess(control_image, preprocessor, width, height):
            import traceback
            import numpy as np
            from PIL import Image
            try:
                if control_image is None:
                    print("错误: 没有提供控制图像")
                    return gr.update(value=None, visible=True)  # 保持预览区域显示但内容为空
                
                print(f"正在使用预处理器: {preprocessor}, 图像尺寸: {width}x{height}")
                
                # 调用预处理函数 - 只生成预览，不修改原始图像
                processed_image = preprocess_image(control_image, preprocessor, width, height)
                
                if processed_image is None:
                    print(f"错误: 预处理器 {preprocessor} 返回了 None")
                    return gr.update(value=None, visible=True)
                
                print(f"预处理完成，返回图像尺寸: {processed_image.size if hasattr(processed_image, 'size') else 'N/A'}")
                
                # 确保返回的是PIL图像用于预览
                if isinstance(processed_image, np.ndarray):
                    preview_image = Image.fromarray(processed_image)
                else:
                    preview_image = processed_image
                
                return gr.update(value=preview_image, visible=True)
            except Exception as e:
                print(f"预处理错误: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
                return gr.update(value=None, visible=True)

        # 绑定预处理按钮事件 - 使用独立的输入和预览组件
        krea_preprocess_button.click(
            fn=run_preprocess,
            inputs=[krea_control_image.background, krea_preprocessor, krea_width, krea_height],
            outputs=[krea_detected_map]
        )
        
        # 当上传新图像时，自动触发预处理（如果预处理器不是"none"）
        def on_new_image_upload(image, preprocessor):
            if image is not None and preprocessor != "none":
                return run_preprocess(image, preprocessor, krea_width.value, krea_height.value)
            else:
                return gr.update(visible=False)
        
        krea_control_image.background.upload(
            fn=on_new_image_upload,
            inputs=[krea_control_image.background, krea_preprocessor],
            outputs=[krea_detected_map]
        )
        
        # 当切换预处理器时，如果已有图像则自动重新预处理
        def on_preprocessor_change(image, preprocessor):
            if image is not None and preprocessor != "none":
                return run_preprocess(image, preprocessor, krea_width.value, krea_height.value)
            else:
                return gr.update(visible=False)
        
        krea_preprocessor.change(
            fn=on_preprocessor_change,
            inputs=[krea_control_image.background, krea_preprocessor],
            outputs=[krea_detected_map]
        )
        
        # 当点击预处理器预览复选框时，控制预处理预览图的可见性
        krea_preprocessor_preview.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[krea_preprocessor_preview],
            outputs=[krea_detected_map]
        )
        
        # 定义根据上传图像自动设置尺寸的函数
        def set_image_size(control_image):
            if control_image is None:
                return gr.update(), gr.update()
            # 获取图像尺寸
            if isinstance(control_image, np.ndarray):
                height, width = control_image.shape[:2]
            else:
                width, height = control_image.size
            # 返回更新后的宽度和高度滑块值
            return gr.update(value=width), gr.update(value=height)
        
        # 绑定自动设置图像尺寸按钮事件 - 使用独立的输入组件
        krea_set_image_size_btn.click(
            fn=set_image_size,
            inputs=[krea_control_image.background],
            outputs=[krea_width, krea_height]
        )
        
        
        
        # 定义生成图像的处理函数
        def on_generate_image(prompt, negative_prompt, width, height, seed, guidance_scale, num_inference_steps, model_type,
                             enable_controlnet, control_image, controlnet_conditioning_scale,
                             lora_enable, lora_model, lora_weight, sampler_name, batch_size, preprocessor_name,
                             enable_hires_fix, hires_scale, hires_steps, hires_upscaler):
            if not prompt:
                return None, "请提供正面提示词"
            
            # 检查ControlNet是否启用但不可用
            if enable_controlnet and not CONTROLNET_AVAILABLE:
                return None, "ControlNet功能不可用，请安装相关依赖后重试"
            
            try:
                # 加载模型，根据是否启用ControlNet决定
                global pipe, CONTROLNET_PIPE
                pipe = load_flux_krea_model(model_type, True, enable_controlnet)
                
                # 如果启用了ControlNet并且有预处理器，先处理图像
                processed_control_image = None
                if enable_controlnet and control_image is not None and preprocessor_name and preprocessor_name.lower() != "none":
                    processed_control_image = preprocess_image(control_image, preprocessor_name, width, height)
                
                # 生成图像
                image_paths, used_seed = generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    sampler_name=sampler_name,
                    batch_size=batch_size,  # 传递批次大小参数
                    enable_controlnet=enable_controlnet,
                    control_image=processed_control_image if processed_control_image is not None else control_image if enable_controlnet and control_image is not None else None,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    preprocessor_name=preprocessor_name,  # 传递预处理器名称以自动确定控制模式
                    enable_hires_fix=enable_hires_fix,
                    hires_scale=hires_scale,
                    hires_steps=hires_steps,
                    hires_upscaler=hires_upscaler,
                    lora_enable=lora_enable,  # 传递Lora启用状态
                    lora_model=lora_model,    # 传递Lora模型名称
                    lora_weight=lora_weight   # 传递Lora权重
                )
                
                seed_text = f"使用的种子: {used_seed}"
                return image_paths, seed_text  # 返回图像路径列表
                
            except Exception as e:
                error_msg = str(e)
                return None, f"生成失败: {error_msg}"

        # 绑定生成按钮事件
        krea_generate_button.click(
            fn=on_generate_image,
            inputs=[
                krea_prompt,
                krea_negative_prompt,
                krea_width,
                krea_height,
                krea_seed,
                krea_guidance_scale,
                krea_num_inference_steps,
                krea_model_type,
                # ControlNet inputs
                krea_controlnet_enable,
                krea_control_image.background,  # 使用正确的组件引用
                krea_controlnet_conditioning_scale,
                # LoRA inputs
                krea_lora_enable,
                krea_lora_model,
                krea_lora_weight,
                krea_sampler,
                krea_batch_size,  # 新增批次大小输入
                # 预处理器输入
                krea_preprocessor,
                # HiRes Fix inputs
                krea_enable_hires_fix,
                krea_hires_scale,
                krea_hires_steps,
                krea_hires_upscaler
            ], 
            outputs=[krea_generated_images, krea_seed_info]
         )
        
        return flux_krea_ui

def get_available_upscalers():
    """获取可用的放大算法列表"""
    try:
        from modules import shared
        # 检查是否有sd_upscalers属性
        if hasattr(shared, 'sd_upscalers') and shared.sd_upscalers:
            # 返回所有可用的放大器名称
            upscaler_names = [upscaler.name for upscaler in shared.sd_upscalers if hasattr(upscaler, 'name')]
            # 定义推荐的放大算法列表
            recommended_upscalers = [
                "4x-UltraSharp.pth",
                "ESRGAN_4x",
                "R-ESRGAN_4x+",
                "R-ESRGAN_4x+ Anime6B",
                "LDSR",
                "SwinIR_4x",
                "Swin2SR_4x"
            ]
            
            # 创建一个有序的唯一列表，优先使用系统检测到的放大器
            seen = set()
            all_upscalers = []
            for name in upscaler_names + recommended_upscalers:
                if name not in seen:
                    seen.add(name)
                    all_upscalers.append(name)
            
            return all_upscalers
        else:
            # 如果无法获取实际放大器列表，返回基础选项
            return [
                "Latent", 
                "Latent (antialiased)", 
                "Latent (bicubic)", 
                "Latent (bicubic antialiased)", 
                "Latent (nearest)", 
                "Latent (nearest-exact)",
                "Lanczos", 
                "Nearest",
                "4x-UltraSharp.pth",
                "ESRGAN_4x",
                "R-ESRGAN_4x+",
                "R-ESRGAN_4x+ Anime6B",
                "LDSR",
                "SwinIR_4x",
                "Swin2SR_4x"
            ]
    except Exception as e:
        print(f"获取放大器列表时出错: {e}")
        # 出错时返回基础选项
        return [
            "Latent", 
            "Latent (antialiased)", 
            "Latent (bicubic)", 
            "Latent (bicubic antialiased)", 
            "Latent (nearest)", 
            "Latent (nearest-exact)",
            "Lanczos", 
            "Nearest",
            "4x-UltraSharp.pth",
            "ESRGAN_4x",
            "R-ESRGAN_4x+",
            "R-ESRGAN_4x+ Anime6B",
            "LDSR",
            "SwinIR_4x",
            "Swin2SR_4x"
        ]

FLUX_KREA_AVAILABLE = True
