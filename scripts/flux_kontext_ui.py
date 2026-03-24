import gradio as gr
import numpy as np
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import gc
import os
import sys
import time
from modules import shared
import datetime

# 导入发送到分镜功能
try:
    from scripts.storyboard_assistant import send_to_storyboard
    STORYBOARD_AVAILABLE = True
except Exception as e:
    print(f"⚠️ 导入分镜助手失败：{e}")
    STORYBOARD_AVAILABLE = False

# 尝试导入angle_selector模块
try:
    import importlib.util
    import os
    from pathlib import Path
    
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    angle_selector_path = current_dir / "kontext_angle_selector.py"

    if angle_selector_path.exists():
        spec = importlib.util.spec_from_file_location("kontext_angle_selector", str(angle_selector_path))
        angle_selector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(angle_selector_module)
        create_angle_visualization_component = angle_selector_module.create_kontext_angle_visualization_component
        ANGLE_SELECTOR_AVAILABLE = True
    else:
        create_angle_visualization_component = None
        ANGLE_SELECTOR_AVAILABLE = False
except Exception as e:
    print(f"[WARNING] 多角度提示词可视化选择器模块导入失败: {e}")
    create_angle_visualization_component = None
    ANGLE_SELECTOR_AVAILABLE = False

# 尝试导入diffusers相关模块
try:
    from diffusers import (
        FluxKontextPipeline, 
        FluxTransformer2DModel, 
        GGUFQuantizationConfig,
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"Diffusers库导入失败: {e}")
    DIFFUSERS_AVAILABLE = False

# 尝试导入nunchaku相关模块
try:
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.lora.flux.compose import compose_lora
    from nunchaku.utils import get_precision
    NUNCHAKU_AVAILABLE = True

except ImportError as e:
    print(f"Nunchaku库导入失败: {e}")
    NUNCHAKU_AVAILABLE = False

# 检查LoRA支持
try:
    from diffusers import DiffusionPipeline
    LORA_SUPPORTED = True
except ImportError:
    LORA_SUPPORTED = False
    print("警告: LoRA支持不可用，因为diffusers版本不支持LoRA功能")

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

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast
)

# 导入NunchakuT5EncoderModel用于加载量化T5模型
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nunchaku-2_lora_concat'))
    from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel
    NUNCHAKU_T5_AVAILABLE = True
except ImportError as e:
    print(f"Nunchaku T5编码器导入失败: {e}")
    NUNCHAKU_T5_AVAILABLE = False

NUNCHAKU_MODEL_CONFIGS = {
    "int4": "svdq-int4_r32-flux.1-kontext-dev.safetensors",
    "fp4": "svdq-fp4_r32-flux.1-kontext-dev.safetensors"
}

POSE_CONFIG_FILENAME = "saved_edits.txt"

MAX_SEED = np.iinfo(np.int32).max
MAX_INPUT_IMAGES = 2

FLUX_KONTEXT_LOADED = False
pipe = None

LOADED_LORA = None
LOADED_LORA_WEIGHT = 0.0

SELECTED_MODEL = None

def load_lora_weights(pipe, lora_path, weight=0.5):
    """加载LoRA权重到模型中"""
    global LOADED_LORA, LOADED_LORA_WEIGHT
    
    try:
        full_lora_path = os.path.join(shared.models_path, "Lora", lora_path)
        
        if not os.path.exists(full_lora_path):
            print(f"LoRA文件不存在: {full_lora_path}")
            return False
        
        if NUNCHAKU_AVAILABLE:
            try:
                composed_lora = compose_lora([(full_lora_path, weight)])
                pipe.transformer.update_lora_params(composed_lora)
                LOADED_LORA = full_lora_path
                LOADED_LORA_WEIGHT = weight
                print(f"成功使用Nunchaku加载LoRA模型: {full_lora_path} (权重: {weight})")
                return True
            except Exception as e:
                print(f"Nunchaku LoRA加载失败，回退到标准方法: {e}")
        
        if hasattr(pipe, 'load_lora_weights'):
            pipe.load_lora_weights(full_lora_path, adapter_name="default")
            pipe.set_adapters("default", weight)
            LOADED_LORA = full_lora_path
            LOADED_LORA_WEIGHT = weight
            print(f"成功加载LoRA模型: {full_lora_path} (权重: {weight})")
            return True
        else:
            print("当前模型不支持LoRA加载")
            return False
    except Exception as e:
        print(f"加载LoRA模型失败: {e}")
        return False


def unload_lora_weights(pipe):
    """卸载LoRA权重"""
    global LOADED_LORA, LOADED_LORA_WEIGHT
    
    try:
        if NUNCHAKU_AVAILABLE and hasattr(pipe.transformer, 'update_lora_params'):
            try:
                pipe.transformer.update_lora_params({})
                LOADED_LORA = None
                LOADED_LORA_WEIGHT = 0.0
                print("成功使用Nunchaku卸载LoRA模型")
                return True
            except Exception as e:
                print(f"Nunchaku LoRA卸载失败，回退到标准方法: {e}")
        
        if hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
            LOADED_LORA = None
            LOADED_LORA_WEIGHT = 0.0
            print("成功卸载LoRA模型")
            return True
        else:
            print("当前模型不支持LoRA卸载")
            return False
    except Exception as e:
        print(f"卸载LoRA模型失败: {e}")
        return False

def prepare_model_for_lora(pipe):
    """在加载LoRA之前准备模型"""
    return True

def process_uploaded_files(files):
    """处理上传的文件，返回文件列表和图像预览"""
    if files is None:
        files = []
    elif not isinstance(files, list):
        files = [files]
        
    print(f"处理 {len(files)} 个文件")
    
    valid_files = []
    for file in files:
        try:
            if hasattr(file, 'name') and file.name and os.path.exists(file.name):
                valid_files.append(file)
                print(f"成功加载图像: {file.name}")
            else:
                print(f"跳过无效文件: {file}")
        except Exception as e:
            print(f"处理文件时出错: {e}")
            if hasattr(file, 'name'):
                print(f"文件路径: {file.name}")
            pass
    
    print(f"有效文件数: {len(valid_files)}")
    
    preview_data = []
    for i, file in enumerate(valid_files):
        try:
            img = Image.open(file.name)
            preview_data.append(img)
        except Exception as e:
            print(f"处理预览图像时出错: {e}")
            try:
                img = Image.open(file.name)
                preview_data.append(img)
            except Exception:
                pass
    
    return preview_data


def process_images_for_inference(input_files):
    """处理输入图像文件"""
    if input_files is None:
        input_files = []
    elif not isinstance(input_files, list):
        input_files = [input_files]
    
    if len(input_files) == 0:
        outputs_dir = os.path.join(shared.data_path, "outputs")
        if os.path.exists(outputs_dir):
            for root, dirs, files in os.walk(outputs_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_path = os.path.join(root, file)
                        input_files.append(file_path)
                    break
                if len(input_files) >= 4:
                    break
    
    if len(input_files) == 0:
        raise gr.Error("请至少上传一张图像或确保outputs目录中有图像文件")
    
    valid_images = []
    for file_obj in input_files:
        try:
            file_path = None
            if hasattr(file_obj, 'name') and file_obj.name:
                file_path = file_obj.name
            elif isinstance(file_obj, str):
                file_path = file_obj
            else:
                continue
                
            if file_path is not None and os.path.exists(file_path):
                img = Image.open(file_path)
                valid_images.append(img)
        except Exception as e:
            print(f"处理图像文件时出错: {e}")
            pass
    
    if not valid_images:
        raise gr.Error("请至少上传一张有效图像")
    
    return valid_images, None, None



def find_existing_model(filename):
    """在固定路径中查找现有模型文件"""
    try:
        if not shared.models_path or not os.path.exists(shared.models_path):
            print(f"警告: shared.models_path 无效或不存在: {shared.models_path}")
            return None
            
        model_dir = os.path.join(shared.models_path, 'FLUX.1-Kontext-dev')
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            return None
            
        model_path = os.path.join(model_dir, filename)
        
        if os.path.exists(model_path):
            print(f"找到模型文件: {model_path}")
            return model_path
        else:
            print(f"模型文件不存在: {model_path}")
            return None
            
    except Exception as e:
        print(f"查找模型文件时出错: {e}")
        return None


def load_nunchaku_model_with_quantized_t5(enable_cpu_offload=True, precision="fp4"):
    """加载Nunchaku优化的FLUX.1-Kontext模型，支持量化T5文本编码器"""
    global FLUX_KONTEXT_LOADED, pipe
    
    try:
        # 获取模型文件名
        if precision not in NUNCHAKU_MODEL_CONFIGS:
            try:
                precision = get_precision()
                print(f"检测到的精度: {precision}")
            except:
                print("无法检测精度，使用默认fp4")
                precision = "fp4"
        
        model_filename = NUNCHAKU_MODEL_CONFIGS.get(precision, NUNCHAKU_MODEL_CONFIGS["fp4"])
        model_path = os.path.join(
            shared.models_path,
            'FLUX.1-Kontext-dev',
            model_filename
        )
        
        print(f"尝试加载Nunchaku模型: {model_path}")
        
        if not os.path.exists(model_path):
            alt_model_path = os.path.join(
                shared.models_path,
                'Nunchaku',
                model_filename
            )
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
                print(f"在备用路径找到模型: {model_path}")
            else:
                # 修改：提供更具体的错误信息，明确指出缺少哪个模型文件
                raise Exception(f"缺少必需的Nunchaku模型文件: {model_filename}，请确保该文件存在于 {os.path.join(shared.models_path, 'FLUX.1-Kontext-dev')} 目录中")
        
        # 使用正确的Nunchaku模型加载方式
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            model_path,
            offload=enable_cpu_offload
        )
        print("成功加载Nunchaku优化的Transformer")
        
        # 使用FluxKontextPipeline.from_pretrained方法加载完整pipeline
        full_model_path = os.path.join(shared.models_path, 'FLUX.1-Kontext-dev')
        
        # 检查是否存在量化T5模型
        quantized_t5_path = os.path.join(full_model_path, "awq-int4-flux.1-t5xxl.safetensors")
        text_encoder_2_path = os.path.join(full_model_path, "text_encoder_2")
        
        if os.path.exists(quantized_t5_path) and NUNCHAKU_T5_AVAILABLE:
            print(f"加载量化T5模型从路径: {quantized_t5_path}")
            try:
                # 使用NunchakuT5EncoderModel加载量化模型
                text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(quantized_t5_path)
                print("成功加载量化T5模型")
            except Exception as e:
                print(f"加载量化T5模型失败，回退到标准T5模型: {e}")
                import traceback
                traceback.print_exc()
                
                # 回退到标准T5模型加载方式
                if os.path.exists(text_encoder_2_path):
                    text_encoder_2 = T5EncoderModel.from_pretrained(
                        text_encoder_2_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
                    print("成功加载标准T5模型")
                else:
                    raise Exception(f"T5文本编码器路径不存在: {text_encoder_2_path}")
        elif os.path.exists(text_encoder_2_path):
            print(f"加载标准T5模型从路径: {text_encoder_2_path}")
            try:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    text_encoder_2_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                print("成功加载标准T5模型")
            except Exception as e:
                print(f"加载T5模型失败: {e}")
                import traceback
                traceback.print_exc()
                # 修改：提供更具体的错误信息
                raise Exception(f"无法加载T5文本编码器，请检查 {text_encoder_2_path} 目录中的文件是否完整: {str(e)}")
        else:
            # 修改：提供更具体的错误信息
            raise Exception(f"T5文本编码器路径不存在: {text_encoder_2_path}，请确保该目录包含T5模型文件")
        
        # 加载其他必要组件
        pipe = FluxKontextPipeline.from_pretrained(
            full_model_path,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16
        )
        
        # 启用CPU卸载
        if enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
            print("已启用Sequential CPU卸载")
        
        FLUX_KONTEXT_LOADED = True
        print("成功加载Nunchaku优化的FLUX.1-Kontext模型")
        return pipe
        
    except Exception as e:
        error_msg = f"加载Nunchaku模型时出错: {str(e)}"
        print(error_msg)
        # 不再回退到GGUF模型，直接返回None让调用方处理错误
        return None

def load_flux_kontext_model(selected_model="Nunchaku fp4", enable_cpu_offload=True):
    """加载FLUX.1-Kontext模型"""
    global SELECTED_MODEL, FLUX_KONTEXT_LOADED, pipe, LOADED_LORA, LOADED_LORA_WEIGHT
    
    try:
        print(f"正在加载模型: {selected_model}")
        print(f"CPU卸载: {enable_cpu_offload}")
        
        # 重置LoRA状态
        LOADED_LORA = None
        LOADED_LORA_WEIGHT = 0.0
        
        # 如果模型已加载且类型相同，检查是否需要重新加载
        if pipe is not None and SELECTED_MODEL == selected_model and FLUX_KONTEXT_LOADED:
            # 检查CPU卸载设置是否改变
            # 注意：这里简化处理，实际可能需要更复杂的逻辑
            print("模型已加载，跳过重复加载")
            return pipe
            
        # 清理现有模型
        if pipe is not None:
            del pipe
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            FLUX_KONTEXT_LOADED = False
            
        SELECTED_MODEL = selected_model
        
        # 修复模型选择逻辑，确保Nunchaku选项正确映射到对应的模型
        if selected_model.startswith("Nunchaku") and NUNCHAKU_AVAILABLE:
            print("使用Nunchaku优化的FLUX.1-Kontext模型")
            # 根据用户选择的精度加载对应的Nunchaku模型，并传递enable_cpu_offload参数
            precision = "int4" if "int4" in selected_model.lower() else "fp4"
            result = load_nunchaku_model_with_quantized_t5(enable_cpu_offload=enable_cpu_offload, precision=precision)
            
            # 如果Nunchaku模型加载成功，直接返回结果
            if result is not None:
                FLUX_KONTEXT_LOADED = True
                print("Nunchaku模型加载完成")
                return result
            else:
                # 修改：提供更具体的错误信息
                raise Exception("Nunchaku模型加载失败，请检查模型文件是否存在且完整，特别是 svdq-int4_r32-flux.1-kontext-dev.safetensors 或 svdq-fp4_r32-flux.1-kontext-dev.safetensors 文件")
        elif selected_model.startswith("Nunchaku") and not NUNCHAKU_AVAILABLE:
            # 如果Nunchaku不可用，则直接报错
            raise Exception("Nunchaku模型不可用，请安装Nunchaku库支持")
        
        # 移除了GGUF模型处理逻辑，只保留Nunchaku模型

    except Exception as e:
        # 修改：保持详细的错误信息
        error_msg = str(e)
        print(f"加载模型时出错: {error_msg}")
        FLUX_KONTEXT_LOADED = False
        # 直接抛出原始异常，而不是包装成通用信息
        raise e


def fix_model_device_consistency(pipe):
    """修复模型组件的设备一致性问题"""
    try:
        # 不需要手动处理设备一致性，使用pipeline自带的CPU卸载功能
        return None
    except Exception as e:
        print(f"修复模型设备一致性时出错: {e}")
        return None


def generate_edit_series(
    input_images,
    selected_edits, 
    seed=42, 
    randomize_seed=False, 
    guidance_scale=2.5, 
    num_inference_steps=10,
    enable_cpu_offload=True,
    model_type="Nunchaku fp4",
    **kwargs
):
    """生成一系列不同编辑变体，仅支持单图编辑"""
    global pipe
    
    if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
        pipe = load_flux_kontext_model(model_type, True)
        if pipe is None:
            raise gr.Error("模型加载失败，请检查模型文件是否完整")
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    if not input_images:
        raise gr.Error("请上传至少一张图像。")
    
    if not isinstance(input_images, list):
        input_images = [input_images]
    
    valid_images = []
    for img in input_images:
        if img is not None:
            valid_images.append(img)
    
    if len(valid_images) == 0:
        raise gr.Error("请上传有效的图像。")
    
    if len(valid_images) > MAX_INPUT_IMAGES:
        valid_images = valid_images[:MAX_INPUT_IMAGES]
        gr.Warning(f"仅使用前两张图像，最多支持{MAX_INPUT_IMAGES}张输入图像")
    
    input_images = valid_images
    
    edits_to_generate = []
    
    if selected_edits:
        edits_to_generate.extend(selected_edits)
    
    if not edits_to_generate:
        raise gr.Error("请选择至少一个编辑项。")
    
    edits_to_generate = list(dict.fromkeys(edits_to_generate))
    
    print("输入图像尺寸:")
    original_sizes = []
    for i, img in enumerate(input_images):
        width, height = img.size
        original_sizes.append((width, height))
        print(f"图像 {i+1}: {width}x{height}")
    
    # 使用原始图像尺寸，保持宽高比
    target_width, target_height = input_images[0].size
    
    print(f"图像目标尺寸: {target_width}x{target_height}")
    
    # 显示显存信息（仅在启用CUDA时）
    if torch.cuda.is_available():
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            free_memory = total_memory - reserved_memory
            print(f"总显存: {total_memory:.2f} GB")
            print(f"已保留显存: {reserved_memory:.2f} GB")
            print(f"已分配显存: {allocated_memory:.2f} GB")
            print(f"可用显存: {free_memory:.2f} GB")
        except Exception as e:
            print(f"获取显存信息失败: {e}")
    
    images_output = []
    seeds_used = []
    
    for i, input_image in enumerate(input_images):
        # 获取原始图像尺寸
        original_width, original_height = input_image.size
        print(f"图像 {i+1} 尺寸: {original_width}x{original_height}")
        
        current_seed = seed + i if not randomize_seed else random.randint(0, MAX_SEED)
        seeds_used.append(current_seed)
        
        for j, edit_prompt in enumerate(edits_to_generate):
            final_prompt = f"image editing variation {j+1}: {edit_prompt}, high quality, detailed, maintain original subject identity, professional photo"
            print(f"图像 {i+1} 第 {j+1} 个变体，使用的提示词: {final_prompt}")
            print(f"使用的种子: {current_seed}")
            
            try:
                # 在CPU上创建生成器，避免设备不一致问题
                generator = torch.Generator(device="cpu")
                generator.manual_seed(current_seed)
                
                # 使用torch.no_grad上下文管理器减少内存使用
                with torch.no_grad():
                    # 在调用pipeline时明确指定设备和原始图像尺寸
                    image = pipe(
                        image=input_image, 
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=original_width,
                        height=original_height
                    ).images[0]
                
                images_output.append(image)
                
            except RuntimeError as e:
                if "CUDA error: CUBLAS_STATUS_ALLOC_FAILED" in str(e) or "out of memory" in str(e).lower():
                    print("检测到CUDA内存分配失败...")
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(current_seed)
                        
                        image = pipe(
                            image=input_image, 
                            prompt=final_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            width=original_width,
                            height=original_height
                        ).images[0]
                        
                        images_output.append(image)
                    except Exception as fallback_error:
                        print(f"处理失败: {fallback_error}")
                        print("跳过当前编辑项并继续处理下一个")
                elif "Expected all tensors to be on the same device" in str(e):
                    print("检测到设备不一致错误...")
                    try:
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(current_seed)
                        
                        image = pipe(
                            image=input_image, 
                            prompt=final_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            width=original_width,
                            height=original_height
                        ).images[0]
                        
                        images_output.append(image)
                    except Exception as fallback_error:
                        print(f"修复设备不一致后仍然失败: {fallback_error}")
                        print("跳过当前编辑项并继续处理下一个")
                else:
                    raise e
            except Exception as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print("检测到设备不一致错误...")
                    try:
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(current_seed)
                        
                        image = pipe(
                            image=input_image, 
                            prompt=final_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            width=original_width,
                            height=original_height
                        ).images[0]
                        
                        images_output.append(image)
                    except Exception as fallback_error:
                        print(f"修复设备不一致后仍然失败: {fallback_error}")
                        print("跳过当前编辑项并继续处理下一个")
                else:
                    print(f"处理过程中出现未预期错误: {e}")
                    print("跳过当前编辑项并继续处理下一个")
    
    if not images_output:
        raise gr.Error("未能生成任何图像，请检查输入和参数设置。")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return images_output, seeds_used, timestamp


def create_kontext_edits(selected_edits_str):
    """根据选择的编辑类型创建编辑提示词列表"""
    if not selected_edits_str:
        return []
    
    # 将字符串转换为列表
    if isinstance(selected_edits_str, str):
        selected_edits = [edit.strip() for edit in selected_edits_str.split(',')]
    else:
        selected_edits = selected_edits_str
    
    # 创建最终的编辑列表
    edits = list(selected_edits)
    
    return edits


def generate_dual_context_image(
    image_1,
    selected_edits,
    seed=42,
    randomize_seed=False,
    guidance_scale=2.5,
    num_inference_steps=15,
    enable_cpu_offload=True,
    model_type="Nunchaku fp4"
):
    """
    生成基于单张图像的编辑变体
    
    Args:
        image_1: 输入图像
        selected_edits: 编辑指令列表
    """
    global pipe
    
    if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
        pipe = load_flux_kontext_model(model_type, True)
        if pipe is None:
            raise gr.Error("模型加载失败，请检查模型文件是否完整")
    
    if randomize_seed:
        base_seed = random.randint(0, MAX_SEED)
    else:
        base_seed = seed
    
    if image_1 is None:
        raise gr.Error("请上传图像。")
    
    edits_to_generate = []
    
    if selected_edits:
        edits_to_generate.extend(selected_edits)
    
    if not edits_to_generate:
        raise gr.Error("请选择至少一个编辑项。")
    
    edits_to_generate = list(dict.fromkeys(edits_to_generate))
    
    print("输入图像尺寸:")
    width, height = image_1.size
    print(f"图像 1: {width}x{height}")
    
    generated_images = []
    all_used_seeds = []
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 为每个编辑指令生成一个变体
    for i, edit in enumerate(edits_to_generate):
        if randomize_seed:
            current_seed = random.randint(0, MAX_SEED)
        else:
            current_seed = seed
        
        if edit.strip():
            final_prompt = f"image editing: {edit.strip()}, high quality, detailed, maintain original subject identity, professional photo"
        else:
            final_prompt = f"image editing, high quality, detailed, maintain original subject identity, professional photo"
        
        final_prompt = final_prompt[:200]
        
        print(f"图像编辑 第 {i+1} 个变体，使用的提示词: {final_prompt}")
        print(f"使用的种子: {current_seed}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            # 在CPU上创建生成器，避免设备不一致问题
            generator = torch.Generator(device="cpu")
            generator.manual_seed(current_seed)
            
            # 使用原始图像尺寸
            image = pipe(
                image=image_1,
                prompt=final_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                width=width,
                height=height
            ).images[0]
            
        except RuntimeError as e:
            if "CUDA error: CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                print("检测到CUDA内存分配失败...")
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(current_seed)
                    
                    image = pipe(
                        image=image_1,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=width,
                        height=height
                    ).images[0]
                except Exception as fallback_error:
                    print(f"处理失败: {fallback_error}")
                    print("跳过当前编辑项并继续处理下一个")
                    continue
            elif "Expected all tensors to be on the same device" in str(e):
                print("检测到设备不一致错误...")
                try:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(current_seed)
                    
                    image = pipe(
                        image=image_1,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=width,
                        height=height
                    ).images[0]
                    
                    generated_images.append(image)
                except Exception as fallback_error:
                    print(f"修复设备不一致后仍然失败: {fallback_error}")
                    print("跳过当前编辑项并继续处理下一个")
                    continue
            else:
                raise e
        except Exception as e:
            if "Expected all tensors to be on the same device" in str(e):
                print("检测到设备不一致错误...")
                try:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(current_seed)
                    
                    image = pipe(
                        image=image_1,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=width,
                        height=height
                    ).images[0]
                    
                    generated_images.append(image)
                except Exception as fallback_error:
                    print(f"修复设备不一致后仍然失败: {fallback_error}")
                    print("跳过当前编辑项并继续处理下一个")
                    continue
            else:
                print(f"处理过程中出现未预期错误: {e}")
                print("跳过当前编辑项并继续处理下一个")
                continue
        
        generated_images.append(image)
        all_used_seeds.append(current_seed)
        
        try:
            save_dir = os.path.join(shared.data_path, "outputs", "flux-kontext")
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"edited_image_{timestamp}_var{i+1}.png"
            save_path = os.path.join(save_dir, filename)
            
            image.save(save_path)
            print(f"生成的图像已保存到: {save_path}")
        except Exception as e:
            print(f"保存图像时发生错误: {e}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return generated_images, ", ".join(map(str, all_used_seeds))


def open_flux_output_dir():
    """打开FLUX输出目录"""
    output_dir = os.path.join(shared.data_path, "outputs", "flux-kontext")
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
        print(f"打开目录失败: {e}")


def create_flux_kontext_ui():
    """创建FLUX.1-Kontext UI界面"""
    # 使用Blocks包装所有UI元素
    with gr.Blocks() as flux_kontext_ui:
        with gr.Row():
            # 左半边：参数设置区域
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        dual_image_1 = gr.Image(
                            label="上传图像", 
                            type="pil",
                            height=300
                        )
                    
                    with gr.Row():
                        dual_model_choices = ["Nunchaku fp4 (50系）", "Nunchaku int4 (非50系）"] if NUNCHAKU_AVAILABLE else []
                        
                        if not dual_model_choices:
                            dual_model_choices = ["Nunchaku fp4 (50系）", "Nunchaku int4 (非50系）"]
                            # 如果Nunchaku不可用，显示错误信息
                            print("警告：Nunchaku库不可用，但仍显示在选项中")
                    
                        dual_model_type = gr.Dropdown(
                            label="模型选择",
                            choices=dual_model_choices,
                            value="Nunchaku fp4 (50系）" if NUNCHAKU_AVAILABLE else "Nunchaku fp4 (50系）",
                            info="Nunchaku提供更好的性能和更低的显存需求。fp4为浮点4位量化，int4为整数4位量化。"
                        )
                    
                    with gr.Row():
                        dual_lora_enable = gr.Checkbox(
                            label="启用LoRA",
                            value=False,
                            info="启用LoRA模型以修改生成风格"
                        )
                        dual_lora_model = gr.Dropdown(
                            label="LoRA模型选择",
                            choices=list_lora_models(),
                            value=list_lora_models()[0] if list_lora_models() else "",
                            interactive=False  # 默认不可交互
                        )
                        
                    with gr.Row():
                        dual_lora_weight = gr.Number(
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
                    
                    dual_lora_enable.change(
                        fn=update_lora_interactive,
                        inputs=dual_lora_enable,
                        outputs=dual_lora_model
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
                        outputs=dual_lora_model
                    )
                    
                    with gr.Group():
                        gr.Markdown("**编辑设置**")
                        edit_textbox = gr.Textbox(
                            label="编辑指令",
                            placeholder="请输入编辑指令，例如：在图像右侧添加一只猫",
                            lines=3,
                            max_lines=5
                        )

                        # 添加多角度提示词可视化选择器（如果可用）
                        if ANGLE_SELECTOR_AVAILABLE:
                            with gr.Accordion("多角度提示词可视化选择器", open=False):
                                angle_selector_component = create_angle_visualization_component(edit_textbox)

                    with gr.Group():
                        with gr.Row():
                            dual_seed = gr.Number(
                                label="随机种子",
                                value=0,
                                precision=0,
                                info="设置随机种子以获得可重现的结果，0表示随机"
                            )
                        
                        with gr.Row():
                            dual_guidance_scale = gr.Slider(
                                label="CFG引导数",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.1,
                                value=3.5,
                                info="控制生成图像与提示词的一致性，数值越高越严格遵循提示词"
                            )
                            
                            dual_num_inference_steps = gr.Slider(
                                label="推理步数",
                                minimum=10,
                                maximum=50,
                                step=1,
                                value=20,
                                info="控制生成图像的质量和计算时间"
                            )
            
            # 右半边：生成相关控件区域
            with gr.Column(scale=1):
                # 生成按钮
                dual_generate_button = gr.Button("生成图像", variant="primary", size="lg")
                
                # 生成结果展示
                dual_generated_gallery = gr.Gallery(
                    label="生成结果",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    object_fit="contain",
                    height="auto",
                    preview=True
                )
                
                # 添加到分镜按钮
                if STORYBOARD_AVAILABLE:
                    with gr.Row():
                        send_to_storyboard_btn = gr.Button(
                            "📤 发送到分镜",
                            variant="secondary",
                            visible=True
                        )
                    send_status = gr.Textbox(label="发送状态", interactive=False, visible=True)
                
                # 种子信息
                dual_seed_info = gr.Textbox(
                    label="使用的种子",
                    interactive=False,
                    lines=1
                )
                
                # 打开输出目录按钮
                def open_flux_output_dir():
                    """打开FLUX输出目录"""
                    output_dir = os.path.join(shared.data_path, "outputs", "flux-kontext")
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
                        print(f"打开目录失败: {e}")
                
                open_output_dir_button = gr.Button("打开输出目录")
                open_output_dir_button.click(
                    fn=open_flux_output_dir,
                    inputs=[],
                    outputs=[]
                )

        # 定义生成图像的处理函数
        def on_generate_edit_series(*args):
            # 解析参数
            image = args[0]
            edit_prompt = args[1]
            seed = args[2]
            guidance_scale = args[3]
            num_inference_steps = args[4]
            model_type = args[5]
            lora_enable = args[6]
            lora_model = args[7]
            lora_weight = args[8]
            
            if image is None:
                return [], "请上传图像"
            
            if not edit_prompt:
                return [], "请提供编辑指令"
            
            try:
                # 检查必要依赖是否可用
                if not DIFFUSERS_AVAILABLE:
                    raise RuntimeError("缺少必要的依赖库: diffusers")
                
                # 加载模型，默认启用CPU卸载
                global pipe
                pipe = load_flux_kontext_model(model_type, enable_cpu_offload=True)
                
                # 如果启用了LoRA，加载LoRA模型
                if lora_enable and lora_model:
                    try:
                        # 获取transformer对象
                        transformer = pipe.transformer
                        
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
                
                # 生成图像
                # 修复参数传递问题，将image包装成列表，edit_prompt包装成列表
                images, used_seeds, timestamp = generate_edit_series(
                    input_images=[image],  # 修正参数名和格式
                    selected_edits=[edit_prompt],  # 修正参数名和格式
                    seed=seed,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    enable_cpu_offload=True,  # 默认启用CPU卸载
                    model_type=model_type
                )
                
                # 使用第一个种子作为显示种子
                seed_text = f"使用的种子: {used_seeds[0] if used_seeds else seed}"
                print(f"返回 {len(images)} 张图像到结果展示组件")
                for i, img in enumerate(images):
                    print(f"图像 {i+1}: 类型={type(img)}, 尺寸={img.size if hasattr(img, 'size') else 'N/A'}")
                
                return images, seed_text
            
            except Exception as e:
                error_msg = str(e)
                print(f"生成图像时出错: {error_msg}")
                return [], f"生成失败: {error_msg}"

        # 绑定双图像编辑生成按钮事件
        dual_generate_button.click(
            fn=on_generate_edit_series,
            inputs=[
                dual_image_1, 
                edit_textbox,
                dual_seed, 
                dual_guidance_scale, 
                dual_num_inference_steps,
                dual_model_type,
                dual_lora_enable,
                dual_lora_model,
                dual_lora_weight
            ], 
            outputs=[dual_generated_gallery, dual_seed_info]
        )
        
        # 发送到分镜功能
        if STORYBOARD_AVAILABLE:
            def send_kontext_gallery_to_storyboard(images):
                """将画廊中的图片发送到分镜助手"""
                if not images or len(images) == 0:
                    return "❌ 没有可发送的图片"
                
                messages = []
                last_index = -1
                last_target_page = 1
                
                for img in images:
                    # PIL Image 需要保存为临时文件
                    import tempfile
                    if hasattr(img, 'save'):
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            img.convert('RGB').save(tmp, quality=85)
                            img_path = tmp.name
                    else:
                        img_path = img
                    
                    result = send_to_storyboard(img_path)
                    success = result.get('success', False)
                    message = result.get('message', '')
                    index = result.get('index', -1)
                    last_target_page = result.get('target_page', 1)
                    
                    if success:
                        messages.append(f"✅ {message}")
                        last_index = index
                    else:
                        messages.append(f"❌ {message}")
                
                if last_index >= 0:
                    return f"已处理 {len(images)} 张图片，最后添加到分镜 #{last_index + 1}（第 {last_target_page} 页）"
                else:
                    return "处理失败，请查看控制台日志"
            
            send_to_storyboard_btn.click(
                fn=send_kontext_gallery_to_storyboard,
                inputs=[dual_generated_gallery],
                outputs=[send_status]
            )
        
        print("FLUX.1-Kontext UI 创建完成")

    # 返回包装好的UI容器
    return flux_kontext_ui


FLUX_KONTEXT_AVAILABLE = True
