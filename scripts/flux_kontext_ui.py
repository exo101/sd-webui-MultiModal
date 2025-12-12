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
    print("Nunchaku库已找到，将使用优化的FLUX.1-Kontext模型")
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

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast
)


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

def list_lora_models():
    """列出所有可用的LoRA模型"""
    lora_dir = os.path.join(shared.models_path, "Lora")
    
    if not os.path.exists(lora_dir):
        return []
    
    lora_files = []
    for file in os.listdir(lora_dir):
        if file.endswith(".safetensors") or file.endswith(".pt") or file.endswith(".bin"):
            lora_files.append(file)
    
    return lora_files

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


def load_nunchaku_model(enable_cpu_offload=False, precision=None):
    """加载Nunchaku优化的FLUX.1-Kontext模型"""
    global pipe, FLUX_KONTEXT_LOADED
    
    try:
        if precision is None:
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
                # 尝试直接查找模型文件
                direct_model_filenames = [
                    'svdq-fp4_r32-flux.1-kontext-dev.safetensors',
                    'svdq-int4_r32-flux.1-kontext-dev.safetensors'
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
        
        # 使用正确的Nunchaku模型加载方式，启用offload参数
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            model_path, 
            offload=enable_cpu_offload  # 根据enable_cpu_offload参数决定是否启用offload
        )
        print("成功加载Nunchaku优化的Transformer")
        
        # 使用FluxKontextPipeline.from_pretrained方法加载完整pipeline
        full_model_path = os.path.join(shared.models_path, 'FLUX.1-Kontext-dev')
        
        # 加载T5文本编码器并确保它在CPU上
        text_encoder_2_path = os.path.join(full_model_path, "text_encoder_2")
        if os.path.exists(text_encoder_2_path):
            print(f"加载T5模型从路径: {text_encoder_2_path}")
            try:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    text_encoder_2_path, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map={"": "cpu"},  # 强制T5编码器在CPU上运行
                )
                print("成功加载T5模型")
            except Exception as e:
                print(f"加载T5模型失败: {e}")
                import traceback
                traceback.print_exc()
                raise Exception("无法加载T5文本编码器")
        else:
            raise Exception("T5文本编码器路径不存在")
        
        # 加载其他必要组件
        pipe = FluxKontextPipeline.from_pretrained(
            full_model_path, 
            transformer=transformer,
            text_encoder_2=text_encoder_2,  # 明确指定T5编码器
            torch_dtype=torch.bfloat16
        )
        
        # 根据enable_cpu_offload参数决定是否启用CPU卸载
        if enable_cpu_offload:
            # 使用Nunchaku推荐的sequential CPU offload方式
            pipe.enable_sequential_cpu_offload()
            print("已启用Sequential CPU卸载")
        else:
            # 只有在不启用CPU卸载时才移动到CUDA设备
            try:
                pipe = pipe.to("cuda")
                print("模型已移动到CUDA设备")
            except Exception as e:
                print(f"将模型移动到CUDA时出错: {e}")
                if hasattr(pipe, 'enable_sequential_cpu_offload'):
                    pipe.enable_sequential_cpu_offload()
                    print("改为启用Sequential CPU卸载")
                else:
                    print("无法启用CPU卸载，模型可能无法正常运行")
        
        # 强制将T5编码器固定在CPU上
        try:
            if hasattr(pipe, 'text_encoder_2'):
                # 检查是否在meta设备上，如果是则使用特殊方法移动
                if next(pipe.text_encoder_2.parameters()).device == torch.device('meta'):
                    # 使用to_empty方法处理meta设备上的模型
                    pipe.text_encoder_2 = pipe.text_encoder_2.to_empty(device="cpu")
                else:
                    pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
                print("T5文本编码器已固定在CPU上")
        except Exception as e:
            print(f"将T5文本编码器固定在CPU上时出错: {e}")
        
        FLUX_KONTEXT_LOADED = True
        print("成功加载Nunchaku优化的FLUX.1-Kontext模型")
        return pipe
        
    except Exception as e:
        error_msg = f"加载Nunchaku模型时出错: {str(e)}"
        print(error_msg)
        # 不再回退到GGUF模型，直接返回None让调用方处理错误
        return None

def load_flux_kontext_model(selected_model="Q2_K", enable_cpu_offload=False):
    """加载FLUX.1-Kontext GGUF模型"""
    global pipe, FLUX_KONTEXT_LOADED, SELECTED_MODEL
    
    # 检查必要依赖
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("无法加载模型：缺少必要的diffusers库依赖")
    
    if pipe is not None and SELECTED_MODEL == selected_model and FLUX_KONTEXT_LOADED:
        print("使用已缓存的模型")
        return pipe
    
    try:
        if pipe is not None:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            pipe = None
            FLUX_KONTEXT_LOADED = False
            
        SELECTED_MODEL = selected_model
        
        # 修复模型选择逻辑，确保Nunchaku选项正确映射到对应的模型
        if selected_model.startswith("Nunchaku") and NUNCHAKU_AVAILABLE:
            print("使用Nunchaku优化的FLUX.1-Kontext模型")
            # 根据用户选择的精度加载对应的Nunchaku模型
            if selected_model == "Nunchaku int4":
                # 直接使用int4精度
                result = load_nunchaku_model(enable_cpu_offload, "int4")
            else:
                # 默认使用fp4精度或其他Nunchaku选项
                result = load_nunchaku_model(enable_cpu_offload, "fp4")
            
            # 如果Nunchaku模型加载成功，直接返回结果
            if result is not None:
                FLUX_KONTEXT_LOADED = True
                print("Nunchaku模型加载完成")
                return result
            else:
                # 如果Nunchaku模型加载失败，直接报错而不是回退到GGUF模型
                raise Exception("Nunchaku模型加载失败，请检查模型文件是否存在且完整")
        elif selected_model.startswith("Nunchaku") and not NUNCHAKU_AVAILABLE:
            # 如果Nunchaku不可用，则直接报错
            raise Exception("Nunchaku模型不可用，请安装Nunchaku库支持")
        
        # 移除了GGUF模型处理逻辑，只保留Nunchaku模型
        raise Exception("GGUF模型支持已被移除，请使用Nunchaku模型")
        
        # 只有在需要加载GGUF模型时才执行下面的代码
        model_filename = GGUF_FILENAMES.get(selected_model, GGUF_FILENAMES["Q2_K"])
        model_path = find_existing_model(model_filename)
            
        print(f"选择的模型: {selected_model}")
        print(f"模型文件名: {model_filename}")
        print(f"找到的模型路径: {model_path}")
        
        if not model_path:
            raise Exception(f"模型文件 {model_filename} 不存在，请确保已下载模型文件到正确目录")
        
        print(f"正在加载模型: {model_path}")
        
        full_model_path = os.path.join(
            shared.models_path,
            'FLUX.1-Kontext-dev'
        )
        
        print(f"完整模型路径: {full_model_path}")
        print(f"完整模型路径是否存在: {os.path.exists(full_model_path)}")
        print(f"model_index.json是否存在: {os.path.exists(os.path.join(full_model_path, 'model_index.json'))}")
        
        scheduler_path = os.path.join(full_model_path, "scheduler")
        if os.path.exists(scheduler_path):
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
            print("成功加载调度器")
        else:
            scheduler = FlowMatchEulerDiscreteScheduler()
            print("使用默认调度器")
        
        vae_path = os.path.join(full_model_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
            print("成功加载VAE")
        else:
            raise Exception("VAE模型不存在")
        
        text_encoder_path = os.path.join(full_model_path, "text_encoder")
        if os.path.exists(text_encoder_path):
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
            print("成功加载CLIP文本编码器")
        else:
            raise Exception("CLIP文本编码器不存在")
        
        tokenizer_path = os.path.join(full_model_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            print("成功加载CLIP分词器")
        else:
            raise Exception("CLIP分词器不存在")
        
        # 加载T5文本编码器和分词器并确保它在CPU上
        text_encoder_2_path = os.path.join(full_model_path, "text_encoder_2")
        tokenizer_2_path = os.path.join(full_model_path, "tokenizer_2")
        if os.path.exists(text_encoder_2_path) and os.path.exists(tokenizer_2_path):
            print(f"加载T5模型从路径: {text_encoder_2_path}")
            try:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    text_encoder_2_path, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map={"": "cpu"},  # 强制T5编码器在CPU上运行
                )
                print("成功加载T5模型")
                
                # 加载T5分词器
                tokenizer_2 = T5TokenizerFast.from_pretrained(tokenizer_2_path)
                print("成功加载T5分词器")
            except Exception as e:
                print(f"加载T5模型失败: {e}")
                import traceback
                traceback.print_exc()
                raise Exception("无法加载T5文本编码器")
        else:
            raise Exception("T5文本编码器或分词器路径不存在")
        
        # 检查是否是GGUF量化模型
        try:
            # 尝试不带参数初始化GGUFQuantizationConfig
            gguf_config = GGUFQuantizationConfig()
        except Exception as e:
            # 如果不带参数初始化失败，则不使用量化配置
            gguf_config = None
            print(f"GGUFQuantizationConfig初始化失败: {e}")

        # 加载Transformer模型
        transformer_path = os.path.join(full_model_path, "transformer")
        if os.path.exists(transformer_path):
            try:
                # 尝试加载GGUF格式的Transformer
                transformer_kwargs = {
                    "torch_dtype": torch.bfloat16
                }
                # 只有在gguf_config有效时才添加quantization_config参数
                if gguf_config is not None:
                    transformer_kwargs["quantization_config"] = gguf_config
                    
                transformer = FluxTransformer2DModel.from_pretrained(
                    transformer_path,
                    **transformer_kwargs
                )
                print("成功加载GGUF Transformer")
            except Exception as e:
                print(f"GGUF Transformer加载失败: {e}")
                # 回退到普通加载方式
                try:
                    transformer = FluxTransformer2DModel.from_pretrained(
                        transformer_path,
                        torch_dtype=torch.bfloat16
                    )
                    print("成功加载普通Transformer")
                except Exception as e2:
                    print(f"普通Transformer加载也失败: {e2}")
                    raise Exception("Transformer模型加载失败")
        else:
            raise Exception("Transformer模型不存在")
        
        # 构建pipeline
        try:
            pipe = FluxKontextPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
            )
            
            # 根据enable_cpu_offload参数决定是否启用CPU卸载
            if enable_cpu_offload:
                if hasattr(pipe, 'enable_sequential_cpu_offload'):
                    pipe.enable_sequential_cpu_offload()
                    print("已启用Sequential CPU卸载")
                elif hasattr(pipe, 'enable_model_cpu_offload'):
                    pipe.enable_model_cpu_offload()
                    print("已启用Model CPU卸载")
                else:
                    print("警告: 模型不支持CPU卸载")
            else:
                # 只有在不启用CPU卸载时才移动到CUDA设备
                try:
                    pipe = pipe.to("cuda")
                    print("模型已移动到CUDA设备")
                except Exception as e:
                    print(f"将模型移动到CUDA时出错: {e}")
                    if hasattr(pipe, 'enable_sequential_cpu_offload'):
                        pipe.enable_sequential_cpu_offload()
                        print("改为启用Sequential CPU卸载")
                    else:
                        print("无法启用CPU卸载，模型可能无法正常运行")
        except Exception as e:
            print(f"模型移动到CUDA失败: {e}")
            # 回退到CPU模式
            pipe = FluxKontextPipeline(
                scheduler=scheduler,
                vae=vae.to("cpu"),
                text_encoder=text_encoder.to("cpu"),
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2.to("cpu"),
                tokenizer_2=tokenizer_2,
                transformer=transformer.to("cpu"),
            )
            pipe.enable_model_cpu_offload()
            print("改为启用CPU卸载")
        
        # 强制将T5编码器固定在CPU上，避免其试图使用GPU
        try:
            if hasattr(pipe, 'text_encoder_2'):
                pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
                print("T5文本编码器已固定在CPU上")
        except Exception as e:
            print(f"将T5文本编码器固定在CPU上时出错: {e}")
        
        try:
            pipe.vae.enable_slicing()
            print("已启用VAE切片")
        except Exception as e:
            print(f"启用VAE切片失败: {e}")
            
        try:
            pipe.vae.enable_tiling()
            print("已启用VAE平铺")
        except Exception as e:
            print(f"启用VAE平铺失败: {e}")
        
        if LOADED_LORA is not None and os.path.exists(LOADED_LORA):
            load_lora_weights(pipe, LOADED_LORA, LOADED_LORA_WEIGHT)
        
        fix_model_device_consistency(pipe)
        
        FLUX_KONTEXT_LOADED = True
        print("模型加载完成")
        return pipe
        
    except Exception as e:
        error_msg = f"加载 GGUF 管道时出错: {e}"
        print(error_msg)
        FLUX_KONTEXT_LOADED = False
        return None


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
    enable_cpu_offload=False,
    model_type="Q2_K",
    **kwargs  # 添加这一行以接受额外参数
):
    """生成一系列不同编辑变体，支持单图或双图编辑"""
    global pipe
    
    if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
        pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
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
    
    # 显示显存信息
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        free_memory = total_memory - reserved_memory
        print(f"总显存: {total_memory:.2f} GB")
        print(f"已保留显存: {reserved_memory:.2f} GB")
        print(f"已分配显存: {allocated_memory:.2f} GB")
        print(f"可用显存: {free_memory:.2f} GB")
    
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
                # 根据是否启用CPU卸载来选择设备
                device = "cpu" if enable_cpu_offload else ("cuda" if torch.cuda.is_available() else "cpu")
                generator = torch.Generator(device=device)
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
                    print("检测到CUDA内存分配失败，尝试启用CPU卸载...")
                    try:
                        pipe.enable_model_cpu_offload()
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(current_seed)
                        
                        # 强制将文本编码器组件移到CPU
                        if hasattr(pipe, 'text_encoder'):
                            pipe.text_encoder = pipe.text_encoder.to("cpu")
                        if hasattr(pipe, 'text_encoder_2'):
                            pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
                        
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
                        print(f"启用CPU卸载后仍然失败: {fallback_error}")
                        print("跳过当前编辑项并继续处理下一个")
                else:
                    raise e
            except torch.cuda.OutOfMemoryError:
                print("出现内存不足，尝试启用CPU卸载...")
                try:
                    pipe.enable_model_cpu_offload()
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(current_seed)
                    
                    # 强制将文本编码器组件移到CPU
                    if hasattr(pipe, 'text_encoder'):
                        pipe.text_encoder = pipe.text_encoder.to("cpu")
                    if hasattr(pipe, 'text_encoder_2'):
                        pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
                    
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
                except Exception as resize_error:
                    print(f"启用CPU卸载后仍然失败: {resize_error}")
                    print("跳过当前编辑项并继续处理下一个")
            except Exception as e:
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
    edits = []
    for edit in selected_edits:
        if edit in DEFAULT_EDIT_OPTIONS:
            edits.append(edit)
        elif edit in STYLE_EDIT_OPTIONS:
            edits.append(edit)
        elif edit in COLOR_EDIT_OPTIONS:
            edits.append(edit)
        elif edit in COMPOSITION_EDIT_OPTIONS:
            edits.append(edit)
        elif edit in EFFECT_EDIT_OPTIONS:
            edits.append(edit)
        elif edit in SCENE_EDIT_OPTIONS:
            edits.append(edit)
        elif edit in QUALITY_EDIT_OPTIONS:
            edits.append(edit)
        else:
            # 自定义编辑
            edits.append(edit)
    
    return edits


def generate_dual_context_image(
    image_1,
    image_2,
    selected_edits,
    seed=42,
    randomize_seed=False,
    guidance_scale=2.5,
    num_inference_steps=15,
    enable_cpu_offload=False,
    model_type="Q2_K"
):
    """
    生成融合两张图像上下文的新图像
    
    Args:
        image_1: 第一张参考图像
        image_2: 第二张参考图像
        selected_edits: 编辑指令列表
        其他参数与generate_edit_series相同
    """
    global pipe
    
    if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
        pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
        if pipe is None:
            raise gr.Error("模型加载失败，请检查模型文件是否完整")
    
    # 移除重复的设备一致性检查，因为load_flux_kontext_model已经处理了
    
    if randomize_seed:
        base_seed = random.randint(0, MAX_SEED)
    else:
        base_seed = seed
    
    if image_1 is None or image_2 is None:
        raise gr.Error("请上传两张图像。")
    
    edits_to_generate = []
    
    if selected_edits:
        edits_to_generate.extend(selected_edits)
    
    if not edits_to_generate:
        raise gr.Error("请选择至少一个编辑项。")
    
    edits_to_generate = list(dict.fromkeys(edits_to_generate))
    
    print("输入图像尺寸:")
    original_sizes = []
    for i, img in enumerate([image_1, image_2]):
        width, height = img.size
        original_sizes.append((width, height))
        print(f"图像 {i+1}: {width}x{height}")
    
    # 移除手动计算目标尺寸的代码，让模型自己处理尺寸
    
    generated_images = []
    all_used_seeds = []
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - reserved_memory - allocated_memory
        
        print(f"总显存: {total_memory / 1024**3:.2f} GB")
        print(f"已保留显存: {reserved_memory / 1024**3:.2f} GB")
        print(f"已分配显存: {allocated_memory / 1024**3:.2f} GB")
        print(f"可用显存: {free_memory / 1024**3:.2f} GB")
    
    # 创建组合图像，但不改变原始图像的尺寸
    # 获取两个图像的尺寸
    width_1, height_1 = image_1.size
    width_2, height_2 = image_2.size
    
    # 创建一个新的图像，高度为两个图像高度之和，宽度为两个图像中较大的那个
    combined_width = max(width_1, width_2)
    combined_height = height_1 + height_2
    combined_image = Image.new('RGB', (combined_width, combined_height))
    
    # 将图像粘贴到组合图像上
    # 第一张图像放在顶部
    offset_x_1 = (combined_width - width_1) // 2  # 居中放置
    combined_image.paste(image_1, (offset_x_1, 0))
    
    # 第二张图像放在底部
    offset_x_2 = (combined_width - width_2) // 2  # 居中放置
    combined_image.paste(image_2, (offset_x_2, height_1))
    
    for i, edit in enumerate(edits_to_generate):
        if randomize_seed:
            current_seed = random.randint(0, MAX_SEED)
        else:
            current_seed = seed
        
        if edit.strip():
            final_prompt = f"Combine the context from both reference images with the following edit: {edit.strip()}, high quality, detailed, maintain original subject identity, professional photo"
        else:
            final_prompt = f"Combine the context from both reference images, high quality, detailed, maintain original subject identity, professional photo"
        
        final_prompt = final_prompt[:200]
        
        print(f"融合图像 第 {i+1} 个变体，使用的提示词: {final_prompt}")
        print(f"使用的种子: {current_seed}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() and not enable_cpu_offload else "cpu")
            generator.manual_seed(current_seed)
            
            # 不传递width和height参数，让模型自己处理尺寸
            image = pipe(
                image=combined_image,
                prompt=final_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            
        except RuntimeError as e:
            if "CUDA error: CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                print("检测到CUDA内存分配失败，尝试启用CPU卸载...")
                try:
                    pipe.enable_model_cpu_offload()
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(current_seed)
                    
                    image = pipe(
                        image=combined_image,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    ).images[0]
                except Exception as fallback_error:
                    print(f"启用CPU卸载后仍然失败: {fallback_error}")
                    print("跳过当前编辑项并继续处理下一个")
            else:
                raise e
        except torch.cuda.OutOfMemoryError:
            print("出现内存不足，尝试启用CPU卸载...")
            try:
                pipe.enable_model_cpu_offload()
                gc.collect()
                torch.cuda.empty_cache()
                
                generator = torch.Generator(device="cpu")
                generator.manual_seed(current_seed)
                
                image = pipe(
                        image=combined_image,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    ).images[0]
            except Exception as e:
                print(f"启用CPU卸载后仍然失败: {e}")
                print("跳过当前编辑项并继续处理下一个")
        except Exception as e:
            print(f"处理过程中出现未预期错误: {e}")
            print("跳过当前编辑项并继续处理下一个")
            continue
        
        generated_images.append(image)
        all_used_seeds.append(current_seed)
        
        try:
            save_dir = os.path.join(shared.data_path, "outputs", "flux-kontext")
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"dual_context_image_{timestamp}_var{i+1}.png"
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
                        
                        dual_enable_cpu_offload = gr.Checkbox(
                            label="启用CPU卸载 (节省显存)",
                            value=False,
                            info="将部分模型组件移动到CPU以节省显存，但会降低推理速度。如果出现显存不足错误，请启用此选项。"
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
                        dual_lora_weight = gr.Slider(
                            label="LoRA权重",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
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
            enable_cpu_offload = args[6]
            lora_enable = args[7]
            lora_model = args[8]
            lora_weight = args[9]
            
            if image is None:
                return [], "请上传图像"
            
            if not edit_prompt:
                return [], "请提供编辑指令"
            
            try:
                # 检查必要依赖是否可用
                if not DIFFUSERS_AVAILABLE:
                    raise RuntimeError("缺少必要的依赖库: diffusers")
                
                # 加载模型
                global pipe
                pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
                
                # 如果启用了LoRA，加载LoRA模型
                if lora_enable and lora_model:
                    # 注意：这里需要根据实际情况实现LoRA加载逻辑
                    print(f"LoRA功能暂未完全实现: {lora_model}")
                
                # 生成图像
                # 修复参数传递问题，将image包装成列表，edit_prompt包装成列表
                images, used_seeds, timestamp = generate_edit_series(
                    input_images=[image],  # 修正参数名和格式
                    selected_edits=[edit_prompt],  # 修正参数名和格式
                    seed=seed,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    enable_cpu_offload=enable_cpu_offload,
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
                # 移除dual_randomize_seed参数
                dual_guidance_scale, 
                dual_num_inference_steps,
                dual_model_type,
                dual_enable_cpu_offload,
                dual_lora_enable,
                dual_lora_model,
                dual_lora_weight
            ], 
            outputs=[dual_generated_gallery, dual_seed_info]
        )
        
        print("FLUX.1-Kontext UI 创建完成")

    # 返回包装好的UI容器
    return flux_kontext_ui


FLUX_KONTEXT_AVAILABLE = True
