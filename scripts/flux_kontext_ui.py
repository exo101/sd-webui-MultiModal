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

# 添加缺失的导入
from diffusers import (
    FluxKontextPipeline, 
    FluxTransformer2DModel, 
    GGUFQuantizationConfig,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)

# 尝试导入LoRA支持相关模块
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

# 模型文件映射
GGUF_FILENAMES = {
    "Q2_K": "flux1-kontext-dev-Q2_K.gguf",
    "Q4_K_S": "flux1-kontext-dev-Q4_K_S.gguf",
    "Q5_K_M": "flux1-kontext-dev-Q5_K_M.gguf",
    "Q6_K": "flux1-kontext-dev-Q6_K.gguf",
    "Q8_0": "flux1-kontext-dev-Q8_0.gguf"
}

# 保存编辑配置的文件名
POSE_CONFIG_FILENAME = "saved_edits.txt"

# 定义常量
MAX_SEED = np.iinfo(np.int32).max

# 全局变量
FLUX_KONTEXT_LOADED = False
pipe = None
SELECTED_MODEL = "Q2_K"  # 默认选择Q2_K模型（参数量最小，适合测试）
text_encoder_2_cache = None  # 添加T5模型缓存

# LoRA相关全局变量
LOADED_LORA = None
LOADED_LORA_WEIGHT = 0.0

def list_lora_models():
    """列出所有可用的LoRA模型"""
    # 修改LoRA路径为WebUI的models/Lora目录
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
        # 构建完整的LoRA文件路径
        full_lora_path = os.path.join(shared.models_path, "Lora", lora_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_lora_path):
            print(f"LoRA文件不存在: {full_lora_path}")
            return False
        
        # 使用diffusers的load_lora_weights方法加载LoRA权重
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

# 添加一个新的函数来处理LoRA加载前的FP8检查
def prepare_model_for_lora(pipe):
    """在加载LoRA之前准备模型"""
    # 移除了FP8检查，使用标准的模型处理方式
    return True

def process_uploaded_files(files):
    """处理上传的文件，返回文件列表和图像预览"""
    # 确保files是列表类型
    if files is None:
        files = []
    elif not isinstance(files, list):
        files = [files]
        
    print(f"处理 {len(files)} 个文件")
    
    # 从文件对象中提取图像用于预览
    valid_files = []
    for file in files:
        try:
            # 确保文件对象有效
            if hasattr(file, 'name') and file.name and os.path.exists(file.name):
                valid_files.append(file)
                print(f"成功加载图像: {file.name}")
            else:
                print(f"跳过无效文件: {file}")
        except Exception as e:
            print(f"处理文件时出错: {e}")
            # 即使单个文件出错，也要保留有效的文件
            if hasattr(file, 'name'):
                print(f"文件路径: {file.name}")
            pass
    
    print(f"有效文件数: {len(valid_files)}")
    
    # 直接返回预览图像，不添加序号标识
    preview_data = []
    for i, file in enumerate(valid_files):
        try:
            img = Image.open(file.name)
            preview_data.append(img)
        except Exception as e:
            print(f"处理预览图像时出错: {e}")
            # 如果处理标记失败，直接使用原图
            try:
                img = Image.open(file.name)
                preview_data.append(img)
            except Exception:
                pass  # 跳过无法打开的图像
    
    # 返回预览数据，保持上传组件可见并可以继续上传
    return preview_data


def process_images_for_inference(input_files):
    """处理输入图像文件"""
    # 确保input_files是列表类型
    if input_files is None:
        input_files = []
    elif not isinstance(input_files, list):
        input_files = [input_files]
    
    # 如果没有上传文件，则检查outputs目录中的图像
    if len(input_files) == 0:
        # 设置输入图像路径为WebUI的outputs目录
        outputs_dir = os.path.join(shared.data_path, "outputs")
        if os.path.exists(outputs_dir):
            # 获取outputs目录中的图像文件
            for root, dirs, files in os.walk(outputs_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_path = os.path.join(root, file)
                        input_files.append(file_path)
                        if len(input_files) >= 4:  # 限制最多4张图像
                            break
                if len(input_files) >= 4:
                    break
    
    if len(input_files) == 0:
        raise gr.Error("请至少上传一张图像或确保outputs目录中有图像文件")
    
    # 处理输入图像文件
    valid_images = []
    for file_obj in input_files:
        try:
            # 处理文件对象或文件路径
            if hasattr(file_obj, 'name') and file_obj.name:
                file_path = file_obj.name  # Gradio文件对象
            elif isinstance(file_obj, str):
                file_path = file_obj  # 文件路径字符串
            else:
                continue  # 其他情况
                
            if file_path is not None and os.path.exists(file_path):
                img = Image.open(file_path)
                valid_images.append(img)
        except Exception as e:
            print(f"处理图像文件时出错: {e}")
            pass
    
    if not valid_images:
        raise gr.Error("请至少上传一张有效图像")
    
    # 返回原始图像，让管道自动处理尺寸
    return valid_images, None, None


# ==================== 模型管理 ====================

def find_existing_model(filename):
    """在固定路径中查找现有模型文件"""
    try:
        # 确保shared.models_path有效
        if not shared.models_path or not os.path.exists(shared.models_path):
            print(f"警告: shared.models_path 无效或不存在: {shared.models_path}")
            return None
            
        # 构造模型目录路径
        model_dir = os.path.join(shared.models_path, 'FLUX.1-Kontext-dev')
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            return None
            
        # 构造完整的模型文件路径
        model_path = os.path.join(model_dir, filename)
        
        # 检查文件是否存在
        if os.path.exists(model_path):
            print(f"找到模型文件: {model_path}")
            return model_path
        else:
            print(f"模型文件不存在: {model_path}")
            return None
            
    except Exception as e:
        print(f"查找模型文件时出错: {e}")
        return None


def load_flux_kontext_model(selected_model="Q2_K", enable_cpu_offload=False):
    """加载FLUX.1-Kontext GGUF模型"""
    global pipe, FLUX_KONTEXT_LOADED, SELECTED_MODEL
    
    # 检查是否可以使用缓存的模型
    if pipe is not None and SELECTED_MODEL == selected_model and FLUX_KONTEXT_LOADED:
        print("使用已缓存的模型")
        return pipe
    
    try:
        # 清理现有模型（仅在需要重新加载时）
        if pipe is not None:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            pipe = None
            FLUX_KONTEXT_LOADED = False
            
        # 更新选择的模型
        SELECTED_MODEL = selected_model
        
        # 获取模型文件路径
        model_filename = GGUF_FILENAMES.get(selected_model, GGUF_FILENAMES["Q2_K"])
        model_path = find_existing_model(model_filename)
        
        print(f"选择的模型: {selected_model}")
        print(f"模型文件名: {model_filename}")
        print(f"找到的模型路径: {model_path}")
        
        if not model_path:
            raise Exception(f"模型文件 {model_filename} 不存在，请确保已下载模型文件到正确目录")
        
        print(f"正在加载模型: {model_path}")
        
        # 使用固定路径加载完整模型组件
        # 修改模型路径为WebUI的models/FLUX.1-Kontext-dev目录
        full_model_path = os.path.join(
            shared.models_path,
            'FLUX.1-Kontext-dev'
        )
        
        print(f"完整模型路径: {full_model_path}")
        print(f"完整模型路径是否存在: {os.path.exists(full_model_path)}")
        print(f"model_index.json是否存在: {os.path.exists(os.path.join(full_model_path, 'model_index.json'))}")
        
        # 加载调度器
        scheduler_path = os.path.join(full_model_path, "scheduler")
        if os.path.exists(scheduler_path):
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
            print("成功加载调度器")
        else:
            scheduler = FlowMatchEulerDiscreteScheduler()
            print("使用默认调度器")
        
        # 加载VAE
        vae_path = os.path.join(full_model_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
            print("成功加载VAE")
        else:
            raise Exception("VAE模型不存在")
        
        # 加载text_encoder
        text_encoder_path = os.path.join(full_model_path, "text_encoder")
        if os.path.exists(text_encoder_path):
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
            print("成功加载CLIP文本编码器")
        else:
            raise Exception("CLIP文本编码器不存在")
        
        # 加载tokenizer
        tokenizer_path = os.path.join(full_model_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            print("成功加载CLIP分词器")
        else:
            raise Exception("CLIP分词器不存在")
        
        # 加载text_encoder_2 (T5)
        text_encoder_2_path = os.path.join(full_model_path, "text_encoder_2")
        if os.path.exists(text_encoder_2_path):
            print(f"加载T5模型从路径: {text_encoder_2_path}")
            try:
                # 检查是否存在FP8格式的T5模型
                fp8_model_path = os.path.join(text_encoder_2_path, "t5xxl_fp8_e4m3fn.safetensors")
                print(f"检查FP8模型路径: {fp8_model_path}")
                print(f"FP8模型文件是否存在: {os.path.exists(fp8_model_path)}")
                
                # 使用标准方式加载T5模型
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    text_encoder_2_path, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                )
                print("成功加载T5模型")
                
                # 确保text_encoder_2的所有子模块都在同一设备上
                try:
                    # 检查是否有任何参数在错误的设备上
                    for name, param in text_encoder_2.named_parameters():
                        if param.device != torch.device('cpu'):
                            # 如果已经在CUDA上，确保所有参数在同一设备上
                            break
                except:
                    pass
                    
            except Exception as e:
                print(f"加载T5模型失败: {e}")
                import traceback
                traceback.print_exc()
                raise Exception("无法加载T5文本编码器")
        else:
            raise Exception("T5文本编码器路径不存在")
        
        # 加载tokenizer_2
        tokenizer_2_path = os.path.join(full_model_path, "tokenizer_2")
        if os.path.exists(tokenizer_2_path):
            tokenizer_2 = T5TokenizerFast.from_pretrained(tokenizer_2_path)
            print("成功加载T5分词器")
        else:
            raise Exception("T5分词器不存在")
        
        # 加载transformer (GGUF)
        print(f"正在加载GGUF模型: {model_path}")
        try:
            transformer = FluxTransformer2DModel.from_single_file(
                model_path,
                config=os.path.join(full_model_path, "transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            print("成功加载GGUF模型")
        except Exception as e:
            print(f"加载GGUF模型失败: {e}")
            raise Exception("无法加载Transformer模型")
        
        # 使用本地组件创建管道
        pipe = FluxKontextPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        
        # 移除了ensure_text_encoder_2_device_consistency函数
        
        # 简化设备管理，根据显存情况决定是否启用CPU卸载
        try:
            if enable_cpu_offload:
                # 启用CPU卸载以节省显存
                pipe.enable_model_cpu_offload()
                print("已启用模型CPU卸载以节省显存")
            else:
                # 尝试将模型移动到GPU
                pipe = pipe.to("cuda")
                print("模型已移动到CUDA设备")
        except Exception as e:
            print(f"设备管理失败: {e}")
            # 如果无法移动到GPU，保持模型在CPU上
            pipe = pipe.to("cpu")
            print("模型保留在CPU上")
        
        # 根据用户选择启用VAE优化
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
        
        # 根据用户选择决定T5模型的位置
        if enable_cpu_offload:
            # 如果启用了CPU卸载，则将T5模型保留在CPU上以节省显存
            try:
                pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
                print("T5模型已移动到CPU以节省显存")
            except Exception as e:
                print(f"无法将T5模型移动到CPU: {e}")
        else:
            # 如果未启用CPU卸载，则尝试将T5模型移动到GPU上以提高性能
            try:
                # 显存管理优化：先清理GPU缓存
                torch.cuda.empty_cache()
                
                # 尝试将T5模型移动到GPU
                pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
                print("T5模型已移动到GPU以提高性能")
            except Exception as e:
                print(f"无法将T5模型移动到GPU: {e}")
                # 出现错误时，安全地将模型保留在CPU上
                try:
                    # 清理GPU缓存后再尝试移动到CPU
                    torch.cuda.empty_cache()
                    pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
                    print("显存不足，T5模型已保留在CPU上")
                except Exception as fallback_error:
                    print(f"无法将T5模型移动到CPU: {fallback_error}")
                    # 最后回退方案：保持模型在当前设备上
                    print("T5模型保留在当前设备上")
        
        # 确保所有文本编码器都在正确的设备上
        try:
            # 获取text_encoder_2的设备
            text_encoder_2_device = next(pipe.text_encoder_2.parameters()).device
            print(f"text_encoder_2当前设备: {text_encoder_2_device}")
            
            # 确保text_encoder也在相同的设备上
            if hasattr(pipe, 'text_encoder'):
                text_encoder_device = next(pipe.text_encoder.parameters()).device
                print(f"text_encoder当前设备: {text_encoder_device}")
                
                # 如果设备不一致，将text_encoder移动到与text_encoder_2相同的设备
                if text_encoder_device != text_encoder_2_device:
                    pipe.text_encoder = pipe.text_encoder.to(text_encoder_2_device)
                    print(f"已将text_encoder移动到 {text_encoder_2_device}")
        except Exception as e:
            print(f"处理文本编码器设备一致性时出错: {e}")
        
        # 如果有已加载的LoRA，重新加载它
        if LOADED_LORA is not None and os.path.exists(LOADED_LORA):
            load_lora_weights(pipe, LOADED_LORA, LOADED_LORA_WEIGHT)
        
        # 确保所有模型组件的设备一致性
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
        # 获取主要设备（以text_encoder_2为准）
        main_device = next(pipe.text_encoder_2.parameters()).device
        print(f"主要设备: {main_device}")
        
        # 确保text_encoder在相同设备上
        if hasattr(pipe, 'text_encoder'):
            text_encoder_device = next(pipe.text_encoder.parameters()).device
            if text_encoder_device != main_device:
                pipe.text_encoder = pipe.text_encoder.to(main_device)
                print(f"已将text_encoder移动到 {main_device}")
        
        # 确保其他组件也在相同设备上
        components_to_check = ['vae', 'transformer']
        for component_name in components_to_check:
            if hasattr(pipe, component_name):
                component = getattr(pipe, component_name)
                if hasattr(component, 'parameters') and hasattr(component, 'to'):
                    try:
                        component_device = next(component.parameters()).device
                        if component_device != main_device:
                            setattr(pipe, component_name, component.to(main_device))
                            print(f"已将{component_name}移动到 {main_device}")
                    except Exception as e:
                        print(f"处理{component_name}设备时出错: {e}")
        
        return main_device
    except Exception as e:
        print(f"修复模型设备一致性时出错: {e}")
        return None

def generate_edit_series(
    input_images,  # 改为接受图像列表以支持单图和双图处理
    selected_edits, 
    seed=42, 
    randomize_seed=False, 
    guidance_scale=2.5, 
    num_inference_steps=20,
    enable_cpu_offload=False,
    model_type="Q2_K"
):
    """生成一系列不同编辑变体，支持单图或双图编辑"""
    global pipe
    
    # 确保模型已加载
    if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
        pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
        if pipe is None:
            raise gr.Error("模型加载失败，请检查模型文件是否完整")
    
    # 确保所有模型组件的设备一致性
    fix_model_device_consistency(pipe)
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # 确保至少有一张输入图像
    if not input_images:
        raise gr.Error("请上传至少一张图像。")
    
    # 如果传入的是单张图像，转换为列表格式
    if not isinstance(input_images, list):
        input_images = [input_images]
    
    # 验证所有输入图像都有效
    valid_images = []
    for img in input_images:
        if img is not None:
            valid_images.append(img)
    
    if len(valid_images) == 0:
        raise gr.Error("请上传有效的图像。")
    
    # 截断输入图像数量，避免过多输入导致内存问题
    MAX_INPUT_IMAGES = 2  # 限制最多两张图像
    if len(valid_images) > MAX_INPUT_IMAGES:
        valid_images = valid_images[:MAX_INPUT_IMAGES]
        gr.Warning(f"仅使用前两张图像，最多支持{MAX_INPUT_IMAGES}张输入图像")
    
    input_images = valid_images
    
    # 确定要生成的编辑项
    edits_to_generate = []
    
    # 处理选中的编辑列表
    if selected_edits:
        edits_to_generate.extend(selected_edits)
    
    if not edits_to_generate:
        raise gr.Error("请选择至少一个编辑项。")
    
    # 去重同时保持顺序
    edits_to_generate = list(dict.fromkeys(edits_to_generate))
    
    # 获取输入图像尺寸
    print("输入图像尺寸:")
    original_sizes = []
    for i, img in enumerate(input_images):
        width, height = img.size
        original_sizes.append((width, height))
        print(f"图像 {i+1}: {width}x{height}")
    
    # 计算目标尺寸（基于第一张图像的宽高比）
    def make_multiple_of_64(size):
        return max(64, (size // 64) * 64)  # 确保最小尺寸为64
    
    # 为每张图像单独计算保持其原始宽高比的尺寸
    target_sizes = []
    max_area = 1024 ** 2
    
    for img in input_images:
        orig_width, orig_height = img.size
        orig_aspect_ratio = orig_width / orig_height
        
        # 基于原始尺寸计算目标尺寸，保持宽高比
        if orig_width > orig_height:
            # 横向图像
            new_width = min(1024, make_multiple_of_64(orig_width))
            new_height = make_multiple_of_64(int(new_width / orig_aspect_ratio))
        else:
            # 纵向或正方形图像
            new_height = min(1024, make_multiple_of_64(orig_height))
            new_width = make_multiple_of_64(int(new_height * orig_aspect_ratio))
        
        # 确保面积不超过限制
        current_area = new_width * new_height
        if current_area > max_area:
            scale_factor = (max_area / current_area) ** 0.5
            new_width = make_multiple_of_64(int(new_width * scale_factor))
            new_height = make_multiple_of_64(int(new_height * scale_factor))
        
        target_sizes.append((new_width, new_height))
        print(f"图像目标尺寸: {new_width}x{new_height}")
    
    generated_images = []  # 存储所有生成的图像
    all_used_seeds = []  # 存储所有使用的种子
    
    # 生成前清理GPU内存
    torch.cuda.empty_cache()
    
    # 为每张输入图像生成编辑变体
    for img_idx, (input_image, (target_width, target_height)) in enumerate(zip(input_images, target_sizes)):
        # 为每张图像生成独立的基础种子
        base_seed = seed + img_idx * 10000 if not randomize_seed else random.randint(0, MAX_SEED)
        
        print(f"图像 {img_idx+1} 调整后尺寸: {target_width}x{target_height}")
        
        # 为当前图像生成所有编辑变体
        for i, edit in enumerate(edits_to_generate):
            # 为每个编辑项生成独立的种子，确保结果多样化
            if randomize_seed:
                current_seed = random.randint(0, MAX_SEED)
            else:
                # 即使不随机化种子，也为每个编辑项使用不同的种子值
                current_seed = base_seed + i * 1000  # 使用较大间隔确保差异
            
            # 构建更有效的提示词，确保关键词能更好地影响生成结果
            # 添加编辑项编号，增强提示词的独特性
            if edit.strip():
                # 如果用户提供了具体的编辑指令，使用更明确的格式
                # 将用户输入的编辑内容放在更突出的位置，并添加编号标识
                final_prompt = f"image editing variation {i+1}: {edit.strip()}, high quality, detailed, maintain original subject identity, professional photo"
            else:
                # 如果没有提供具体指令，使用通用的高质量图像编辑提示
                final_prompt = f"high quality image editing variation {i+1}, detailed, maintain original subject identity, professional photo"
            
            # 确保提示词不会过长，避免截断重要信息
            if len(final_prompt) > 200:  # 适当增加长度限制
                final_prompt = final_prompt[:200]
            
            print(f"图像 {img_idx+1} 第 {i+1} 个变体，使用的提示词: {final_prompt}")
            print(f"使用的种子: {current_seed}")
            
            # 在每次生成前清理缓存，确保独立生成
            torch.cuda.empty_cache()
            
            try:
                # 使用调整后的尺寸生成图像，同时保持原始宽高比
                # 先调整输入图像到目标尺寸
                resized_input = input_image.resize((target_width, target_height), Image.LANCZOS)
                
                # 为每次生成创建独立的生成器
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
                generator.manual_seed(current_seed)
                
                image = pipe(
                    image=resized_input, 
                    prompt=final_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=target_width,     # 使用当前图像的宽度
                    height=target_height,   # 使用当前图像的高度
                    generator=generator,
                    _auto_resize=False  # 禁用自动调整尺寸
                ).images[0]
                
            except torch.cuda.OutOfMemoryError:
                print(f"使用调整后尺寸时出现内存不足，尝试更小尺寸...")
                # 如果出现内存不足，尝试使用更小的尺寸，但保持宽高比
                width, height = target_width, target_height  # 使用当前图像的尺寸
                # 限制最大边长为768，同时保持宽高比（更保守的尺寸以避免内存问题）
                max_size = 768
                if width > max_size or height > max_size:
                    if width > height:
                        temp_width = max_size
                        temp_height = int(height * (max_size / width))
                    else:
                        temp_height = max_size
                        temp_width = int(width * (max_size / height))
                    
                    # 确保新尺寸是64的倍数
                    temp_width = max(64, (temp_width // 64) * 64)
                    temp_height = max(64, (temp_height // 64) * 64)
                    
                    # 让pipeline处理最终的尺寸调整
                    resized_input = input_image.resize((temp_width, temp_height), Image.LANCZOS)
                else:
                    # 如果尺寸已经很小，直接使用调整后的图像
                    resized_input = input_image.resize((target_width, target_height), Image.LANCZOS)
                    temp_width, temp_height = target_width, target_height  # 使用当前图像的尺寸
                
                # 在再次尝试之前清理GPU内存
                torch.cuda.empty_cache()
                
                # 为每次生成创建独立的生成器
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
                generator.manual_seed(current_seed)
                
                try:
                    image = pipe(
                        image=resized_input, 
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        width=temp_width,    # 使用调整后的宽度
                        height=temp_height,  # 使用调整后的高度
                        generator=generator,
                        _auto_resize=False  # 禁用自动调整尺寸
                    ).images[0]
                except torch.cuda.OutOfMemoryError:
                    # 如果仍然内存不足，启用CPU卸载并重试
                    print("再次出现内存不足，启用CPU卸载...")
                    try:
                        pipe.enable_model_cpu_offload()
                        image = pipe(
                            image=resized_input, 
                            prompt=final_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            width=temp_width,    # 使用调整后的宽度
                            height=temp_height,  # 使用调整后的高度
                            generator=generator,
                            _auto_resize=False  # 禁用自动调整尺寸
                        ).images[0]
                    except Exception as e:
                        print(f"启用CPU卸载后仍然失败: {e}")
                        # 如果所有尝试都失败了，跳过这个编辑项并继续处理下一个
                        print("跳过当前编辑项并继续处理下一个")
                        continue  # 继续处理下一个编辑项而不是退出整个函数
            
            # 只有在成功生成图像后才将其添加到结果列表中
            generated_images.append(image)
            all_used_seeds.append(current_seed)
            
            # 保存生成的图像到项目目录
            try:
                # 创建保存目录
                save_dir = os.path.join(shared.models_path, "FLUX.1-Kontext-dev", "outputs")
                os.makedirs(save_dir, exist_ok=True)
                
                # 生成文件名
                timestamp = int(time.time())
                filename = f"generated_image_{timestamp}_img{img_idx+1}_var{i+1}.png"
                save_path = os.path.join(save_dir, filename)
                
                # 保存图像
                image.save(save_path)
                print(f"生成的图像已保存到: {save_path}")
            except Exception as e:
                print(f"保存图像时出错: {e}")
            
            # 在生成之间清理缓存
            torch.cuda.empty_cache()
    
    # 返回所有生成的图像和使用的种子
    return generated_images, ", ".join(map(str, all_used_seeds))


def generate_dual_context_image(
    image_1,
    image_2,
    selected_edits,
    seed=42,
    randomize_seed=False,
    guidance_scale=2.5,
    num_inference_steps=20,
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
    
    # 确保模型已加载
    if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
        pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
        if pipe is None:
            raise gr.Error("模型加载失败，请检查模型文件是否完整")
    
    # 确保模型组件在相同设备上
    device = fix_model_device_consistency(pipe)
    if device is None:
        print("设备一致性检查失败，尝试启用CPU卸载...")
        try:
            pipe.enable_model_cpu_offload()
            device = fix_model_device_consistency(pipe)
            if device is None:
                raise gr.Error("设备一致性检查失败，即使启用了CPU卸载")
        except Exception as e:
            print(f"启用CPU卸载失败: {e}")
            raise gr.Error("无法解决设备一致性问题")
    
    if randomize_seed:
        base_seed = random.randint(0, MAX_SEED)
    else:
        base_seed = seed
    
    if image_1 is None or image_2 is None:
        raise gr.Error("请上传两张图像。")
    
    # 确定要生成的编辑项
    edits_to_generate = []
    
    # 处理选中的编辑列表
    if selected_edits:
        edits_to_generate.extend(selected_edits)
    
    if not edits_to_generate:
        raise gr.Error("请选择至少一个编辑项。")
    
    # 去重同时保持顺序
    edits_to_generate = list(dict.fromkeys(edits_to_generate))
    
    # 获取输入图像尺寸
    print("输入图像尺寸:")
    original_sizes = []
    for i, img in enumerate([image_1, image_2]):
        width, height = img.size
        original_sizes.append((width, height))
        print(f"图像 {i+1}: {width}x{height}")
    
    # 计算目标尺寸（为每张图像单独计算保持其原始宽高比的尺寸）
    def make_multiple_of_64(size):
        return max(64, (size // 64) * 64)  # 确保最小尺寸为64
    
    # 为每张图像单独计算保持其原始宽高比的尺寸
    target_sizes = []
    max_area = 1024 ** 2
    
    for img in [image_1, image_2]:
        orig_width, orig_height = img.size
        orig_aspect_ratio = orig_width / orig_height
        
        # 基于原始尺寸计算目标尺寸，保持宽高比
        if orig_width > orig_height:
            # 横向图像
            new_width = min(1024, make_multiple_of_64(orig_width))
            new_height = make_multiple_of_64(int(new_width / orig_aspect_ratio))
        else:
            # 纵向或正方形图像
            new_height = min(1024, make_multiple_of_64(orig_height))
            new_width = make_multiple_of_64(int(new_height * orig_aspect_ratio))
        
        # 确保面积不超过限制
        current_area = new_width * new_height
        if current_area > max_area:
            scale_factor = (max_area / current_area) ** 0.5
            new_width = make_multiple_of_64(int(new_width * scale_factor))
            new_height = make_multiple_of_64(int(new_height * scale_factor))
        
        target_sizes.append((new_width, new_height))
        print(f"图像目标尺寸: {new_width}x{new_height}")
    
    # 使用第一张图像的目标尺寸作为融合图像的尺寸
    target_width, target_height = target_sizes[0]
    
    generated_images = []
    all_used_seeds = []
    
    # 生成前清理GPU内存
    torch.cuda.empty_cache()
    
    # 为融合图像生成编辑变体
    for i, edit in enumerate(edits_to_generate):
        # 为每个编辑项生成独立的种子，确保结果多样化
        if randomize_seed:
            current_seed = random.randint(0, MAX_SEED)
        else:
            # 即使不随机化种子，也为每个编辑项使用不同的种子值
            current_seed = base_seed + i * 1000  # 使用较大间隔确保差异
        
        # 构建更有效的提示词，确保关键词能更好地影响生成结果
        if edit.strip():
            # 如果用户提供了具体的编辑指令，使用更明确的格式
            final_prompt = f"Combine the context from both reference images with the following edit: {edit.strip()}, high quality, detailed, maintain original subject identity, professional photo"
        else:
            # 如果没有提供具体指令，使用通用的高质量图像编辑提示
            final_prompt = f"Combine the context from both reference images, high quality, detailed, maintain original subject identity, professional photo"
        
        # 确保提示词不会过长，避免截断重要信息
        if len(final_prompt) > 200:  # 适当增加长度限制
            final_prompt = final_prompt[:200]
        
        print(f"融合图像 第 {i+1} 个变体，使用的提示词: {final_prompt}")
        print(f"使用的种子: {current_seed}")
        
        # 在每次生成前清理缓存，确保独立生成
        torch.cuda.empty_cache()
        
        try:
            # 调整输入图像到目标尺寸（保持各自的宽高比）
            resized_image_1 = image_1.resize(target_sizes[0], Image.LANCZOS)
            resized_image_2 = image_2.resize(target_sizes[1], Image.LANCZOS)
            
            # 创建一个包含两张图像的组合图像作为输入
            # 这里我们创建一个上下排列的组合图像
            combined_width = target_width  # 使用图像1的宽度作为组合图像宽度
            combined_height = target_height * 2  # 高度是图像1高度的两倍
            
            # 确保组合图像尺寸是64的倍数
            combined_width = make_multiple_of_64(combined_width)
            combined_height = make_multiple_of_64(combined_height)
            
            combined_image = Image.new('RGB', (combined_width, combined_height))
            # 调整图像1到指定尺寸
            resized_image_1 = resized_image_1.resize((combined_width, target_height), Image.LANCZOS)
            # 调整图像2到相同宽度但保持其原始宽高比
            aspect_ratio_2 = image_2.size[1] / image_2.size[0]  # height/width
            image_2_new_height = int(combined_width * aspect_ratio_2)
            resized_image_2 = image_2.resize((combined_width, image_2_new_height), Image.LANCZOS)
            
            combined_image.paste(resized_image_1, (0, 0))
            combined_image.paste(resized_image_2, (0, target_height))
            
            # 为每次生成创建独立的生成器
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() and not enable_cpu_offload else "cpu")
            generator.manual_seed(current_seed)
            
            # 生成图像时使用图像1的尺寸，确保输出保持正确的宽高比
            image = pipe(
                image=combined_image,
                prompt=final_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=target_width,
                height=target_height,
                generator=generator,
                _auto_resize=False  # 禁用自动调整尺寸
            ).images[0]
            
        except torch.cuda.OutOfMemoryError:
            # 如果出现内存不足，尝试使用更小的尺寸
            width, height = target_width, target_height
            # 限制最大边长为768
            max_size = 768
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # 确保新尺寸是64的倍数
                new_width = max(64, (new_width // 64) * 64)
                new_height = max(64, (new_height // 64) * 64)
            else:
                new_width, new_height = width, height
            
            # 保持宽高比调整尺寸
            new_width = make_multiple_of_64(new_width)
            new_height = make_multiple_of_64(new_height)
            
            combined_width = new_width
            combined_height = new_height * 2  # 高度是宽度的两倍以容纳两张图像
            
            print(f"使用调整后尺寸时出现内存不足，尝试更小尺寸: {new_width}x{new_height}...")
            
            # 在再次尝试之前清理GPU内存
            torch.cuda.empty_cache()
            
            # 调整输入图像到新尺寸
            resized_image_1 = image_1.resize((new_width, new_height), Image.LANCZOS)
            # 调整图像2到相同宽度但保持其原始宽高比
            aspect_ratio_2 = image_2.size[1] / image_2.size[0]  # height/width
            image_2_new_height = int(new_width * aspect_ratio_2)
            resized_image_2 = image_2.resize((new_width, image_2_new_height), Image.LANCZOS)
            
            # 创建组合图像
            combined_image = Image.new('RGB', (new_width, new_height * 2))
            combined_image.paste(resized_image_1, (0, 0))
            combined_image.paste(resized_image_2, (0, new_height))
            
            # 为每次生成创建独立的生成器
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() and not enable_cpu_offload else "cpu")
            generator.manual_seed(current_seed)
            
            try:
                image = pipe(
                    image=combined_image,
                    prompt=final_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=new_width,
                    height=new_height,
                    generator=generator,
                    _auto_resize=False  # 禁用自动调整尺寸
                ).images[0]
            except torch.cuda.OutOfMemoryError:
                # 如果仍然内存不足，启用CPU卸载并重试
                print("再次出现内存不足，启用CPU卸载...")
                try:
                    pipe.enable_model_cpu_offload()
                    image = pipe(
                        image=combined_image,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        width=new_width,
                        height=new_height,
                        generator=generator,
                        _auto_resize=False  # 禁用自动调整尺寸
                    ).images[0]
                except Exception as e:
                    # 如果所有尝试都失败了，跳过这个编辑项并继续处理下一个
                    print(f"生成图像时发生错误: {e}")
                    continue  # 继续处理下一个编辑项而不是退出整个函数
        
        # 只有在成功生成图像后才将其添加到结果列表中
        generated_images.append(image)
        all_used_seeds.append(current_seed)
        
        # 保存生成的图像到项目目录
        try:
            # 创建保存目录
            save_dir = os.path.join(shared.models_path, "FLUX.1-Kontext-dev", "outputs")
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = int(time.time())
            filename = f"dual_context_image_{timestamp}_var{i+1}.png"
            save_path = os.path.join(save_dir, filename)
            
            # 保存图像
            image.save(save_path)
            print(f"生成的图像已保存到: {save_path}")
        except Exception as e:
            print(f"保存图像时发生错误: {e}")
        
        # 在生成之间清理缓存
        torch.cuda.empty_cache()
    
    # 返回所有生成的图像和使用的种子
    return generated_images, ", ".join(map(str, all_used_seeds))

# ==================== UI创建函数 ====================

def create_flux_kontext_ui():
    """创建FLUX.1-Kontext UI界面"""
    with gr.Group():
        with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("单图编辑生成"):
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    portrait_input_image = gr.Image(
                                        label="上传图像", 
                                        type="pil",
                                        height=400
                                    )
                                    
                                with gr.Row():
                                    portrait_model_type = gr.Dropdown(
                                        label="模型选择",
                                        choices=["Q2_K", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
                                        value="Q2_K",
                                        info="Q2_K 显存占用最小 (推荐 6-8GB 显存), Q6_K 平衡质量与显存占用, Q8_0 质量最高 (需要 12GB+ 显存)"
                                    )
                                    
                                    portrait_enable_cpu_offload = gr.Checkbox(
                                        label="启用CPU卸载 (节省显存)",
                                        value=False,
                                        info="将部分模型组件移动到CPU以节省显存，但会降低推理速度。如果出现显存不足错误，请启用此选项。"
                                    )
                                
                                # 添加LoRA控制组件
                                with gr.Row():
                                    portrait_lora_enable = gr.Checkbox(
                                        label="启用LoRA",
                                        value=False,
                                        info="启用LoRA模型以修改生成风格"
                                    )
                                    portrait_lora_model = gr.Dropdown(
                                        label="LoRA模型选择",
                                        choices=list_lora_models(),
                                        value=list_lora_models()[0] if list_lora_models() else "",
                                        interactive=True
                                    )
                                    portrait_lora_weight = gr.Slider(
                                        label="LoRA权重",
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.05,
                                        value=0.5,
                                        info="控制LoRA模型的影响强度"
                                    )
                                
                                # 添加LoRA刷新按钮
                                portrait_lora_refresh = gr.Button("刷新LoRA模型列表")
                                
                                # LoRA刷新按钮事件处理
                                def refresh_lora_models():
                                    """刷新LoRA模型列表"""
                                    lora_list = list_lora_models()
                                    return gr.update(choices=lora_list, value=lora_list[0] if lora_list else "")
                                
                                portrait_lora_refresh.click(
                                    fn=refresh_lora_models,
                                    inputs=[],
                                    outputs=[portrait_lora_model]
                                )
                                
                                # 创建可编辑的编辑选项组件
                                pose_components = []
                                with gr.Group():
                                    # 只保留一个编辑选项组件
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            with gr.Group():
                                                with gr.Row():
                                                    textbox = gr.Textbox(
                                                        label="编辑内容",
                                                        value="",  # 确保默认为空
                                                        interactive=True,
                                                        container=True,
                                                        lines=2
                                                    )
                                            pose_components.append(textbox)
                                    
                                    # 收集所有编辑文本框组件
                                    pose_textboxes = list(pose_components)
                                    
                                    # 移除了编辑选项的清除、保存和加载按钮
                                    # 移除了相关的函数定义和按钮点击事件
                                
                                with gr.Accordion("高级设置", open=False):
                                    with gr.Column():
                                        portrait_seed = gr.Slider(
                                            label="种子值",
                                            minimum=0,
                                            maximum=MAX_SEED,
                                            step=1,
                                            value=42,
                                            info="控制生成图像的随机性"
                                        )
                                        
                                        portrait_randomize_seed = gr.Checkbox(
                                            label="为每个编辑随机化种子", 
                                            value=True,
                                            info="为每个编辑使用不同的随机种子"
                                        )
                                        
                                        with gr.Row():
                                            portrait_guidance_scale = gr.Slider(
                                                label="指导强度",
                                                minimum=1,
                                                maximum=10,
                                                step=0.1,
                                                value=2.5,
                                                info="控制生成图像与提示词的匹配程度"
                                            )
                                            
                                            portrait_num_inference_steps = gr.Slider(
                                                label="推理步数",
                                                minimum=10,
                                                maximum=50,
                                                step=1,
                                                value=20,
                                                info="控制生成图像的质量和计算时间"
                                            )
                                
                                # 添加一个占位符以保持布局平衡
                                gr.Box()
                                
                                portrait_generate_button = gr.Button("生成系列结果", variant="primary", size="lg")
                            
                            with gr.Group():
                                with gr.Row():
                                    generated_gallery = gr.Gallery(
                                        label="生成结果",
                                        columns=3,
                                        rows=2,
                                        object_fit="contain",
                                        height="auto"
                                    )
                                    
                                with gr.Row():
                                    seed_info = gr.Textbox(
                                        label="使用的种子",
                                        interactive=False,
                                        lines=2
                                    )
                    
                    # 添加双图像编辑标签页
                    with gr.TabItem("双图像编辑"):
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    dual_image_1 = gr.Image(
                                        label="上传图像 1", 
                                        type="pil",
                                        height=300
                                    )
                                    
                                    dual_image_2 = gr.Image(
                                        label="上传图像 2", 
                                        type="pil",
                                        height=300
                                    )
                                
                                with gr.Row():
                                    dual_model_type = gr.Dropdown(
                                        label="模型选择",
                                        choices=["Q2_K", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
                                        value="Q2_K",
                                        info="Q2_K 显存占用最小 (推荐 6-8GB 显存), Q6_K 平衡质量与显存占用, Q8_0 质量最高 (需要 12GB+ 显存)"
                                    )
                                    
                                    dual_enable_cpu_offload = gr.Checkbox(
                                        label="启用CPU卸载 (节省显存)",
                                        value=False,
                                        info="将部分模型组件移动到CPU以节省显存，但会降低推理速度。如果出现显存不足错误，请启用此选项。"
                                    )
                                
                                # 添加LoRA控制组件
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
                                        interactive=True
                                    )
                                    dual_lora_weight = gr.Slider(
                                        label="LoRA权重",
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.05,
                                        value=0.5,
                                        info="控制LoRA模型的影响强度"
                                    )
                                
                                # 添加处理模式选择
                                with gr.Row():
                                    dual_processing_mode = gr.Radio(
                                        label="处理模式",
                                        choices=["独立处理（分别编辑两张图像）", "融合处理（结合两张图像上下文生成新图像）"],
                                        value="独立处理（分别编辑两张图像）",
                                        info="选择如何处理双图像编辑"
                                    )
                                
                                # 添加LoRA刷新按钮
                                dual_lora_refresh = gr.Button("刷新LoRA模型列表")
                                
                                # LoRA刷新按钮事件处理
                                def refresh_dual_lora_models():
                                    """刷新LoRA模型列表"""
                                    lora_list = list_lora_models()
                                    return gr.update(choices=lora_list, value=lora_list[0] if lora_list else "")
                                
                                dual_lora_refresh.click(
                                    fn=refresh_dual_lora_models,
                                    inputs=[],
                                    outputs=[dual_lora_model]
                                )
                                # 双图像编辑选项
                                dual_edit_components = []
                                with gr.Group():
                                    # 只保留一个编辑选项组件
                                    with gr.Row(equal_height=False):
                                        with gr.Column(scale=1):
                                            # 第一个编辑选项
                                            with gr.Group():
                                                with gr.Row():
                                                    textbox1 = gr.Textbox(
                                                        label="编辑内容",
                                                        value="",  # 确保默认为空
                                                        interactive=True,
                                                        container=True,
                                                        lines=2
                                                    )
                                            dual_edit_components.append(textbox1)
                                    
                                    # 双图像编辑文本框组件
                                    dual_textboxes = list(dual_edit_components)
                                
                            with gr.Accordion("高级设置", open=False):
                                with gr.Column():
                                    dual_seed = gr.Slider(
                                        label="种子值",
                                        minimum=0,
                                        maximum=MAX_SEED,
                                        step=1,
                                        value=42,
                                        info="控制生成图像的随机性"
                                    )
                                        
                                    dual_randomize_seed = gr.Checkbox(
                                        label="为每个编辑随机化种子", 
                                        value=True,
                                        info="为每个编辑使用不同的随机种子"
                                    )
                                        
                                    with gr.Row():
                                        dual_guidance_scale = gr.Slider(
                                            label="指导强度",
                                            minimum=1,
                                            maximum=10,
                                            step=0.1,
                                            value=2.5,
                                            info="控制生成图像与提示词的匹配程度"
                                        )
                                            
                                        dual_num_inference_steps = gr.Slider(
                                            label="推理步数",
                                            minimum=10,
                                            maximum=50,
                                            step=1,
                                            value=20,
                                            info="控制生成图像的质量和计算时间"
                                            )
                                
                                # 添加一个占位符以保持布局平衡
                                gr.Box()
                                
                                dual_generate_button = gr.Button("生成双图像编辑结果", variant="primary", size="lg")
                            
                            with gr.Group():
                                with gr.Row():
                                    dual_generated_gallery = gr.Gallery(
                                        label="生成结果",
                                        columns=2,
                                        rows=2,
                                        object_fit="contain",
                                        height="auto"
                                    )
                                    
                                with gr.Row():
                                    dual_seed_info = gr.Textbox(
                                        label="使用的种子",
                                        interactive=False,
                                        lines=2
                                    )
                
                # 编辑系列生成事件处理
                def on_generate_edit_series(input_image, *args):
                    # 解析参数
                    edit_args = args[:-9]  # 编辑组件参数
                    seed = args[-9] 
                    randomize_seed = args[-8]
                    guidance_scale = args[-7]
                    num_inference_steps = args[-6]
                    model_type = args[-5]
                    enable_cpu_offload = args[-4]
                    lora_enable = args[-3]  # 新增LoRA启用参数
                    lora_model = args[-2]   # 新增LoRA模型参数
                    lora_weight = args[-1]  # 新增LoRA权重参数
                    
                    if input_image is None:
                        return [], "请上传图像。"
                    
                    # 解析编辑选项
                    selected_edits = []
                    
                    # 打印调试信息
                    print(f"编辑组件参数数量: {len(edit_args)}")
                    # 只有一个编辑组件 (现在只有textbox，checkbox已移除)
                    if len(edit_args) >= 1:
                        textbox_value = edit_args[0]
                        print(f"编辑: textbox='{textbox_value}'")
                        
                        # 只有当文本框有内容时才添加
                        if textbox_value and textbox_value.strip():
                            selected_edits.append(textbox_value)
                    
                    if not selected_edits:
                        return [], "请至少选择一个编辑项。"
                    
                    print(f"生成编辑系列，选中的编辑项: {selected_edits}")
                    
                    # 如果启用了LoRA，加载LoRA模型
                    global pipe
                    lora_path = None
                    if lora_enable and lora_model:
                        # 构建完整的LoRA文件路径
                        full_lora_path = os.path.join(shared.models_path, "Lora", lora_model)
                        
                        # 检查文件是否存在
                        if os.path.exists(full_lora_path):
                            # 确保模型已加载
                            if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
                                pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
                            
                            if pipe is not None:
                                # 在加载LoRA之前准备模型
                                prepare_model_for_lora(pipe)
                                # 卸载当前LoRA（如果有）
                                unload_lora_weights(pipe)
                                # 加载新LoRA - 传递模型文件名而非完整路径
                                load_lora_weights(pipe, lora_model, lora_weight)
                        else:
                            print(f"LoRA模型文件不存在: {full_lora_path}")
                    
                    # 确保输入图像是列表格式
                    if not isinstance(input_image, list):
                        input_image = [input_image]
                    
                    images, seeds = generate_edit_series(
                        input_image, selected_edits,
                        seed, randomize_seed, guidance_scale, num_inference_steps,
                        enable_cpu_offload, model_type
                    )
                    
                    seed_text = f"使用的种子: {seeds}"
                    
                    print(f"返回 {len(images)} 张图像到结果展示组件")
                    for i, img in enumerate(images):
                        print(f"图像 {i+1}: 类型={type(img)}, 尺寸={img.size if hasattr(img, 'size') else 'N/A'}")
                    
                    return images, seed_text
                
                # 双图像编辑生成事件处理
                def on_generate_dual_edit(image_1, image_2, *args):
                    # 解析参数
                    edit_args = args[:-10]  # 编辑组件参数 (2个编辑组件，每个包含checkbox和textbox，共4个参数)
                    seed = args[-10] 
                    randomize_seed = args[-9]
                    guidance_scale = args[-8]
                    num_inference_steps = args[-7]
                    model_type = args[-6]
                    enable_cpu_offload = args[-5]
                    lora_enable = args[-4]  # 新增LoRA启用参数
                    lora_model = args[-3]   # 新增LoRA模型参数
                    lora_weight = args[-2]  # 新增LoRA权重参数
                    processing_mode = args[-1]  # 新增处理模式参数
                    
                    if image_1 is None or image_2 is None:
                        return [], "请上传两张图像。"
                    
                    # 解析编辑选项
                    selected_edits = []
                    
                    # 处理可编辑的编辑组件
                    # 处理单个编辑组件
                    for i in range(len(edit_args)):
                        textbox_value = edit_args[i]
                        
                        # 只有当文本框有内容时才添加
                        if textbox_value and textbox_value.strip():
                            selected_edits.append(textbox_value)
                    
                    if not selected_edits:
                        return [], "请至少选择一个编辑项。"
                    
                    print(f"双图像编辑，选中的编辑项: {selected_edits}")
                    
                    # 如果启用了LoRA，加载LoRA模型
                    global pipe
                    lora_path = None
                    if lora_enable and lora_model:
                        # 构建完整的LoRA文件路径
                        full_lora_path = os.path.join(shared.models_path, "Lora", lora_model)
                        
                        # 检查文件是否存在
                        if os.path.exists(full_lora_path):
                            # 确保模型已加载
                            if pipe is None or SELECTED_MODEL != model_type or not FLUX_KONTEXT_LOADED:
                                pipe = load_flux_kontext_model(model_type, enable_cpu_offload)
                            
                            if pipe is not None:
                                # 在加载LoRA之前准备模型
                                prepare_model_for_lora(pipe)
                                # 卸载当前LoRA（如果有）
                                unload_lora_weights(pipe)
                                # 加载新LoRA - 传递模型文件名而非完整路径
                                load_lora_weights(pipe, lora_model, lora_weight)
                        else:
                            print(f"LoRA模型文件不存在: {full_lora_path}")
                    
                    # 根据处理模式选择不同的处理方式
                    if processing_mode == "融合处理（结合两张图像上下文生成新图像）":
                        # 使用新的双图像融合生成功能
                        images, seeds = generate_dual_context_image(
                            image_1, image_2, selected_edits,
                            seed, randomize_seed, guidance_scale, num_inference_steps,
                            enable_cpu_offload, model_type
                        )
                        seed_text = f"使用的种子: {seeds}"
                    else:
                        # 为两张图像分别生成编辑结果（原有功能）
                        all_images = []
                        all_seeds = []
                        
                        # 生成第一张图像的编辑结果
                        images_1, seeds_1 = generate_edit_series(
                            [image_1], selected_edits,
                            seed, randomize_seed, guidance_scale, num_inference_steps,
                            enable_cpu_offload, model_type
                        )
                        all_images.extend(images_1)
                        all_seeds.append(seeds_1)
                        
                        # 生成第二张图像的编辑结果
                        images_2, seeds_2 = generate_edit_series(
                            [image_2], selected_edits,
                            seed + 1000 if not randomize_seed else seed, 
                            randomize_seed, guidance_scale, num_inference_steps,
                            enable_cpu_offload, model_type
                        )
                        all_images.extend(images_2)
                        all_seeds.append(seeds_2)
                        
                        images = all_images
                        seed_text = f"图像1种子: {seeds_1} | 图像2种子: {seeds_2}"
                    
                    print(f"双图像编辑返回 {len(images)} 张图像到结果展示组件")
                    for i, img in enumerate(images):
                        print(f"图像 {i+1}: 类型={type(img)}, 尺寸={img.size if hasattr(img, 'size') else 'N/A'}")
                    
                    return images, seed_text
                
                # 收集所有编辑组件作为输入
                # 修复单图编辑输入收集逻辑
                pose_inputs = []
                for i, textbox in enumerate(pose_components):
                    pose_inputs.extend([textbox])
                
                dual_inputs = []
                for i, textbox in enumerate(dual_edit_components):
                    dual_inputs.extend([textbox])
                
                portrait_generate_button.click(
                    fn=on_generate_edit_series,
                    inputs=[
                        portrait_input_image, 
                        *pose_inputs,
                        portrait_seed, 
                        portrait_randomize_seed, 
                        portrait_guidance_scale, 
                        portrait_num_inference_steps,
                        portrait_model_type, 
                        portrait_enable_cpu_offload,
                        portrait_lora_enable,
                        portrait_lora_model,
                        portrait_lora_weight
                    ],
                    outputs=[generated_gallery, seed_info]
                )
                
                dual_generate_button.click(
                    fn=on_generate_dual_edit,
                    inputs=[
                        dual_image_1,
                        dual_image_2,
                        *dual_inputs,
                        dual_seed, 
                        dual_randomize_seed, 
                        dual_guidance_scale, 
                        dual_num_inference_steps,
                        dual_model_type, 
                        dual_enable_cpu_offload,
                        dual_lora_enable,
                        dual_lora_model,
                        dual_lora_weight,
                        dual_processing_mode
                    ],
                    outputs=[dual_generated_gallery, dual_seed_info]
                )
                
                # 移除了双图像编辑选项的清除、保存和加载按钮的事件绑定
                # 移除了相关的函数定义和按钮点击事件
                
                # LoRA相关按钮事件处理
                def refresh_lora_models():
                    """刷新LoRA模型列表"""
                    lora_list = list_lora_models()
                    return gr.update(choices=lora_list, value=lora_list[0] if lora_list else "")
                
                portrait_lora_refresh.click(
                    fn=refresh_lora_models,
                    inputs=[],
                    outputs=[portrait_lora_model]
                )
                
                dual_lora_refresh.click(
                    fn=refresh_dual_lora_models,
                    inputs=[],
                    outputs=[dual_lora_model]
                )
                
                # 返回所有组件引用，供外部访问
                return {
                    # 批量编辑系列返回组件
                    "portrait_input_image": portrait_input_image,
                    "portrait_model_type": portrait_model_type,
                    "portrait_enable_cpu_offload": portrait_enable_cpu_offload,
                    "portrait_seed": portrait_seed,
                    "portrait_randomize_seed": portrait_randomize_seed,
                    "portrait_guidance_scale": portrait_guidance_scale,
                    "portrait_num_inference_steps": portrait_num_inference_steps,
                    "portrait_generate_button": portrait_generate_button,
                    "generated_gallery": generated_gallery,
                    "seed_info": seed_info,
                    # 双图像编辑返回组件
                    "dual_image_1": dual_image_1,
                    "dual_image_2": dual_image_2,
                    "dual_model_type": dual_model_type,
                    "dual_enable_cpu_offload": dual_enable_cpu_offload,
                    "dual_seed": dual_seed,
                    "dual_randomize_seed": dual_randomize_seed,
                    "dual_guidance_scale": dual_guidance_scale,
                    "dual_num_inference_steps": dual_num_inference_steps,
                    "dual_generate_button": dual_generate_button,
                    "dual_generated_gallery": dual_generated_gallery,
                    "dual_seed_info": dual_seed_info,
                    # LoRA相关返回组件
                    "portrait_lora_enable": portrait_lora_enable,
                    "portrait_lora_model": portrait_lora_model,
                    "portrait_lora_weight": portrait_lora_weight,
                    "portrait_lora_refresh": portrait_lora_refresh,
                    "dual_lora_enable": dual_lora_enable,
                    "dual_lora_model": dual_lora_model,
                    "dual_lora_weight": dual_lora_weight,
                    "dual_lora_refresh": dual_lora_refresh,
                    # 处理模式组件
                    "dual_processing_mode": dual_processing_mode,
                    # 单图编辑组件
                    "pose_textbox": pose_textboxes[0] if pose_textboxes else None,
                    # 双图编辑组件
                    "dual_textbox": dual_textboxes[0] if dual_textboxes else None,
            }

# 模块可用性标记
FLUX_KONTEXT_AVAILABLE = True
