import gradio as gr
import torch
import os
import gc
import time
from modules import shared
from modules import sd_samplers, sd_schedulers

# 尝试导入diffusers相关模块
try:
    from diffusers import (
        FluxPipeline, 
        FlowMatchEulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        UniPCMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers库未找到")

# 尝试导入nunchaku相关模块
try:
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    NUNCHAKU_AVAILABLE = True
    print("Nunchaku库已找到")
except ImportError:
    NUNCHAKU_AVAILABLE = False
    print("Nunchaku库未找到")

# 全局变量存储当前加载的模型和相关信息
pipe = None
SELECTED_MODEL = None
FLUX_KREA_LOADED = False

# 根据依赖库是否可用决定插件是否可用
FLUX_KREA_AVAILABLE = DIFFUSERS_AVAILABLE and NUNCHAKU_AVAILABLE

# 支持的采样器列表（使用WebUI原生采样器）
def get_webui_samplers():
    """获取WebUI原生采样器列表"""
    try:
        # 使用WebUI原生采样器
        samplers = [sampler.name for sampler in sd_samplers.visible_samplers()]
        return samplers if samplers else ["Euler"]
    except Exception as e:
        print(f"获取WebUI采样器失败: {e}")
        # 回退到默认采样器列表
        return ["Euler", "DPM++ 2M", "Euler Ancestral", "UniPC"]

# 支持的调度器列表（使用WebUI原生调度器）
def get_webui_schedulers():
    """获取WebUI原生调度器列表"""
    try:
        # 使用WebUI原生调度器
        schedulers = [scheduler.label for scheduler in sd_schedulers.schedulers]
        return schedulers if schedulers else ["Default"]
    except Exception as e:
        print(f"获取WebUI调度器失败: {e}")
        # 回退到默认调度器列表
        return ["Default"]

def load_flux_krea_model(model_type, enable_cpu_offload=False):
    """加载FLUX.1-krea模型"""
    global pipe, SELECTED_MODEL, FLUX_KREA_LOADED
    
    # 检查必要依赖是否可用
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("缺少必要的依赖库: diffusers")
    
    if not NUNCHAKU_AVAILABLE:
        raise RuntimeError("缺少必要的依赖库: nunchaku")
    
    # 如果已经加载了相同类型的模型，则直接返回
    if pipe is not None and SELECTED_MODEL == model_type and FLUX_KREA_LOADED:
        print(f"FLUX.1-krea模型 {model_type} 已经加载")
        return pipe
    
    # 清理现有模型以释放显存
    if pipe is not None:
        del pipe
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
            offload=enable_cpu_offload  # 根据enable_cpu_offload参数决定是否启用offload
        )
        
        # 构建本地模型路径
        local_model_path = os.path.join(shared.models_path, "FLUX.1-Kontext-dev")
        
        # 创建FLUX管道 - 使用本地模型路径而不是Hub路径
        pipe = FluxPipeline.from_pretrained(
            local_model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # 根据选项决定是否启用CPU卸载
        if enable_cpu_offload:
            print("启用Sequential CPU卸载以节省显存")
            pipe.enable_sequential_cpu_offload()
        else:
            try:
                pipe = pipe.to("cuda")
                print("模型已移动到CUDA设备")
            except Exception as e:
                print(f"将模型移动到CUDA时出错: {e}")
                pipe.enable_sequential_cpu_offload()
                print("改为启用Sequential CPU卸载")
        
        # 标记模型已加载
        SELECTED_MODEL = model_type
        FLUX_KREA_LOADED = True
        print("FLUX.1-krea模型加载完成")
        return pipe
        
    except Exception as e:
        print(f"加载FLUX.1-krea模型时出错: {e}")
        pipe = None
        SELECTED_MODEL = None
        FLUX_KREA_LOADED = False
        raise e

def update_sampler(sampler_name):
    """更新采样器"""
    global pipe
    
    if pipe is None:
        return
    
    try:
        # 查找WebUI采样器配置
        sampler = None
        for s in sd_samplers.visible_samplers():
            if s.name == sampler_name:
                sampler = s
                break
        
        if sampler and hasattr(sampler, 'constructor') and sampler.constructor:
            # 使用WebUI采样器的构造函数
            if hasattr(pipe, 'scheduler'):
                # 获取当前调度器配置
                config = pipe.scheduler.config
                # 创建新的调度器实例
                pipe.scheduler = sampler.constructor.from_config(config)
                print(f"采样器已更新为: {sampler_name}")
            else:
                print(f"管道不支持调度器更新")
        else:
            # 回退到原有的采样器映射
            scheduler_class = SAMPLER_MAP.get(sampler_name)
            if scheduler_class:
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                print(f"采样器已更新为: {sampler_name}")
            else:
                print(f"未知的采样器: {sampler_name}")
    except Exception as e:
        print(f"更新采样器时出错: {e}")

def generate_image(prompt, negative_prompt="", width=1024, height=1024, 
                   guidance_scale=3.5, num_inference_steps=20, seed=0, 
                   sampler_name="Euler", batch_size=1):
    """生成图像"""
    global pipe
    
    if pipe is None:
        raise ValueError("模型未加载，请先加载模型")
    
    # 更新采样器
    update_sampler(sampler_name)
    
    # 设置随机种子
    if seed == 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    try:
        # 生成图像
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images
        
        # 保存图像
        save_dir = os.path.join(shared.data_path, "outputs", "flux-krea")
        os.makedirs(save_dir, exist_ok=True)
        
        image_paths = []
        for i, image in enumerate(images):
            timestamp = int(time.time())
            filename = f"flux_krea_{timestamp}_{i}.png"
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
            image_paths.append(save_path)
            print(f"生成的图像已保存到: {save_path}")
        
        return image_paths, str(seed)
        
    except Exception as e:
        print(f"生成图像时出错: {e}")
        raise e

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
                
                with gr.Accordion("模型设置", open=False):
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
                                info="Nunchaku提供更好的性能和更低的显存需求。fp4为浮点4位量化（适用于50系显卡），int4为整数4位量化（适用于非50系显卡）"
                            )
                            
                            krea_enable_cpu_offload = gr.Checkbox(
                                label="启用CPU卸载 (节省显存)",
                                value=False,
                                info="将部分模型组件移动到CPU以节省显存，但会降低推理速度。如果出现显存不足错误，请启用此选项。"
                            )
                        
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
                            krea_lora_weight = gr.Slider(
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
                    
                    with gr.Row():
                        krea_sampler = gr.Dropdown(
                            label="采样器",
                            choices=get_webui_samplers(),
                            value="Euler" if "Euler" in get_webui_samplers() else get_webui_samplers()[0],
                            info="选择图像生成的采样算法"
                        )
                        
                        # 使用WebUI原生调度器
                        krea_scheduler = gr.Dropdown(
                            label="调度器",
                            choices=get_webui_schedulers(),
                            value="Default",
                            info="调度器控制噪声调度策略"
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

        # 定义生成图像的处理函数
        def on_generate_image(*args):
            # 解析参数
            prompt = args[0]
            negative_prompt = args[1]
            width = args[2]
            height = args[3]
            seed = args[4]
            guidance_scale = args[5]
            num_inference_steps = args[6]
            model_type = args[7]
            enable_cpu_offload = args[8]
            lora_enable = args[9]
            lora_model = args[10]
            lora_weight = args[11]
            sampler_name = args[12]
            batch_size = args[13]  # 新增批次大小参数
            
            if not prompt:
                return None, "请提供正面提示词"
            
            try:
                # 检查必要依赖是否可用
                if not DIFFUSERS_AVAILABLE:
                    raise RuntimeError("缺少必要的依赖库: diffusers")
                
                if not NUNCHAKU_AVAILABLE:
                    raise RuntimeError("缺少必要的依赖库: nunchaku")
                
                # 加载模型
                global pipe
                pipe = load_flux_krea_model(model_type, enable_cpu_offload)
                
                # 如果启用了LoRA，加载LoRA模型
                if lora_enable and lora_model:
                    # 注意：这里需要根据实际情况实现LoRA加载逻辑
                    pass  # LoRA功能暂未实现，静默处理
                
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
                    batch_size=batch_size  # 传递批次大小参数
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
                krea_enable_cpu_offload,
                krea_lora_enable,
                krea_lora_model,
                krea_lora_weight,
                krea_sampler,
                krea_batch_size  # 新增批次大小输入
            ], 
            outputs=[krea_generated_images, krea_seed_info]
        )
        
        return flux_krea_ui

FLUX_KREA_AVAILABLE = True