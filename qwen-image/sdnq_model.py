import torch
import gc
from pathlib import Path
import os


def load_sdnq_model(model_path, torch_dtype=torch.bfloat16):
    """
    专门用于加载SDNQ量化模型的函数
    :param model_path: 模型路径
    :param torch_dtype: 模型数据类型，默认为bfloat16
    :return: 加载成功的pipeline对象，失败则返回None
    """
    try:
        print(f"正在加载SDNQ模型: {model_path}")
        
        # 导入SDNQ相关库
        import diffusers
        from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
        
        # 尝试从sdnq.common导入，如果失败则使用备用方案检测triton
        try:
            from sdnq.common import use_torch_compile as triton_is_available_internal
            triton_is_available = triton_is_available_internal
        except ImportError:
            # 如果导入失败，使用备用检测方法
            try:
                import triton
                triton_version = triton.__version__
                # 检查是否为支持的版本
                from packaging import version
                triton_is_available = version.parse(triton_version) >= version.parse("2.1.0")
                print(f"检测到Triton版本: {triton_version}, 可用: {triton_is_available}")
            except ImportError:
                triton_is_available = False
                print("未找到Triton，将使用PyTorch Eager模式")
        
        from sdnq.loader import apply_sdnq_options_to_model
        
        # 确保模型路径是Path对象或字符串
        if isinstance(model_path, Path):
            model_path_str = str(model_path)
        else:
            model_path_str = model_path
        
        # 使用diffusers加载QwenImagePipeline
        print("使用diffusers加载QwenImagePipeline...")
        # 将已弃用的torch_dtype参数替换为dtype
        pipe = diffusers.QwenImagePipeline.from_pretrained(
            model_path_str,
            dtype=torch_dtype
        )
        
        print("应用SDNQ量化选项到transformer...")
        # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            print("已启用INT8 MatMul优化")
        else:
            print("未启用INT8 MatMul优化（Triton不可用或无GPU）")
        
        print("SDNQ模型加载成功")
        return pipe
        
    except ImportError as e:
        print(f"导入SDNQ相关库失败: {e}")
        print("Triton不可用，跳过量化优化，使用基础模型...")
        # 如果sdnq库不可用，尝试使用基础模型加载
        try:
            import diffusers
            # 确保模型路径是Path对象或字符串
            if isinstance(model_path, Path):
                model_path_str = str(model_path)
            else:
                model_path_str = model_path
            
            # 使用diffusers加载QwenImagePipeline
            pipe = diffusers.QwenImagePipeline.from_pretrained(
                model_path_str,
                dtype=torch_dtype
            )
            
            print("基础模型加载成功")
            return pipe
        except Exception as fallback_error:
            print(f"基础模型加载也失败: {fallback_error}")
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"加载SDNQ模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_sdnq_generation(
    pipe,
    prompt,
    negative_prompt,
    width,
    height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    seed=42,
    num_images_per_prompt=1,
    control_image=None,
    controlnet_conditioning_scale=1.0,
    control_guidance_start=0.0,
    control_guidance_end=1.0
):
    """
    使用SDNQ模型生成图像
    :param pipe: 模型管道
    :param prompt: 正面提示词
    :param negative_prompt: 负面提示词
    :param width: 图像宽度
    :param height: 图像高度
    :param num_inference_steps: 推理步数
    :param true_cfg_scale: CFG缩放
    :param seed: 随机种子
    :param num_images_per_prompt: 每次提示生成的图像数量
    :param control_image: ControlNet控制图像
    :param controlnet_conditioning_scale: ControlNet强度
    :param control_guidance_start: ControlNet开始时间
    :param control_guidance_end: ControlNet结束时间
    :return: 生成的图像列表
    """
    try:
        print(f"使用SDNQ模型生成图像，尺寸: ({width}, {height})")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"批量生成数量: {num_images_per_prompt}")
        
        # 根据模型设备选择合适的生成器设备
        device = next(pipe.transformer.parameters()).device if hasattr(pipe, 'transformer') and pipe.transformer is not None else torch.device("cpu")
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # 检查pipeline类型，确定是否启用了ControlNet
        from diffusers import QwenImageControlNetPipeline
        use_controlnet = control_image is not None and isinstance(pipe, QwenImageControlNetPipeline)
        
        if use_controlnet:
            print(f"启用ControlNet，强度: {controlnet_conditioning_scale}, "
                  f"开始: {control_guidance_start}, 结束: {control_guidance_end}")
        elif control_image is not None:
            print("警告: 提供了control_image，但当前pipeline不支持ControlNet，忽略ControlNet参数")
        
        # 检查是否为量化模型，如果是，则不直接转换数据类型
        is_quantized_model = (
            hasattr(pipe, 'transformer') and 
            any('awq' in str(param.dtype).lower() or 'int8' in str(param.dtype).lower() 
                for param in pipe.transformer.parameters())
        )
        
        if is_quantized_model:
            print("检测到量化模型，跳过数据类型转换以避免不支持的操作")
        else:
            # 确保所有模型组件使用相同的数据类型，解决"mat1 and mat2 must have the same dtype"问题
            target_dtype = torch.bfloat16  # 使用bfloat16，这是SDNQ模型的默认类型
            
            # 检查并统一各组件的数据类型
            if hasattr(pipe, 'transformer') and pipe.transformer.dtype != target_dtype:
                print(f"调整transformer数据类型从 {pipe.transformer.dtype} 到 {target_dtype}")
                pipe.transformer = pipe.transformer.to(target_dtype)
            
            if hasattr(pipe, 'text_encoder') and pipe.text_encoder.dtype != target_dtype:
                print(f"调整text_encoder数据类型从 {pipe.text_encoder.dtype} 到 {target_dtype}")
                pipe.text_encoder = pipe.text_encoder.to(target_dtype)
            
            if hasattr(pipe, 'vae') and pipe.vae.dtype != target_dtype:
                print(f"调整vae数据类型从 {pipe.vae.dtype} 到 {target_dtype}")
                pipe.vae = pipe.vae.to(target_dtype)
        
        # 如果启用了ControlNet，确保ControlNet组件的数据类型与主模型一致
        if use_controlnet and hasattr(pipe, 'controlnet'):
            if is_quantized_model:
                # 对于量化模型，检查ControlNet的数据类型是否与主模型一致
                main_model_dtype = next(pipe.transformer.parameters()).dtype
                controlnet_dtype = next(pipe.controlnet.parameters()).dtype
                
                if main_model_dtype != controlnet_dtype:
                    print(f"检测到ControlNet与主模型数据类型不匹配: {controlnet_dtype} vs {main_model_dtype}")
                    # 由于是量化模型，不能直接转换数据类型，但可以尝试将controlnet移动到相同设备
                    pipe.controlnet = pipe.controlnet.to(device=next(pipe.transformer.parameters()).device, dtype=main_model_dtype)
                    print(f"已将ControlNet调整到与主模型相同的数据类型和设备")
            else:
                # 对于非量化模型，统一ControlNet的数据类型
                if hasattr(pipe, 'controlnet') and pipe.controlnet.dtype != target_dtype:
                    print(f"调整ControlNet数据类型从 {pipe.controlnet.dtype} 到 {target_dtype}")
                    pipe.controlnet = pipe.controlnet.to(target_dtype)
        
        # 在生成前应用内存优化
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 设置内存碎片整理频率
                torch.cuda.set_per_process_memory_fraction(0.9)  # 限制使用90%的GPU内存
            
            # 启用多种内存优化技术
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            
            # 启用模型分片（如果支持）
            if hasattr(pipe, 'enable_model_cpu_offload'):
                # 不立即启用CPU卸载，而是作为后备选项
                pass
                
            if hasattr(pipe, 'enable_sequential_cpu_offload'):
                # 对于较小的GPU，可以启用顺序CPU卸载
                pass
                
        except Exception as e:
            print(f"应用内存优化时出现警告: {e}")
        
        # 生成参数
        base_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator
        }
        
        # 添加ControlNet参数（如果适用）
        if use_controlnet:
            base_kwargs.update({
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end
            })
        
        # 执行生成（带内存不足重试机制）
        try:
            output = pipe(**base_kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e) or "Allocation" in str(e) or "insufficient" in str(e):
                print(f"生成错误，尝试修复: {e}")
                
                # 如果是内存错误，尝试内存优化
                try:
                    # 清理缓存
                    torch.cuda.empty_cache()
                    
                    # 启用更激进的内存管理
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        print("启用模型CPU卸载...")
                        pipe.enable_model_cpu_offload()
                    
                    # 进一步减少批处理大小和步数
                    base_kwargs["num_inference_steps"] = max(1, num_inference_steps // 2)
                    base_kwargs["num_images_per_prompt"] = 1
                    
                    # 重试生成
                    output = pipe(**base_kwargs)
                    print(f"重试成功，使用调整后的参数")
                except Exception as retry_e:
                    print(f"重试失败: {retry_e}")
                    # 尝试更多内存释放手段
                    try:
                        # 清理Python垃圾
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # 尝试更小的图片尺寸（如果图片太大）
                        if width * height > 1024 * 1024:  # 超过1百万像素
                            # 暂时缩小图片尺寸重试
                            reduced_width = int(width * 0.8)
                            reduced_height = int(height * 0.8)
                            base_kwargs["width"] = reduced_width
                            base_kwargs["height"] = reduced_height
                            
                            print(f"尝试使用较小的图片尺寸: ({reduced_width}, {reduced_height})")
                            output = pipe(**base_kwargs)
                            print(f"使用减小尺寸重试成功")
                        else:
                            # 如果已经很小了，就不再缩小
                            raise retry_e
                    except Exception as final_e:
                        print(f"SDNQ图像生成失败: {final_e}")
                        # 恢复管道设置，以防CPU卸载影响后续使用
                        try:
                            if hasattr(pipe, 'maybe_free_model_hooks') and callable(getattr(pipe, 'maybe_free_model_hooks')):
                                pipe.maybe_free_model_hooks()
                        except:
                            pass
                        raise final_e
            elif "Cannot generate a cpu tensor from a generator of type cuda" in str(e):
                print("检测到设备不匹配错误，重新使用CPU生成器...")
                # 使用CPU生成器重试
                cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
                base_kwargs["generator"] = cpu_generator
                
                # 重试生成
                output = pipe(**base_kwargs)
                print(f"使用CPU生成器生成成功")
            elif "Casting a quantized model to a new 'dtype' is unsupported" in str(e):
                print("检测到量化模型数据类型转换错误，跳过数据类型转换...")
                # 对于量化模型，我们不能转换数据类型，只能使用模型原有的数据类型
                # 直接重试生成，不进行数据类型转换
                output = pipe(**base_kwargs)
                print(f"跳过数据类型转换后生成成功")
            elif "mat1 and mat2 must have the same dtype" in str(e):
                print("检测到数据类型不匹配错误，尝试修复量化模型数据类型...")
                # 这个错误可能发生在ControlNet启用时，尝试确保所有组件兼容
                if use_controlnet and hasattr(pipe, 'controlnet'):
                    main_model_dtype = next(pipe.transformer.parameters()).dtype
                    controlnet_dtype = next(pipe.controlnet.parameters()).dtype
                    
                    print(f"主模型数据类型: {main_model_dtype}, ControlNet数据类型: {controlnet_dtype}")
                    
                    # 确保ControlNet和主模型使用相同的数据类型
                    if main_model_dtype != controlnet_dtype:
                        # 对于量化模型，直接转换数据类型是不支持的，但是我们可以尝试重新调整
                        # 将controlnet移动到与transformer相同的设备和数据类型
                        pipe.controlnet = pipe.controlnet.to(device=next(pipe.transformer.parameters()).device, dtype=main_model_dtype)
                        print(f"已将ControlNet调整到与主模型相同的数据类型和设备: {main_model_dtype}")
                    
                    # 重试生成
                    output = pipe(**base_kwargs)
                    print(f"数据类型同步后生成成功")
                else:
                    print("由于是量化模型，无法进行数据类型转换，跳过此步骤")
                    raise e
            else:
                raise e
        
        # 确保返回的是图像列表
        images = output.images if hasattr(output, 'images') else [output]
        
        return images
    except Exception as e:
        print(f"SDNQ图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None
