def preprocess_for_qwen_image_controlnet(image_path, preprocessor_type):
    """为Qwen-Image-ControlNet预处理图像"""
    try:
        if not image_path or not os.path.exists(image_path):
            print(f"预处理图像路径无效: {image_path}")
            return None
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        # 移除调试日志: print(f"开始使用预处理器 {preprocessor_type} 处理图像: {image_path}")
        
        # 使用WebUI的预处理器管理系统
        try:
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
                    # 减少调试日志: print(f"已添加路径到sys.path: {path}")
            
            # 导入WebUI的预处理器管理模块
            from modules_forge.shared import supported_preprocessors
            from modules_forge.initialization import initialize_forge
            
            # 初始化Forge系统
            initialize_forge()
            
            # 手动导入inpaint预处理器以确保预处理器被正确加载
            try:
                import forge_preprocessor_inpaint.scripts.preprocessor_inpaint
                # 减少调试日志: print("成功加载forge_preprocessor_inpaint模块")
            except Exception as e:
                # 减少调试日志: print(f"加载forge_preprocessor_inpaint模块时出错: {e}")
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
                        # 减少调试日志: print("手动注册inpaint_only预处理器成功")
                    
                    if not inpaint_global_harmonious_registered:
                        inpaint_preprocessor = PreprocessorInpaint()
                        add_supported_preprocessor(inpaint_preprocessor)
                        # 减少调试日志: print("手动注册inpaint_global_harmonious预处理器成功")
                    
                    if not inpaint_lama_registered:
                        inpaint_lama_preprocessor = PreprocessorInpaintLama()
                        add_supported_preprocessor(inpaint_lama_preprocessor)
                        # 减少调试日志: print("手动注册inpaint_lama预处理器成功")
                        
                except Exception as manual_register_error:
                    # 减少调试日志: print(f"手动注册inpaint预处理器失败: {manual_register_error}")
                    pass
            
            # 手动导入legacy_preprocessors以确保预处理器被正确加载
            try:
                import forge_legacy_preprocessors.scripts.legacy_preprocessors
                # 减少调试日志: print("成功加载legacy_preprocessors模块")
            except Exception as e:
                # 减少调试日志: print(f"加载legacy_preprocessors模块时出错: {e}")
                pass
            
            # 尝试直接使用预处理器类型名称获取预处理器对象
            # 根据WebUI源码，预处理器的名称就是其在supported_preprocessors中的键
            # 移除调试日志: print(f"尝试查找预处理器: {preprocessor_type}")
            
            # 特殊处理"none"预处理器 - 直接返回原始图像
            if preprocessor_type.lower() in ["none", "无", "none (default)"]:
                # 移除调试日志: print("使用无预处理模式，直接返回原始图像")
                if isinstance(image, np.ndarray):
                    return image
                else:
                    return np.array(image)
            
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
                
                for variant in variants:
                    if variant in supported_preprocessors:
                        preprocessor = supported_preprocessors[variant]
                        # 移除调试日志: print(f"通过变体名称找到预处理器: {variant}")
                        break
            
            # 特殊处理"inpaint_only"预处理器名称
            # 在某些情况下，用户可能使用"Inpaint Only"而不是"inpaint_only"
            if preprocessor is None and preprocessor_type.lower().replace(" ", "_") in ["inpaint_only", "inpaintonly"]:
                # 尝试查找"inpaint_only"
                if "inpaint_only" in supported_preprocessors:
                    preprocessor = supported_preprocessors["inpaint_only"]
                    # 移除调试日志: print("通过特殊处理找到预处理器: inpaint_only")
            
            # 如果还是找不到，直接报错而不是回退到canny
            if preprocessor is None:
                print(f"错误：未找到预处理器 {preprocessor_type}")
                raise ValueError(f"未找到预处理器: {preprocessor_type}，请检查预处理器名称是否正确")
            
            # 移除调试日志: print(f"成功找到预处理器: {preprocessor.name}")
            
            # 使用预处理器处理图像
            # 注意：WebUI预处理器通常接受RGB格式的numpy数组，值范围为0-255
            # 移除调试日志: print(f"使用预处理器 {preprocessor.name} 处理图像: {image_path}")
            
            # 确保图像数据是正确的格式
            if isinstance(image, np.ndarray):
                # 如果图像是numpy数组格式
                image_array = image
            else:
                # 如果图像是PIL Image格式
                image_array = np.array(image)
            
            # 确保图像是RGB格式
            if len(image_array.shape) == 2:
                # 灰度图转RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                # RGBA转RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] == 3:
                # 已经是RGB格式
                pass
            else:
                # 其他情况，假设是BGR格式转RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # 调用预处理器处理图像
            # 注意：不同的预处理器可能有不同的参数要求
            try:
                # 尝试以不同方式调用预处理器
                if hasattr(preprocessor, '__call__'):
                    # 大多数预处理器是可调用对象
                    # 检查预处理器需要的参数并提供默认值
                    import inspect
                    sig = inspect.signature(preprocessor.__call__)
                    kwargs = {}
                    
                    # 为常见参数提供默认值
                    if 'resolution' in sig.parameters:
                        kwargs['resolution'] = 512
                    if 'slider_1' in sig.parameters:
                        # 对于Canny预处理器，slider_1是低阈值
                        kwargs['slider_1'] = 100 if preprocessor.name == 'canny' else None
                    if 'slider_2' in sig.parameters:
                        # 对于Canny预处理器，slider_2是高阈值
                        kwargs['slider_2'] = 200 if preprocessor.name == 'canny' else None
                    if 'slider_3' in sig.parameters:
                        kwargs['slider_3'] = None
                    
                    # 确保所有数字参数都不是None
                    if 'slider_1' in kwargs and kwargs['slider_1'] is None:
                        # 检查预处理器是否需要特定的默认值
                        if hasattr(preprocessor, 'slider_1') and preprocessor.slider_1 is not None:
                            if hasattr(preprocessor.slider_1, 'gradio_update_kwargs'):
                                kwargs['slider_1'] = preprocessor.slider_1.gradio_update_kwargs.get('value', 0)
                        else:
                            kwargs['slider_1'] = 0
                    
                    if 'slider_2' in kwargs and kwargs['slider_2'] is None:
                        # 检查预处理器是否需要特定的默认值
                        if hasattr(preprocessor, 'slider_2') and preprocessor.slider_2 is not None:
                            if hasattr(preprocessor.slider_2, 'gradio_update_kwargs'):
                                kwargs['slider_2'] = preprocessor.slider_2.gradio_update_kwargs.get('value', 0)
                        else:
                            kwargs['slider_2'] = 0
                    
                    processed_image_array = preprocessor(image_array, **kwargs)
                else:
                    # 一些预处理器可能需要特殊的调用方式
                    processed_image_array = preprocessor(image_array)
                
                # 移除调试日志: print("预处理器调用成功")
                
                # 确保输出是正确的格式
                if isinstance(processed_image_array, tuple):
                    # 有些预处理器返回元组，第一个元素是图像
                    processed_image_array = processed_image_array[0]
                
                # 确保输出是numpy数组
                if not isinstance(processed_image_array, np.ndarray):
                    raise ValueError(f"预处理器返回了意外的类型: {type(processed_image_array)}")
                
                # 检查输出是否为空
                if processed_image_array.size == 0:
                    raise ValueError("预处理器返回了空数组")
                
                # 确保输出是3通道RGB图像
                if len(processed_image_array.shape) == 2:
                    # 灰度图转RGB
                    processed_image = cv2.cvtColor(processed_image_array, cv2.COLOR_GRAY2RGB)
                elif processed_image_array.shape[2] == 1:
                    # 单通道转RGB
                    processed_image = cv2.cvtColor(processed_image_array.squeeze(), cv2.COLOR_GRAY2RGB)
                elif processed_image_array.shape[2] == 3:
                    # 已经是RGB格式
                    processed_image = processed_image_array
                elif processed_image_array.shape[2] == 4:
                    # RGBA转RGB
                    processed_image = cv2.cvtColor(processed_image_array, cv2.COLOR_RGBA2RGB)
                else:
                    # 其他情况，默认使用原始输出
                    processed_image = processed_image_array
                
                # 确保输出数组是非空的
                if processed_image is not None and processed_image.size > 0:
                    return processed_image
                else:
                    print("预处理器返回了空结果")
                    return None
                
            except Exception as process_error:
                print(f"使用WebUI预处理器时出错: {process_error}")
                # 出错时不再回退，直接抛出异常
                raise
            
        except Exception as e:
            print(f"使用WebUI预处理器时出错: {e}")
            import traceback
            traceback.print_exc()
            # 不再回退到默认处理，直接抛出异常
            raise
        
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
        # 不再返回None，直接抛出异常
        raise