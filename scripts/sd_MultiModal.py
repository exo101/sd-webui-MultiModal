import os
import torch
from modules import devices
import gradio as gr
import numpy as np
import datetime
from pathlib import Path
from modules import script_callbacks
import webbrowser
import subprocess
import sys
import time
import importlib

# 添加scripts目录到系统路径，确保模块可以被正确加载
scripts_dir = Path(__file__).parent
if str(scripts_dir) is not None and str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# 尝试导入各个功能模块
def import_modules():
    """尝试导入所有必要的模块，并返回包含这些模块的命名空间对象"""
    def _import_and_register_modules():
        # 确保当前脚本目录在 Python 路径中
        script_dir = str(scripts_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
        try: from flux_kontext_ui import create_flux_kontext_ui, FLUX_KONTEXT_AVAILABLE
        except ImportError: 
            create_flux_kontext_ui = None
            FLUX_KONTEXT_AVAILABLE = False
            
        try: from announcement import create_announcement_module
        except ImportError:
            create_announcement_module = None
            
        try: 
            from qwen_image_ui import create_qwen_image_ui, QWEN_IMAGE_MODULE_AVAILABLE
        except ImportError: 
            create_qwen_image_ui = None
            QWEN_IMAGE_MODULE_AVAILABLE = False
            
        try: 
            from qwen_image_edit import create_qwen_image_edit_ui, QWEN_IMAGE_EDIT_MODULE_AVAILABLE
        except ImportError: 
            create_qwen_image_edit_ui = None
            QWEN_IMAGE_EDIT_MODULE_AVAILABLE = False
            
        
        # 尝试导入FLUX.1-krea模块
        try: 
            from flux_krea_ui import create_flux_krea_ui, FLUX_KREA_AVAILABLE
        except ImportError: 
            create_flux_krea_ui = None
            FLUX_KREA_AVAILABLE = False

        # 尝试导入FLUX.2-klein模块
        try: 
            from flux_klein_ui import create_flux_klein_ui, FLUX_KLEIN_AVAILABLE
        except ImportError: 
            create_flux_klein_ui = None
            FLUX_KLEIN_AVAILABLE = False

         # 添加Z-Image模块导入
        try: 
            from z_image_ui import create_z_image_ui as z_image_create_func, Z_IMAGE_MODULE_AVAILABLE
        except ImportError: 
            z_image_create_func = None
            Z_IMAGE_MODULE_AVAILABLE = False
            
            
        # 添加Qwen API模块导入
        try: 
            from qwen_api_ui import create_qwen_api_ui
        except ImportError: 
            create_qwen_api_ui = None

        # 添加Z-Image正式版模块导入
        try: 
            from z_image_deployer import create_z_image_deploy_ui, Z_IMAGE_MODULE_AVAILABLE as Z_IMAGE_DEPLOY_MODULE_AVAILABLE
        except ImportError: 
            create_z_image_deploy_ui = None
            Z_IMAGE_DEPLOY_MODULE_AVAILABLE = False

        # 添加美学提升模块导入
        try:
            from aesthetic_enhancement_ui import create_aesthetic_enhancement_ui, QWEN_MODULE_AVAILABLE as AESTHETIC_QWEN_AVAILABLE
        except ImportError:
            create_aesthetic_enhancement_ui = None
            AESTHETIC_QWEN_AVAILABLE = False

        # 添加分镜助手模块导入
        try:
            from storyboard_assistant import create_storyboard_assistant_module
        except ImportError:
            create_storyboard_assistant_module = None

        # 返回命名空间对象
        import types
        namespace = types.SimpleNamespace()
        namespace.create_flux_kontext_ui = create_flux_kontext_ui
        namespace.FLUX_KONTEXT_AVAILABLE = FLUX_KONTEXT_AVAILABLE
        namespace.create_announcement_module = create_announcement_module
        namespace.create_qwen_image_ui = create_qwen_image_ui
        namespace.QWEN_IMAGE_MODULE_AVAILABLE = QWEN_IMAGE_MODULE_AVAILABLE
        namespace.create_qwen_image_edit_ui = create_qwen_image_edit_ui
        namespace.QWEN_IMAGE_EDIT_MODULE_AVAILABLE = QWEN_IMAGE_EDIT_MODULE_AVAILABLE
        namespace.create_flux_krea_ui = create_flux_krea_ui
        namespace.FLUX_KREA_AVAILABLE = FLUX_KREA_AVAILABLE
        namespace.create_flux_klein_ui = create_flux_klein_ui
        namespace.FLUX_KLEIN_AVAILABLE = FLUX_KLEIN_AVAILABLE

        # 添加 Z-Image 模块到命名空间
        try:
            from z_image_ui import create_z_image_ui as z_image_create_func, Z_IMAGE_MODULE_AVAILABLE
        except ImportError: 
            z_image_create_func = None
            Z_IMAGE_MODULE_AVAILABLE = False

        namespace.create_z_image_ui = z_image_create_func
        namespace.Z_IMAGE_MODULE_AVAILABLE = Z_IMAGE_MODULE_AVAILABLE
        
        # 添加Z-Image正式版模块到命名空间
        try: 
            from z_image_deployer import create_z_image_deploy_ui, Z_IMAGE_MODULE_AVAILABLE as Z_IMAGE_DEPLOY_MODULE_AVAILABLE
        except ImportError: 
            create_z_image_deploy_ui = None
            Z_IMAGE_DEPLOY_MODULE_AVAILABLE = False

        namespace.create_z_image_deploy_ui = create_z_image_deploy_ui
        namespace.Z_IMAGE_DEPLOY_MODULE_AVAILABLE = Z_IMAGE_DEPLOY_MODULE_AVAILABLE
        
        # 添加Qwen API模块到命名空间
        try: 
            from qwen_api_ui import create_qwen_api_ui, QWEN_API_AVAILABLE
        except ImportError: 
            create_qwen_api_ui = None
            QWEN_API_AVAILABLE = False

        namespace.create_qwen_api_ui = create_qwen_api_ui
        namespace.QWEN_API_AVAILABLE = QWEN_API_AVAILABLE

        # 添加 FLUX_KLEIN 模块到命名空间
        try: 
            from flux_klein_ui import create_flux_klein_ui, FLUX_KLEIN_AVAILABLE
        except ImportError: 
            create_flux_klein_ui = None
            FLUX_KLEIN_AVAILABLE = False

        namespace.create_flux_klein_ui = create_flux_klein_ui
        namespace.FLUX_KLEIN_AVAILABLE = FLUX_KLEIN_AVAILABLE

        # 添加美学提升模块到命名空间
        try: 
            from aesthetic_enhancement_ui import create_aesthetic_enhancement_ui, QWEN_MODULE_AVAILABLE as AESTHETIC_QWEN_AVAILABLE
        except ImportError: 
            create_aesthetic_enhancement_ui = None
            AESTHETIC_QWEN_AVAILABLE = False
      
        namespace.create_aesthetic_enhancement_ui = create_aesthetic_enhancement_ui
        namespace.AESTHETIC_QWEN_AVAILABLE = AESTHETIC_QWEN_AVAILABLE
        
        # 添加分镜助手模块到命名空间
        try: 
            from storyboard_assistant import create_storyboard_assistant_module
        except ImportError: 
            create_storyboard_assistant_module = None
      
        namespace.create_storyboard_assistant_module = create_storyboard_assistant_module
        return namespace
        
    return _import_and_register_modules()

# 尝试导入所有模块
imported_modules = import_modules()

# 将导入的模块赋值给变量，方便在后续代码中使用
create_flux_kontext_ui = imported_modules.create_flux_kontext_ui
FLUX_KONTEXT_AVAILABLE = imported_modules.FLUX_KONTEXT_AVAILABLE
create_announcement_module = imported_modules.create_announcement_module
create_qwen_image_ui = imported_modules.create_qwen_image_ui
QWEN_IMAGE_MODULE_AVAILABLE = imported_modules.QWEN_IMAGE_MODULE_AVAILABLE
create_qwen_image_edit_ui = imported_modules.create_qwen_image_edit_ui
QWEN_IMAGE_EDIT_MODULE_AVAILABLE = imported_modules.QWEN_IMAGE_EDIT_MODULE_AVAILABLE
create_flux_krea_ui = imported_modules.create_flux_krea_ui
FLUX_KREA_AVAILABLE = imported_modules.FLUX_KREA_AVAILABLE
create_flux_klein_ui = imported_modules.create_flux_klein_ui
FLUX_KLEIN_AVAILABLE = imported_modules.FLUX_KLEIN_AVAILABLE
create_z_image_ui = imported_modules.create_z_image_ui
Z_IMAGE_MODULE_AVAILABLE = imported_modules.Z_IMAGE_MODULE_AVAILABLE
create_z_image_deploy_ui = imported_modules.create_z_image_deploy_ui
Z_IMAGE_DEPLOY_MODULE_AVAILABLE = imported_modules.Z_IMAGE_DEPLOY_MODULE_AVAILABLE
create_qwen_api_ui = imported_modules.create_qwen_api_ui
QWEN_API_AVAILABLE = imported_modules.QWEN_API_AVAILABLE
create_aesthetic_enhancement_ui = imported_modules.create_aesthetic_enhancement_ui
AESTHETIC_QWEN_AVAILABLE = imported_modules.AESTHETIC_QWEN_AVAILABLE
create_storyboard_assistant_module = imported_modules.create_storyboard_assistant_module

def MultiModal_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            # 重要公告标签页
            with gr.TabItem("1 资源汇总"):
                # 使用延迟渲染避免出现空模块问题   
                announcement_ui = create_announcement_module()
                if "markdown_content" in announcement_ui:
                    announcement_ui["markdown_content"]
            
            # 添加 FLUX 系列标签页
            with gr.TabItem("2.FLUX系列图像生成与编辑"):
                with gr.Tabs():
                    with gr.TabItem("kontext图像编辑"):
                        try:
                            if 'FLUX_KONTEXT_AVAILABLE' in globals() and FLUX_KONTEXT_AVAILABLE:
                                flux_kontext_components = create_flux_kontext_ui()
                                
                                if not flux_kontext_components:
                                    gr.Markdown("FLUX.1-Kontext模块加载失败")
                            else:
                                gr.Markdown("FLUX.1-Kontext模块当前不可用，可能是因为缺少模型文件或依赖项。")
                        except Exception as e:
                            gr.Markdown(f"FLUX.1-Kontext模块初始化错误: {e}")
                            import traceback
                            traceback.print_exc()
                
                    with gr.TabItem("FLUX.1-图像生成"):
                        try:
                            # 检查FLUX.1-krea模块是否可用
                            flux_krea_available = globals().get('FLUX_KREA_AVAILABLE', False)
                            if flux_krea_available:
                                # 创建FLUX.1-krea UI组件
                                flux_krea_components = create_flux_krea_ui()
                                
                                if not flux_krea_components:
                                    gr.Markdown("FLUX.1-krea模块加载失败")
                            else:
                                gr.Markdown("FLUX.1-krea模块当前不可用，可能是因为缺少模型文件或依赖项。")
                        except Exception as e:
                            gr.Markdown(f"FLUX.1-krea模块初始化错误: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    with gr.TabItem("FLUX.2-klein-图像生成"):
                        try:
                            # 检查FLUX.2-klein模块是否可用
                            flux_klein_available = globals().get('FLUX_KLEIN_AVAILABLE', False)
                            if flux_klein_available:
                                # 创建FLUX.2-klein UI组件
                                flux_klein_components = create_flux_klein_ui()
                                
                                if not flux_klein_components:
                                    gr.Markdown("FLUX.2-klein模块加载失败")
                            else:
                                gr.Markdown("FLUX.2-klein模块当前不可用，可能是因为缺少模型文件或依赖项。")
                        except Exception as e:
                            gr.Markdown(f"FLUX.2-klein模块初始化错误: {e}")
                            import traceback
                            traceback.print_exc()
            # 添加 Qwen Image 标签页（如果可用）
            if 'QWEN_IMAGE_MODULE_AVAILABLE' in globals() and QWEN_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("3.Qwen Image图像生成与编辑"):
                    try:
                        with gr.Tabs():
                            with gr.TabItem("文生图"):
                                # 创建 Qwen Image 文生图 UI 组件
                                qwen_image_components = create_qwen_image_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_image_components:
                                    gr.Markdown("Qwen Image文生图模块加载失败")
                            
                            with gr.TabItem("图像编辑"):
                                # 创建 Qwen Image 图像编辑 UI 组件
                                qwen_image_edit_components = create_qwen_image_edit_ui() if 'create_qwen_image_edit_ui' in globals() and QWEN_IMAGE_EDIT_MODULE_AVAILABLE else None
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_image_edit_components:
                                    gr.Markdown("Qwen Image图像编辑模块加载失败")
                            
                            with gr.TabItem("Qwen图像生成与编辑API调用"):
                                # 创建 Qwen API UI 组件
                                qwen_api_components = create_qwen_api_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not qwen_api_components:
                                    gr.Markdown("Qwen API模块加载失败")
                    except Exception as e:
                        gr.Markdown(f"Qwen Image模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'QWEN_IMAGE_MODULE_AVAILABLE' in globals() and not QWEN_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("5.nunchaku加速-Qwen Image图像生成与编辑"):
                    gr.Markdown("Qwen Image模块当前不可用，可能是因为缺少模型文件或依赖项。")
            
            # 添加 Z-Image-Turbo 标签页（如果可用）
            if 'Z_IMAGE_MODULE_AVAILABLE' in globals() and Z_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("4.Z-Image-Turbo图像生成"):
                    try:
                        with gr.Tabs():
                            with gr.TabItem("文生图"):
                                # 创建 Z-Image-Turbo 文生图 UI 组件
                                z_image_components = create_z_image_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not z_image_components:
                                    gr.Markdown("Z-Image-Turbo文生图模块加载失败")
                    except Exception as e:
                        gr.Markdown(f"Z-Image-Turbo模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'Z_IMAGE_MODULE_AVAILABLE' in globals() and not Z_IMAGE_MODULE_AVAILABLE:
                with gr.TabItem("6.Z-Image-Turbo图像生成"):
                    gr.Markdown("Z-Image-Turbo模块当前不可用，可能是因为缺少模型文件或依赖项。")
            
            # 添加 Z-Image 正式版标签页（作为第7个标签页）
            if 'Z_IMAGE_DEPLOY_MODULE_AVAILABLE' in globals() and Z_IMAGE_DEPLOY_MODULE_AVAILABLE:
                with gr.TabItem("5.Z-Image（base）"):
                    try:
                        with gr.Tabs():
                            with gr.TabItem("文生图"):
                                # 创建 Z-Image 正式版 文生图 UI 组件
                                z_image_deploy_components = create_z_image_deploy_ui()
                                
                                # 组件已经自动显示，无需额外处理
                                if not z_image_deploy_components:
                                    gr.Markdown("Z-Image（正式版）文生图模块加载失败")
                                    
                            with gr.TabItem("图生图"):
                                # 图生图功能在正式版UI中已经通过TabItem集成
                                gr.Markdown("图生图功能已在左侧标签中提供")
                    except Exception as e:
                        gr.Markdown(f"Z-Image（正式版）模块初始化错误: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'Z_IMAGE_DEPLOY_MODULE_AVAILABLE' in globals() and not Z_IMAGE_DEPLOY_MODULE_AVAILABLE:
                with gr.TabItem("7.Z-Image（正式版）"):
                    gr.Markdown("Z-Image（正式版）模块当前不可用，可能是因为缺少模型文件或依赖项。")
            
            # 添加第 8 个标签页：美学提升模块
            with gr.TabItem("6.美学提升"):
                try:
                    if create_aesthetic_enhancement_ui is not None:
                        # 创建美学提升 UI 组件
                        aesthetic_ui = create_aesthetic_enhancement_ui()
                        
                        if not aesthetic_ui:
                            gr.Markdown("美学提升模块加载失败")
                    else:
                        gr.Markdown("美学提升模块当前不可用。")
                except Exception as e:
                    gr.Markdown(f"美学提升模块初始化错误：{e}")
                    import traceback
                    traceback.print_exc()
            
            # 添加第 9 个标签页：分镜助手（集成所有功能的工作流管理）
            with gr.TabItem("7.分镜助手"):
                try:
                    if create_storyboard_assistant_module is not None:
                        # 创建分镜助手 UI 组件（不需要传递参数）
                        storyboard_assistant_ui = create_storyboard_assistant_module()
                        
                        # 组件已经自动显示，无需额外处理
                        if not storyboard_assistant_ui:
                            gr.Markdown("分镜助手模块加载失败")
                    else:
                        gr.Markdown("分镜助手模块当前不可用。")
                except Exception as e:
                    gr.Markdown(f"分镜助手模块初始化错误：{e}")
                    import traceback
                    traceback.print_exc()

            # 注释掉已移动的标签页：wan系列视频生成API调用 (10 -> 7)
    return [(ui, "多模态图像处理15", "MultiModal_vision_tab")]
                  
script_callbacks.on_ui_tabs(MultiModal_tab)

# 在 WebUI 启动时在后台日志中显示插件信息和使用声明
def on_app_started(*args, **kwargs):
    print("=" * 60)
    print("多模态图像处理插件 - forge 版本专用")
    print("开发者：鸡肉爱土豆")
    print("网址：https://space.bilibili.com/403361177")
    print("声明：为创作者提供更便捷更强大无复杂工作流的插件")
    print()
    print("集成功能：")
    print("- nunchuku 加速模型")
    print("- FLUX.2-klein 图像生成与编辑")
    print("- Z-Image-Turbo图像生成")
    print("- Qwen 图像生成与编辑")
    print("- Qwen 系列 api 调用")
    print("- Qwen 系列图像识别与语言交互")
    print("- 美学提升 - 构图素材库（新增）")
    print("- 分镜助手 - 一体化工作流管理（新增）")
    print()
    print("使用须知：使用此插件者请合法使用 AI。")
    print("=" * 60)

script_callbacks.on_app_started(on_app_started)

# 检查模块状态
modules_status = {
    'flux_kontext': FLUX_KONTEXT_AVAILABLE,
    'qwen_image': QWEN_IMAGE_MODULE_AVAILABLE,
    'qwen_image_edit': QWEN_IMAGE_EDIT_MODULE_AVAILABLE,
    'qwen_api': QWEN_API_AVAILABLE,
    'flux_krea': FLUX_KREA_AVAILABLE,
    'z_image': Z_IMAGE_MODULE_AVAILABLE,
    'storyboard_assistant': create_storyboard_assistant_module is not None,

}
