import gradio as gr
from pathlib import Path
import os
import json
import datetime
from PIL import Image
import io

# ==================== 发送到分镜功能 ====================
# 这个函数可以被其他图像生成模块调用，用于将生成的图片发送到分镜助手

def send_to_storyboard(image_path, description=""):
    """
    将生成的图片发送到分镜助手
    
    Args:
        image_path: 图片路径、PIL Image 对象、或包含路径的元组/字典
        description: 分镜描述（可选）
    
    Returns:
        dict: {
            'success': bool,  # 成功标志
            'message': str,   # 消息
            'index': int,     # 分镜索引（从 0 开始）
            'total_count': int,  # 总分镜数
            'target_page': int   # 应该在第几页显示（最后一页）
        }
    """
    try:
        # 获取分镜数据目录
        data_dir = Path(__file__).parent / "storyboard_data"
        data_dir.mkdir(exist_ok=True)
        
        storyboard_file = data_dir / "storyboard.json"
        
        # 加载分镜数据
        def load_storyboard_data():
            if storyboard_file.exists():
                try:
                    with open(storyboard_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            return []
                        return json.loads(content)
                except Exception as e:
                    print(f"⚠️ 加载分镜数据失败：{e}")
                    return []
            return []
        
        # 保存分镜数据
        def save_storyboard_data(data):
            try:
                with open(storyboard_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"❌ 保存分镜数据失败：{e}")
                return False
        
        # 提取真实图片路径（处理元组、字典等情况）
        def extract_image_path(path):
            """从各种格式中提取真实的图片路径"""
            if path is None:
                return None
            elif isinstance(path, Image.Image):
                # PIL Image 对象
                return path
            elif isinstance(path, dict):
                # 字典格式，提取 'name' 字段
                return path.get('name')
            elif isinstance(path, (list, tuple)):
                # 元组或列表，取第一个元素（通常是路径）
                if len(path) > 0:
                    return extract_image_path(path[0])
                return None
            elif isinstance(path, str):
                # 字符串路径
                return path
            else:
                # 其他类型，尝试直接返回
                return path
        
        # 处理图片函数（不压缩，直接复制）
        def process_image_for_storyboard(image_path):
            """将图片复制到分镜临时目录，保持原始质量和尺寸"""
            try:
                # 提取真实路径
                real_path = extract_image_path(image_path)
                
                if real_path is None:
                    return None
                
                if isinstance(real_path, Image.Image):
                    img = real_path.convert('RGB')
                elif isinstance(real_path, str) and os.path.exists(real_path):
                    img = Image.open(real_path).convert('RGB')
                else:
                    print(f"⚠️ 无效的图片路径：{real_path}")
                    return None
                
                # 保存到临时目录（保持原始质量）
                output_dir = data_dir / "temp_images"
                output_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"storyboard_{timestamp}.png"
                
                img.save(output_path, "PNG")  # 使用 PNG 格式保存，无损质量
                return str(output_path)
            
            except Exception as e:
                print(f"⚠️ 处理图片失败：{e}")
                import traceback
                traceback.print_exc()
                return None
        
        # 处理图片
        processed_path = process_image_for_storyboard(image_path)
        if not processed_path:
            return {
                'success': False,
                'message': "❌ 图片处理失败",
                'index': -1,
                'total_count': 0,
                'target_page': 1
            }
        
        # 加载现有数据
        storyboard_data = load_storyboard_data()
        
        # 计算新的分镜索引
        new_index = len(storyboard_data)
        
        # 添加新的分镜记录
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_storyboard = {
            "id": new_index,
            "image_path": processed_path,
            "aspect_ratio": "16:9 (宽屏)",
            "description": description if description else "",
            "timestamp": timestamp
        }
        
        storyboard_data.append(new_storyboard)
        
        # 保存数据
        success = save_storyboard_data(storyboard_data)
        
        # 计算分页信息
        STORYBOARDS_PER_PAGE = 9  # 每页 9 个宫格
        total_count = len(storyboard_data)
        target_page = max(1, (total_count + STORYBOARDS_PER_PAGE - 1) // STORYBOARDS_PER_PAGE)
        
        if success:
            return {
                'success': True,
                'message': f"✅ 已添加到分镜 #{new_index + 1}",
                'index': new_index,
                'total_count': total_count,
                'target_page': target_page
            }
        else:
            return {
                'success': False,
                'message': "❌ 保存分镜数据失败",
                'index': new_index,
                'total_count': total_count,
                'target_page': target_page
            }
    
    except Exception as e:
        print(f"❌ 发送到分镜失败：{e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f"❌ 错误：{str(e)}",
            'index': -1,
            'total_count': 0,
            'target_page': 1
        }


def create_send_to_storyboard_button(image_component=None, description_component=None):
    """
    创建发送到分镜的按钮组件
    
    Args:
        image_component: 图像组件（用于获取生成的图片）
        description_component: 描述文本框组件（可选，用于获取分镜描述）
    
    Returns:
        ToolButton: 发送到分镜按钮
    """
    send_btn = ToolButton(
        value="📤 发送到分镜",
        visible=True,
        elem_classes=["tool-button"],
        tooltip="将当前生成的图片发送到分镜助手"
    )
    
    return send_btn


def create_storyboard_assistant_module():
    """
    分镜助手 - 专业的剧本与分镜管理系统
    
    核心架构：
    1. 左侧：剧本文字排版（多故事、多角色）
    2. 右侧：25 宫格分镜墙（可视化编排）
    """
    
    # 分页常量
    CHARACTERS_PER_PAGE = 6  # 角色每页显示数量
    STORYBOARDS_PER_PAGE = 9  # 分镜每页显示数量（3×3 布局）
    
    result = {}
    
    # 数据文件路径
    data_dir = Path(__file__).parent / "storyboard_data"
    data_dir.mkdir(exist_ok=True)
    
    script_file = data_dir / "script.json"
    storyboard_file = data_dir / "storyboard.json"
    
    # 图片压缩函数
    def compress_image(image_path, max_size=1024, quality=85):
        """
        压缩图片，保持宽高比
        
        Args:
            image_path: 原图路径
            max_size: 最大边长（像素）
            quality: JPEG 质量 (1-100)
        
        Returns:
            压缩后的图片路径
        """
        try:
            if not image_path or not os.path.exists(image_path):
                return None
            
            # 打开图片
            img = Image.open(image_path)
            
            # 转换为 RGB 模式（移除 alpha 通道）
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 计算缩放比例
            width, height = img.size
            if max(width, height) > max_size:
                ratio = max_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 生成压缩后的文件名
            original_name = Path(image_path).stem
            compressed_path = data_dir / "character_images" / f"{original_name}_compressed.jpg"
            compressed_path.parent.mkdir(exist_ok=True)
            
            # 保存压缩图片
            img.save(compressed_path, 'JPEG', quality=quality, optimize=True)
            
            return str(compressed_path)
        except Exception as e:
            print(f"图片压缩失败：{e}")
            return None
    
    # 加载/保存数据的辅助函数
    def load_data(file_path, default=None):
        if default is None:
            default = {"stories": {}} if file_path == script_file else []
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 处理空文件的情况
                    if not content.strip():
                        print(f"警告：文件 {file_path} 为空，使用默认值")
                        return default
                    data = json.loads(content)
                    
                    # 数据迁移：如果存在全局 characters 字段，迁移到各故事下
                    if file_path == script_file and "characters" in data and "stories" in data:
                        # 检查是否有全局角色数据需要迁移
                        global_chars = data.get("characters", {})
                        if isinstance(global_chars, (list, dict)) and len(global_chars) > 0:
                            # 将所有角色迁移到第一个故事（如果有）
                            stories = data.get("stories", {})
                            if stories:
                                # 取第一个故事作为默认归属
                                first_story_key = list(stories.keys())[0]
                                if first_story_key not in data["stories"]:
                                    data["stories"][first_story_key] = {}
                                
                                if "characters" not in data["stories"][first_story_key]:
                                    data["stories"][first_story_key]["characters"] = {}
                                
                                # 合并角色数据
                                if isinstance(global_chars, list):
                                    for char in global_chars:
                                        if "name" in char:
                                            name = char["name"]
                                            data["stories"][first_story_key]["characters"][name] = char
                                elif isinstance(global_chars, dict):
                                    for name, char in global_chars.items():
                                        data["stories"][first_story_key]["characters"][name] = char
                                
                                # 清除全局 characters 字段
                                del data["characters"]
                                save_data(file_path, data)
                                print(f"✅ 已将全局角色数据迁移到故事《{first_story_key}》")
                    
                    return data
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析失败：{file_path} - {e}，使用默认值")
            return default
        except Exception as e:
            print(f"加载数据失败：{e}")
        return default
    
    def save_data(file_path, data):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存数据失败：{e}")
            return False
    
    # 分页相关函数
    
    def get_page_characters(current_story, page=1):
        """获取指定故事的指定页的角色列表"""
        script_data = load_data(script_file)
        
        # 如果没有选中故事，返回空列表
        if not current_story:
            return [], 1, 0
        
        stories = script_data.get("stories", {})
        
        # 获取当前故事的角色列表
        characters = {}
        if current_story in stories:
            characters = stories[current_story].get("characters", {})
        
        # 兼容旧数据格式（列表）和新格式（字典）
        if isinstance(characters, list):
            # 旧格式：characters 是列表 [{"name": "艾伦", ...}, ...]
            char_names = [char.get("name", "") for char in characters if "name" in char]
        elif isinstance(characters, dict):
            # 新格式：characters 是字典 {"艾伦": {...}, ...}
            char_names = list(characters.keys())
        else:
            char_names = []
        
        # 计算分页
        total_count = len(char_names)
        total_pages = max(1, (total_count + CHARACTERS_PER_PAGE - 1) // CHARACTERS_PER_PAGE)
        page = max(1, min(page, total_pages))
        
        # 获取当前页的角色列表
        start_index = (page - 1) * CHARACTERS_PER_PAGE
        end_index = start_index + CHARACTERS_PER_PAGE
        page_chars = char_names[start_index:end_index]
        
        return page_chars, page, total_pages
    
    # 画幅比例预设
    aspect_ratios = [
        "16:9 (宽屏)", "9:16 (竖屏)", "1:1 (正方形)",
        "4:3 (标准)", "3:4 (竖版)", "21:9 (超宽屏)"
    ]
    
    with gr.Blocks() as ui:
        gr.Markdown("""
        # 🎬 分镜助手 - 专业版
        
        **左侧**：剧本文字排版（多故事管理 · 多角色小传） | **右侧**：25 宫格分镜墙（可视化编排 · 即点即编）
        
        ---
        """)
        
        # 主界面：左右分栏
        with gr.Row():
            
            # ========== 左侧：剧本创作区 ==========
            with gr.Column(scale=1, min_width=500):
                gr.Markdown("### 📖 剧本管理")
                
                # 故事选择器
                with gr.Row():
                    story_selector = gr.Dropdown(
                        label="选择故事",
                        choices=[],
                        interactive=True,
                        allow_custom_value=True,
                        scale=3
                    )
                    new_story_btn = gr.Button("➕ 新建", size="sm", scale=1)
                
                # 剧本文本编辑区
                with gr.Box():
                    gr.Markdown("**📝 剧本文本编辑器**")
                    
                    # 故事标题和类型
                    with gr.Row():
                        story_title_input = gr.Textbox(
                            label="故事标题",
                            placeholder="输入故事名称",
                            lines=1,
                            value=""
                        )
                        story_genre = gr.Dropdown(
                            label="题材类型",
                            choices=[
                                "奇幻", "科幻", "爱情", "动作", "悬疑",
                                "恐怖", "喜剧", "冒险", "历史", "其他"
                            ],
                            value="奇幻",
                            interactive=True
                        )
                    
                    # 完整的剧本编辑器（用于编辑）
                    full_script_editor = gr.Textbox(
                        label="完整剧本编辑器",
                        placeholder="【格式示例】\n\n第一幕 - 场景名称\n时间：日/夜/黄昏\n地点：具体场所\n人物：出场角色列表\n\n△ 动作描述或环境描写\n（括号内为情绪提示）\n\n角色 A:（情绪）台词内容...\n角色 B:回应台词...\n\n---\n\n第二幕 - 新的场景\n...",
                        lines=20,
                        max_lines=50,
                        visible=True
                    )
                    
                # 剧本操作按钮
                with gr.Row():
                    save_script_btn = gr.Button("💾 保存剧本", variant="primary", size="md")
                    delete_story_btn = gr.Button("🗑️ 删除此故事", variant="stop", size="md")
                    export_script_btn = gr.Button("📄 导出剧本", variant="secondary", size="md")
                
                # 角色小传管理（可折叠）
                with gr.Accordion("👥 角色小传（展开查看）", open=False):
                    gr.Markdown("**多角色档案管理** - 属于当前故事的角色列表（每页最多 6 个角色）")
                    
                    # 分页控制栏
                    with gr.Row():
                        prev_page_btn = gr.Button("◀ 上一页", size="sm", variant="secondary")
                        page_indicator = gr.Number(label="当前页", value=1, interactive=False, precision=0)
                        next_page_btn = gr.Button("下一页 ▶", size="sm", variant="secondary")
                    
                    # 角色选择器
                    with gr.Row():
                        char_selector = gr.Dropdown(
                            label="选择角色",
                            choices=[],
                            interactive=True,
                            allow_custom_value=True,
                            scale=4
                        )
                        new_char_btn = gr.Button("➕ 新建角色", size="sm", scale=1)
                    
                    # 角色配图和编辑器（左右分栏）
                    with gr.Row():
                        # 左侧：角色配图
                        with gr.Column(scale=1):
                            character_image = gr.Image(
                                label="角色配图",
                                height=400,
                                sources=["upload", "clipboard"],
                                type="filepath"
                            )
                            
                            # 图片操作按钮
                            with gr.Row():
                                delete_char_img_btn = gr.Button("🗑️ 删除配图", size="sm", variant="stop")
                        
                        # 右侧：角色文本编辑器
                        with gr.Column(scale=1):
                            character_editor = gr.Textbox(
                                label="角色详情编辑器",
                                placeholder="""【角色名称】

年龄：
性别：

【外貌特征】
身高、体型、发型、五官特点、穿着风格...

【性格特征】
内向/外向、勇敢/怯懦、乐观/悲观、优点、缺点...

【背景故事】
出身、家庭背景、成长经历、重要事件、转折点...

【行为动机】
表面目标、内心渴望、深层需求、人生信条...

【角色关系】
与其他角色的关系网...

【备注】
其他补充信息...""",
                                lines=18,
                                max_lines=30
                            )
                    
                    # 角色操作按钮
                    with gr.Row():
                        save_char_btn = gr.Button("💾 保存角色档案", variant="primary", size="md")
                        delete_char_btn = gr.Button("🗑️ 删除此角色", variant="stop", size="md")
                
                # 剧本状态显示
                script_status = gr.Textbox(
                    label="操作状态",
                    interactive=False,
                    visible=True
                )
            
            # ========== 右侧：25 宫格分镜墙 ==========
            with gr.Column(scale=2, min_width=800):
                gr.Markdown("### 🎨 分镜墙（每页 9 个）")
                
                # 顶部工具栏和分页控制
                with gr.Row():
                    export_all_btn = gr.Button("📤 导出分镜", size="lg")
                
                # 分页控制栏
                with gr.Row():
                    storyboard_prev_page_btn = gr.Button("◀ 上一页", size="sm", variant="secondary")
                    storyboard_current_page_num = gr.Number(label="当前页", value=1, interactive=False, precision=0)
                    storyboard_total_pages_num = gr.Number(label="总页数", value=1, interactive=False, precision=0)
                    storyboard_next_page_btn = gr.Button("下一页 ▶", size="sm", variant="secondary")
                    add_storyboard_btn = gr.Button("➕ 添加分镜", size="sm", variant="primary")
                
                # 分镜使用说明
                gr.Markdown("**使用说明**：点击宫格图片可上传/替换图片，支持剪贴板粘贴。每个宫格都有独立的删除按钮。使用分页导航查看更多分镜。")
                
                # 辅助函数：创建单个宫格组件
                def create_storyboard_cell(index):
                    with gr.Box() as cell_box:
                        # 使用 HTML 组件显示编号（可以被动态更新）
                        cell_label = gr.HTML(
                            value=f'<div style="text-align: center; font-weight: bold; margin-bottom: 5px;">#{index}</div>'
                        )
                        
                        # 图片区域 - 支持上传和剪贴板
                        cell_img = gr.Image(
                            label=f"分镜{index}",
                            height=160,
                            show_label=False,
                            sources=["upload", "clipboard"],
                            type="filepath"
                        )
                        
                        cell_annotation = gr.Textbox(
                            placeholder=f"注释...",
                            lines=2,
                            max_lines=2,
                            show_label=False,
                            container=False
                        )
                        
                        # 宫格操作按钮 - 只保留删除
                        with gr.Row():
                            cell_delete = gr.Button(
                                "🗑️ 删除",
                                size="xs",
                                variant="stop"
                            )
                    return cell_img, cell_annotation, cell_delete, cell_label

                # 初始化存储列表（每页 9 个宫格，3×3 布局）
                gallery_cells = []
                cell_annotations = []
                cell_delete_btns = []
                cell_labels = []  # 存储 HTML 标签组件
                
                # 使用 3 列 Grid 布局创建 9 个宫格（3 行×3 列）
                for row_idx in range(3):
                    with gr.Row():
                        for col_idx in range(3):
                            cell_index = row_idx * 3 + col_idx + 1
                            img, ann, del_btn, label_html = create_storyboard_cell(cell_index)
                            gallery_cells.append(img)
                            cell_annotations.append(ann)
                            cell_delete_btns.append(del_btn)
                            cell_labels.append(label_html)
                
                # 底部操作栏（全局操作）
                with gr.Row():
                    clear_all_btn = gr.Button("🗑️ 清空全部", variant="stop", size="md")
                
                # 状态栏
                status_bar = gr.Textbox(
                    label="操作状态",
                    interactive=False,
                    lines=1
                )
                
                # 隐藏的刷新触发器（用于解决无限刷新问题）
                refresh_trigger = gr.Number(value=0, visible=False)
        
        # ========== 事件处理 ==========
        
        # --- 1. 剧本管理功能 ---
        
        def refresh_story_list():
            """刷新故事列表"""
            script_data = load_data(script_file)
            story_names = list(script_data.get("stories", {}).keys())
            return gr.update(choices=story_names, value=story_names[0] if story_names else "")
        
        def load_story(story_name):
            """加载选中的故事"""
            script_data = load_data(script_file)
            stories = script_data.get("stories", {})
            
            if story_name in stories:
                story = stories[story_name]
                return [
                    story.get("title", story_name),
                    story.get("genre", "奇幻"),
                    story.get("script", "")  # 使用 script 字段而不是 content
                ]
            else:
                return ["", "奇幻", ""]
        
        def save_story(title, genre, content, current_story):
            """保存当前故事"""
            if not title.strip():
                return "❌ 请输入故事标题", gr.update()
            
            script_data = load_data(script_file)
            
            # 防御性检查：确保 scriptData 包含必要的键
            if "stories" not in script_data or not isinstance(script_data.get("stories"), dict):
                script_data["stories"] = {}
            
            # 如果当前没有选择故事，创建新的
            story_key = current_story if current_story else title
            
            script_data["stories"][story_key] = {
                "title": title,
                "genre": genre,
                "script": content,  # 使用 script 字段存储剧本文本
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            save_data(script_file, script_data)
            
            # 更新故事列表
            story_names = list(script_data["stories"].keys())
            
            return f"✅ 故事《{title}》已保存", gr.update(choices=story_names, value=story_key)
        
        def create_new_story():
            """新建故事"""
            # 清空所有输入框，准备创建新故事
            return [
                "",  # 清空下拉框选择
                "",  # 清空故事标题
                "奇幻",  # 重置类型为默认值
                ""   # 清空剧本编辑器
            ]
        
        def delete_story(current_story):
            """删除当前故事"""
            if not current_story:
                return "❌ 请先选择一个故事", gr.update()
            
            script_data = load_data(script_file)
            
            # 防御性检查：确保 scriptData 包含必要的键
            if "stories" not in script_data or not isinstance(script_data.get("stories"), dict):
                return "❌ 没有可用的故事", gr.update()
            
            if current_story in script_data["stories"]:
                del script_data["stories"][current_story]
                save_data(script_file, script_data)
                
                story_names = list(script_data["stories"].keys())
                return f"✅ 已删除故事《{current_story}》", gr.update(choices=story_names, value=story_names[0] if story_names else "")
            
            return "❌ 故事不存在", gr.update()
        
        def export_script(current_story):
            """导出剧本为 TXT 文件（包含该故事的所有角色）"""
            if not current_story:
                return "❌ 请先选择一个故事"
            
            script_data = load_data(script_file)
            
            # 防御性检查
            if "stories" not in script_data or not isinstance(script_data.get("stories"), dict):
                return "❌ 没有可用的故事"
            
            if current_story not in script_data["stories"]:
                return "❌ 故事不存在"
            
            story = script_data["stories"][current_story]
            title = story.get("title", current_story)
            genre = story.get("genre", "奇幻")
            content = story.get("script", "")  # 使用 script 字段读取剧本文本
            
            try:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                export_dir = data_dir / "script_exports" / f"{title}_{timestamp}"
                export_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建 TXT 文件
                txt_file = export_dir / f"{title}.txt"
                
                # 复制角色配图到导出目录
                characters = story.get("characters", {})
                if isinstance(characters, dict) and characters:
                    images_dir = export_dir / "character_images"
                    images_dir.mkdir(exist_ok=True)
                    
                    for char_name, char_data in characters.items():
                        img_path = char_data.get("image_path")
                        if img_path and os.path.exists(img_path):
                            try:
                                # 复制图片到导出目录
                                import shutil
                                img_filename = os.path.basename(img_path)
                                new_img_path = images_dir / img_filename
                                shutil.copy2(img_path, new_img_path)
                                print(f"✅ 已复制角色 {char_name} 的配图：{new_img_path}")
                            except Exception as e:
                                print(f"⚠️ 复制角色 {char_name} 的配图失败：{e}")
                
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 50 + "\n")
                    f.write(f"剧本：《{title}》\n")
                    f.write(f"题材类型：{genre}\n")
                    f.write(f"导出时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # 导出当前故事的角色档案
                    if isinstance(characters, dict) and characters:
                        f.write("\n【角色档案】\n")
                        f.write("-" * 50 + "\n")
                        for char_name, char_data in characters.items():
                            char_content = char_data.get("content", "").strip()
                            img_path = char_data.get("image_path", "")
                            
                            f.write(f"\n【{char_name}】\n")
                            if img_path:
                                f.write(f"📷 配图：{os.path.basename(img_path)}\n")
                            if char_content:
                                # 如果内容第一行已经包含角色名，跳过该行避免重复
                                lines = char_content.split('\n')
                                if lines and (lines[0].startswith(char_name + '，') or 
                                             lines[0].startswith(char_name + ':') or 
                                     lines[0].startswith(char_name + ':')):
                                    # 跳过第一行（包含角色名的行），从第二行开始输出
                                    remaining_content = '\n'.join(lines[1:])
                                    if remaining_content.strip():
                                        f.write(f"{remaining_content}\n")
                                else:
                                    # 第一行不包含角色名，原样输出
                                    f.write(f"{char_content}\n")
                            f.write("-" * 50 + "\n")
                    
                    f.write("\n【完整剧本】\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(content)
                
                return f"✅ 剧本《{title}》已导出到：{export_dir}"
            except Exception as e:
                return f"❌ 导出失败：{e}"
        
        # --- 2. 角色管理功能 ---
        
        def refresh_character_list(current_story, page=1):
            """刷新角色选择器（仅显示当前故事的角色）"""
            script_data = load_data(script_file)
            
            # 如果没有选中故事，返回空列表
            if not current_story:
                return gr.update(choices=[], value=""), 1, 0
            
            stories = script_data.get("stories", {})
            
            # 获取当前故事的角色列表
            characters = {}
            if current_story in stories:
                characters = stories[current_story].get("characters", {})
            
            # 兼容旧数据格式（列表）和新格式（字典）
            if isinstance(characters, list):
                # 旧格式：characters 是列表 [{"name": "艾伦", ...}, ...]
                char_names = [char.get("name", "") for char in characters if "name" in char]
            elif isinstance(characters, dict):
                # 新格式：characters 是字典 {"艾伦": {...}, ...}
                char_names = list(characters.keys())
            else:
                char_names = []
            
            # 计算分页
            total_count = len(char_names)
            total_pages = max(1, (total_count + CHARACTERS_PER_PAGE - 1) // CHARACTERS_PER_PAGE)
            page = max(1, min(page, total_pages))
            
            return gr.update(choices=char_names, value=char_names[0] if char_names else ""), page, total_pages
        
        def save_character(editor_content, char_selector, current_story, char_image):
            """保存角色档案到当前故事"""
            # 优先使用用户在下拉框中选择的名称
            if char_selector and char_selector.strip():
                char_name = char_selector.strip()
            else:
                # 如果下拉框为空，尝试从编辑器内容中提取角色名（第一行）
                lines = editor_content.strip().split('\n')
                extracted_name = ""
                
                for line in lines:
                    line = line.strip()
                    # 跳过空行和特殊标记行
                    if not line or line.startswith('[') or line.startswith('年龄') or line.startswith('性别'):
                        continue
                    
                    # 提取冒号前的内容作为角色名
                    if ':' in line:
                        extracted_name = line.split(':')[0].strip()
                    elif ':' in line:
                        extracted_name = line.split(':')[0].strip()
                    else:
                        # 如果没有冒号，取整行作为角色名（但要截断过长的部分）
                        extracted_name = line.strip()
                    
                    # 角色名不应该太长（通常不超过 20 个字符）
                    if len(extracted_name) > 20:
                        extracted_name = extracted_name[:20].strip()
                    
                    break
                
                char_name = extracted_name
            
            if not char_name:
                return "❌ 无法识别角色名称，请在第一行输入角色名（如：陈旭）", gr.update(), None
            
            if not current_story:
                return "❌ 请先选择或创建一个故事", gr.update(), None
            
            script_data = load_data(script_file)
            
            # 防御性检查：确保 stories 存在
            if "stories" not in script_data or not isinstance(script_data.get("stories"), dict):
                script_data["stories"] = {}
            
            # 确保当前故事存在
            if current_story not in script_data["stories"]:
                script_data["stories"][current_story] = {
                    "title": current_story,
                    "genre": "奇幻",
                    "content": "",
                    "characters": {},
                    "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # 压缩图片
            compressed_image_path = None
            if char_image:
                compressed_image_path = compress_image(char_image)
            
            character = {
                "name": char_name,
                "content": editor_content,
                "image_path": compressed_image_path,
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 确保当前故事有 characters 字段
            if "characters" not in script_data["stories"][current_story]:
                script_data["stories"][current_story]["characters"] = {}
            
            current_characters = script_data["stories"][current_story]["characters"]
            
            # 兼容旧数据格式（列表）和新格式（字典）
            if isinstance(current_characters, list):
                # 如果是列表格式，转换为字典格式
                characters_dict = {}
                for char in current_characters:
                    if "name" in char:
                        name = char["name"]
                        characters_dict[name] = char
                script_data["stories"][current_story]["characters"] = characters_dict
                current_characters = characters_dict
            
            # 确保 characters 是字典
            if not isinstance(current_characters, dict):
                script_data["stories"][current_story]["characters"] = {}
                current_characters = {}
            
            current_characters[char_name] = character
            save_data(script_file, script_data)
            
            # 更新角色列表（仅当前故事的角色）- 确保只返回角色名
            char_names = list(current_characters.keys())
            
            # 返回时，确保下拉框的 value 只是角色名，不是完整内容
            return f"✅ 角色 {char_name} 已保存到故事《{current_story}》", gr.update(choices=char_names, value=char_name), compressed_image_path
        
        def load_character(char_name, current_story):
            """加载当前故事的角色信息"""
            if not current_story or not char_name:
                return "", None
            
            script_data = load_data(script_file)
            stories = script_data.get("stories", {})
            
            if current_story not in stories:
                return "", None
            
            characters = stories[current_story].get("characters", {})
            
            # 兼容旧数据格式（列表）
            if isinstance(characters, list):
                # 在列表中查找匹配的角色名
                for char in characters:
                    if char.get("name") == char_name:
                        return char.get("content", ""), char.get("image_path")
                return "", None
            
            # 新数据格式（字典）
            if char_name in characters:
                char = characters[char_name]
                return (
                    char.get("content", ""),
                    char.get("image_path")
                )
            
            return "", None
        
        def create_new_character():
            """新建角色"""
            return gr.update(value="")
        
        def delete_character(char_name, current_story):
            """删除当前故事的角色"""
            if not char_name:
                return "❌ 请先选择一个角色", gr.update(), None
            
            if not current_story:
                return "❌ 请先选择一个故事", gr.update(), None
            
            script_data = load_data(script_file)
            stories = script_data.get("stories", {})
            
            if current_story not in stories:
                return "❌ 故事不存在", gr.update(), None
            
            characters = stories[current_story].get("characters", {})
            
            # 兼容旧数据格式（列表）
            if isinstance(characters, list):
                # 从列表中删除
                for i, char in enumerate(characters):
                    if char.get("name") == char_name:
                        del characters[i]
                        save_data(script_file, script_data)
                        char_names = [c.get("name", "") for c in characters if "name" in c]
                        return f"✅ 已删除角色 {char_name}", gr.update(choices=char_names, value=char_names[0] if char_names else ""), None
            
            # 新数据格式（字典）
            if char_name in characters:
                del characters[char_name]
                save_data(script_file, script_data)
                char_names = list(characters.keys())
                return f"✅ 已删除角色 {char_name}", gr.update(choices=char_names, value=char_names[0] if char_names else ""), None
            
            return "❌ 角色不存在", gr.update(), None
        
        def delete_character_image(current_story, char_name):
            """删除角色的配图"""
            if not current_story or not char_name:
                return "❌ 请先选择角色", None
            
            script_data = load_data(script_file)
            stories = script_data.get("stories", {})
            
            if current_story not in stories:
                return "❌ 故事不存在", None
            
            characters = stories[current_story].get("characters", {})
            
            if isinstance(characters, dict) and char_name in characters:
                # 删除图片文件
                old_image = characters[char_name].get("image_path")
                if old_image and os.path.exists(old_image):
                    try:
                        os.remove(old_image)
                    except:
                        pass
                
                # 更新数据
                characters[char_name]["image_path"] = None
                save_data(script_file, script_data)
                
                return f"✅ 已删除角色 {char_name} 的配图", None
            
            return "❌ 角色不存在", None
        
        # --- 3. 分镜管理功能 ---
        
        def handle_image_upload(img_path, cell_index):
            """处理宫格图片上传（从 Image 组件）"""
            if not img_path:
                return f"❌ 未选择图片", *refresh_gallery()
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            storyboard_data = load_data(storyboard_file)
            
            # 确保列表足够长
            while len(storyboard_data) <= cell_index:
                storyboard_data.append({})
            
            # 更新指定宫格
            storyboard_data[cell_index] = {
                "id": cell_index,
                "image_path": img_path,
                "aspect_ratio": "16:9 (宽屏)",
                "description": storyboard_data[cell_index].get("description", "") if cell_index < len(storyboard_data) else "",
                "timestamp": timestamp
            }
            
            save_data(storyboard_file, storyboard_data)
            
            return f"✅ 已更新分镜 #{cell_index + 1}", *refresh_gallery()
        
        def delete_cell(cell_index):
            """删除指定宫格的分镜"""
            storyboard_data = load_data(storyboard_file)
            
            if cell_index < len(storyboard_data):
                # 清空该宫格数据
                storyboard_data[cell_index] = {
                    "id": cell_index,
                    "image_path": None,
                    "aspect_ratio": "",
                    "description": "",
                    "timestamp": ""
                }
                
                save_data(storyboard_file, storyboard_data)
                return f"✅ 已删除分镜 #{cell_index + 1}", *refresh_gallery()
            else:
                return "⚠️ 该位置没有分镜", *(refresh_gallery())
        
        def update_annotation(text, cell_index):
            """更新宫格注释"""
            storyboard_data = load_data(storyboard_file)
            
            # 确保列表足够长
            while len(storyboard_data) <= cell_index:
                storyboard_data.append({})
            
            # 更新注释
            if cell_index < len(storyboard_data):
                storyboard_data[cell_index]["description"] = text
            else:
                storyboard_data[cell_index] = {
                    "id": cell_index,
                    "image_path": None,
                    "aspect_ratio": "",
                    "description": text,
                    "timestamp": ""
                }
            
            save_data(storyboard_file, storyboard_data)
            return f"✅ 已更新注释 #{cell_index + 1}"
        
        def refresh_gallery(current_page=1):
            """刷新分镜墙显示（带分页）"""
            storyboard_data = load_data(storyboard_file)
            
            # 计算分页
            total_count = len(storyboard_data)
            total_pages = max(1, (total_count + STORYBOARDS_PER_PAGE - 1) // STORYBOARDS_PER_PAGE)
            current_page = max(1, min(current_page, total_pages))
            
            # 获取当前页的分镜数据
            start_idx = (current_page - 1) * STORYBOARDS_PER_PAGE
            end_idx = start_idx + STORYBOARDS_PER_PAGE
            
            # 分别返回 9 个图片和 9 个注释（共 18 个值）
            images = []
            annotations = []
            
            for i in range(STORYBOARDS_PER_PAGE):
                global_index = start_idx + i
                if global_index < len(storyboard_data):
                    item = storyboard_data[global_index]
                    img_path = item.get("image_path")
                    annotation = item.get("description", "")[:50]  # 摘要
                    
                    if img_path and os.path.exists(img_path):
                        images.append(img_path)
                        annotations.append(annotation)
                    else:
                        images.append(None)
                        annotations.append("")
                else:
                    images.append(None)
                    annotations.append("")
            
            # 生成当前页各宫格的标签（显示全局索引）
            labels = []
            for i in range(STORYBOARDS_PER_PAGE):
                global_index = start_idx + i + 1  # 从 1 开始计数
                labels.append(f'<div style="text-align: center; font-weight: bold; margin-bottom: 5px;">#{global_index}</div>')
            
            # 返回 27 个独立的值（9 个图片 + 9 个注释 + 9 个标签）+ 页码信息
            return (*images, *annotations, *labels, current_page, total_pages)
        
        def get_cell_labels(current_page):
            """生成当前页各宫格的标签（显示全局索引）"""
            labels = []
            start_idx = (current_page - 1) * STORYBOARDS_PER_PAGE
            for i in range(STORYBOARDS_PER_PAGE):
                global_index = start_idx + i + 1  # 从 1 开始计数
                labels.append(f"**#{global_index}**")
            return labels
        
        def get_next_available_index():
            """获取下一个可用的分镜索引"""
            storyboard_data = load_data(storyboard_file)
            return len(storyboard_data)
        
        def add_new_storyboard():
            """添加新的空白分镜"""
            storyboard_data = load_data(storyboard_file)
            
            new_index = len(storyboard_data)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 添加新的空白分镜记录
            storyboard_data.append({
                "id": new_index,
                "image_path": None,
                "aspect_ratio": "16:9 (宽屏)",
                "description": "",
                "timestamp": timestamp
            })
            
            save_data(storyboard_file, storyboard_data)
            
            # 计算应该在哪一页显示
            total_count = len(storyboard_data)
            total_pages = max(1, (total_count + STORYBOARDS_PER_PAGE - 1) // STORYBOARDS_PER_PAGE)
            
            # 返回消息和刷新画廊
            return f"✅ 已添加分镜 #{new_index + 1}", *refresh_gallery(total_pages)
        
        def storyboard_prev_page_action(current_page):
            """分镜墙上一页操作"""
            storyboard_data = load_data(storyboard_file)
            total_count = len(storyboard_data)
            total_pages = max(1, (total_count + STORYBOARDS_PER_PAGE - 1) // STORYBOARDS_PER_PAGE)
            new_page = max(1, current_page - 1)
            new_page = min(new_page, total_pages)
            
            # 返回刷新后的画廊数据和页码信息
            return (*refresh_gallery(new_page), new_page, total_pages)
        
        def storyboard_next_page_action(current_page):
            """分镜墙下一页操作"""
            storyboard_data = load_data(storyboard_file)
            total_count = len(storyboard_data)
            total_pages = max(1, (total_count + STORYBOARDS_PER_PAGE - 1) // STORYBOARDS_PER_PAGE)
            new_page = min(total_pages, current_page + 1)
            
            # 返回刷新后的画廊数据和页码信息
            return (*refresh_gallery(new_page), new_page, total_pages)
        
        def clear_all():
            """清空全部分镜"""
            save_data(storyboard_file, [])
            return "✅ 已清空全部分镜", True
        
        def export_all():
            """导出全部分镜数据和图片到 TXT 文档"""
            try:
                import shutil
                
                storyboard_data = load_data(storyboard_file)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                export_dir = data_dir / f"storyboard_export_{timestamp}"
                export_dir.mkdir(exist_ok=True)
                
                # 导出 TXT 文本文件
                txt_file = export_dir / "storyboard_content.txt"
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 50 + "\n")
                    f.write("分镜脚本\n")
                    f.write(f"导出时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    has_content = False
                    for i, item in enumerate(storyboard_data):
                        if i >= 25:  # 只导出前 25 个分镜
                            break
                        
                        img_path = item.get("image_path")
                        description = item.get("description", "")
                        
                        # 只有当有图片或注释时才导出
                        if img_path or description:
                            has_content = True
                            f.write(f"【分镜 #{i+1}】\n")
                            
                            if img_path and os.path.exists(img_path):
                                # 复制图片到导出目录
                                img_filename = Path(img_path).name
                                dest_img_path = export_dir / img_filename
                                shutil.copy2(img_path, dest_img_path)
                                f.write(f"图片：{img_filename}\n")
                            
                            if description:
                                f.write(f"注释：{description}\n")
                            
                            f.write("\n")
                    
                    if not has_content:
                        f.write("暂无分镜内容\n")
                
                return f"✅ 已导出到：{export_dir}\n包含：分镜文本 (storyboard_content.txt) + 图片文件"
            except Exception as e:
                return f"❌ 导出失败：{e}"
        
        # ========== 事件绑定 ==========
        
        # --- 剧本管理事件 ---
        story_selector.change(
            fn=load_story,
            inputs=[story_selector],
            outputs=[story_title_input, story_genre, full_script_editor]
        ).then(
            fn=refresh_character_list,
            inputs=[story_selector, page_indicator],
            outputs=[char_selector, page_indicator, page_indicator]
        )
        
        new_story_btn.click(
            fn=create_new_story,
            outputs=[story_selector, story_title_input, story_genre, full_script_editor]
        ).then(
            fn=lambda: (gr.update(choices=[], value=""), 1, 0),
            outputs=[char_selector, page_indicator, page_indicator]
        )
        
        save_script_btn.click(
            fn=save_story,
            inputs=[story_title_input, story_genre, full_script_editor, story_selector],
            outputs=[script_status, story_selector]
        )
        
        delete_story_btn.click(
            fn=delete_story,
            inputs=[story_selector],
            outputs=[script_status, story_selector]
        )
        
        export_script_btn.click(
            fn=export_script,
            inputs=[story_selector],
            outputs=[script_status]
        )
        
        # --- 角色管理事件 ---
        
        def prev_page_action(current_story, current_page):
            """上一页操作"""
            if not current_story:
                return gr.update(choices=[], value=""), "", None, 1, 0
            
            new_page = max(1, current_page - 1)
            page_chars, page, total_pages = get_page_characters(current_story, new_page)
            
            # page_chars 已经是角色名称列表（字符串）
            char_names = page_chars
            first_char_name = char_names[0] if char_names else ""
            
            # 加载第一个角色的详情
            if first_char_name:
                content, img_path = load_character(first_char_name, current_story)
                return (
                    gr.update(choices=char_names, value=first_char_name),
                    content,
                    img_path,
                    page,
                    total_pages
                )
            else:
                return gr.update(choices=[], value=""), "", None, page, total_pages
        
        def next_page_action(current_story, current_page):
            """下一页操作"""
            if not current_story:
                return gr.update(choices=[], value=""), "", None, 1, 0
            
            _, _, total_pages = get_page_characters(current_story, current_page)
            new_page = min(total_pages, current_page + 1)
            page_chars, page, total_pages = get_page_characters(current_story, new_page)
            
            # page_chars 已经是角色名称列表（字符串）
            char_names = page_chars
            first_char_name = char_names[0] if char_names else ""
            
            # 加载第一个角色的详情
            if first_char_name:
                content, img_path = load_character(first_char_name, current_story)
                return (
                    gr.update(choices=char_names, value=first_char_name),
                    content,
                    img_path,
                    page,
                    total_pages
                )
            else:
                return gr.update(choices=[], value=""), "", None, page, total_pages
        
        # 当故事选择器变化时，同时刷新角色列表
        story_selector.change(
            fn=load_story,
            inputs=[story_selector],
            outputs=[story_title_input, story_genre, full_script_editor]
        ).then(
            fn=refresh_character_list,
            inputs=[story_selector, page_indicator],
            outputs=[char_selector, page_indicator, page_indicator]
        )
        
        new_story_btn.click(
            fn=create_new_story,
            outputs=[story_selector, story_title_input, story_genre, full_script_editor]
        ).then(
            fn=lambda: (gr.update(choices=[], value=""), 1, 0),
            outputs=[char_selector, page_indicator, page_indicator]
        )
        
        # 分页按钮事件
        prev_page_btn.click(
            fn=prev_page_action,
            inputs=[story_selector, page_indicator],
            outputs=[char_selector, character_editor, character_image, page_indicator, page_indicator]
        )
        
        next_page_btn.click(
            fn=next_page_action,
            inputs=[story_selector, page_indicator],
            outputs=[char_selector, character_editor, character_image, page_indicator, page_indicator]
        )
        
        char_selector.change(
            fn=load_character,
            inputs=[char_selector, story_selector],
            outputs=[character_editor, character_image]
        )
        
        new_char_btn.click(
            fn=lambda: (gr.update(choices=[], value=""), "", None),
            outputs=[char_selector, character_editor, character_image]
        )
        
        save_char_btn.click(
            fn=save_character,
            inputs=[character_editor, char_selector, story_selector, character_image],
            outputs=[script_status, char_selector, character_image]
        )
        
        delete_char_btn.click(
            fn=delete_character,
            inputs=[char_selector, story_selector],
            outputs=[script_status, char_selector, character_image]
        )
        
        delete_char_img_btn.click(
            fn=delete_character_image,
            inputs=[story_selector, char_selector],
            outputs=[script_status, character_image]
        )
        
        # --- 分镜管理事件 (全局工具栏) ---
        
        # 添加新分镜按钮
        add_storyboard_btn.click(
            fn=add_new_storyboard,
            outputs=[status_bar] + gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
        )
        
        export_all_btn.click(
            fn=export_all,
            outputs=[status_bar]
        )
        
        # 分页导航
        storyboard_prev_page_btn.click(
            fn=storyboard_prev_page_action,
            inputs=[storyboard_current_page_num],
            outputs=gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
        )
        
        storyboard_next_page_btn.click(
            fn=storyboard_next_page_action,
            inputs=[storyboard_current_page_num],
            outputs=gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
        )
        
        clear_all_btn.click(
            fn=clear_all,
            outputs=[status_bar, storyboard_current_page_num]
        ).then(
            fn=lambda: refresh_gallery(1),
            outputs=gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
        )
        
        # --- 分镜管理事件 (宫格级别) ---
        
        # 定义删除单个宫格的通用处理函数（移到循环外）
        def delete_single_cell_with_page(current_page, cell_idx):
            """删除指定宫格的分镜（带页码信息）"""
            storyboard_data = load_data(storyboard_file)
            global_index = (current_page - 1) * STORYBOARDS_PER_PAGE + cell_idx
            
            if global_index < len(storyboard_data):
                # 清空该宫格数据
                storyboard_data[global_index] = {
                    "id": global_index,
                    "image_path": None,
                    "aspect_ratio": "",
                    "description": "",
                    "timestamp": ""
                }
                
                save_data(storyboard_file, storyboard_data)
                # refresh_gallery 已经返回了 current_page 和 total_pages，不需要额外添加
                return (f"✅ 已删除分镜 #{global_index + 1}", *refresh_gallery(current_page))
            else:
                return (f"⚠️ 该位置没有分镜", *refresh_gallery(current_page))
        
        # 定义删除单个宫格的通用处理函数（移到循环外）
        def delete_single_cell_with_page(current_page, cell_idx):
            """删除指定宫格的分镜（带页码信息）"""
            storyboard_data = load_data(storyboard_file)
            global_index = (current_page - 1) * STORYBOARDS_PER_PAGE + cell_idx
            
            if global_index < len(storyboard_data):
                # 清空该宫格数据
                storyboard_data[global_index] = {
                    "id": global_index,
                    "image_path": None,
                    "aspect_ratio": "",
                    "description": "",
                    "timestamp": ""
                }
                
                save_data(storyboard_file, storyboard_data)
                # refresh_gallery 已经返回了 current_page 和 total_pages，不需要额外添加
                return (f"✅ 已删除分镜 #{global_index + 1}", *refresh_gallery(current_page))
            else:
                return (f"⚠️ 该位置没有分镜", *refresh_gallery(current_page))
        
        # 为每个宫格的删除按钮绑定事件 - 使用新的页码组件 storyboard_current_page_num
        for i, btn in enumerate(cell_delete_btns):
            # 使用 lambda 包装，将 idx=i 作为固定参数传递
            btn.click(
                fn=lambda current_page, idx=i: delete_single_cell_with_page(current_page, idx),
                inputs=[storyboard_current_page_num],
                outputs=[status_bar] + gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
            )
        
        # 定义图片上传的通用处理函数（移到循环外）
        def on_image_upload_with_page(file, current_page, cell_idx):
            """处理图片上传（带页码信息）"""
            if not file:
                return (f"✅ 分镜 #{cell_idx + 1} 已清空", *refresh_gallery(current_page))
            
            # 获取文件路径
            file_path = file if isinstance(file, str) else file.name
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            storyboard_data = load_data(storyboard_file)
            
            # 计算全局索引（当前页的偏移量 + 单元格索引）
            global_index = (current_page - 1) * STORYBOARDS_PER_PAGE + cell_idx
            
            # 确保列表足够长
            while len(storyboard_data) <= global_index:
                storyboard_data.append({})
            
            # 更新指定宫格
            storyboard_data[global_index] = {
                "id": global_index,
                "image_path": file_path,
                "aspect_ratio": "16:9 (宽屏)",
                "description": storyboard_data[global_index].get("description", "") if global_index < len(storyboard_data) else "",
                "timestamp": timestamp
            }
            
            save_data(storyboard_file, storyboard_data)
            
            # refresh_gallery 已经返回了 current_page 和 total_pages，不需要额外添加
            return (f"✅ 已更新分镜 #{global_index + 1}", *refresh_gallery(current_page))
        
        # 为每个宫格的图片组件绑定事件（上传）- 使用新的页码组件 storyboard_current_page_num
        for i, img_component in enumerate(gallery_cells):
            # 使用 lambda 包装，将 idx=i 作为固定参数传递
            img_component.upload(
                fn=lambda file, current_page, idx=i: on_image_upload_with_page(file, current_page, idx),
                inputs=[img_component, storyboard_current_page_num],
                outputs=[status_bar] + gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
            )
        
        # 为每个宫格的注释文本框绑定事件（失去焦点时自动保存）
        for i, textbox in enumerate(cell_annotations):
            def update_annotation(text, current_page, idx):
                """更新宫格注释"""
                storyboard_data = load_data(storyboard_file)
                
                # 计算全局索引（当前页的偏移量 + 单元格索引）
                global_index = (current_page - 1) * STORYBOARDS_PER_PAGE + idx
                
                # 确保列表足够长
                while len(storyboard_data) <= global_index:
                    storyboard_data.append({})
                
                # 更新注释
                if global_index < len(storyboard_data):
                    storyboard_data[global_index]["description"] = text
                else:
                    storyboard_data[global_index] = {
                        "id": global_index,
                        "image_path": None,
                        "aspect_ratio": "",
                        "description": text,
                        "timestamp": ""
                    }
                
                save_data(storyboard_file, storyboard_data)
                return f"✅ 已更新注释 #{global_index + 1}"
            
            textbox.change(
                fn=lambda text, current_page, idx=i: update_annotation(text, current_page, idx),
                inputs=[textbox, storyboard_current_page_num],
                outputs=[status_bar]
            )
        
        # 清空全部按钮
        clear_all_btn.click(
            fn=clear_all,
            outputs=[status_bar, storyboard_current_page_num]
        ).then(
            fn=lambda: refresh_gallery(1),
            outputs=gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
        )
        
        # 初始化加载
        ui.load(fn=refresh_story_list, outputs=[story_selector])
        ui.load(
            fn=lambda: refresh_gallery(1),
            outputs=gallery_cells + cell_annotations + cell_labels + [storyboard_current_page_num, storyboard_total_pages_num]
        )
        ui.load(
            fn=lambda: (gr.update(choices=[], value=""), 1, 0),
            outputs=[char_selector, page_indicator, page_indicator]
        )
        
        # 存储组件引用
        result["gallery_cells"] = gallery_cells
    
    return result
