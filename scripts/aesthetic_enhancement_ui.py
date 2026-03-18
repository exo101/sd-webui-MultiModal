import os
import gradio as gr
from pathlib import Path
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Qwen 模块可用性检查标志
QWEN_MODULE_AVAILABLE = False

# 确保 scripts 目录在 Python 路径中
try:
    scripts_dir = Path(__file__).parent
    scripts_dir_str = str(scripts_dir)
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)
        logger.info(f"已将 {scripts_dir_str} 添加到 Python 路径")
    
    # 检查 qwen_analysis_ui.py 文件是否存在
    qwen_file = scripts_dir / "qwen_analysis_ui.py"
    if qwen_file.exists():
        logger.info(f"✅ 找到 qwen_analysis_ui.py 文件：{qwen_file}")
        # 尝试导入模块
        try:
            import qwen_analysis_ui
            QWEN_MODULE_AVAILABLE = True
            logger.info("✅ Qwen 分析模块加载成功")
        except Exception as e:
            logger.error(f"❌ Qwen 分析模块导入失败：{e}", exc_info=True)
    else:
        logger.warning(f"❌ qwen_analysis_ui.py 文件不存在：{qwen_file}")
except Exception as e:
    logger.error(f"❌ Qwen 分析模块检查失败：{e}", exc_info=True)

# 获取素材目录路径
SCRIPTS_DIR = Path(__file__).parent
AESTHETIC_ENHANCEMENT_DIR = SCRIPTS_DIR.parent / "Aesthetic-Enhancement"

# ==================== 构图技巧模块 ====================

# 构图类型映射（文件名到中文标题）
COMPOSITION_TITLES = {
    "S 型构图.png": "S 型构图",
    "U 字型构图.png": "U 字型构图",
    "x 形式构图.png": "X 形式构图",
    "三角形构图.png": "三角形构图",
    "三角形构图 2.png": "三角形构图 2",
    "九宫格构图.png": "九宫格构图",
    "口字形构图.png": "口字形构图",
    "向心式构图.png": "向心式构图",
    "对角线构图.png": "对角线构图",
    "对角线构图 2.png": "对角线构图 2",
    "引导线构图.png": "引导线构图",
    "放射式构图.png": "放射式构图",
    "点景构图.png": "点景构图",
    "环形式构图.png": "环形式构图",
}

# 构图说明
COMPOSITION_DESCRIPTIONS = {
    "S 型构图": "优雅流畅的曲线构图，营造韵律感和动感",
    "U 字型构图": "稳定的 U 形结构，突出中心主体",
    "X 形式构图": "交叉对称的视觉引导，增强画面张力",
    "三角形构图": "稳定均衡的经典构图，适用于多种场景",
    "九宫格构图": "黄金分割变体，将主体置于交点位置",
    "口字形构图": "框架式构图，聚焦内部主体",
    "向心式构图": "所有元素向中心汇聚，强化视觉焦点",
    "对角线构图": "斜线分割画面，创造动态平衡",
    "引导线构图": "利用线条引导视线，增强纵深感",
    "放射式构图": "从中心向外发散，展现扩张感",
    "点景构图": "点睛之笔，小元素提升整体效果",
    "环形式构图": "环形包围结构，营造围合感",
}


def get_composition_images():
    """获取所有构图素材图片路径"""
    composition_dir = AESTHETIC_ENHANCEMENT_DIR / "构图技巧"
    
    if not composition_dir.exists():
        logger.warning(f"构图技巧素材目录不存在：{composition_dir}")
        return []
    
    image_files = []
    for img_path in composition_dir.glob("*.png"):
        filename = img_path.name
        title = COMPOSITION_TITLES.get(filename, filename.replace(".png", ""))
        image_files.append({
            "path": str(img_path),
            "title": title,
            "description": COMPOSITION_DESCRIPTIONS.get(title, "")
        })
    
    logger.info(f"加载了 {len(image_files)} 个构图素材")
    return sorted(image_files, key=lambda x: x["title"])


# ==================== 打光技巧模块 ====================

# 打光类型映射
LIGHTING_TITLES = {
    "丁达尔光.png": "丁达尔光",
    "丁达尔光 2.png": "丁达尔光 2",
    "伦勃朗光.png": "伦勃朗光",
    "侧逆光.png": "侧逆光",
    "侧顺光.png": "侧顺光",
    "光源构图.png": "光源构图",
    "底光.png": "底光",
    "正逆光.png": "正逆光",
    "顶光.png": "顶光",
    "顺光.png": "顺光",
    "鬼光.png": "鬼光",
}

# 打光说明
LIGHTING_DESCRIPTIONS = {
    "丁达尔光": "光线穿过介质产生的光束效果，增强空间层次感",
    "伦勃朗光": "经典的三角光照明，塑造立体感和戏剧性",
    "侧逆光": "从侧后方照射，勾勒轮廓，分离主体与背景",
    "侧顺光": "从侧前方照射，均匀照亮主体，展现细节",
    "底光": "从下方照射，营造神秘或恐怖氛围",
    "正逆光": "从正后方照射，形成剪影或轮廓光效果",
    "顶光": "从上方照射，模拟自然光或聚光灯效果",
    "顺光": "正面照射，亮度均匀但缺乏层次",
    "鬼光": "特殊角度的诡异照明，营造阴森氛围",
    "光源构图": "利用光源位置引导视觉焦点",
}


def get_lighting_images():
    """获取所有打光素材图片路径"""
    lighting_dir = AESTHETIC_ENHANCEMENT_DIR / "打光技巧"
    
    if not lighting_dir.exists():
        logger.warning(f"打光技巧素材目录不存在：{lighting_dir}")
        return []
    
    image_files = []
    for img_path in lighting_dir.glob("*.png"):
        filename = img_path.name
        title = LIGHTING_TITLES.get(filename, filename.replace(".png", ""))
        image_files.append({
            "path": str(img_path),
            "title": title,
            "description": LIGHTING_DESCRIPTIONS.get(title, "")
        })
    
    logger.info(f"加载了 {len(image_files)} 个打光素材")
    return sorted(image_files, key=lambda x: x["title"])


# ==================== UI 组件创建 ====================

def create_composition_card(image_info, index):
    """创建单个构图卡片"""
    # 将路径转换为 URL 格式以在 HTML 中使用
    img_url = f"file={image_info['path']}"
    
    with gr.Group():
        # 使用 HTML img 标签以便绑定点击事件
        gr.HTML(f"""
        <div class="gallery-card" data-index="{index}" data-title="{image_info['title']}" data-description="{image_info['description']}" data-src="{image_info['path']}">
            <div class="gallery-image-container">
                <img src="{img_url}" alt="{image_info['title']}" class="gallery-image" />
                <div class="gallery-overlay">
                    <span class="gallery-zoom-icon">🔍</span>
                    <span class="gallery-zoom-text">点击放大</span>
                </div>
            </div>
            <div class="gallery-info">
                <div class="gallery-title">{image_info['title']}</div>
                <div class="gallery-description">{image_info['description']}</div>
            </div>
        </div>
        """)


def create_lighting_card(image_info, index):
    """创建单个打光卡片"""
    # 将路径转换为 URL 格式以在 HTML 中使用
    img_url = f"file={image_info['path']}"
    
    with gr.Group():
        # 使用 HTML img 标签以便绑定点击事件
        gr.HTML(f"""
        <div class="gallery-card" data-index="{index}" data-title="{image_info['title']}" data-description="{image_info['description']}" data-src="{image_info['path']}">
            <div class="gallery-image-container">
                <img src="{img_url}" alt="{image_info['title']}" class="gallery-image" />
                <div class="gallery-overlay">
                    <span class="gallery-zoom-icon">🔍</span>
                    <span class="gallery-zoom-text">点击放大</span>
                </div>
            </div>
            <div class="gallery-info">
                <div class="gallery-title">{image_info['title']}</div>
                <div class="gallery-description">{image_info['description']}</div>
            </div>
        </div>
        """)


def create_composition_tab():
    """创建构图技巧标签页"""
    gr.Markdown("""
    # 📐 构图技巧
    
    学习经典构图法则，提升作品美学品质。构图是画面的骨架，决定了作品的视觉结构和美感基础。
    
    **使用建议**：
    - 观察每个构图的视觉引导线和元素排布
    - 理解不同构图传达的情感和视觉效果
    - 在实际创作中灵活运用多种构图技巧
    """)
    
    # 获取所有构图图片
    composition_images = get_composition_images()
    
    if not composition_images:
        gr.Markdown("⚠️ 未找到构图素材，请检查素材目录是否正确配置。")
        return
    
    # 画廊网格容器
    with gr.Column(scale=1):
        gr.HTML("""
        <div class='gallery-container' style='width: 100%; margin: 0 auto;'>
        """)
        for i, img_info in enumerate(composition_images):
            create_composition_card(img_info, i)
        gr.HTML("</div>")
    
    # 构图技巧详解
    with gr.Accordion("📖 构图技巧详解", open=False):
        gr.Markdown("""
        ## 常见构图技巧与应用
        
        ### 1. S 型构图
        - **特点**: 曲线优美，富有韵律感
        - **适用**: 风景、人像、静物
        - **效果**: 优雅、流动、柔美
        
        ### 2. 三角形构图
        - **特点**: 稳定均衡，层次分明
        - **适用**: 群像、建筑、山景
        - **效果**: 稳固、和谐、庄重
        
        ### 3. 九宫格构图
        - **特点**: 符合黄金分割比例
        - **适用**: 通用构图法则
        - **效果**: 自然舒适，视觉平衡
        
        ### 4. 对角线构图
        - **特点**: 动态感强，打破平衡
        - **适用**: 运动、街拍、创意摄影
        - **效果**: 活力、动感、张力
        
        ### 5. 引导线构图
        - **特点**: 强烈的纵深感和方向性
        - **适用**: 道路、河流、走廊
        - **效果**: 延伸、聚焦、沉浸
        
        ### 6. 向心式构图
        - **特点**: 所有元素指向中心
        - **适用**: 团体照、圆形建筑
        - **效果**: 凝聚、聚焦、统一
        
        ### 7. 框架式构图 (口字形)
        - **特点**: 前景形成画框
        - **适用**: 门窗、洞口、树枝
        - **效果**: 聚焦主体、增加层次
        
        ### 8. 放射式构图
        - **特点**: 从中心向外扩散
        - **适用**: 阳光、花朵、爆炸
        - **效果**: 扩张、活力、冲击力
        
        ## 实践建议
        1. **多观察**: 分析优秀作品的构图规律
        2. **勤练习**: 有意识地运用不同构图法则
        3. **善变通**: 根据实际情况灵活组合多种构图
        4. **敢突破**: 在掌握规则后尝试创新
        """)


def create_lighting_tab():
    """创建打光技巧标签页"""
    gr.Markdown("""
    # 💡 打光技巧
    
    掌握光影艺术，塑造画面氛围。光线是摄影和绘画的灵魂，决定了作品的情感表达和视觉效果。
    
    **使用建议**：
    - 理解不同光位的特点和情感表达
    - 学会组合多种光源创造丰富层次
    - 根据主题选择合适的布光方案
    """)
    
    # 获取所有打光图片
    lighting_images = get_lighting_images()
    
    if not lighting_images:
        gr.Markdown("⚠️ 未找到打光素材，请检查素材目录是否正确配置。")
        return
    
    # 画廊网格容器
    with gr.Column(scale=1):
        gr.HTML("""
        <div class='gallery-container' style='width: 100%; margin: 0 auto;'>
        """)
        for i, img_info in enumerate(lighting_images):
            create_lighting_card(img_info, i)
        gr.HTML("</div>")
    
    # 打光技巧详解
    with gr.Accordion("💡 打光技巧详解", open=False):
        gr.Markdown("""
        ## 常见光位与效果
        
        ### 1. 顺光 (正面光)
        - **特点**: 光线从正面照射被摄体
        - **效果**: 亮度均匀，色彩饱和，但缺乏立体感
        - **适用**: 证件照、产品拍摄
        
        ### 2. 侧顺光 (前侧光)
        - **特点**: 光线从侧前方 45°照射
        - **效果**: 展现明暗过渡，增强立体感
        - **适用**: 人像、静物、建筑
        
        ### 3. 侧逆光 (后侧光)
        - **特点**: 光线从侧后方照射
        - **效果**: 勾勒轮廓，分离主体与背景
        - **适用**: 人像发丝光、物体轮廓强调
        
        ### 4. 逆光
        - **特点**: 光线从正后方照射
        - **效果**: 形成剪影或明亮轮廓
        - **适用**: 剪影摄影、透明物体
        
        ### 5. 顶光
        - **特点**: 光线从上方垂直照射
        - **效果**: 模拟正午阳光或聚光灯
        - **适用**: 舞台摄影、特殊氛围
        
        ### 6. 底光
        - **特点**: 光线从下方照射
        - **效果**: 营造诡异、神秘氛围
        - **适用**: 恐怖片、特殊创意
        
        ### 7. 伦勃朗光
        - **特点**: 侧上方 45°，面部形成三角光斑
        - **效果**: 经典戏剧性用光，立体感强
        - **适用**: 人像摄影、古典油画
        
        ### 8. 丁达尔效应
        - **特点**: 光线穿过介质形成可见光束
        - **效果**: 增强空间层次，营造梦幻氛围
        - **适用**: 森林、教堂、舞台
        
        ## 布光原则
        1. **主光**: 确定主要光源方向和强度
        2. **辅光**: 补充阴影，降低反差
        3. **轮廓光**: 分离主体，增强层次
        4. **背景光**: 营造环境氛围
        5. **装饰光**: 点缀细节，画龙点睛
        
        ## 实践建议
        - 从单灯开始练习，理解光的基本特性
        - 逐步增加灯位，掌握多灯配合
        - 善用反光板和柔光设备
        - 观察自然光的变化规律
        """)


def create_aesthetic_enhancement_ui():
    """创建美学提升模块 UI（包含构图和打光两个子标签）"""
    
    # 添加自定义 CSS
    custom_css = """
    /* 画廊网格容器 - 响应式布局 */
    .gallery-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        padding: 20px;
        background: #f9f9f9;
        border-radius: 8px;
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
        box-sizing: border-box;
        justify-content: center;
        justify-items: center;
    }
    
    /* 画廊卡片 */
    .gallery-card {
        background: #fff;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
        cursor: pointer;
        margin: 0;
        width: 100%;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* 图片容器 - 保持固定宽高比 */
    .gallery-image-container {
        position: relative;
        width: 100%;
        padding-top: 75%; /* 4:3 宽高比 */
        overflow: hidden;
        background: #f0f0f0;
    }
    
    /* 画廊图片样式 */
    .gallery-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.2s ease;
    }
    
    .gallery-card:hover .gallery-image {
        transform: scale(1.05);
    }
    
    /* 悬停遮罩层 */
    .gallery-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        opacity: 0;
        transition: opacity 0.2s ease;
        color: white;
        font-size: 14px;
    }
    
    .gallery-card:hover .gallery-overlay {
        opacity: 1;
    }
    
    .gallery-zoom-icon {
        font-size: 24px;
        margin-bottom: 4px;
    }
    
    .gallery-zoom-text {
        font-size: 12px;
        font-weight: bold;
    }
    
    /* 信息区域 */
    .gallery-info {
        padding: 10px 12px;
        text-align: center;
        background: #fff;
    }
    
    .gallery-title {
        font-weight: bold;
        font-size: 14px;
        color: #333;
        margin-bottom: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .gallery-description {
        font-size: 12px;
        color: #666;
        line-height: 1.4;
        font-style: italic;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* 模态框样式 */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85);
        display: none;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        cursor: zoom-out;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-image {
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        animation: zoomIn 0.3s ease;
    }
    
    @keyframes zoomIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .modal-close-hint {
        position: absolute;
        top: 20px;
        right: 30px;
        color: white;
        font-size: 32px;
        font-weight: bold;
        cursor: pointer;
        z-index: 10000;
        transition: color 0.2s ease;
    }
    
    .modal-close-hint:hover {
        color: #ff6b6b;
    }
    
    .modal-info {
        position: absolute;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        font-size: 16px;
        text-align: center;
        background: rgba(0, 0, 0, 0.7);
        padding: 12px 24px;
        border-radius: 8px;
        max-width: 80%;
        backdrop-filter: blur(10px);
    }
    
    .modal-info strong {
        display: block;
        font-size: 18px;
        margin-bottom: 6px;
    }
    
    /* 响应式适配 */
    @media (max-width: 1200px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        }
    }
    
    @media (max-width: 768px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            padding: 16px;
        }
        
        .gallery-info {
            padding: 8px 10px;
        }
        
        .gallery-title {
            font-size: 13px;
        }
        
        .gallery-description {
            font-size: 11px;
        }
    }
    
    @media (max-width: 480px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            padding: 12px;
        }
        
        .gallery-image-container {
            padding-top: 100%; /* 正方形 */
        }
        
        .gallery-title {
            font-size: 12px;
        }
        
        .gallery-description {
            display: none;
        }
    }
    """

    # 添加模态框 HTML
    gr.HTML("""
    <div id="imageModal" class="modal-overlay">
        <span class="modal-close-hint" onclick="closeModal()">&times;</span>
        <img id="modalImg" class="modal-image" src="" alt="">
        <div id="modalInfo" class="modal-info"></div>
    </div>
    """)
    
    # 添加自定义 JavaScript
    gr.HTML("""
    <script>
    let currentImageInfo = null;
    
    function openModal(imagePath, title, description) {
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImg');
        const modalInfo = document.getElementById('modalInfo');
        
        if (!modal || !modalImg || !modalInfo) {
            console.error('模态框元素未找到');
            return;
        }
        
        // 将文件路径转换为 Gradio 可用的 URL
        const fileUrl = imagePath.startsWith('file=') ? imagePath : `file=${imagePath}`;
        modalImg.src = fileUrl;
        modalInfo.innerHTML = `<strong>${title}</strong><br>${description || ''}`;
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        
        currentImageInfo = { path: imagePath, title, description };
        console.log('✅ 模态框已打开:', title);
    }
    
    function closeModal() {
        const modal = document.getElementById('imageModal');
        if (!modal) return;
        
        modal.style.display = 'none';
        document.body.style.overflow = '';
        currentImageInfo = null;
        console.log('❌ 模态框已关闭');
    }
    
    // 绑定画廊卡片点击事件
    function bindGalleryClicks() {
        try {
            console.log('🎨 开始绑定画廊卡片点击事件...');
            
            // 查找所有画廊卡片
            const cards = document.querySelectorAll('.gallery-card');
            console.log('📊 找到', cards.length, '个画廊卡片');
            
            cards.forEach((card, index) => {
                const title = card.getAttribute('data-title') || '';
                const description = card.getAttribute('data-description') || '';
                const src = card.getAttribute('data-src') || '';
                
                if (!src) {
                    console.warn('⚠️ 跳过无效卡片 (无 src):', index);
                    return;
                }
                
                // 为整个卡片添加点击事件
                card.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🖼️ 卡片被点击:', title, '| 索引:', index);
                    openModal(src, title, description);
                });
                
                console.log('✓ 已绑定卡片:', title);
            });
            
            console.log('✅ 画廊点击事件绑定完成，共绑定', cards.length, '个卡片');
        } catch (error) {
            console.error('❌ 绑定画廊点击事件失败:', error);
        }
    }
    
    // 延迟绑定以确保 Gradio 渲染完成
    function initGallery() {
        setTimeout(() => {
            bindGalleryClicks();
        }, 1000);
    }
    
    // 页面加载后绑定事件
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initGallery);
    } else {
        initGallery();
    }
    
    // 支持 Tab 切换后重新绑定
    document.addEventListener('click', function(e) {
        if (e.target.closest('.tab-nav')) {
            setTimeout(() => {
                bindGalleryClicks();
            }, 500);
        }
    });
    
    // ESC 键关闭模态框
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
    
    // 点击模态框背景关闭
    const modalElement = document.getElementById('imageModal');
    if (modalElement) {
        modalElement.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModal();
            }
        });
    }
    </script>
    """)

    # 创建标签页
    with gr.Blocks(css=custom_css) as demo:
        with gr.Tab("📐 构图技巧"):
            create_composition_tab()
        with gr.Tab("💡 打光技巧"):
            create_lighting_tab()
        
        # AI 智能分析 Tab - 直接在 Tab 内部实现 Qwen UI
        if QWEN_MODULE_AVAILABLE:
            # 导入 Qwen 模块的核心组件和函数
            try:
                from qwen_analysis_ui import (
                    DEFAULT_QWEN_MODEL, 
                    test_ollama_connection,
                    analyze_single_image,
                    batch_analyze_images,
                    extract_video_frames,
                    get_comprehensive_analysis_prompt,
                    get_composition_only_prompt,
                    get_lighting_only_prompt,
                    get_shot_only_prompt
                )
                
                with gr.Tab("🎬 AI 智能分析"):
                    gr.Markdown("""
                    # 🎬 AI 智能画面分析（Qwen3.5-VL）
                    
                    基于 **本地部署的 Qwen3.5 多模态大模型**，对图片和视频进行专业的画面分析
                    
                    ### 分析维度
                    - **📐 构图分析**: 九宫格、三角形、S 型等经典构图识别
                    - **💡 灯光分析**: 光位、光比、氛围营造技巧
                    - **🎥 分镜分析**: 景别、角度、镜头语言解读
                    - **🎨 色彩情绪**: 色调、配色方案、情感表达
                    - **💬 改进建议**: 专业摄影师视角的优化建议
                    
                    > 💡 **提示**: 使用本地 Ollama 服务，无需 API Key，完全免费！
                    """)
                    
                    # 连接状态检查
                    connection_status = gr.State()
                    
                    def init_connection():
                        success, message = test_ollama_connection()
                        return message
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 1️⃣ 系统配置")
                            
                            # 连接测试按钮
                            test_btn = gr.Button("🔌 测试 Ollama 连接", variant="secondary")
                            connection_info = gr.Textbox(
                                label="连接状态",
                                lines=5,
                                value="点击按钮测试连接...",
                                interactive=False
                            )
                            
                            # 模型选择
                            model_selector = gr.Dropdown(
                                choices=[
                                    "qwen3.5:9b",
                                    "qwen3.5:4b", 
                                    "qwen3.5:2b",
                                    "huihui_ai/qwen3.5-abliterated:9B",
                                    "huihui_ai/qwen3.5-abliterated:4B",
                                    "huihui_ai/qwen3.5-abliterated:2B",
                                ],
                                value=DEFAULT_QWEN_MODEL,
                                label="选择 Qwen3.5 模型",
                                info="推荐使用 4b 版本，平衡速度和质量"
                            )
                            
                            gr.Markdown("### 2️⃣ 选择分析模式")
                            
                            analysis_mode = gr.Radio(
                                choices=[
                                    ("🖼️ 图片分析", "image"),
                                    ("🎬 视频分析", "video")
                                ],
                                value="image",
                                label="分析模式"
                            )
                            
                            analysis_type = gr.Radio(
                                choices=[
                                    ("🔍 综合分析", "comprehensive"),
                                    ("📐 仅构图分析", "composition"),
                                    ("💡 仅灯光分析", "lighting"),
                                    ("🎥 仅分镜分析", "shot")
                                ],
                                value="comprehensive",
                                label="分析类型"
                            )
                            
                            # 图片输入
                            image_input = gr.Image(
                                type="filepath",
                                label="上传图片",
                                height=300
                            )
                            
                            # 视频输入
                            video_input = gr.Video(
                                label="上传视频",
                                height=300,
                                visible=False
                            )
                            
                            # 关键帧提取模式选择
                            keyframe_mode = gr.Radio(
                                choices=[
                                    ("🎯 智能镜头检测（推荐）", "smart"),
                                    ("📏 固定间隔抽帧", "fixed")
                                ],
                                value="smart",
                                label="关键帧提取模式",
                                info="智能模式会自动识别镜头切换，只抽取有变化的画面",
                                visible=False
                            )
                            
                            frame_interval = gr.Slider(
                                minimum=1,
                                maximum=120,
                                value=30,
                                step=1,
                                label="抽帧间隔（帧数）",
                                info="每隔多少帧抽取一帧（仅在固定间隔模式下使用）",
                                visible=False
                            )
                            
                            analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 3️⃣ 分析结果")
                            
                            result_output = gr.Textbox(
                                label="分析报告",
                                lines=25,
                                show_copy_button=True,
                                placeholder="分析结果将显示在这里..."
                            )
                            
                            # 帧预览（视频分析时使用）
                            frame_gallery = gr.Gallery(
                                label="提取的帧预览",
                                columns=4,
                                height="auto",
                                visible=False
                            )
                    
                    # 事件绑定
                    
                    # 初始化连接测试
                    demo.load(fn=init_connection, inputs=[], outputs=[connection_info])
                    
                    # 手动测试连接
                    test_btn.click(fn=init_connection, inputs=[], outputs=[connection_info])
                    
                    # 切换分析模式时的 UI 更新
                    def update_inputs(mode):
                        return (
                            gr.update(visible=(mode == "image")),  # image_input
                            gr.update(visible=(mode == "video")),  # video_input
                            gr.update(visible=(mode == "video")),  # keyframe_mode
                            gr.update(visible=(mode == "video" and False))  # frame_interval (仅在固定间隔模式显示)
                        )
                    
                    analysis_mode.change(
                        fn=update_inputs,
                        inputs=[analysis_mode],
                        outputs=[image_input, video_input, keyframe_mode, frame_interval]
                    )
                    
                    # 切换关键帧模式时的 UI 更新
                    def update_frame_interval_visibility(mode, keyframe_mode_val):
                        show_interval = (mode == "video" and keyframe_mode_val == "fixed")
                        return gr.update(visible=show_interval)
                    
                    keyframe_mode.change(
                        fn=update_frame_interval_visibility,
                        inputs=[analysis_mode, keyframe_mode],
                        outputs=[frame_interval]
                    )
                    
                    # 执行分析
                    def run_analysis(model, mode, analysis_type_val, image, video, keyframe_mode_val, interval):
                        if mode == "image":
                            if not image:
                                return "❌ 请先上传图片", gr.update(visible=False)
                            
                            result = analyze_single_image(image, analysis_type_val, model)
                            
                            if result.get("success"):
                                return result.get("analysis", "分析完成"), gr.update(visible=False)
                            else:
                                return f"❌ 分析失败\n\n{result.get('analysis', '')}", gr.update(visible=False)
                        
                        elif mode == "video":
                            if not video:
                                return "❌ 请先上传视频", gr.update(visible=False)
                            
                            # 根据模式提取视频帧
                            if keyframe_mode_val == "smart":
                                logger.info("使用智能镜头检测模式提取关键帧")
                                # 智能模式：传递 0 作为间隔参数，触发镜头检测
                                frame_paths = extract_video_frames(video, "temp/video_frames", frame_interval=0)
                            else:
                                logger.info(f"使用固定间隔模式提取关键帧（间隔：{interval}帧）")
                                frame_paths = extract_video_frames(video, "temp/video_frames", frame_interval=interval)
                            
                            if not frame_paths:
                                return "❌ 视频处理失败", gr.update(visible=False, value=[])
                            
                            # 批量分析
                            report = batch_analyze_images(frame_paths, analysis_type_val, model)
                            
                            # 返回结果和帧预览
                            return report, gr.update(visible=True, value=frame_paths[:16])  # 最多显示 16 帧
                    
                    analyze_btn.click(
                        fn=run_analysis,
                        inputs=[model_selector, analysis_mode, analysis_type, image_input, video_input, keyframe_mode, frame_interval],
                        outputs=[result_output, frame_gallery]
                    )
                    
                    # 使用说明
                    with gr.Accordion("📖 使用说明", open=False):
                        gr.Markdown("""
                        ### 快速开始
                        
                        #### 前置要求
                        
                        1. **安装 Ollama**: 从 https://ollama.com 下载并安装
                        2. **安装Qwen3.5 模型**: 运行命令 `ollama run qwen3.5:4b`
                        3. **启动 Ollama 服务**: 确保 Ollama 在后台运行
                        
                        #### 图片分析
                        
                        1. 选择"🖼️ 图片分析"模式
                        2. 上传要分析的图片
                        3. 选择分析类型（推荐"综合分析"）
                        4. 点击"开始分析"
                        
                        #### 视频分析
                        
                        1. 选择"🎬 视频分析"模式
                        2. 上传要分析的视频
                        3. **选择关键帧提取模式**：
                           - **🎯 智能镜头检测（推荐）**：自动识别场景切换，只抽取有变化的画面
                             - 短镜头（<1 秒）：提取 1 帧
                             - 中等镜头（1-3 秒）：提取 2 帧
                             - 长镜头（>3 秒）：提取 3 帧
                             - **优势**：避免重复分析相似画面，聚焦关键场景变化
                           - **📏 固定间隔抽帧**：传统的等间隔抽帧方式
                             - 可自定义抽帧间隔（如每 30 帧抽 1 帧）
                             - 适合需要均匀采样的特殊场景
                        4. 选择分析类型
                        5. 点击"开始分析"
                        
                        ### 💡 智能镜头检测原理
                        
                        1. **特征提取**：计算每帧的 HSV 颜色直方图
                        2. **相似度对比**：比较相邻帧的相关系数
                        3. **镜头识别**：相似度骤降处判定为镜头切换
                        4. **关键帧选择**：从每个镜头中抽取代表性帧
                        
                        **效果对比**：
                        - ❌ 传统方式：30 分钟视频 → 60 帧（大量重复）
                        - ✅ 智能方式：30 分钟视频 → 15-25 帧（每个镜头仅 1-3 帧）
                        
                        ### 模型选择建议
                        
                        - **qwen3.5:2b**: 速度最快，适合低显存显卡（4-6GB）
                        - **qwen3.5:4b**: 平衡速度和质量，推荐（8GB+ 显存）
                        - **qwen3.5:9b**: 质量最高，速度较慢（12GB+ 显存）
                        
                        ### 常见问题
                        
                        **Q: 提示"无法连接到 Ollama"**
                        A: 请确保 Ollama 服务正在运行，可以在开始菜单搜索"Ollama"启动它。
                        
                        **Q: 分析时间很长**
                        A: 首次运行需要加载模型到显存，可能需要 1-2 分钟。后续分析会快很多。
                        
                        **Q: 显存不足**
                        A: 尝试使用更小的模型版本（如 qwen3.5:2b），或降低图片分辨率。
                        
                        **Q: 分析结果不理想**
                        A: 可以尝试：
                        - 更换分析类型（如只分析构图或灯光）
                        - 使用更大的模型版本
                        - 提供更高质量的图片
                        
                        ### 技术细节
                        
                        - **API 端点**: http://localhost:11434/api/chat
                        - **模型格式**: Ollama Chat Format
                        - **图像编码**: Base64
                        - **超时设置**: 120 秒（适应高分辨率图像分析）
                        """)
            except Exception as e:
                logger.error(f"❌ Qwen 分析模块加载失败：{e}")
                with gr.Tab("🎬 AI 智能分析"):
                    gr.Markdown(f"""
                    ### ❌ Qwen 分析模块加载失败
                    
                    错误信息：{str(e)}
                    
                    请检查以下事项：
                    
                    1. **文件存在**: 确认 `qwen_analysis_ui.py` 文件位于 scripts 目录中
                    2. **依赖安装**: 运行 `pip install requests opencv-python`
                    3. **Ollama 服务**: 确保 Ollama 已启动且 Qwen3.5 模型已安装
                    
                    **安装步骤**:
                    ```bash
                    # 1. 安装 Ollama
                    访问 https://ollama.com 下载安装
                    
                    # 2. 安装Qwen3.5 模型
                    ollama run qwen3.5:4b
                    
                    # 3. 安装 Python 依赖
                    pip install requests opencv-python
                    ```
                    """)
        else:
            with gr.Tab("🎬 AI 智能分析"):
                gr.Markdown("""
                ### ⚠️ Qwen 分析模块未正确安装
                
                请检查以下事项：
                
                1. **文件存在**: 确认 `qwen_analysis_ui.py` 文件位于 scripts 目录中
                2. **依赖安装**: 运行 `pip install requests opencv-python`
                3. **Ollama 服务**: 确保 Ollama 已启动且 Qwen3.5 模型已安装
                
                **安装步骤**:
                ```bash
                # 1. 安装 Ollama
                访问 https://ollama.com 下载安装
                
                # 2. 安装Qwen3.5 模型
                ollama run qwen3.5:4b
                
                # 3. 安装 Python 依赖
                pip install requests opencv-python
                ```
                """)
    
    return demo
