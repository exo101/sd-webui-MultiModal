import os
import gradio as gr
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
    with gr.Column():
        gr.HTML("""
        <div class='gallery-container'>
        """)
        for i, img_info in enumerate(composition_images):
            create_composition_card(img_info, i)
        gr.HTML("</div>")
    
    # 构图技巧详解
    with gr.Accordion("📖 构图技巧详解", open=False):
        gr.Markdown("""
        ## 常见构图技巧与应用
        
        ### 1. S型构图
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
    with gr.Column():
        gr.HTML("""
        <div class='gallery-container'>
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
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px;
        padding: 20px;
        background: #f9f9f9;
        border-radius: 12px;
    }
    
    /* 画廊卡片 */
    .gallery-card {
        background: #fff;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0;
    }
    
    .gallery-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
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
        object-fit: cover; /* 填充容器，保持比例 */
        transition: transform 0.3s ease;
    }
    
    .gallery-card:hover .gallery-image {
        transform: scale(1.08);
    }
    
    /* 悬停遮罩层 */
    .gallery-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        opacity: 0;
        transition: opacity 0.3s ease;
        color: white;
        font-size: 18px;
    }
    
    .gallery-card:hover .gallery-overlay {
        opacity: 1;
    }
    
    .gallery-zoom-icon {
        font-size: 32px;
        margin-bottom: 8px;
    }
    
    .gallery-zoom-text {
        font-size: 14px;
        font-weight: bold;
    }
    
    /* 信息区域 */
    .gallery-info {
        padding: 12px;
        text-align: center;
        background: #fff;
    }
    
    .gallery-title {
        font-weight: bold;
        font-size: 15px;
        color: #333;
        margin-bottom: 6px;
    }
    
    .gallery-description {
        font-size: 13px;
        color: #666;
        line-height: 1.5;
        font-style: italic;
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
    @media (max-width: 768px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 12px;
            padding: 12px;
        }
        
        .gallery-info {
            padding: 8px;
        }
        
        .gallery-title {
            font-size: 14px;
        }
        
        .gallery-description {
            font-size: 12px;
        }
    }
    
    @media (max-width: 480px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 8px;
        }
        
        .gallery-image-container {
            padding-top: 100%; /* 正方形 */
        }
    }
    """
    
    with gr.Blocks(analytics_enabled=False, css=custom_css) as ui:
        gr.Markdown("""
        # 🎨 美学提升 - 构图与打光素材库
        
        学习经典美学法则，提升作品艺术品质。本模块提供构图技巧和打光技巧两大类素材资源。
        
        **学习建议**：
        - 构图是画面的骨架，决定视觉结构
        - 打光是画面的灵魂，塑造情感氛围
        - 两者结合，创造优秀的视觉作品
        """)
        
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
        
        with gr.Tabs():
            with gr.TabItem("📐 构图技巧"):
                create_composition_tab()
            
            with gr.TabItem("💡 打光技巧"):
                create_lighting_tab()
    
    return ui


if __name__ == "__main__":
    # 测试运行
    print("美学提升模块测试")
    images = get_composition_images()
    print(f"找到 {len(images)} 个构图素材:")
    for img in images:
        print(f"  - {img['title']}: {img['path']}")
