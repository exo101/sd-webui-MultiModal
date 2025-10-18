import os
import numpy as np
import torch
import gradio as gr
import cv2
from collections import OrderedDict
import random
from PIL import Image
import copy
import datetime

# 假设已安装 segment_anything 库，并且可以导入真实模型
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    

def initialize_sam_model(model_type=None):
    """初始化SAM模型"""
    if not SAM_AVAILABLE:
        print("无法找到 segment_anything 库，请先安装依赖")
        return None
        
    # 定义模型文件路径 - 使用WebUI的models目录 (models/sam)
    # 通过环境变量或默认方式获取WebUI根目录
    webui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_dir = os.path.join(webui_root, "models", "sam")
    model_paths = {
        "vit_h": os.path.join(model_dir, "sam_vit_h_4b8939.pth"),  # 2.38G
        "vit_l": os.path.join(model_dir, "sam_vit_l_0b3195.pth")   # 1.25G
    }
    
    # 如果没有指定模型类型，则查找第一个存在的模型
    if model_type is None:
        for m_type, m_path in model_paths.items():
            if os.path.exists(m_path):
                model_type = m_type
                print(f"找到模型文件: {m_type} ({m_path})")
                break
        else:
            # 如果没有找到任何模型文件
            print("未找到任何模型文件:")
            for m_type, m_path in model_paths.items():
                print(f"  {m_type}: {m_path}")
            print("请确保至少一个模型文件已下载并放置在正确路径下")
            return None
    
    model_path = model_paths.get(model_type, model_paths["vit_h"])
    
    # 检查指定的模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保模型文件已下载并放置在正确路径下")
        return None
    
    try:
        # 加载真实的 SAM 模型
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用官方提供的自动分割生成器和预测器
        mask_generator = SamAutomaticMaskGenerator(sam)
        predictor = SamPredictor(sam)
        
        print(f"成功加载模型: {model_type}")
        return mask_generator, sam, predictor  # 返回预测器用于点分割
    except Exception as e:
        print(f"SAM模型初始化失败: {str(e)}")
        return None

# 不在模块加载时初始化模型，改为在实际使用时初始化
sam_components = None  # 延迟初始化，在需要时才加载模型

def save_segmentation_results(results, segmentation_type):
    """
    保存分割结果到插件目录
    
    Args:
        results: 分割结果列表，包含PIL图像对象
        segmentation_type: 分割类型（"point_segmentation" 或 "random_segmentation"）
    """
    try:
        # 创建保存目录
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(plugin_dir, "segmentation_results", segmentation_type)
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳用于文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存每个结果
        saved_count = 0
        for i, result in enumerate(results):
            if result is not None:
                # 生成文件名
                filename = f"{segmentation_type}_{timestamp}_{i+1}.png"
                filepath = os.path.join(save_dir, filename)
                
                # 保存图像
                result.save(filepath, "PNG")
                print(f"已保存分割结果: {filepath}")
                saved_count += 1
        
        print(f"总共保存了 {saved_count} 个分割结果到: {save_dir}")
        
    except Exception as e:
        print(f"保存分割结果时出错: {e}")
        import traceback
        traceback.print_exc()


# 存储原始图像
original_image = None
points_state = []  # 存储当前标记点

def remove_points_from_image(image_np, points, point_radius=15):
    """从图像中移除标记点区域"""
    if len(points) == 0 or image_np is None:
        return image_np
        
    image_copy = image_np.copy()
    
    # 确保图像有正确的维度
    if len(image_copy.shape) < 3:
        image_copy = np.expand_dims(image_copy, axis=-1)
    
    # 为每个点创建一个遮罩区域并将其设置为透明或与背景融合
    for point in points:
        x, y = int(point[0]), int(point[1])
        
        # 创建圆形遮罩
        center_y, center_x = y, x
        
        # 计算受影响的区域
        y_min = max(0, y - point_radius)
        y_max = min(image_copy.shape[0], y + point_radius + 1)
        x_min = max(0, x - point_radius)
        x_max = min(image_copy.shape[1], x + point_radius + 1)
        
        # 创建网格坐标
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        # 计算距离中心点的距离
        distances = (yy - center_y) ** 2 + (xx - center_x) ** 2
        
        # 创建遮罩
        mask = distances <= point_radius ** 2
        
        # 如果图像有alpha通道，则将该区域设为透明
        if image_copy.shape[2] == 4:
            image_copy[y_min:y_max, x_min:x_max][mask, 3] = 0
        # 如果没有alpha通道，则将该区域设为黑色
        else:
            # 对于RGB图像，直接设置所有通道
            for c in range(image_copy.shape[2]):
                image_copy[y_min:y_max, x_min:x_max][mask, c] = 0
            
    return image_copy

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    """显示mask叠加在原图上的效果"""
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        # 确保图像和mask的维度匹配
        if image.ndim == 3 and mask.ndim == 2:
            # 对于RGB图像，应用mask到所有三个通道
            for c in range(min(3, image.shape[2])):  # 只处理存在的通道
                image[:, :, c] = np.where(mask, 
                                          image[:, :, c] * (1 - alpha) + 255 * color[c] * alpha, 
                                          image[:, :, c])
        elif image.ndim == 3 and mask.ndim == 3 and mask.shape[2] == 1:
            # 对于单通道mask，应用到所有三个通道
            mask_2d = mask[:, :, 0]
            for c in range(min(3, image.shape[2])):
                image[:, :, c] = np.where(mask_2d, 
                                          image[:, :, c] * (1 - alpha) + 255 * color[c] * alpha, 
                                          image[:, :, c])
    return image.astype(np.uint8)

def create_mask_output(image_np, masks):
    """创建mask输出图像，只返回最终的分割图像"""
    global points_state
    print(f"Creating output image, received {len(masks)} masks")
    print(f"Masks shape: {masks.shape}, image shape: {image_np.shape}")
    
    matted_images = []  # 只需要最终的分割图像
    for i, mask in enumerate(masks):
        # 确保只处理最多6个mask
        if i >= 6:
            print("Reached maximum number of masks (6), stopping processing")
            break
            
        # 获取mask的2D表示
        try:
            mask_2d = np.any(mask, axis=0)
        except Exception as e:
            print(f"Error processing mask {i}: {e}")
            continue
            
        print(f"Processing mask {i}, shape: {mask_2d.shape}, non-zero pixels: {np.count_nonzero(mask_2d)}")
        
        # 检查mask是否为空
        if np.count_nonzero(mask_2d) == 0:
            print(f"Skipping empty mask {i}")
            continue
            
        # 检查mask维度是否正确
        if mask_2d.ndim != 2:
            print(f"Skipping invalid mask {i} with wrong dimensions: {mask_2d.ndim}")
            continue
            
        # 检查图像形状是否匹配
        if mask_2d.shape[0] != image_np.shape[0] or mask_2d.shape[1] != image_np.shape[1]:
            print(f"Skipping mask {i} with mismatched dimensions. Mask: {mask_2d.shape}, Image: {image_np.shape}")
            continue
            
        try:
            # matted_images包含透明背景的原图，不移除标记点，因为结果图像中不应包含标记点
            if image_np.shape[2] == 4:
                # 图像已经有alpha通道
                image_np_copy = copy.deepcopy(image_np)
                image_np_copy[~mask_2d, 3] = 0
                # 不在结果图像中移除标记点，因为结果图像中本就不应该包含标记点
                matted_images.append(Image.fromarray(image_np_copy))
            else:
                # 图像没有alpha通道，需要添加
                image_np_copy = copy.deepcopy(image_np)
                # 添加alpha通道
                alpha_channel = np.full((image_np_copy.shape[0], image_np_copy.shape[1]), 255, dtype=np.uint8)
                alpha_channel[~mask_2d] = 0
                image_with_alpha = np.dstack([image_np_copy, alpha_channel])
                # 不在结果图像中移除标记点，因为结果图像中本就不应该包含标记点
                matted_images.append(Image.fromarray(image_with_alpha))
        except Exception as e:
            print(f"Error creating output for mask {i}: {e}")
            continue
            
    print(f"Generated {len(matted_images)} output images")
    # 只返回最终的分割图像
    # 过滤掉无效图像（尺寸为0的图像）
    filtered_images = []
    for img in matted_images:
        if img is not None and img.size[0] > 0 and img.size[1] > 0:
            filtered_images.append(img)
        else:
            print(f"Skipping invalid image: size={img.size if img is not None else 'None'}")
    
    print(f"Returning {len(filtered_images)} valid images after filtering")
    return filtered_images

def point_segmentation(img, points, model_type="vit_h"):
    """基于标记点的分割实现，每个点独立分割，只保留前两个掩码，并保存结果到插件目录"""
    global sam_components, points_state, original_image
    
    # 更新points_state
    points_state = points
    
    # 如果模型尚未初始化，则在此时初始化
    if sam_components is None:
        sam_components = initialize_sam_model(model_type)
        
    # 如果模型仍然不可用，返回错误信息
    if sam_components is None or not SAM_AVAILABLE:
        gr.Warning("无法初始化SAM模型，请确保模型文件已下载并放置在正确路径下")
        return []
    
    if img is None:
        gr.Warning("请先上传图片")
        return []
    
    # 如果没有标记点，返回错误信息
    if len(points) == 0:
        gr.Warning("请至少添加一个标记点")
        return []
    
    try:
        # 获取SAM组件
        mask_generator, sam, predictor = sam_components
        
        # 使用原始图像进行分割，而不是带标记点的图像
        segmentation_img = original_image if original_image is not None else img
        
        # 转换图像格式
        if segmentation_img.ndim == 3 and segmentation_img.shape[2] == 4:
            img_rgb = cv2.cvtColor(segmentation_img, cv2.COLOR_BGRA2RGB)
        elif segmentation_img.ndim == 2:
            img_rgb = cv2.cvtColor(segmentation_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = segmentation_img[..., :3]
        
        # 设置图像
        predictor.set_image(img_rgb)
        
        # 为每个点独立进行分割
        all_masks = []
        all_scores = []
        
        for point in points:
            # 准备单个点的坐标和标签
            input_points = np.array([point])
            input_labels = np.array([1])  # 1表示正点
            
            # 执行分割
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            print(f"Point SAM为点 {point} 生成 {len(masks)} 个mask，scores: {scores}")
            
            # 检查mask是否有效
            if masks is not None and len(masks) > 0:
                # 如果masks是列表或元组，转换为numpy数组
                if isinstance(masks, (list, tuple)):
                    masks = np.array(masks)
                
                # 只保留前两个掩码（删除第三个较大的掩码）
                if masks.shape[0] > 2:
                    masks = masks[:2]  # 只保留前两个掩码
                    scores = scores[:2] if len(scores) > 2 else scores
                
                # 确保mask具有正确的维度
                if masks.ndim == 3:
                    # SAM返回的masks形状为 (num_masks, height, width)
                    # 我们需要将其转换为 (num_masks, 1, height, width) 以与create_mask_output兼容
                    masks = masks[:, None, ...]
                elif masks.ndim == 4:
                    # masks已经是正确的形状，但只保留前两个
                    masks = masks[:2] if masks.shape[0] > 2 else masks
                else:
                    continue  # 跳过无效的mask
                
                # 添加到总结果中
                all_masks.append(masks)
                all_scores.extend(scores[:2] if len(scores) > 2 else scores)
        
        if not all_masks:
            gr.Warning("未能生成有效的分割结果")
            return []
        
        # 合并所有点的mask
        combined_masks = np.concatenate(all_masks, axis=0)
        print(f"总共生成 {len(combined_masks)} 个mask（已删除较大的第三个掩码）")
        
        # 使用create_mask_output函数处理结果，使用原始图像而不是带标记点的图像
        results = create_mask_output(segmentation_img, combined_masks)
        
        # 过滤掉空的图像结果并去重
        filtered_results = []
        seen_hashes = set()
        
        for img_result in results:
            # 检查图像是否有效
            if img_result is not None:
                # 检查PIL图像的尺寸
                if hasattr(img_result, 'size') and img_result.size[0] > 0 and img_result.size[1] > 0:
                    # 进一步检查是否包含有效像素
                    img_array = np.array(img_result)
                    if img_array.ndim >= 2 and np.sum(img_array) > 0:
                        # 创建图像的哈希值用于去重
                        img_hash = hash(img_array.tobytes())
                        if img_hash not in seen_hashes:
                            seen_hashes.add(img_hash)
                            filtered_results.append(img_result)
        
        if len(filtered_results) == 0:
            gr.Warning("未能生成有效的分割结果")
            return []
        
        # 保存结果到插件目录
        save_segmentation_results(filtered_results, "point_segmentation")
        
        print(f"Returning {len(filtered_results)} valid results")
        # 限制返回结果数量为6个
        return filtered_results[:6]
        
    except Exception as e:
        gr.Warning(f"点分割过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def random_segmentation(img, model_type="vit_h"):
    """基于SAM的自动随机分割实现，并保存结果到插件目录"""
    global sam_components
    
    # 如果模型尚未初始化，则在此时初始化
    if sam_components is None:
        sam_components = initialize_sam_model(model_type)
        
    # 如果模型仍然不可用，返回错误信息
    if sam_components is None or not SAM_AVAILABLE:
        gr.Warning("无法初始化SAM模型，请确保模型文件已下载并放置在正确路径下")
        return []
    
    if img is None:
        gr.Warning("请先上传图片")
        return []
    
    try:
        # 获取SAM自动分割生成器
        mask_generator, sam, predictor = sam_components
        
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 使用真实的自动分割生成器
        annotations = mask_generator.generate(img)
        
        # 引入随机性，确保每次运行生成不同结果
        random.shuffle(annotations)  # 打乱mask顺序以实现随机性
        
        print(f"Auto SAM生成 {len(annotations)} 个mask")
        
        H, W, _ = img.shape
        results = []
        
        # 创建已使用区域的掩码
        mask_used = np.zeros((H, W), dtype=bool)
        remaining_area = np.ones((H, W), dtype=bool)
        
        valid_masks = []
        
        # 动态调整面积阈值范围，增加随机性
        base_min = 0.005  # 基础最小阈值 0.5%
        base_max = 0.3    # 基础最大阈值 30%
        random_factor = random.uniform(0.8, 1.2)  # 随机因子
        min_threshold = (H * W) * (base_min * random_factor)
        max_threshold = (H * W) * (base_max * random_factor)
        
        # 第一次筛选：收集所有有效mask并计算重叠度
        for idx, annotation in enumerate(annotations):
            current_seg = annotation['segmentation']
            
            # 计算与已有mask的重叠度
            overlap_ratio = 0
            if mask_used.any():
                overlap_area = np.logical_and(current_seg, mask_used)
                overlap_ratio = np.sum(overlap_area) / np.sum(current_seg)
            
            # 计算剩余区域中的面积
            remaining_seg = np.logical_and(current_seg, remaining_area)
            area = np.sum(remaining_seg)
            
            # 动态调整重叠度阈值
            overlap_threshold = random.uniform(0.2, 0.4)
            
            # 过滤条件：面积在阈值范围内且重叠度要小
            if min_threshold <= area <= max_threshold and overlap_ratio < overlap_threshold:
                valid_masks.append((area, overlap_ratio, idx, annotation, remaining_seg))
        
        # 动态调整面积分组边界
        small_threshold = random.uniform(0.03, 0.07)
        medium_threshold = random.uniform(0.12, 0.18)
        
        # 按面积分组，确保选择不同大小范围的mask
        area_groups = {
            'small': [],    # 小面积
            'medium': [],   # 中等面积
            'large': []     # 大面积
        }
        
        for mask in valid_masks:
            area = mask[0] / (H * W)  # 计算相对面积比例
            if area < small_threshold:
                area_groups['small'].append(mask)
            elif area < medium_threshold:
                area_groups['medium'].append(mask)
            else:
                area_groups['large'].append(mask)
        
        # 动态调整每组选择的mask数量
        total_masks = 6  # 限制为6个结果
        group_counts = {}
        remaining_count = total_masks
        
        # 随机分配每组的mask数量
        for group in ['small', 'medium', 'large']:
            if group == 'large':
                group_counts[group] = remaining_count
            else:
                count = random.randint(1, min(2, remaining_count - 1))
                group_counts[group] = count
                remaining_count -= count
        
        # 从每个组中随机选择mask
        results = []
        used_indices = set()
        
        for group_name, group_masks in area_groups.items():
            if not group_masks:  # 如果该组没有mask，跳过
                continue
                
            random.shuffle(group_masks)  # 随机打乱每组内的mask
            count = 0
            target = group_counts[group_name]
            
            for area, overlap_ratio, idx, annotation, current_seg in group_masks:
                if count >= target or len(results) >= total_masks:
                    break
                    
                if idx in used_indices:
                    continue
                
                # 计算与当前所有已选mask的总重叠度
                total_overlap = np.sum(np.logical_and(current_seg, mask_used)) / np.sum(current_seg)
                
                # 动态调整接受阈值
                accept_threshold = random.uniform(0.25, 0.35)
                
                # 如果重叠度较低，则接受这个mask
                if total_overlap < accept_threshold:
                    used_indices.add(idx)
                    mask_used = np.logical_or(mask_used, current_seg)
                    remaining_area = np.logical_not(mask_used)
                    results.append((idx, annotation, current_seg))
                    count += 1
        
        # 如果结果不足预期数量，从剩余的mask中随机补充
        remaining_slots = total_masks - len(results)
        if remaining_slots > 0:
            remaining_masks = [m for m in valid_masks if m[2] not in used_indices]
            random.shuffle(remaining_masks)
            
            for mask in remaining_masks[:remaining_slots]:
                _, _, idx, annotation, current_seg = mask
                results.append((idx, annotation, current_seg))
        
        # 最终随机打乱结果顺序
        random.shuffle(results)
        
        # 处理每个有效的mask
        final_output = []
        seen_hashes = set()  # 用于去重
        
        for idx, annotation, current_seg in results:
            # 创建带有透明通道的图像
            result_img = np.zeros((H, W, 4), dtype=np.uint8)
            result_img[..., :3] = img
            
            # 将当前mask区域设置为不透明
            result_img[current_seg, 3] = 255
            
            # 计算边界框
            coords = np.argwhere(current_seg)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 随机调整边界扩展范围
            padding = random.randint(3, 8)
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(H-1, y_max + padding)
            x_max = min(W-1, x_max + padding)
            
            # 提取ROI区域
            cropped_img = result_img[y_min:y_max+1, x_min:x_max+1]
            
            # 转换为PIL图像并检查是否重复
            pil_image = Image.fromarray(cropped_img)
            img_array = np.array(pil_image)
            img_hash = hash(img_array.tobytes())
            
            # 只有在不重复的情况下才添加
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                final_output.append(pil_image)
            
            # 如果已经收集了足够的唯一结果，则停止
            if len(final_output) >= 6:
                break
        
        # 如果结果不足6个，但有更多未处理的结果，继续处理直到达到6个或没有更多结果
        if len(final_output) < 6:
            for idx, annotation, current_seg in results[len(final_output):]:
                if len(final_output) >= 6:
                    break
                    
                # 创建带有透明通道的图像
                result_img = np.zeros((H, W, 4), dtype=np.uint8)
                result_img[..., :3] = img
                
                # 将当前mask区域设置为不透明
                result_img[current_seg, 3] = 255
                
                # 计算边界框
                coords = np.argwhere(current_seg)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # 随机调整边界扩展范围
                padding = random.randint(3, 8)
                y_min = max(0, y_min - padding)
                x_min = max(0, x_min - padding)
                y_max = min(H-1, y_max + padding)
                x_max = min(W-1, x_max + padding)
                
                # 提取ROI区域
                cropped_img = result_img[y_min:y_max+1, x_min:x_max+1]
                
                # 转换为PIL图像并检查是否重复
                pil_image = Image.fromarray(cropped_img)
                img_array = np.array(pil_image)
                img_hash = hash(img_array.tobytes())
                
                # 只有在不重复的情况下才添加
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    final_output.append(pil_image)
        
        # 保存结果到插件目录
        save_segmentation_results(final_output, "random_segmentation")
        
        return final_output[:6]  # 确保最多返回6个结果
        
    except Exception as e:
        gr.Warning(f"自动分割过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def create_sam_segmentation():
    """创建Segment Anything图像分割界面组件"""
    global sam_components, original_image
    
    # 只有在用户实际使用功能时才尝试初始化模型
    if sam_components is None:
        # 可以选择在这里初始化默认模型，或者在用户第一次点击运行时初始化
        pass
    
    with gr.Row():
        gr.Markdown("## Segment Anything 模型参数说明")
    with gr.Row():
        gr.Markdown("""
        - **sam_vit_h_4b8939**: 最大模型，精度最高，但需要更多显存
        - **sam_vit_l_0b3195**: 中等模型，精度和资源消耗的平衡
        """)
    
    with gr.Row():
        sam_input = gr.Image(type="numpy", label="上传图像", interactive=True, height=400)
        sam_output = gr.Gallery(label="分割结果", columns=4, object_fit="contain")
    
    with gr.Row():
        model_type = gr.Dropdown(
            label="选择模型",
            choices=[("vit_h (2.38G)", "vit_h"), ("vit_l (1.25G)", "vit_l")],
            value="vit_h",
            interactive=True
        )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 自动分割")
            gr.Markdown("自动分割随机不同的区域")
            auto_run_button = gr.Button("运行自动分割")
            auto_clear_button = gr.Button("清除结果")
            
        
        with gr.Column():
            gr.Markdown("### 手动分割")
            gr.Markdown("在上传图像点击添加标记点进行分割，边缘瑕疵通过图生图进行修复）")
            point_run_button = gr.Button("根据标记点运行分割")
            point_clear_button = gr.Button("清除标记点和结果")
    
    # 用于存储标记点的状态变量
    points_state = gr.State([])
    
    def clear_results(img):
        """清除所有分割结果"""
        return [img] if img is not None else []
    
    def change_model(model_type):
        """切换模型"""
        global sam_components
        sam_components = initialize_sam_model(model_type)
        return f"模型已切换到: {model_type}"
    
    def save_original_image(img):
        """保存原始图像"""
        global original_image
        if img is not None:
            original_image = img.copy()
        return img
    
    def add_point(evt: gr.SelectData, img, points):
        """添加正点"""
        global original_image, points_state
        if img is None:
            return img, points
    
        # 保存原始图像（如果还没有保存）
        if original_image is None:
            original_image = img.copy()
        
        # 获取点击坐标
        x, y = evt.index
        point = [x, y]
        
        # 添加点
        points = points + [point]
        points_state = points  # 更新全局points_state
        
        # 在图像上绘制点 (增大点的大小)
        img_copy = img.copy()
        # 绘制所有点（红色）- 增大点的大小从5到12，增加边框使其更清晰
        for p in points:
            cv2.circle(img_copy, (p[0], p[1]), 12, (0, 0, 255), -1)  # 增大点的半径
            cv2.circle(img_copy, (p[0], p[1]), 15, (255, 255, 255), 2)  # 添加白色边框使点更清晰
    
        return img_copy, points
    
    def clear_all_points(img):
        """清除所有标记点"""
        global original_image
        # 恢复原始图像并清除点列表
        if original_image is not None:
            return original_image.copy(), []
        elif img is not None:
            return img.copy(), []
        else:
            return None, []
    
    # 绑定事件
    auto_run_button.click(
        random_segmentation,
        inputs=[sam_input, model_type],
        outputs=[sam_output]
    )
    
    auto_clear_button.click(
        clear_results,
        inputs=[sam_input],
        outputs=[sam_output]
    )
    
    point_run_button.click(
        point_segmentation,
        inputs=[sam_input, points_state, model_type],
        outputs=[sam_output]
    )
    
    point_clear_button.click(
        clear_all_points,
        inputs=[sam_input],
        outputs=[sam_input, points_state]
    )
    
    # 保存上传的原始图像
    sam_input.upload(
        save_original_image,
        inputs=[sam_input],
        outputs=[sam_input]
    )
    
    # 点击添加正点
    sam_input.select(
        add_point,
        inputs=[sam_input, points_state],
        outputs=[sam_input, points_state]
    )
    
    model_type.change(
        change_model,
        inputs=[model_type],
        outputs=[]
    )

    return {
        "sam_input": sam_input,
        "sam_output": sam_output,
        "model_type": model_type,
        "auto_run_button": auto_run_button,
        "auto_clear_button": auto_clear_button,
        "point_run_button": point_run_button,
        "point_clear_button": point_clear_button,
        "points_state": points_state
    }


def create_sam_ui():
    """
    创建图像分割UI模块的入口函数
    """
    return create_sam_segmentation()
