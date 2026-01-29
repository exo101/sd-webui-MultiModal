
"""
Qwen Image Extension - 配置和工具模块
用于模块化处理qwen_image_scripts.py中的功能
"""

import sys
from pathlib import Path

# 添加必要的路径
webui_root = Path(__file__).parent.parent.parent.parent
extensions_builtin = webui_root / "extensions-builtin"
forge_preprocessors = extensions_builtin / "forge_legacy_preprocessors"

paths_to_add = [
    str(webui_root),
    str(extensions_builtin),
    str(forge_preprocessors)
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

# 预处理器类型映射（UI显示名称到内部标识符）
CONTROLNET_PREPROCESSOR_DISPLAY_TO_INTERNAL = {
    # Canny 类别
    "[Canny] Canny": "canny",
    # Depth 类别
    "[Depth] Depth Midas": "depth_midas",
    "[Depth] Depth Leres": "depth_leres",
    "[Depth] Depth Leres++": "depth_leres++",
    "[Depth] Depth Anything": "depth_anything",
    "[Depth] Depth Anything V2": "depth_anything_v2",
    "[Depth] Depth Hand Refiner": "depth_hand_refiner",
    "[Depth] Depth Marigold": "depth_marigold",
    "[Depth] Depth Zoe": "depth_zoe",
    # Pose 类别
    "[Pose] Openpose Full": "openpose_full",
    "[Pose] Openpose": "openpose",
    "[Pose] Openpose Face": "openpose_face",
    "[Pose] Openpose Faceonly": "openpose_faceonly",
    "[Pose] Openpose Hand": "openpose_hand",
    "[Pose] DW Openpose Full": "dw_openpose_full",
    "[Pose] Animal Openpose": "animal_openpose",
    "[Pose] Densepose (purple bg & purple torso)": "densepose",
    "[Pose] Densepose Parula (black bg & blue torso)": "densepose_parula",
    # Lineart 类别
    "[Lineart] Lineart Standard (from white bg & black line)": "lineart_standard",
    "[Lineart] Lineart Realistic": "lineart_realistic",
    "[Lineart] Lineart Coarse": "lineart_coarse",
    "[Lineart] Lineart Anime": "lineart_anime",
    "[Lineart] Lineart Anime Denoise": "lineart_anime_denoise",
    # Softedge 类别
    "[Softedge] Scribble Pidinet": "scribble_pidinet",
    "[Softedge] Softedge Pidinet": "softedge_pidinet",
    "[Softedge] Softedge Pidinet Safe": "softedge_pidinet_safe",
    "[Softedge] Softedge Pidinstruct": "softedge_pidinstruct",
    "[Softedge] Softedge Hed": "softedge_hed",
    "[Softedge] Softedge Hedsafe": "softedge_hedsafe",
    # Inpaint 类别
    "[Inpaint] Inpaint Only": "inpaint_only",
    # 直接名称映射（为了兼容性）
    "canny": "canny",
    "depth_midas": "depth_midas",
    "depth_leres": "depth_leres",
    "depth_leres++": "depth_leres++",
    "depth_anything": "depth_anything",
    "depth_anything_v2": "depth_anything_v2",
    "depth_hand_refiner": "depth_hand_refiner",
    "depth_marigold": "depth_marigold",
    "depth_zoe": "depth_zoe",
    "openpose_full": "openpose_full",
    "openpose": "openpose",
    "openpose_face": "openpose_face",
    "openpose_faceonly": "openpose_faceonly",
    "openpose_hand": "openpose_hand",
    "dw_openpose_full": "dw_openpose_full",
    "animal_openpose": "animal_openpose",
    "densepose": "densepose",
    "densepose_parula": "densepose_parula",
    "lineart_standard": "lineart_standard",
    "lineart_realistic": "lineart_realistic",
    "lineart_coarse": "lineart_coarse",
    "lineart_anime": "lineart_anime",
    "lineart_anime_denoise": "lineart_anime_denoise",
    "scribble_pidinet": "scribble_pidinet",
    "softedge_pidinet": "softedge_pidinet",
    "softedge_pidinet_safe": "softedge_pidinet_safe",
    "softedge_pidinstruct": "softedge_pidinstruct",
    "softedge_hed": "softedge_hed",
    "softedge_hedsafe": "softedge_hedsafe",
    "inpaint_only": "inpaint_only",
    # 特殊值
    "None": "none",
    "none": "none",
    "": "none"  # 空字符串也视为"none"
}