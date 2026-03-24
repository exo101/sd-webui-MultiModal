# Stable Diffusion WebUI Forge 多模态图像处理插件

## 系统要求

- NVIDIA RTX 显卡可用范围
-  30 系 3080，3090，3090ti （虽然能用矿卡居多，不推荐）
-  40 系 4060ti，4070，4070ti，4070super，4080，4090
-  50 系 5060ti，5070，5070ti，5080，5090
- 显存：至少 12GB
- 内存：推荐 DDR5 64GB 
     
## 核心功能

- 🌟 **FLUX.2-klein 图像编辑**: 集成 FLUX.2-klein 进行上下文感知的图像编辑
- 🌟 **Qwen Image图像生成与编辑**: 支持复杂文本渲染和精确图像编辑
- 🎨 **Z-Image-Turbo 图像生成**: 高效高质量的图像生成模型
- 🎨 **Z-Image (base)**: Z-Image 正式版，支持文生图和图生图
- ✨ **美学提升**: 构图素材库与美学优化
- 🎬 **分镜助手**: 一体化工作流管理，剧本与分镜创作

## 已剥离的独立插件

以下功能已分离为独立插件，可根据需求选择安装：

### 1. sd-webui-sam-matting (图像分割/抠图/清理)
- ✂️ **智能抠图**: 基于 rembg 实现一键背景移除
- 🖌️ **图像分割**: 集成 Segment Anything Model (SAM) 进行精确图像分割
- 🧹 **图像清理**: 提供图像清理和修复功能
- **安装地址**: `extensions/sd-webui-sam-matting/`

### 2. sd-webui-qwen-vision-chat (图像识别与语言交互)
- 🖼️ **图像识别**: 支持 QwenVL视觉模型进行图像理解
- 💬 **语言交互**: 支持 Qwen 系列语言模型对话
- 📝 **快捷描述**: 自动生成提示词和图像描述
- 🏷️ **标签管理**: 批量处理关键词标签
- 🖼️ **图像管理**: 图片预览与管理工具
- **安装地址**: `extensions/sd-webui-qwen-vision-chat/`

## 各项目配置显存要求

- FLUX.2-klein-4B: 显存 10GB
- Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32: 显存 14GB
- nunchuku 加速 Qwen-Image-Edit-2501: 显存 10GB / 内存 64GB
- nunchuku 加速 Qwen-Image: 显存 10GB / 内存 64GB
- Qwen3VL: 显存 10GB
- Z-Image-Turbo: 显存 12GB
- nunchaku 加速-FLUX.1-Kontext: 显存 8GB
- nunchaku 加速 FLUX: 显存 8GB
- XL: 显存 8GB
- Cleaner: 显存 6GB 以上 (需安装 sam-matting 插件)
- Segment Anything: 显存 10GB (需安装 sam-matting 插件)

个人主页：[https://space.bilibili.com/403361177](https://space.bilibili.com/403361177?spm_id_from=333.40164.0.0) 

AI交流群qq：1054090769 整合包与模型在群公告

## 模型与插件下载

- 旧整合包已不适用日益更新的 AI 应用与 50 系显卡，我为此更新了新整合包环境，补充落后的 webui forge 生态
- 下载 sd-webui-MultiModal 插件文件到 `/extensions` 目录
- 凡是涉及 nunchaku 加速模型都区分着 50 系显卡与非 50 系显卡的模型，需要从魔搭社区进行搜索 nunchaku
- 大部分模型都在 qq 群内的百度或夸克网盘分享

## 插件目录结构

<img width="685" height="477" alt="QQ20251030-223846" src="https://github.com/user-attachments/assets/a1ded7d7-3311-4d15-98b7-34dc3dbcd108" />

## 模型目录配置

| 整合包目录 | 模型目录 | 子目录 | 说明 |
|---------------------|--------|-----------|-----------|
| `sd-webui-forge-aki` | `models` | `adetailer` | 修脸插件模型 |
| `sd-webui-forge-aki` | `models` | `cleaner` | 图像清理模型目录 (需 sam-matting 插件) |
| `sd-webui-forge-aki` | `models` | `RealESRGAN` | 高清放大算法目录 |
| `sd-webui-forge-aki` | `models` | `ESRGAN` | 高清放大算法目录 |
| `sd-webui-forge-aki` | `models` | `lora` | LoRA 微调模型目录 |
| `sd-webui-forge-aki` | `models` | `qwen-image` | qwen 模型与组件总目录 |
| `sd-webui-forge-aki` | `models` | `FLUX.2-klein` | flux2 模型目录 |
| `sd-webui-forge-aki` | `models` | `FLUX.1-Kontext-dev` | nunchuku 量化 flux 系列模型目录 |
| `sd-webui-forge-aki` | `models` | `ControlNet` | ControlNet 控制模型目录 |
| `sd-webui-forge-aki` | `models` | `ControlNetPreprocessor` | ControlNet 预处理器目录 |
| `sd-webui-forge-aki` | `models` | `sam` | 图像分割模型目录 (需 sam-matting 插件) |
| `sd-webui-forge-aki` | `models` | `Stable-diffusion` | 传统 flux/XL/1.5 模型目录 |
| `sd-webui-forge-aki` | `models` | `Tongyi-MAl` | Z-Image 模型目录 |
| `sd-webui-forge-aki` | `models` | `vae` | 图像编解码模型 |

## PS 插件目录

| PS 目录 | 插件目录 | 子目录 | 说明 |
|---------------------|--------|-----------|-----------|
| `Adobe Photoshop 2024` | `Plug-ins` | `sd-ppp_PS` | ps 插件目录 |
| `Adobe Photoshop 2024` | `Plug-ins` | `Auto.Photoshop.SD.plugin_v1.4.1` | ps 插件目录 |

## 更新日志

### 最新更新

- **模块化重构**: 为避免插件功能过多导致内存增加，MultiModal 插件已拆分为多个独立插件
  - `sd-webui-MultiModal`: 专注图像处理与 AI 绘画模型
  - `sd-webui-sam-matting`: 图像分割/抠图/清理功能
  - `sd-webui-qwen-vision-chat`: 图像识别与语言交互功能
  
- 增加 Flash Attention 加速轮子提速 Z-Image，qwen，flux，FLUX.2-klein，XL 等模型生成时间
- 增加 SageAttention 加速轮子提速 Z-Image，qwen，flux，FLUX.2-klein，XL 等模型生成时间
- 增加 nunchaku-qwen-2511 模型
- 增加 nunchaku-qwen-2512 模型
- 增加 Z-Image base 支持
- 增加 FLUX.2-klein-4b 模型 fp8 模型，nunchaku-Z-Image-Turbo，lora支持
 
### 历史更新

- 添加Z-Image-Turbo fp8 模型支持 lora支持，图生图支持
- 添加qwen，wan 系列 api 调用模型功能（qwenmix，qwenEdit，wan2.6，wan2.5 文生视频，图生视频，首尾帧等）
- 为 qwen，flux，Z-Image，等一众模型界面添加多角度提示词可视化选择器插件

- 添加 Qwen-Image-Edit-2511-ControlNet 支持
- 添加Z-Image-Turbo nuchaku 量化模型支持（transformer 主模型量化），显存降低 速度提升
- 添加 nunchaku flux ControlNet 支持（补全缺失模块）
- 整合包增加 ui 交互指南，降低使用难度，科普参数
- 添加图像识别模型 qwen3VL8b，qwen3VL4b，qwen3VL2b
- 添加 Qwen-Image-Edit-2511-SDNQ-uint4 量化模型支持
- 添加 nunchaku-qwen-image，lora ControlNet 模块，同时实现了深度，姿势，线稿，软边缘
- 添加 nunchaku-qwen-image-edit-2509 lora，ControlNet 功能实现，深度，线稿，姿势，局部编辑
- 添加 nunchaku qwen-image与 qwen-image-edit 2509

## 核心功能详解

### 1. FLUX.2-klein 图像编辑

FLUX.2-klein模型集成文生图、图像编辑、局部编辑、扩图等多位一体的编辑模型

**教程链接**: [https://www.bilibili.com/video/BV1gEzCBqEpA/](https://www.bilibili.com/video/BV1gEzCBqEpA/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37)

<img width="1787" height="886" alt="image" src="https://github.com/user-attachments/assets/3b999a97-ac93-41f3-80f5-b12304280ae5" />
<img width="1784" height="831" alt="image" src="https://github.com/user-attachments/assets/22a5efa9-2bfb-47b7-b4b9-d7cb7a993514" />
<img width="1749" height="831" alt="image" src="https://github.com/user-attachments/assets/29208ee2-77f6-43a3-b045-07c81eef1e56" />

### 2. Qwen Image图像生成与编辑

**模型演示教程**: [https://www.bilibili.com/video/BV1zn4TzKEdW/](https://www.bilibili.com/video/BV1zn4TzKEdW/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37)

- **qwen-image**: 基本文字生成，擅长中文理解，参数大的特点
- **qwen-image-edit plus**: 具备编辑图像能力，实现多种编辑效果
- 参考了 nunchaku 优化方法，生成时间与配置压力大幅度减少
- 模型分为适用于（非 50 系列显卡之前的用户）和（50 系列显卡）的用户

#### 加速模型详情页
- qwen-image 文生图加速主模型：[https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image/summary](https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image/summary)
- qwen-image-edit编辑加速主模型：[https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/summary](https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/summary)

<img width="1825" height="765" alt="88" src="https://github.com/user-attachments/assets/03327093-bb00-4a5f-ad11-a3ed31aaa90b" />

#### 模型版本说明

Lightning 模型是专门为快速推理设计的，训练时使用了特定的 CFG 设置：
- Lightning 模型 CFG 设置为 1
- 普通模型是完整训练的模型，对 CFG 参数更宽容，可以使用较高的 CFG 值为 4

**50 系模型**:
- `svdq-fp4_r128-qwen-image-lightningv1.1-8steps`
- `svdq-fp4_r128-qwen-image.safetensors` - 推理步数至少 15 往上，引导数是 4

**非 50 系模型**:
- `svdq-int4_r128-qwen-image-edit-2509.safetensors` - 推理步数至少 15 往上，引导数是 4
- `svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors` - 推理步数 10，引导数是 1

由于 nunchaku qwen 模型是一种量化的优化策略模型，完整版模型有 20B 参数，qwen lora 模型权重需要调整为 1.5 才能生效。

<img width="1256" height="898" alt="QQ20251023-190930" src="https://github.com/user-attachments/assets/a430135c-dc93-4515-b69a-34fa0e4d751f" /> 
<img width="1226" height="836" alt="QQ20251023-190809" src="https://github.com/user-attachments/assets/6db3520d-266e-4c75-9dbf-2cd972e572f4" />

**注意**: 
- 模型目录内的 `qwenimage` 与 `qwen-image-edit` 是主模型
- 编辑模型最多支持上传三张图像，但多图编辑能力弱于单图编辑能力

<img width="762" height="495" alt="24542525" src="https://github.com/user-attachments/assets/f8e58477-3e33-478c-ac0f-495da4adea4e" />
<img width="1474" height="960" alt="图层 2" src="https://github.com/user-attachments/assets/e6dcf697-2d5e-4612-80fd-732bf7afb4f9" />

以 qwen-image为例 5070ti 显卡，迭代步数 10，生成时间为 30-50 秒之间。

#### Qwen Image ControlNet

**示例教程**: [https://www.bilibili.com/video/BV13PsHz4E2C/](https://www.bilibili.com/video/BV13PsHz4E2C/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37)

Qwen 使用方式与 XL ControlNet 并无差别，得益于 qwen 模型的优化能力，生成效果与质量要远比 XL 好得多。

- 点击爆炸图标可预览预处理器结果
- 权重 0.7-1 之间
- 预处理器与模型都在网盘中 `Qwen-Image-ControlNet-Union`
- 这是一个综合 ControlNet 模型，同时具备深度、姿势、线稿、软边缘

<img width="1805" height="918" alt="23325" src="https://github.com/user-attachments/assets/2c6de0b0-7b72-4aba-aba7-2ff90368176e" />

**使用方法**:
- 在 ControlNet 中上传图像不能上传超过 1500 像素的图像，超过后会爆显存
- 使用 QQ 或微信截图，Ctrl+V 粘贴到上传图像的位置就行，这样就不必在 PS 中处理尺寸的问题了

<img width="871" height="515" alt="2545676" src="https://github.com/user-attachments/assets/2a2bf747-2035-4723-83e1-4bb18f7e42f0" />

在这些预处理器中只有属于pose、深度、线稿，以及属于他们的变体 qwen ControlNet 才支持，其余不支持，这是 qwen 官方训练 ControlNet 决定的。

#### Qwen Image Edit 2509 ControlNet

编辑模型自带 ControlNet 只需加载预处理器就可以控制图像，保持人物一致性，变化姿态，转换场景构图，编辑文字是个强大的多功能模型。

- 编辑模型可以同时使用自身的微调 lora 模型和 qwen-image lora 模型
- 可保持人物不变的情况下改变风格

### 3. Z-Image 系列

#### Z-Image-Turbo
- 高效快速的图像生成模型
- 支持 LoRA 和 ControlNet
- 支持文生图和图生图

#### Z-Image (base)
- Z-Image 正式版本
- 更稳定的生成质量
- 完整的文生图和图生图功能

### 4. 美学提升

- 构图素材库
- 美学优化建议
- 专业的构图指导

### 5. 分镜助手

- 一体化工作流管理
- 剧本创作与管理
- 分镜设计与可视化
- 完整的创作流程支持

---

**使用声明**: 使用此插件者请合法使用 AI。
