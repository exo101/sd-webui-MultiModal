# Qwen-Image 模块说明

本目录包含 Qwen-Image 模型及相关组件，用于在 Stable Diffusion WebUI Forge 中实现文本到图像生成和图像编辑功能。

## 目录结构

```
qwen-image/
├── demo/                           # 示例脚本目录
│   ├── qwen-image-edit-2509.py     # 图像编辑示例脚本
│   └── qwen-image-lightning.py     # 文本到图像生成示例脚本
├── models/                         # 模型文件目录
│   ├── qwenimage/                  # 文本到图像生成模型
│   │   ├── svdq-fp4_r128-qwen-image-lightningv1.0-4steps.safetensors
│   │   ├── svdq-fp4_r128-qwen-image-lightningv1.1-8steps.safetensors
│   │   └── svdq-fp4_r128-qwen-image.safetensors
│   ├── qwen-image-edit/            # 图像编辑模型
│   │   ├── svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors
│   │   ├── svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors
│   │   └── svdq-fp4_r128-qwen-image-edit-2509.safetensors
│   ├── processor/                  # 处理器组件
│   ├── scheduler/                  # 调度器组件
│   ├── text_encoder/               # 文本编码器组件
│   ├── tokenizer/                  # 分词器组件
│   ├── transformer/                # Transformer 组件
│   ├── vae/                        # VAE 组件
│   ├── model_index.json            # 模型索引文件
│   ├── README.md                   # 模型说明文件
│   ├── LICENSE                     # 许可证文件
│   └── .gitattributes              # Git 属性文件
├── nunchaku/                       # Nunchaku 库（Qwen-Image 优化库）
├── outputs/                        # 生成图像输出目录
├── qwen_image_scripts.py           # Qwen-Image 功能核心脚本
└── README.md                       # 本说明文件
```

## 各目录及文件说明

### demo/
包含 Qwen-Image 模型的使用示例脚本：
- `qwen-image-lightning.py`: 展示如何使用 Qwen-Image Lightning 模型进行文本到图像生成
- `qwen-image-edit-2509.py`: 展示如何使用 Qwen-Image Edit 模型进行图像编辑

### models/
模型文件及相关组件目录。

#### models/qwenimage/
文本到图像生成模型文件：
- `svdq-fp4_r128-qwen-image-lightningv1.0-4steps.safetensors`: 4步推理的 Lightning 模型 v1.0
- `svdq-fp4_r128-qwen-image-lightningv1.1-8steps.safetensors`: 8步推理的 Lightning 模型 v1.1
- `svdq-fp4_r128-qwen-image.safetensors`: 标准 Qwen-Image 模型

#### models/qwen-image-edit/
图像编辑模型文件：
- `svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors`: 4步推理的图像编辑模型
- `svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors`: 8步推理的图像编辑模型
- `svdq-fp4_r128-qwen-image-edit-2509.safetensors`: 标准图像编辑模型

### outputs/
图像生成输出目录，所有通过 Qwen-Image 生成的图像都会保存在此目录中。

### qwen_image_scripts.py
Qwen-Image 功能的核心脚本，包含：
- 文本到图像生成功能
- 图像编辑功能
- 模型加载和推理逻辑
- 与 WebUI 的接口函数

## 模型特点

### 推理步数
模型根据推理步数分为两类：
- 4步模型：适合快速生成，质量稍低但速度更快
- 8步模型：生成质量更高，但需要更多推理时间

### Rank 等级
模型文件名中的 `r128` 表示 Rank 等级为 128，提供更好的生成质量。

### 模型版本
不同版本的模型在文件名中有明确标识，如 `lightningv1.0`、`lightningv1.1`、`lightningv2.0` 等。

## 使用说明

1. 通过 WebUI 界面选择相应的模型文件进行文本到图像生成或图像编辑
2. 根据需要选择合适的推理步数（4步或8步）
3. 生成的图像将自动保存在 `outputs/` 目录中
4. 生成信息（如配置参数、生成时间等）也会一并记录

## 系统要求

- 显卡：推荐 NVIDIA RTX 4070 Ti 或更高配置
- 显存：至少 12GB
- 内存：推荐 64GB 或更高
