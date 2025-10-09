# MultiModal-Forge 插件

一个为 Stable Diffusion WebUI Forge 设计的多功能集成插件，集成了图像识别、语言交互、语音克隆，视频处理、图像编辑，清理，分割，抠图等多种AI功能。

个人主页：https://space.bilibili.com/403361177?spm_id_from=333.788.upinfo.detail.click  
WebUI Forge使用介绍：https://www.bilibili.com/video/BV1BCHXzJE1C?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37  
多模态插件使用介绍：https://www.bilibili.com/video/BV16Ta3zFEpn?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37

## 核心功能

- 📚 **资源汇总**: 集中管理各类资源和公告信息
- 🖼️ **图像识别与语言交互**: 支持多种视觉和语言模型，可进行图像描述、内容分析等
- ✂️ **智能抠图**: 基于 rembg 实现一键背景移除
- 🖌️ **图像分割**: 集成 Segment Anything Model (SAM) 进行精确图像分割
- 🧹 **图像清理**: 提供图像清理和修复功能
- 🎬 **视频关键帧提取**: 从视频中提取关键帧用于进一步处理
- 🤖 **数字人视频生成**: 基于 LatentSync 实现音频驱动的数字人视频生成
- 🔊 **TTS语音合成**: 集成 Index-TTS 实现高质量文本转语音
- 🌟 **FLUX.1 图像编辑**: 集成 FLUX.1-Kontext 进行上下文感知的图像编辑

## 功能模块详细介绍

### 1. 资源汇总
- 集中展示重要公告和资源信息
- 提供快速访问各类功能的入口
- 显示插件使用说明和更新日志<img width="1245" height="650" alt="1" src="https://github.com/user-attachments/assets/f9b99645-a76a-43ce-aa27-1d5774e9cfa3" />


### 2. 图像识别与语言交互

#### 核心特性
- 支持多种视觉模型（Qwen-VL、LLaMA-Vision等）
- 支持多种语言模型（Qwen、DeepSeek等）
- 提供快捷提示词模板
- 支持单张和批量图像处理
- 根据显存大小推荐合适的模型（8GB显存推荐1.7B/3B模型，16GB显存可选latest/7B模型）

#### 模型类型
- 图像识别模型：可处理图片输入，支持单张和批量操作
- 语言交互模型：仅支持文本对话，不处理图片

#### 快捷描述功能
提供多种预设提示词模板：
- 自然语言描述文本
- Stable Diffusion提示词
- MidJourney提示词
- 分镜构图描述
- 图生视频描述
- 文生视频描述文本
- 艺术评论分析
- 产品列表描述

#### 使用说明
1. 选择模型类型（图像识别/语言交互）
2. 选择相应的视觉或语言模型
3. 选择上传方式（单张图片/批量图片）
4. 上传图片或输入文本进行交互

### 3. 图像处理工具集

#### 智能抠图
- 基于 rembg 实现高质量背景移除
- 支持透明背景和自定义背景色
- 批量处理功能
- 实时预览效果

#### 图像分割
- 集成 Segment Anything Model (SAM)
- 精确的图像分割功能
- 支持点选和框选分割方式
- 可下载分割结果

#### 图像清理
- 图像去噪和修复功能
- 简单易用的界面
- 支持多种清理模式

### 4. 视频关键帧提取
- 多种提取模式（关键帧/等间隔/场景变化）
- 可调节提取质量
- 支持多种视频格式
- 可预览提取的帧

### 5. 数字人视频生成
- 基于 LatentSync 的音频驱动视频生成
- 支持自定义推理步数和引导尺度
- 需要清晰正面人脸的视频作为输入
- 支持多种音频格式

### 6. Index-TTS语音合成
- 集成 Index-TTS 实现高质量语音合成
- 支持多种语音风格
- 可调节语速、音调等参数
- 支持中文和多语言合成

### 7. FLUX.1-Kontext图像编辑
- 上下文感知的图像编辑功能
- 支持基于文本的图像修改
- 保持图像上下文一致性
- GGUF量化模型优化使用门槛12g显存可用

## 文件夹结构说明

```sd-webui-MultiModal/
├── scripts/                           # 主功能模块脚本目录
├── XYKC_AI/                           # AI模型API接口目录
│   └── XYKC_AI_PyScripts/             # Python脚本接口
├── cleaner/                           # 图像清理独立模块
│   └── models/                        # 图像清理模型目录
│       └── big-lama.safetensors       # 图像清理主模型文件
├── index-tts/                         # Index-TTS语音合成独立模块
│   └── checkpoints/                   # TTS模型检查点目录
│       ├── gpt.pth                    # GPT模型文件
│       ├── s2mel.pth                  # S2MEL模型文件
│       ├── bigvgan_v2_22khz_80band_256x/ # 声码器模型目录
│       ├── w2v-bert-2.0/              # Wav2Vec-BERT模型目录
│       └── ...                        # 其他TTS相关模型文件
├── LatentSync/                        # 数字人视频生成独立模块
│   ├── checkpoints/                   # 主模型检查点目录
│   │   ├── latentsync_unet.pt         # LatentSync主模型文件
│   │   ├── sd-vae-ft-mse/             # VAE模型目录
│   │   │   └── diffusion_pytorch_model.safetensors  # VAE权重文件
│   │   ├── stable_syncnet.pt          # SyncNet模型文件
│   │   └── whisper/                   # Whisper模型目录
│   └── latentsync/                   # 核心代码目录
│       └── checkpoints/              # 辅助模型检查点目录
└── FLUX.1-Kontext/                    # FLUX.1图像编辑独立模块
    ├── models/                        # 图像编辑模型目录
    │   └── FLUX.1-Kontext-dev/        # FLUX.1主模型目录
    │       ├── flux1-kontext-dev-Q6_K.gguf  # GGUF格式主模型文件
    │       ├── transformer/           # Transformer模型组件
    │       │   ├── config.json        # Transformer配置文件
    │       │   └── ...                # 其他Transformer相关文件
    │       ├── vae/                   # VAE模型组件
    │       │   ├── config.json        # VAE配置文件
    │       │   └── diffusion_pytorch_model.safetensors  # VAE权重文件
    │       ├── text_encoder/          # 文本编码器
    │       │   ├── config.json        # 文本编码器配置文件
    │       │   └── model.safetensors  # 文本编码器模型文件
    │       ├── text_encoder_2/        # 第二文本编码器
    │       │   ├── config.json        # 第二文本编码器配置文件
    │       │   ├── model.safetensors  # 第二文本编码器模型文件
    │       │   ├── model-00001-of-00002.safetensors  # 分片模型文件1
    │       │   ├── model-00002-of-00002.safetensors  # 分片模型文件2
    │       │   └── model.safetensors.index.json     # 模型索引文件
    │       ├── tokenizer/             # 分词器
    │       │   ├── merges.txt         # 合并规则文件
    │       │   ├── special_tokens_map.json  # 特殊标记映射文件
    │       │   ├── tokenizer_config.json    # 分词器配置文件
    │       │   └── vocab.json         # 词汇表文件
    │       ├── tokenizer_2/           # 第二分词器
    │       │   ├── special_tokens_map.json  # 特殊标记映射文件
    │       │   ├── spiece.model       # SentencePiece模型文件
    │       │   ├── tokenizer.json     # 分词器文件
    │       │   └── tokenizer_config.json    # 分词器配置文件
    │       ├── scheduler/             # 调度器
    │       │   └── scheduler_config.json    # 调度器配置文件
    │       └── model_index.json       # 模型配置索引文件
    └── lora/                          # LoRA微调模型目录
        ├── Kontext-电商重打光_v1.safetensors      # 电商打光LoRA模型
        └── Kontext游戏资源配色与升级编辑_1.0.safetensors # 游戏资源编辑LoRA模型

## 安装说明

### 前置要求

- Stable Diffusion WebUI Forge 环境  
  旧整合包已不适用日益更新的AI应用与50系显卡，我为此更新了新整合包环境

#### 大陆国内用户
通过网盘分享的文件：更新环境库的webui Forge整合包  
链接: https://pan.baidu.com/s/1FL7WVbcdvHfMtGUs-QOTVg 提取码: jwmu

#### 海外用户
如果使用此插件需更新Forge环境（由于体积较大无法方便上传只能如此了）  
需要以下依赖：
```
diffusers==0.35.1

Torch 2.7.0 (CUDA 12.8)+ xFormers 0.0.30

transformers==4.53.1

huggingface-hub==0.34.4
```

每个项目都可以独立运行既可按需下载，也可全部下载

### 功能模块安装指南

#### 图像识别与语音交互功能
安装Ollama应用程序：https://ollama.com/  
安装(qwen2.5vl)视觉模型与(qwen3)语言模型  
在计算机开始菜单搜索栏输入CMD执行以下命令：
```
ollama run qwen2.5vl:3b

ollama run qwen3:1.7b
```
参数越大响应速度越慢质量越高，模型选择建议：8GB显存选择1.7B或3B模型获得更快响应速度，16GB显存可选择latest或7B模型

#### 图像清理功能
需下载模型big-lama.safetensors  

**大陆国内用户**通过网盘分享的文件：cleaner  
链接: https://pan.baidu.com/s/1P8XlDjPvjFnfu4MumPE9sg 提取码: twqc 

**海外用户**  
https://huggingface.co/kaitte/big-lama/blob/main/big-lama.safetensors

#### 图像分割功能
需下载模型sam_vit_h_4b8939.pth ，sam_vit_l_0b3195.pth  

**大陆国内用户**通过网盘分享的文件：图像分割模型  
链接: https://pan.baidu.com/s/1xiioFavOcrxp3DvXE_mdIQ 提取码: iah5 

**海外用户**  
https://huggingface.co/HCMUE-Research/SAM-vit-h/tree/main  
https://huggingface.co/datasets/Gourieff/ReActor/tree/main/models/sams

#### 语音合成功能
需下载模型  

**大陆国内用户**通过网盘分享的文件：index-tts.7z  
链接: https://pan.baidu.com/s/1i9LYtdWcOZpzKbSsBR04jg 提取码: r79k 

**海外用户**  
https://huggingface.co/IndexTeam/IndexTTS-2/tree/main

#### 数字人视频生成功能
需下载模型  

**大陆国内用户**通过网盘分享的文件：LatentSync.7z  
链接: https://pan.baidu.com/s/18RQoQvH_zqmVX4RtAIAenw 提取码: u55x 

**海外用户**  
https://huggingface.co/ByteDance/LatentSync-1.5/tree/main

#### Kontext图像编辑功能
需下载模型  

**大陆国内用户**通过网盘分享的文件：FLUX.1-Kontext  
链接: https://pan.baidu.com/s/1LiT2OEXdDTA5DeV9SEKfEA 提取码: 73ac 

**海外用户**  
https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/tree/main  
注释（无需下载主模型，已用GGUF量化模型代替主模型）  
https://huggingface.co/bullerwins/FLUX.1-Kontext-dev-GGUF/tree/main

### 最后安装步骤

1. 克隆本仓库到 extensions 目录：
   ```bash
   cd sd-webui-forge-aki-v4.0/extensions
   git clone https://github.com/yourusername/MultiModal-Forge.git
   ```
2. 根据需要下载相应的模型文件

3. 重启 WebUI

## 使用须知

使用此插件者请合法使用AI，不得发表不正当言论，作假新闻，二次销售，之后的一切行为与插件开发者无关。
