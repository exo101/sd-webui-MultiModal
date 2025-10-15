# 多模态SD插件

一个为 Stable Diffusion WebUI Forge 设计的多功能集成插件

## 系统要求

- 显卡：最低推荐 NVIDIA RTX 3060 /中端NVIDIA RTX 4070/中高端NVIDIA RTX 5070ti
- 显存：至少 12GB
- 内存：推荐 32/64GB 
     
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
- 🌟 **Qwen-Image复杂文本渲染和qwen-image-edit-2509精确图像编辑
- 
  插件支持：
- qwen-image-edit
- qwen-image
- qwenVL
- Index-TTS2
- LatentSync
- Ollama,deepseek
- Segment Anything
- FLUX.1-Kontext
- FLUX
- 1.5
- XL
- LoRA
- XL ControlNet
- 不支持FLUX ControlNet
  
个人主页：[https://space.bilibili.com/403361177?spm_id_from=333.788.upinfo.detail.click ](https://space.bilibili.com/403361177?spm_id_from=333.40164.0.0) 

WebUI Forge使用介绍：[https://www.bilibili.com/video/BV1BCHXzJE1C?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37  ](https://www.bilibili.com/video/BV1FWtBzbEiR?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37)

多模态插件使用介绍：[https://www.bilibili.com/video/BV16Ta3zFEpn?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37](https://www.bilibili.com/video/BV1B6xCzNEAT/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37)

所有插件包与整合包下载链接可在视频简介下方查看

### 前置要求

- Stable Diffusion WebUI Forge 环境  
  旧整合包已不适用日益更新的AI应用与50系显卡，我为此更新了新整合包环境，补充落后的webui生态
  https://github.com/exo101/sd-webui-forge-aki-v4.0/tree/main

- 克隆本仓库到 extensions 目录：
   sd-webui-forge-aki-v4.0/extensions


### 2025/10/12 更新多模态SD插件12版本：增加第八个功能标签页 qwen-image与 qwen-image-edit plus
 - <img width="1825" height="765" alt="88" src="https://github.com/user-attachments/assets/03327093-bb00-4a5f-ad11-a3ed31aaa90b" />
 
 qwen模型演示教程
 https://www.bilibili.com/video/BV1zn4TzKEdW/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37

   qwen-image基本文字生成，中文理解，参数大的特点，qwen-image-edit plus具备编辑图像，实现多种编辑效果的模型
   之前一直部署不上webui是因为没有好的优化方法和策略，最近参考了comfyui的nunchaku优化方法，生成时间与配置压力大幅度减少
    为大家带来更加便利的的使用方式，生成成功时会记录配置与参数设置信息
   
 - qwen-image为例
   <img width="861" height="435" alt="122" src="https://github.com/user-attachments/assets/650e86f6-a822-424d-ae60-9fed1f1426aa" />

 - 以编辑模型为例

    <img width="1815" height="854" alt="333" src="https://github.com/user-attachments/assets/37e5f859-263d-478d-ab63-b9d41a682217" />

 -  不融合lightning的 svdq-fp4_r128-qwen-image-edit-2509.safetensors质量最高，生成时间最长

    <img width="866" height="375" alt="111" src="https://github.com/user-attachments/assets/f0601d64-fec4-4efd-b841-e44b3277e246" />
   
 - 融合lightning的8步模型 svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors 质量较好，生成时间中等

    <img width="859" height="359" alt="222" src="https://github.com/user-attachments/assets/b6935a43-1868-4b0b-b8a5-cd0cd3bf4ff2" />
   
 -   在高配置的电脑上体现不出明显的时间差距，迭代步数越高时间越长，质量越高，最高不超过40，
    
     编辑模型最多支持上传三张图像，但多图编辑能力弱于单图编辑能力
    
     <img width="1842" height="947" alt="4444" src="https://github.com/user-attachments/assets/e2329e50-db48-4f1a-9cec-c293933f4993" />
   
## 目录结构

为了更清晰地展示 qwen-image 项目结构，以下为表格形式的思维导图目录结构说明：


| 主目录 | 子目录/文件 | 说明 |
|-------|-----------|------|
| `sd-webui-MultiModal\qwen-image\` | - | qwen-image插件根目录 |
| `sd-webui-MultiModal\qwen-image\` | `demo\` | 示例脚本目录 |
| `sd-webui-MultiModal\qwen-image\` | `models\` | 模型文件目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `qwenimage\` | 文本到图像生成模型目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `qwen-image-edit\` | 图像编辑模型目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `processor\` | 处理器组件目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `scheduler\` | 调度器组件目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `text_encoder\` | 文本编码器组件目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `tokenizer\` | 分词器组件目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `transformer\` | Transformer组件目录 |
| `sd-webui-MultiModal\qwen-image\models\` | `vae\` | VAE组件目录 |
| `sd-webui-MultiModal\qwen-image\` | `outputs\` | 生成图像输出目录 |
| `sd-webui-MultiModal\qwen-image\` | `qwen_image_scripts.py` | Qwen-Image功能核心脚本 |
| `sd-webui-MultiModal\qwen-image\` | `README.md` | 本说明文件 |


<img width="666" height="276" alt="234324" src="https://github.com/user-attachments/assets/56492f90-cd13-4e7c-8826-3e8ea1c003a2" />

<img width="780" height="504" alt="55555" src="https://github.com/user-attachments/assets/ce2cac1f-e7eb-4354-a7c0-cf99f6cb406d" />

- 模型下载说明：从基础模型中下载除了transformer（主模型）以外的，所有辅助模型组件，vae，text_encoder，scheduler，等配置文件

https://www.modelscope.cn/models/Qwen/Qwen-Image/files

qwen-image加速主模型详情页介绍
https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image/summary

qwen-image-edit加速主模型详情页介绍
https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/summary

为什么在webui中要这样下载呢？而不是comfyui的那样的方式，这是个原理性问题，因为diffusion库只认完整的模型组件和预训练模型，为什么不从
官方下载transformer（主模型）呢？因为官方的主模型实在太大了，4090显卡也很吃力，所以才要下载这个加速的主模型

<img width="700" height="805" alt="1233" src="https://github.com/user-attachments/assets/93bc645f-cfb5-45a4-a9ba-e4acd213c783" />

在WebUI Forge环境中安装nunchaku加速依赖，也就是打开D:\sd-webui-forge-aki-v4.0\python目录输入cmd

python -m pip install "D:\下载\nunchaku-1.0.0+torch2.7-cp311-cp311-win_amd64.whl" 

<img width="804" height="689" alt="65656" src="https://github.com/user-attachments/assets/cac2ff7c-88bf-4036-a8cd-02927e0e36c6" />
<img width="706" height="691" alt="234234" src="https://github.com/user-attachments/assets/fcba81c7-2534-4427-a258-4472e4699347" />
<img width="1094" height="414" alt="456536" src="https://github.com/user-attachments/assets/b50e172f-ae44-42cd-9c55-00f7af8235c3" />

### 模型版本
不同版本的模型在文件名中有明确标识，如 `lightningv1.0`、`lightningv1.1`、`lightningv2.0` 等。
 
生成信息（如配置参数、生成时间等）也会一并记录

### 2025-10-09 多模态插件11版本

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
<img width="1851" height="953" alt="3" src="https://github.com/user-attachments/assets/aaaedc60-8b8a-4d13-85e1-64599e71d5b1" />
<img width="1829" height="965" alt="2" src="https://github.com/user-attachments/assets/606bfe39-5b26-4c4a-a400-6aa496a75cb4" />
<img width="1816" height="789" alt="13" src="https://github.com/user-attachments/assets/16ff1933-5ff9-46c4-b533-90fca5e15c44" />



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

### 3. 图像处理工具集

#### 智能抠图
- 基于 rembg 实现高质量背景移除<img width="1782" height="896" alt="7" src="https://github.com/user-attachments/assets/63e9293d-09b2-494a-8ea9-8eaa46aef287" />

- 支持透明背景和自定义背景色
- 批量处理功能<img width="1798" height="679" alt="8" src="https://github.com/user-attachments/assets/9450ffa8-f8ac-4ca0-bccd-7e709f873369" />

- 实时预览效果

#### 图像分割
- 集成 Segment Anything Model (SAM)
- 精确的图像分割功能
- 支持点选和自动分割方式
<img width="1812" height="917" alt="10" src="https://github.com/user-attachments/assets/c0ca0f59-be6e-408c-bb4c-a117d718e588" />
<img width="1816" height="909" alt="9" src="https://github.com/user-attachments/assets/08b633ba-7ed8-4886-a6f3-e22a3bd7cb8e" />

- 可下载分割结果

#### 图像清理
- 图像去噪和修复功能
- 简单易用的界面<img width="1835" height="741" alt="11" src="https://github.com/user-attachments/assets/4ac7c40d-971c-4364-8cc1-73a872fcec79" />

- 支持多种清理模式

### 4. 视频关键帧提取
- 多种提取模式（关键帧/等间隔/场景变化）
- 可调节提取质量
- 支持多种视频格式
- 可预览提取的帧<img width="1809" height="677" alt="12" src="https://github.com/user-attachments/assets/23b3cb3d-c763-4432-894f-fdc84b8c7b9f" />


### 5. 数字人视频生成
- 基于 LatentSync 的音频驱动视频生成
- 支持自定义推理步数和引导尺度
- 需要清晰正面人脸的视频作为输入
- 支持多种音频格式
<img width="1831" height="925" alt="18" src="https://github.com/user-attachments/assets/4b380e69-3814-4078-ac3e-9f228d83bcde" />
https://github.com/user-attachments/assets/587086f5-5204-4953-b37b-5c1c72a97f61



### 6. Index-TTS语音合成
- 集成 Index-TTS 实现高质量语音合成
- 支持多种语音风格
- 可调节语速、音调等参数
- 支持中文和多语言合成
- <img width="1786" height="805" alt="23" src="https://github.com/user-attachments/assets/1318c3fa-c979-4c93-8003-639e5f43f7f6" />
- <img width="1788" height="428" alt="17" src="https://github.com/user-attachments/assets/52ed7801-36f3-4145-9386-f2ae7285ea11" />
   [output_1760002640.wav](https://github.com/user-attachments/files/22794279/output_1760002640.wav)



### 7. FLUX.1-Kontext图像编辑
- 上下文感知的图像编辑功能
- 支持基于文本的图像修改
- 保持图像上下文一致性
- GGUF量化模型优化使用门槛12g显存可用
- 单图编辑
- <img width="729" height="485" alt="22" src="https://github.com/user-attachments/assets/564199b4-bad1-4cfb-9629-88632482c6fa" />
<img width="864" height="1200" alt="generated_image_1756609759_img1_var2" src="https://github.com/user-attachments/assets/93b6c6a8-a8cd-424c-bf4a-2b02a83ca495" />


双图编辑
<img width="1813" height="571" alt="21" src="https://github.com/user-attachments/assets/4df0079b-ff8d-4290-ae16-7e367eb90881" />
<img width="1024" height="1024" alt="dual_context_image_1756582213_var1" src="https://github.com/user-attachments/assets/1bf91812-70a9-4662-aed1-ac6839a274ab" />

## 文件夹结构说明
<img width="660" height="387" alt="14" src="https://github.com/user-attachments/assets/32c734e4-e84f-4909-a020-3fee6abe35ad" />

为了更清晰地展示项目结构，以下为表格形式的思维导图目录结构说明：

### 主插件结构

| 主目录 | 子目录 | 说明 |
|-------|-------|------|
| `sd-webui-MultiModal\` | `scripts\` | 主功能模块脚本目录 |
| `sd-webui-MultiModal\` | `XYKC_AI\` | AI模型API接口目录 |
| `sd-webui-MultiModal\XYKC_AI\` | `XYKC_AI_PyScripts\` | Python脚本接口 |


#### 图像识别与语言交互
安装(qwen2.5vl)视觉模型与(qwen3)语言模型  
在计算机开始菜单搜索栏输入CMD执行以下命令：
ollama run qwen2.5vl:3b
ollama run qwen3:1.7b
参数越大响应速度越慢质量越高，模型选择建议：8GB显存选择1.7B或3B模型获得更快响应速度，16GB显存可选择latest或7B模型

使用语音合成或视频处理功能需下载ffmpeg
开始菜单搜索环境变量
添加C:\ffmpeg\bin到环境变量

<img width="1693" height="734" alt="QQ20251011-134442" src="https://github.com/user-attachments/assets/651fa968-f16d-4084-b6af-db12ac26632d" />

### 图像处理模块

| 主目录 | 子目录 | 说明 |
|-------|-------|------|
| `sd-webui-MultiModal\` | `cleaner\` | 图像清理独立模块 |
| `sd-webui-MultiModal\cleaner\` | `models\` | 图像清理模型目录 |


### 语音合成模块

| 主目录 | 子目录 | 说明 |
|-------|-------|------|
| `sd-webui-MultiModal\` | `index-tts\` | Index-TTS语音合成独立模块 |
| `sd-webui-MultiModal\index-tts\` | `checkpoints\` | TTS模型检查点目录 |


### 数字人视频生成模块

| 主目录 | 子目录 | 说明 |
|-------|-------|------|
| `sd-webui-MultiModal\` | `LatentSync\` | 数字人视频生成独立模块 |
| `sd-webui-MultiModal\LatentSync\` | `checkpoints\` | 主模型检查点目录 |
| `sd-webui-MultiModal\LatentSync\checkpoints\` | `sd-vae-ft-mse\` | VAE模型目录 |
| `sd-webui-MultiModal\LatentSync\checkpoints\` | `whisper\` | Whisper模型目录 |
| `sd-webui-MultiModal\LatentSync\` | `latentsync\` | 核心代码目录 |
| `sd-webui-MultiModal\LatentSync\latentsync\` | `checkpoints\` | 辅助模型检查点目录 |



### 图像编辑模块

| 主目录 | 子目录 | 说明 |
|-------|-------|------|
| `sd-webui-MultiModal\` | `FLUX.1-Kontext\` | FLUX.1图像编辑独立模块 |
| `sd-webui-MultiModal\FLUX.1-Kontext\` | `models\` | 图像编辑模型目录 |
| `sd-webui-MultiModal\FLUX.1-Kontext\models\` | `FLUX.1-Kontext-dev\` | FLUX.1主模型目录 |
| `sd-webui-MultiModal\FLUX.1-Kontext\` | `lora\` | LoRA微调模型目录 |



 重启 WebUI

## 使用须知

使用此插件者请合法使用AI，不得发表不正当言论，作假新闻，二次销售，之后的一切行为与插件开发者无关。
