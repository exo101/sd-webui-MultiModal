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
- 🤖 **数字人视频生成**: 基于 LatentSync 实现音频驱动的数字人唇形同步视频生成
- 🔊 **TTS语音合成**: 集成 Index-TTS 实现高质量文本转语音
- 🌟 **FLUX.1 图像编辑**: 集成 FLUX.1-Kontext 进行上下文感知的图像编辑
- 🌟 **Qwen-Image复杂文本渲染和qwen-image-edit-2509精确图像编辑

## 各项目配置显存要求

- Qwen-Image: 8GB以上
- Qwen3vL:    8GB以上
- LatentSync: 12GB
- Index-TTS:  8GB以上
- FLUX.1-Kontext:  12GB
- FLUX: 10GB
- XL: 8GB
- Cleaner: 4GB以上
- Segment Anything:  8GB

个人主页：[https://space.bilibili.com/403361177?spm_id_from=333.788.upinfo.detail.click ](https://space.bilibili.com/403361177?spm_id_from=333.40164.0.0) 

WebUI Forge使用介绍：[https://www.bilibili.com/video/BV1BCHXzJE1C?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37  ](https://www.bilibili.com/video/BV1FWtBzbEiR?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37)

多模态插件使用介绍：[https://www.bilibili.com/video/BV16Ta3zFEpn?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37](https://www.bilibili.com/video/BV1B6xCzNEAT/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37)

所有插件包与整合包下载链接可在视频简介下方查看

### 前置要求

- Stable Diffusion WebUI Forge 环境  
  旧整合包已不适用日益更新的AI应用与50系显卡，我为此更新了新整合包环境，补充落后的webui生态
  https://github.com/exo101/sd-webui-forge-aki-v4.0/tree/main

- 克隆本仓库到 extensions 目录： sd-webui-forge-aki-v4.4/extensions
- 2025/10/24 webui frogr 整合包更新之4.4版本
- 2025/10/24 支持qwen3VL需要transformers==4.57.0，但会牺牲Index-TTS2语音合成功能使用，因为Index-TTS2需要的是transformers==4.52.1版本，两者存在冲突
- 2025/10/24 更新完善qwen-image ControlNet实现多个变体预处理器
- 2025/10/18 添加qwen-image ControlNet。同时实现了深度，姿势，线稿，软边缘
- 2025/10/12 更新多模态SD插件12版本：增加第八个功能标签页 qwen-image与 qwen-image-edit plus

## 功能模块详细介绍
 ## 插件目录
 
<img width="660" height="387" alt="14" src="https://github.com/user-attachments/assets/32c734e4-e84f-4909-a020-3fee6abe35ad" />
| 整合包目录 | 插件目录 | 插件子目录 |项目代码目录 | 说明 |
|----------|-------|-----------|-----------|-----------|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|` cleaner`|图像分割插件配置文件|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`index-tts2`| 图像清理插件配置文件|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`FLUX.1-Kontext`|图像编辑插件配置文件|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`index-tts2`| Index-TTS语音合成配置文件|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`LatentSync`|数字人视频生成模型与配置文件|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`qwen-image`|模型文件配置文件 |
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`scripts`| 主插件配置文件|
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal`|`XYKC_AI`| 图像识别插件配置文件|

## 插件模型目录
| 整合包目录 | 模型目录 | 子目录 |说明 |
|----------|-----|-----------|-----------|
| `sd-webui-forge-aki`|`models`|`sam`| 图像分割模型目录 |
| `sd-webui-forge-aki`|`models`|`cleaner`| 图像清理模型目录 |
| `sd-webui-forge-aki`|`models`|`index-tts2`| Index-TTS语音合成模型目录 |
| `sd-webui-forge-aki`|`models`|`FLUX.1-Kontext-dev`| 图像编辑模型目录 |
| `sd-webui-forge-aki`|`models`|`lora`|Kontext LoRA微调模型目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image`| qwen模型文件目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image\qwenimage`| qwen文生图主模型目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image\qwen-image-edit`| qwen图像编辑主模型目录 |
| `sd-webui-forge-aki`|`models`|`ControlNet\ Qwen-Image-ControlNet-Union`| qwen ControlNet模型目录 |
| `sd-webui-forge-aki`|`models`|`Qwen3-VL-4B-Instruct` | qwen3 VL模型文件目录 |
| `sd-webui-forge-aki`|`models`|` Codeformer，faceswaplab，insightface，GFPGAN`| 换脸插件模型文件目录 |
| `sd-webui-forge-aki`|`extensions`|`sd-webui-MultiModal\LatentSync`| 数字人视频生成模型目录 |

- 安装OllamaSetup.exe应用程序至C盘
- | `C:`|`ffmpeg\`| 语音与视频合成依赖文件放置c盘根目录 |
  
模型已存至网盘
<img width="1128" height="587" alt="QQ20251023-180519" src="https://github.com/user-attachments/assets/9edfb153-9351-4da5-a30b-685ad9c8891e" />


### 1. 资源汇总

- 集中展示重要公告和资源信息
- 提供快速访问各类功能的入口
- 显示插件使用说明和更新日志<img width="1245" height="650" alt="1" src="https://github.com/user-attachments/assets/f9b99645-a76a-43ce-aa27-1d5774e9cfa3" />

### 2. 图像识别与语言交互
- 支持多种视觉模型（Qwen3VL、LLaMA-Vision等）
- 支持多种语言模型（Qwen、DeepSeek等）
- 提供快捷提示词模板
- 支持单张和批量图像处理
- 根据显存大小推荐合适的模型（8GB显存推荐1.7B/3B模型，16GB显存可选latest/7B模型），参数越大响应速度越慢质量越高
- 安装ollama应用程序 https://ollama.com/search
- 安装(qwen2.5vl)视觉模型与(qwen3)语言模型，在计算机开始菜单搜索栏输入CMD执行以下命令

- 

ollama run qwen2.5vl:3b
ollama run qwen3:1.7b

<img width="1107" height="385" alt="123" src="https://github.com/user-attachments/assets/454cc34a-ca0a-4f4d-a816-539859c484de" />

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
-  使用语音合成或视频处理功能需将ffmpeg放置c盘根目录
- 开始菜单搜索环境变量， 添加C:\ffmpeg\bin到环境变量
<img width="1831" height="925" alt="18" src="https://github.com/user-attachments/assets/4b380e69-3814-4078-ac3e-9f228d83bcde" />
https://github.com/user-attachments/assets/587086f5-5204-4953-b37b-5c1c72a97f61

### 6. Index-TTS语音合成

- 集成 Index-TTS 实现高质量语音合成
- 支持多种语音风格
- 可调节语速、音调等参数
- 支持中文和多语言合成
- 使用语音合成或视频处理功能需将ffmpeg放置c盘根目录
- 开始菜单搜索环境变量， 添加C:\ffmpeg\bin到环境变量
- <img width="722" height="479" alt="QQ20251018-013019" src="https://github.com/user-attachments/assets/4fe32403-16bd-47c2-9639-59390b7cd741" />
- <img width="1693" height="734" alt="QQ20251011-134442" src="https://github.com/user-attachments/assets/651fa968-f16d-4084-b6af-db12ac26632d" />
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

 ### 8. qwen-image图像生成
 - <img width="1825" height="765" alt="88" src="https://github.com/user-attachments/assets/03327093-bb00-4a5f-ad11-a3ed31aaa90b" />

qwen-image文生图加速主模型详情页介绍
https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image/summary

qwen-image-edit编辑加速主模型详情页介绍
https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/summary

模型分为适用于非 Blackwell GPU（50 系列之前的用户）适用于 Blackwell GPU（50 系列）的用户。

我在网盘当中下载的模型是适合50系列模型，如果你是非50系显卡，需要自行下载主模型，其余模型组件不必重新下载
50系显卡除外的用户下载我截图当中的模型
<img width="1256" height="898" alt="QQ20251023-190930" src="https://github.com/user-attachments/assets/a430135c-dc93-4515-b69a-34fa0e4d751f" />
<img width="1226" height="836" alt="QQ20251023-190809" src="https://github.com/user-attachments/assets/6db3520d-266e-4c75-9dbf-2cd972e572f4" />

模型目录内的qwenimage与qwen-image-edit是主模型

<img width="762" height="495" alt="24542525" src="https://github.com/user-attachments/assets/f8e58477-3e33-478c-ac0f-495da4adea4e" />
<img width="1474" height="960" alt="图层 2" src="https://github.com/user-attachments/assets/e6dcf697-2d5e-4612-80fd-732bf7afb4f9" />


### 模型版本

 lightning是加速 8steps 是8步就能生成高质量图像了，当然步数越大时间越长，质量也高，没有此类标识的模型需要增加推理步数和引导数，
 
 svdq-fp4_r128-qwen-image-lightningv1.1-8steps 使用时就是 推理步数可从低到高 8或20，引导数是 1，

 svdq-fp4_r128-qwen-image.safetensors  使用时就是 推理步数至少15往上，引导数是 4 
 
生成信息（如配置参数、生成时间等）也会一并记录
 
 qwen模型演示教程
 https://www.bilibili.com/video/BV1zn4TzKEdW/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37

   qwen-image基本文字生成，中文理解，参数大的特点，qwen-image-edit plus具备编辑图像，实现多种编辑效果的模型
   之前一直部署不上webui是因为没有好的优化方法和策略，最近参考了comfyui的nunchaku优化方法，生成时间与配置压力大幅度减少
    为大家带来更加便利的的使用方式，生成成功时会记录配置与参数设置信息
    ControINet在默认状态下是启用状态，只使用文生图需要关闭启用，这是一个bug
    
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
     
   ### 8. qwen-image ControlNet
   
   qwen 使用方式与XL ControlNet并无差别，得益于qwen模型的优化能力生成效果与质量要远比XL好的多
   
   点击爆炸图标可预览预处理器结果，权重0.7-1之间，与处理器与模型都在网盘中 Qwen-Image-ControlNet-Union

   这是一个综合ControlNet模型，同时具备深度，姿势，线稿，软边缘
    
  ControINet在默认状态下是启用状态，只使用文生图需要关闭启用
  
  在ControINet中上传图像不能上传超过1500像素的图像，超过后会爆显存，使用qq或微信截图，clit+v粘贴到上传图像的位置就行这样就不必在ps中处理尺寸的问题了
     
   <img width="877" height="552" alt="1241214" src="https://github.com/user-attachments/assets/4807196b-3641-46de-b3c3-25d641e9373c" />
   
  <img width="1776" height="941" alt="2344235" src="https://github.com/user-attachments/assets/45ef3c01-689c-44d4-b543-512fbbdf3c08" />

  <img width="1805" height="918" alt="23325" src="https://github.com/user-attachments/assets/2c6de0b0-7b72-4aba-aba7-2ff90368176e" />
 
   在这些预处理器中只有属于pose，深度，线稿，以及属于他们的变体qwen ControINet才支持，其余不支持，这是qwen官方训练ControINet决定的
   
   <img width="871" height="515" alt="2545676" src="https://github.com/user-attachments/assets/2a2bf747-2035-4723-83e1-4bb18f7e42f0" />

### 支持的预处理器类别

#### 1. 深度类 (Depth)
用于从图像中提取深度信息的预处理器：
- depth_midas
- depth_leres
- depth_leres++
- depth_anything
- depth_anything_v2
- depth_hand_refiner
- depth_marigold
- depth_zoe

#### 2. 姿态类 (Pose)
用于检测和提取人体姿态关键点的预处理器：
- openpose_full
- openpose
- openpose_face
- openpose_faceonly
- openpose_hand
- dw_openpose_full
- animal_openpose
- densepose (pruple bg & purple torso)
- densepose_parula (black bg & blue torso)

#### 3. 线稿类 (Lineart)
用于提取或生成线条画的预处理器：
- lineart_standard (from white bg & black line)
- lineart
- lineart_coarse
- lineart_anime
- lineart_anime_denoise
- invert (from white bg & black line)
- scribble_pidinet
- scribble_xdog
- softedge_pidinet
- softedge_pidinet_safe
- softedge_pidinstruct
- softedge_hed
- softedge_hedsafe
- mlsd

#### 4. 其他变体
包括以上三类的变体和增强版本，例如：
- 各种组合形式（如同时包含姿态和手部检测）
- 不同算法实现的相同功能预处理器
- 带有特定后处理效果的变体

### 不支持的预处理器类别

除了上述三类（深度、姿态、线稿）及其变体外的其他预处理器均不支持，包括但不限于：
- 颜色类预处理器
- 语义分割类预处理器
- 法线贴图类预处理器
- 边缘检测类预处理器（除线稿类外）
- 风格迁移类预处理器
- 其他特殊用途预处理器

 重启 WebUI

## 使用须知

使用此插件者请合法使用AI，不得发表不正当言论，作假新闻，二次销售，之后的一切行为与插件开发者无关。
