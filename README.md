# Stable Diffusion WebUI Forge 多模态集成插件

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
- Index-TTS: 10GB以上
- FLUX.1-Kontext: 12GB
- FLUX:10GB
- XL: 8GB
- Cleaner: 4GB以上
- Segment Anything: 8GB

个人主页：[https://space.bilibili.com/403361177?spm_id_from=333.788.upinfo.detail.click ](https://space.bilibili.com/403361177?spm_id_from=333.40164.0.0) 

WebUI Forge安装使用介绍：
[https://www.bilibili.com/video/BV1BCHXzJE1C?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37  ](https://www.bilibili.com/video/BV1FWtBzbEiR?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37)

多模态插件安装使用介绍：
[https://www.bilibili.com/video/BV1DSW4zTEGR?spm_id_from=333.788.player.switch&vd_source=343e49b703fb5b4137cd6c1987846f37&p=2](https://www.bilibili.com/video/BV1DSW4zTEGR?spm_id_from=333.788.player.switch&vd_source=343e49b703fb5b4137cd6c1987846f37&p=2)

WebUI Forge整合包与插件模型下载链接可在视频简介下方查看

### 前置要求

  - 旧整合包已不适用日益更新的AI应用与50系显卡，我为此更新了新整合包环境，补充落后的webui生态
  - https://github.com/exo101/sd-webui-forge-aki-v4.5
  - 下载插件文件到sd-webui-forge-aki-v4.5/extensions目录
    
### 更新内容

2025/10/29
  
- 添加nunchaku qwen lora 功能支持，增加随机种子，生成批次
- 修复ControlNet启用框bug，完善qwen-image ControlNet实现多个变体预处理器，排除不支持的预处理器防止误触
- 修复qwen3VL图像识别返回文本首字符缺失的问题bug
  
2025/10/24
  
- webui frogr 整合包更新之4.4版本，支持了qwen3VL，更新了transformers==4.57.0
- transformers==4.57.0会牺牲Index-TTS2语音合成功能使用，transformers==4.52.1版本会牺牲qwen3VL
- 如果使用Index-TTS2，更改整合包根目录配置requirements_versions文件，降级到transformers==4.52.1版本
- 如果使用qwen3VL，请保持整合包根目录配置requirements_versions文件，transformers==4.57.0
- qwen3VL与Index-TTS2环境库transformers版本两者存在冲突

2025/10/18
 
- 添加qwen-image ControlNet模块，同时实现了深度，姿势，线稿，软边缘
  
2025/10/12
  
- 更新多模态SD插件12版本：增加第八个功能标签页nunchaku qwen-image与 qwen-image-edit 2509

 ## 插件目录
 
<img width="685" height="477" alt="QQ20251030-223846" src="https://github.com/user-attachments/assets/a1ded7d7-3311-4d15-98b7-34dc3dbcd108" />


## 插件模型目录
| 整合包目录 | 模型目录 | 子目录 |说明 |
|---------------------|--------|-----------|-----------|
| `sd-webui-forge-aki`|`models`|`sam`| 图像分割模型目录 |
| `sd-webui-forge-aki`|`models`|`cleaner`| 图像清理模型目录 |
| `sd-webui-forge-aki`|`models`|`index-tts2`| Index-TTS语音合成模型目录 |
| `sd-webui-forge-aki`|`models`|`FLUX.1-Kontext-dev`| 图像编辑模型目录 |
| `sd-webui-forge-aki`|`models`|`lora`|qwen LoRA微调模型目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image`| qwen模型文件目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image\qwenimage`| qwen文生图主模型目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image\qwen-image-edit`| qwen图像编辑主模型目录 |
| `sd-webui-forge-aki`|`models`|`ControlNet\ Qwen-Image-ControlNet-Union`| qwen ControlNet模型目录 |
| `sd-webui-forge-aki`|`models`|`Qwen3-VL-4B-Instruct`|qwen3 VL模型文件目录|
| `sd-webui-forge-aki`|`models`|` Codeformer`| 换脸插件模型文件目录 |
| `sd-webui-forge-aki`|`models`|`faceswaplab`|换脸插件模型文件目录|
| `sd-webui-forge-aki`|`models`|`insightface`|换脸插件模型文件目录|
| `sd-webui-forge-aki`|`models`|`GFPGAN`|换脸插件模型文件目录 |
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
- 示例教程：[https://www.bilibili.com/video/BV1xkMHzkE6n?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37](https://www.bilibili.com/video/BV1NSTKzfEq7?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37)
- 支持多种视觉模型（Qwen3VL、LLaMA-Vision等）
- 支持多种语言模型（Qwen、DeepSeek等）
- 提供快捷提示词模板
- 支持单张和批量图像处理
- 根据显存大小推荐合适的模型（8GB显存推荐1.7B/3B模型，16GB显存可选latest/7B模型），参数越大响应速度越慢质量越高
- 安装ollama应用程序 https://ollama.com/search
- 安装(qwen2.5vl)视觉模型与(qwen3)语言模型，在计算机开始菜单搜索栏输入CMD执行以下命令

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
- 基于 rembg 实现高质量背景移除
- 支持透明背景和自定义背景色，批量处理功能
- <img width="1782" height="896" alt="7" src="https://github.com/user-attachments/assets/63e9293d-09b2-494a-8ea9-8eaa46aef287" />
- <img width="1798" height="679" alt="8" src="https://github.com/user-attachments/assets/9450ffa8-f8ac-4ca0-bccd-7e709f873369" />

#### 图像分割

- 示例教程：https://www.bilibili.com/video/BV143YtzsE1j?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 精确的图像分割功能，支持点选和自动分割方式
<img width="1812" height="917" alt="10" src="https://github.com/user-attachments/assets/c0ca0f59-be6e-408c-bb4c-a117d718e588" />
<img width="1816" height="909" alt="9" src="https://github.com/user-attachments/assets/08b633ba-7ed8-4886-a6f3-e22a3bd7cb8e" />

#### 图像清理
- 示例教程：https://www.bilibili.com/video/BV1YRehz1EBz?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 图像去噪和修复功能，支持多种清理模式
- 简单易用的界面<img width="1835" height="741" alt="11" src="https://github.com/user-attachments/assets/4ac7c40d-971c-4364-8cc1-73a872fcec79" />

### 4. 视频关键帧提取
- 示例教程：https://www.bilibili.com/video/BV1nFarzjExK?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 多种提取模式（关键帧/等间隔/场景变化）
- 可调节提取质量，支持多种视频格式
- 可预览提取的帧<img width="1809" height="677" alt="12" src="https://github.com/user-attachments/assets/23b3cb3d-c763-4432-894f-fdc84b8c7b9f" />

### 5. 数字人视频生成 
- 示例教程：https://www.bilibili.com/video/BV1Vr8XzcE2a?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 基于 LatentSync 的音频驱动视频生成， 需要清晰人脸的视频作为输入，唇形同步
 <img width="1831" height="925" alt="18" src="https://github.com/user-attachments/assets/4b380e69-3814-4078-ac3e-9f228d83bcde" />
 https://github.com/user-attachments/assets/587086f5-5204-4953-b37b-5c1c72a97f61

### 6. Index-TTS语音合成

- 示例教程：https://www.bilibili.com/video/BV1ngpHzvETn?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 集成 Index-TTS 实现高质量语音合成
- 支持中文和多语言合成，可调节语速、音调等参数
- 开始菜单搜索环境变量， 添加C:\ffmpeg\bin到环境变量
- <img width="722" height="479" alt="QQ20251018-013019" src="https://github.com/user-attachments/assets/4fe32403-16bd-47c2-9639-59390b7cd741" />
- <img width="1693" height="734" alt="QQ20251011-134442" src="https://github.com/user-attachments/assets/651fa968-f16d-4084-b6af-db12ac26632d" />
- <img width="1786" height="805" alt="23" src="https://github.com/user-attachments/assets/1318c3fa-c979-4c93-8003-639e5f43f7f6" />
- <img width="1788" height="428" alt="17" src="https://github.com/user-attachments/assets/52ed7801-36f3-4145-9386-f2ae7285ea11" />
   [output_1760002640.wav](https://github.com/user-attachments/files/22794279/output_1760002640.wav)


### 7. FLUX.1-Kontext图像编辑
- 示例教程：https://www.bilibili.com/video/BV1BeaGz8EEC?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 上下文感知的图像编辑功能，基于文本的图像修改
- GGUF量化模型优化使用门槛12g显存可用
- <img width="1813" height="571" alt="21" src="https://github.com/user-attachments/assets/4df0079b-ff8d-4290-ae16-7e367eb90881" />
- <img width="1024" height="1024" alt="dual_context_image_1756582213_var1" src="https://github.com/user-attachments/assets/1bf91812-70a9-4662-aed1-ac6839a274ab" />

 ### 8. qwen-image图像生成介绍
 
 - qwen模型演示教程https://www.bilibili.com/video/BV1zn4TzKEdW/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37
 - qwen-image基本文字生成，中文理解，参数大的特点
 - qwen-image-edit plus具备编辑图像，实现多种编辑效果的模型
 - 参考了的nunchaku优化方法，生成时间与配置压力大幅度减少
 - 在高配置的电脑上体现不出明显的时间差距，迭代步数越高时间越长，质量越高，最高不超过40
 - 模型分为适用于非 Blackwell GPU（50 系列之前的用户）适用于 Blackwell GPU（50 系列）的用户。
 - qwen-image文生图加速主模型详情页介绍
 - https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image/summary
 - qwen-image-edit编辑加速主模型详情页介绍
 - https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/summary
   <img width="1825" height="765" alt="88" src="https://github.com/user-attachments/assets/03327093-bb00-4a5f-ad11-a3ed31aaa90b" />

### 模型版本

 - Lightning模型是专门为快速推理设计的，训练时使用了特定的CFG设置。，Lightning模型设置为1，普通模型是完整训练的模型，对CFG参数更宽容，可以使用较高的CFG值为4，
 - svdq-fp4_r128-qwen-image-lightningv1.1-8steps 使用时就是 推理步数10，引导数是 1
 - svdq-fp4_r128-qwen-image.safetensors  使用时就是 推理步数至少15往上，引导数是 4
 - 由于nunchaku qwen模型是一种量化的优化策略模型，完整版模型有20B参数，qwen lora 模型权重需要调整为1.5才能生效
 - 我在网盘当中下载的模型是适合50系列模型，如果你是非50系显卡，需要自行下载主模型，其余模型组件不必重新下载，50系显卡除外的用户下载我截图当中的模型
 - <img width="1256" height="898" alt="QQ20251023-190930" src="https://github.com/user-attachments/assets/a430135c-dc93-4515-b69a-34fa0e4d751f" /> 
 - <img width="1226" height="836" alt="QQ20251023-190809" src="https://github.com/user-attachments/assets/6db3520d-266e-4c75-9dbf-2cd972e572f4" />

 - 模型目录内的qwenimage与qwen-image-edit是主模型
 - 编辑模型最多支持上传三张图像，但多图编辑能力弱于单图编辑能力
 - <img width="762" height="495" alt="24542525" src="https://github.com/user-attachments/assets/f8e58477-3e33-478c-ac0f-495da4adea4e" />
 - <img width="1474" height="960" alt="图层 2" src="https://github.com/user-attachments/assets/e6dcf697-2d5e-4612-80fd-732bf7afb4f9" />
 - qwen-image为例 <img width="861" height="435" alt="122" src="https://github.com/user-attachments/assets/650e86f6-a822-424d-ae60-9fed1f1426aa" /> 
 - <img width="1815" height="854" alt="333" src="https://github.com/user-attachments/assets/37e5f859-263d-478d-ab63-b9d41a682217" />
 - <img width="866" height="375" alt="111" src="https://github.com/user-attachments/assets/f0601d64-fec4-4efd-b841-e44b3277e246" />
 - <img width="859" height="359" alt="222" src="https://github.com/user-attachments/assets/b6935a43-1868-4b0b-b8a5-cd0cd3bf4ff2" /> 
   
### 8. qwen-image ControlNet 模型

  - 示例教程：https://www.bilibili.com/video/BV13PsHz4E2C/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37
  - qwen 使用方式与XL ControlNet并无差别，得益于qwen模型的优化能力生成效果与质量要远比XL好的多 
  - 点击爆炸图标可预览预处理器结果，权重0.7-1之间，预处理器与模型都在网盘中 Qwen-Image-ControlNet-Union
  - 这是一个综合ControlNet模型，同时具备深度，姿势，线稿，软边缘
  - 在ControINet中上传图像不能上传超过1500像素的图像，超过后会爆显存，使用qq或微信截图，clit+v粘贴到上传图像的位置就行这样就不必在ps中处理尺寸的问题了
  - <img width="1805" height="918" alt="23325" src="https://github.com/user-attachments/assets/2c6de0b0-7b72-4aba-aba7-2ff90368176e" />
  - 在这些预处理器中只有属于pose，深度，线稿，以及属于他们的变体qwen ControINet才支持，其余不支持，这是qwen官方训练ControINet决定的
  - <img width="871" height="515" alt="2545676" src="https://github.com/user-attachments/assets/2a2bf747-2035-4723-83e1-4bb18f7e42f0" />
  
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
- lineart_realistic
- lineart_coarse
- lineart_anime
- lineart_anime_denoise
- invert (from white bg & black line)
- 
 #### 4. 软边缘类 (softedge)
- softedge_pidinet
- softedge_pidinet_safe
- softedge_pidinstruct
- softedge_hed
- softedge_hedsafe

### 不支持的预处理器类别

除了上述三类（深度、姿态、线稿）及其变体外的其他预处理器均不支持，包括但不限于：
- 颜色类预处理器
- 语义分割类预处理器
- 法线贴图类预处理器
- 边缘检测类预处理器（除线稿类外）
- 风格迁移类预处理器
- 其他特殊用途预处理器
- 带有特定后处理效果的变体
 重启 WebUI

## 使用须知

使用此插件者请合法使用AI，不得发表不正当言论，作假新闻，二次销售，之后的一切行为与插件开发者无关。
