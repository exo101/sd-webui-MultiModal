# Stable Diffusion WebUI Forge 多模态集成插件

## 系统要求

- NVIDIA RTX显卡可用范围
-  30系 3080，3090，3090ti （虽然能用矿卡居多，不推荐）
-  40系 4060ti，4070，4070ti，4070super，4080，4090
-  50系 5060ti，5070，5070ti，5080，5090
- 显存：至少 12GB
- 内存：推荐 DDR5 64GB 
     
## 核心功能

- 📚 **资源汇总**: 集中管理各类资源和公告信息
- 🖼️ **图像识别与语言交互**: 支持多种视觉和语言模型，可进行图像描述、内容分析等
- ✂️ **智能抠图**: 基于 rembg 实现一键背景移除
- 🖌️ **图像分割**: 集成 Segment Anything Model (SAM) 进行精确图像分割
- 🧹 **图像清理**: 提供图像清理和修复功能
- 🌟 **FLUX.2 图像编辑**: 集成 FLUX.2-klein 进行上下文感知的图像编辑
- 🌟 **Qwen-Image复杂文本渲染和qwen-image-edit-2509精确图像编辑

## 各项目配置显存要求
- FLUX.2-klein-4B 显存 10GB
- Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32： 显存 14GB
- nuchuku加速 Qwen-Image-Edit-2501: 显存10GB /内存64g
- nuchuku加速 Qwen-Image: 显存10GB /内存64g
- Qwen3vL: 显存10GB
- Z-Image-Turbo:显存12g
- nunchaku加速-FLUX.1-Kontext:显存8GB
- nunchaku加速FLUX:显存8GB
- XL: 显存8GB
- Cleaner: 显存6GB以上
- Segment Anything: 显存10GB

个人主页：[https://space.bilibili.com/403361177?spm_id_from=333.788.upinfo.detail.click ](https://space.bilibili.com/403361177?spm_id_from=333.40164.0.0) 
AI交流群qq：1054090769 整合包与模型在群公告


# 模型与插件下载

  - 旧整合包已不适用日益更新的AI应用与50系显卡，我为此更新了新整合包环境，补充落后的webui forge生态
  - 下载sd-webui-MultiModal插件文件到/extensions目录
  - 图像识别模型需要自行下载ollama应用程序，从命令行下载
  - 凡是涉及nunchaku加速模型都区分着50系显卡与非50系显卡的模型，需要从魔搭社区进行下载搜索nunchaku
  - 大部分模型都在qq群内的百度或夸克网盘分享
    
 ## 插件目录
 
<img width="685" height="477" alt="QQ20251030-223846" src="https://github.com/user-attachments/assets/a1ded7d7-3311-4d15-98b7-34dc3dbcd108" />


## 插件模型目录
| 整合包目录 | 模型目录 | 子目录 |说明 |
|---------------------|--------|-----------|-----------|
| `sd-webui-forge-aki`|`models`|`adetailer`| 修脸插件模型 |
| `sd-webui-forge-aki`|`models`|`cleaner`| 图像清理模型目录 |
| `sd-webui-forge-aki`|`models`|`RealESRGAN`| 高清放大算法目录 |
| `sd-webui-forge-aki`|`models`|`ESRGAN`| 高清放大算法目录 |
| `sd-webui-forge-aki`|`models`|`lora`| LoRA微调模型目录 |
| `sd-webui-forge-aki`|`models`|`qwen-image`| qwen模型与组件总目录 |
| `sd-webui-forge-aki`|`models`|`FLUX.2-klein`| fluX2模型目录 |
| `sd-webui-forge-aki`|`models`|`FLUX.1-Kontext-dev`| nunchuku量化fluX系列模型目录 |
| `sd-webui-forge-aki`|`models`|`ControlNet`| ControlNet控制模型目录 |
| `sd-webui-forge-aki`|`models`|`ControlNetPreprocessor`| ControlNet预处理器目录 |
| `sd-webui-forge-aki`|`models`|`sam`| 图像分割模型目录 |
| `sd-webui-forge-aki`|`models`|`Stable-diffusion`| 传统flux.XL.1.5模型目录 |
| `sd-webui-forge-aki`|`models`|`Tongyi-MAl`| Z-Image模型目录 |
| `sd-webui-forge-aki`|`models`|`vae`| 图像编解码模型 |

## ps插件
| ps目录 | 插件目录 | 子目录 |说明 |
|---------------------|--------|-----------|-----------|
| `Adobe Photoshop 2024`|`Plug-ins`|`sd-ppp_PS`| ps插件目录 |
| `Adobe Photoshop 2024`|`Plug-ins`|`Auto.Photoshop.SD.plugin_v1.4.1`| ps插件目录  |

### 更新内容
- 增加Z-Image正式版支持
- FLUX.2-klein-4b模型 增加，nunchaku-Z-Image-Turbo，lora支持
- 
### 更新内容
- 为了避免插件功能过多导致内存增加，多模态插件分裂成了两个插件，sd-webui-MultiModal只负责图像处理，ai绘画模型
- sd-webui-multimodal-media负责，处理视频，音乐，语音，多媒体，模型位置不变
- 增加 FLUX.2-klein-4b模型，此模型具备文生图，图像编辑，局部编辑，扩图，等多模态能力
- 为每个模型类界面添加了多视角可视化选择器，任务队列功能
- 使用最新模型需到python目录上方输入cmd执行命令 python -m pip install git+https://github.com/huggingface/diffusers
  
### 更新内容
- 添加Z-Image-Turbo fp8模型支持 lora支持，图生图支持
- 添加qwen，wan系列api调用模型功能（qwenmix，qwenEdit，wan2.6，wan2.5文生视频，图生视频，首尾帧等）
- 为qwen，flux，Z-Image，等一众模型界面添加多角度提示词可视化选择器插件
- 
### 更新内容
- 添加Qwen-Image-Edit-2511-ControlNet支持
- 添加Z-Image-Turbo nuchaku量化模型支持（transformer主模型量化），显存降低 速度提升
- 添加nunchuku flux ControlNet 支持 （补全缺失模块）
- 整合包增加ui交互指南，降低使用难度，科普参数
- 添加图像识别模型qwen3VL8b，qwen3VL4b，qwen3VL2b
- 添加Qwen-Image-Edit-2511-SDNQ-uint4量化模型支持
- 添加nunchaku-qwen-image，lora  ControlNet模块，同时实现了深度，姿势，线稿，软边缘
- 添加nunchaku-qwen-image-edit-2509 lora，ControlNet功能实现，深度，线稿，姿势，局部编辑
- 添加nunchaku qwen-image与 qwen-image-edit 2509

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
- 安装ollama应用程序 https://ollama.com/search
- 安装QwenVL视觉模型或Qwen3语言模型，在计算机开始菜单搜索栏输入CMD执行以下命令
- | `安装命令`|`ollama run qwen3-vl:8b`|视觉模型|
- | `安装命令`|`ollama run qwen3-vl:4b`|视觉模型|
- | `安装命令`|`ollama run qwen3-vl:2b`|视觉模型|
- | `安装命令`|`ollama run qwen2.5vl:3b`|视觉模型|
- | `安装命令`|`ollama run qwen3:1.7b`|语言模型|
- | `查看命令`|`ollama list`|模型列表|
- | `删除命令`|`ollama rm 模型名称`|删除模型|
- | `退出命令`|` ctrl+d`|退出模型使用|

  质量，速度，配置呈对应关系，参数大速度慢质量好，参数小速度快质量差
  16g选择8b或4b，12g选择4b或2b
 <img width="1105" height="390" alt="QQ20251101-185455" src="https://github.com/user-attachments/assets/d431810c-ba48-473a-99c8-a8ea90d408d0" />
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
<img width="1828" height="817" alt="43534534" src="https://github.com/user-attachments/assets/913abe19-d8b1-4229-b18d-44f7d3930508" />

#### 图像分割

- 示例教程：https://www.bilibili.com/video/BV143YtzsE1j?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 精确的图像分割功能，支持点选和自动分割方式
<img width="1812" height="917" alt="10" src="https://github.com/user-attachments/assets/c0ca0f59-be6e-408c-bb4c-a117d718e588" />
<img width="1816" height="909" alt="9" src="https://github.com/user-attachments/assets/08b633ba-7ed8-4886-a6f3-e22a3bd7cb8e" />

#### 图像清理
- 示例教程：https://www.bilibili.com/video/BV1YRehz1EBz?spm_id_from=333.788.videopod.sections&vd_source=343e49b703fb5b4137cd6c1987846f37
- 图像去噪和修复功能，支持多种清理模式
- 简单易用的界面<img width="1835" height="741" alt="11" src="https://github.com/user-attachments/assets/4ac7c40d-971c-4364-8cc1-73a872fcec79" />

FLUX.2-klein模型集成文生图，图像编辑，局部编辑，扩图多位一体的编辑模型

教程链接
https://www.bilibili.com/video/BV1gEzCBqEpA/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37

<img width="1787" height="886" alt="image" src="https://github.com/user-attachments/assets/3b999a97-ac93-41f3-80f5-b12304280ae5" />
<img width="1784" height="831" alt="image" src="https://github.com/user-attachments/assets/22a5efa9-2bfb-47b7-b4b9-d7cb7a993514" />
<img width="1749" height="831" alt="image" src="https://github.com/user-attachments/assets/29208ee2-77f6-43a3-b045-07c81eef1e56" />

 ### . qwen-image图像生成介绍
 
 - qwen模型演示教程https://www.bilibili.com/video/BV1zn4TzKEdW/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37
 - qwen-image基本文字生成，中文理解，参数大的特点
 - qwen-image-edit plus具备编辑图像，实现多种编辑效果的模型
 - 参考了的nunchaku优化方法，生成时间与配置压力大幅度减少
 - 模型分为适用于（非50系列显卡之前的用户）适用于（50系列显卡）的用户。
 - qwen-image文生图加速主模型详情页介绍
 - https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image/summary
 - qwen-image-edit编辑加速主模型详情页介绍
 - https://www.modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image-edit-2509/summary
   <img width="1825" height="765" alt="88" src="https://github.com/user-attachments/assets/03327093-bb00-4a5f-ad11-a3ed31aaa90b" />

### 模型版本

 - Lightning模型是专门为快速推理设计的，训练时使用了特定的CFG设置，Lightning模型设置为1，普通模型是完整训练的模型，对CFG参数更宽容，可以使用较高的CFG值为4
   
 - 50系模型
 - svdq-fp4_r128-qwen-image-lightningv1.1-8steps  
 - svdq-fp4_r128-qwen-image.safetensors   推理步数至少15往上，引导数是 4
   
 - 非50系模型
 - svdq-int4_r128-qwen-image-edit-2509.safetensors 推理步数至少15往上，引导数是 4
 - svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors 非50系模型 推理步数10，引导数是 1
 - 由于nunchaku qwen模型是一种量化的优化策略模型，完整版模型有20B参数，qwen lora 模型权重需要调整为1.5才能生效
 - 我在网盘当中下载的模型是适合50系列模型，如果你是非50系显卡，需要自行下载主模型，其余是模型的必备组件，50系显卡除外的用户下载我截图当中的模型
 - <img width="1256" height="898" alt="QQ20251023-190930" src="https://github.com/user-attachments/assets/a430135c-dc93-4515-b69a-34fa0e4d751f" /> 
 - <img width="1226" height="836" alt="QQ20251023-190809" src="https://github.com/user-attachments/assets/6db3520d-266e-4c75-9dbf-2cd972e572f4" />

 - 模型目录内的qwenimage与qwen-image-edit是主模型
 - 编辑模型最多支持上传三张图像，但多图编辑能力弱于单图编辑能力
 - <img width="762" height="495" alt="24542525" src="https://github.com/user-attachments/assets/f8e58477-3e33-478c-ac0f-495da4adea4e" />
 - <img width="1474" height="960" alt="图层 2" src="https://github.com/user-attachments/assets/e6dcf697-2d5e-4612-80fd-732bf7afb4f9" />
 - qwen-image为例5070ti显卡，迭代步数10，生成时间为30-50秒之间
   
   
### qwen-image ControlNet 模型

  - 示例教程：https://www.bilibili.com/video/BV13PsHz4E2C/?spm_id_from=333.1387.homepage.video_card.click&vd_source=343e49b703fb5b4137cd6c1987846f37
  - qwen 使用方式与XL ControlNet并无差别，得益于qwen模型的优化能力生成效果与质量要远比XL好的多 
  - 点击爆炸图标可预览预处理器结果，权重0.7-1之间，预处理器与模型都在网盘中 Qwen-Image-ControlNet-Union
  - 这是一个综合ControlNet模型，同时具备深度，姿势，线稿，软边缘
  - 在ControINet中上传图像不能上传超过1500像素的图像，超过后会爆显存，使用qq或微信截图，clit+v粘贴到上传图像的位置就行这样就不必在ps中处理尺寸的问题了
  - <img width="1805" height="918" alt="23325" src="https://github.com/user-attachments/assets/2c6de0b0-7b72-4aba-aba7-2ff90368176e" />
  - 在这些预处理器中只有属于pose，深度，线稿，以及属于他们的变体qwen ControINet才支持，其余不支持，这是qwen官方训练ControINet决定的
  - <img width="871" height="515" alt="2545676" src="https://github.com/user-attachments/assets/2a2bf747-2035-4723-83e1-4bb18f7e42f0" />
  
### qwen-image-edit-2509 ControlNet
  - 编辑模型自带ControlNet只需加载预处理器就可以控制图像，保持人物一致性，变化姿态，转换场景构图，编辑文字是个强大的多功能模型
  - 编辑模型可以同时使用自身的微调lora模型和qwen-image lora模型，可保持人物不变的情况下载改变风格



使用此插件者请合法使用AI
