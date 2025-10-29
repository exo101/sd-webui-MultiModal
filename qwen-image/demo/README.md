---
license: apache-2.0
language:
- en
library_name: diffusers
pipeline_tag: image-to-image
tags:
- Image-to-Image
- ControlNet
- Diffusers
- QwenImageControlNetPipeline
- Qwen-Image
base_model: Qwen/Qwen-Image
---

# Qwen-Image-ControlNet-Union
This repository provides a unified ControlNet that supports 4 common control types (canny, soft edge, depth, pose) for [Qwen-Image](https://github.com/QwenLM/Qwen-Image).


# Model Cards
- This ControlNet consists of 5 double blocks copied from the pretrained transformer layers.
- We train the model from scratch for 50K steps using a dataset of 10M high-quality general and human images.
- We train at 1328x1328 resolution in BFloat16, batch size=64, learning rate=4e-5. We set the text drop ratio to 0.10.
- This model supports multiple control modes, including canny, soft edge, depth, pose. You can use it just as a normal ControlNet.

# Showcases
<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img src="./conds/canny1.png"   alt="canny"></td>
    <td><img src="./outputs/canny1.png" alt="canny"></td>
  </tr>
  <tr>
    <td><img src="./conds/soft_edge.png"    alt="soft_edge"></td>
    <td><img src="./outputs/soft_edge.png"   alt="soft_edge"></td>
  </tr>
  <tr>
    <td><img src="./conds/depth.png"    alt="depth"></td>
    <td><img src="./outputs/depth.png"   alt="depth"></td>
  </tr>
  <tr>
    <td><img src="./conds/pose.png"    alt="pose"></td>
    <td><img src="./outputs/pose.png"   alt="pose"></td>
  </tr>
</table>

# Inference
```python
import torch
from diffusers.utils import load_image

# https://github.com/huggingface/diffusers/pull/12215
# pip install git+https://github.com/huggingface/diffusers
from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel

base_model = "Qwen/Qwen-Image"
controlnet_model = "InstantX/Qwen-Image-ControlNet-Union"

controlnet = QwenImageControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)

pipe = QwenImageControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# canny
# it is highly suggested to add 'TEXT' into prompt if there are text elements
control_image = load_image("conds/canny.png")
prompt = "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation."
controlnet_conditioning_scale = 1.0

# soft edge
# control_image = load_image("conds/soft_edge.png")
# prompt = "Photograph of a young man with light brown hair jumping mid-air off a large, reddish-brown rock. He's wearing a navy blue sweater, light blue shirt, gray pants, and brown shoes. His arms are outstretched, and he has a slight smile on his face. The background features a cloudy sky and a distant, leafless tree line. The grass around the rock is patchy."
# controlnet_conditioning_scale = 1.0

# depth
# control_image = load_image("conds/depth.png")
# prompt = "A swanky, minimalist living room with a huge floor-to-ceiling window letting in loads of natural light. A beige couch with white cushions sits on a wooden floor, with a matching coffee table in front. The walls are a soft, warm beige, decorated with two framed botanical prints. A potted plant chills in the corner near the window. Sunlight pours through the leaves outside, casting cool shadows on the floor."
# controlnet_conditioning_scale = 1.0

# pose
# control_image = load_image("conds/pose.png")
# prompt = "Photograph of a young man with light brown hair and a beard, wearing a beige flat cap, black leather jacket, gray shirt, brown pants, and white sneakers. He's sitting on a concrete ledge in front of a large circular window, with a cityscape reflected in the glass. The wall is cream-colored, and the sky is clear blue. His shadow is cast on the wall."
# controlnet_conditioning_scale = 1.0

image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    control_image=control_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=control_image.size[0],
    height=control_image.size[1],
    num_inference_steps=30,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
image.save(f"qwenimage_cn_union_result.png")
```

# Inference Setting
You can adjust control strength via controlnet_conditioning_scale.
- Canny: use cv2.Canny, set controlnet_conditioning_scale in [0.8, 1.0]
- Soft Edge: use [AnylineDetector](https://github.com/huggingface/controlnet_aux), set controlnet_conditioning_scale in [0.8, 1.0]
- Depth: use [depth-anything](https://github.com/DepthAnything/Depth-Anything-V2), set controlnet_conditioning_scale in [0.8, 1.0]
- Pose: use [DWPose](https://github.com/IDEA-Research/DWPose/tree/onnx), set controlnet_conditioning_scale in [0.8, 1.0]

We strongly recommend using detailed prompts, especially when include text elements. For example, use "a poster with text 'InstantX Team' on the top" instead of "a poster".

For multiple conditions inference, please refer to [PR](https://github.com/huggingface/diffusers/pull/12215).

# ComfyUI Support
[ComfyUI](https://www.comfy.org/) offers native support for Qwen-Image-ControlNet-Union. Check the [blog](https://blog.comfy.org/p/day-1-support-of-qwen-image-instantx) for more details.

# Community Support
[Liblib AI](https://www.liblib.art/) offers native support for Qwen-Image-ControlNet-Union. [Visit](https://www.liblib.art/sd) for online inference.

# Limitations
We find that the model was unable to preserve some details without explicit 'TEXT' in prompt, such as small font text.

# Acknowledgements
This model is developed by InstantX Team. All copyright reserved.
