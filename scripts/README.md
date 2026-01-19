---
license: apache-2.0
base_model: Qwen/Qwen-Image-Edit-2511
tags:
  - qwen
  - qwen-image-edit
  - qwen-image-edit-2511
  - lora
  - multi-angle
  - camera-angles
  - camera-control
  - image-editing
  - image-to-image
  - gaussian-splatting
  - diffusers
  - fal
language:
  - en
pipeline_tag: image-to-image
library_name: diffusers
---

# Qwen-Image-Edit-2511-Multiple-Angles-LoRA

> **Multi-angle camera control LoRA for Qwen-Image-Edit-2511**
>
> 96 camera positions • Trained on 3000+ Gaussian Splatting renders • Built with [fal.ai](https://fal.ai)

---

## Results

![Camera Animation Results](all_animations_combined.gif)

---

## Highlights

| Feature | Details |
|---------|---------|
| **96 Camera Poses** | 4 elevations × 8 azimuths × 3 distances |
| **3000+ Training Pairs** | Massive dataset for maximum precision |
| **Gaussian Splatting Data** | High-quality 3D-consistent renders |
| **Low-Angle Support** | Proper low-angle (-30°) camera control |
| **Extensively Tested** | More iterations and quality checks |

---

## Camera System Diagrams

![All 96 Poses](poses_96_animated.gif)

![Distance Comparison](poses_96_distance_comparison.png)

---

## Why This LoRA?

**This is the first multi-angle camera control LoRA for [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).**

While Qwen-Image-Edit-2511 has built-in viewpoint capabilities, this LoRA provides:

- **96 precise camera poses** - Exact control over camera position
- **3000+ training pairs** - Massive dataset for maximum accuracy
- **Gaussian Splatting data** - 3D-consistent training for better spatial understanding
- **Low-angle excellence** - Proper support for ground-level and low camera positions (-30°)

---

## Prompt Format

```
<sks> [azimuth] [elevation] [distance]
```

### Quick Examples

```
<sks> front view eye-level shot medium shot
<sks> right side view high-angle shot close-up
<sks> back view low-angle shot wide shot
<sks> front-left quarter view elevated shot medium shot
```

---

## 96 Camera Positions

**4 Elevations × 8 Azimuths × 3 Distances = 96 Poses**

### Azimuths (Horizontal Rotation)

```
                         0° 
                    (front view)
                         │
         315°            │            45°
    (front-left)         │       (front-right)
              ╲          │          ╱
               ╲         │         ╱
                ╲        │        ╱
   270° ─────────────── ● ─────────────── 90°
   (left side)        OBJECT         (right side)
                ╱        │        ╲
               ╱         │         ╲
              ╱          │          ╲
         225°            │            135°
     (back-left)         │       (back-right)
                         │
                        180°
                    (back view)
```

| Angle | Descriptor |
|-------|------------|
| 0° | `front view` |
| 45° | `front-right quarter view` |
| 90° | `right side view` |
| 135° | `back-right quarter view` |
| 180° | `back view` |
| 225° | `back-left quarter view` |
| 270° | `left side view` |
| 315° | `front-left quarter view` |

### Elevations (Vertical Angle)

| Angle | Descriptor | Description |
|-------|------------|-------------|
| -30° | `low-angle shot` | Camera below, looking up |
| 0° | `eye-level shot` | Camera at object level |
| 30° | `elevated shot` | Camera slightly above |
| 60° | `high-angle shot` | Camera high, looking down |

### Distances

| Factor | Descriptor | Usage |
|--------|------------|-------|
| ×0.6 | `close-up` | Details, textures |
| ×1.0 | `medium shot` | Balanced, standard |
| ×1.8 | `wide shot` | Context, environment |

---

## All 96 Prompts Reference

### CLOSE-UP (32 prompts)

<details>
<summary>Click to expand</summary>

**Low-angle (-30°)**
```
<sks> front view low-angle shot close-up
<sks> front-right quarter view low-angle shot close-up
<sks> right side view low-angle shot close-up
<sks> back-right quarter view low-angle shot close-up
<sks> back view low-angle shot close-up
<sks> back-left quarter view low-angle shot close-up
<sks> left side view low-angle shot close-up
<sks> front-left quarter view low-angle shot close-up
```

**Eye-level (0°)**
```
<sks> front view eye-level shot close-up
<sks> front-right quarter view eye-level shot close-up
<sks> right side view eye-level shot close-up
<sks> back-right quarter view eye-level shot close-up
<sks> back view eye-level shot close-up
<sks> back-left quarter view eye-level shot close-up
<sks> left side view eye-level shot close-up
<sks> front-left quarter view eye-level shot close-up
```

**Elevated (30°)**
```
<sks> front view elevated shot close-up
<sks> front-right quarter view elevated shot close-up
<sks> right side view elevated shot close-up
<sks> back-right quarter view elevated shot close-up
<sks> back view elevated shot close-up
<sks> back-left quarter view elevated shot close-up
<sks> left side view elevated shot close-up
<sks> front-left quarter view elevated shot close-up
```

**High-angle (60°)**
```
<sks> front view high-angle shot close-up
<sks> front-right quarter view high-angle shot close-up
<sks> right side view high-angle shot close-up
<sks> back-right quarter view high-angle shot close-up
<sks> back view high-angle shot close-up
<sks> back-left quarter view high-angle shot close-up
<sks> left side view high-angle shot close-up
<sks> front-left quarter view high-angle shot close-up
```

</details>

### MEDIUM SHOT (32 prompts)

<details>
<summary>Click to expand</summary>

**Low-angle (-30°)**
```
<sks> front view low-angle shot medium shot
<sks> front-right quarter view low-angle shot medium shot
<sks> right side view low-angle shot medium shot
<sks> back-right quarter view low-angle shot medium shot
<sks> back view low-angle shot medium shot
<sks> back-left quarter view low-angle shot medium shot
<sks> left side view low-angle shot medium shot
<sks> front-left quarter view low-angle shot medium shot
```

**Eye-level (0°)** — Reference pose: `front view eye-level shot medium shot`
```
<sks> front view eye-level shot medium shot
<sks> front-right quarter view eye-level shot medium shot
<sks> right side view eye-level shot medium shot
<sks> back-right quarter view eye-level shot medium shot
<sks> back view eye-level shot medium shot
<sks> back-left quarter view eye-level shot medium shot
<sks> left side view eye-level shot medium shot
<sks> front-left quarter view eye-level shot medium shot
```

**Elevated (30°)**
```
<sks> front view elevated shot medium shot
<sks> front-right quarter view elevated shot medium shot
<sks> right side view elevated shot medium shot
<sks> back-right quarter view elevated shot medium shot
<sks> back view elevated shot medium shot
<sks> back-left quarter view elevated shot medium shot
<sks> left side view elevated shot medium shot
<sks> front-left quarter view elevated shot medium shot
```

**High-angle (60°)**
```
<sks> front view high-angle shot medium shot
<sks> front-right quarter view high-angle shot medium shot
<sks> right side view high-angle shot medium shot
<sks> back-right quarter view high-angle shot medium shot
<sks> back view high-angle shot medium shot
<sks> back-left quarter view high-angle shot medium shot
<sks> left side view high-angle shot medium shot
<sks> front-left quarter view high-angle shot medium shot
```

</details>

### WIDE SHOT (32 prompts)

<details>
<summary>Click to expand</summary>

**Low-angle (-30°)**
```
<sks> front view low-angle shot wide shot
<sks> front-right quarter view low-angle shot wide shot
<sks> right side view low-angle shot wide shot
<sks> back-right quarter view low-angle shot wide shot
<sks> back view low-angle shot wide shot
<sks> back-left quarter view low-angle shot wide shot
<sks> left side view low-angle shot wide shot
<sks> front-left quarter view low-angle shot wide shot
```

**Eye-level (0°)**
```
<sks> front view eye-level shot wide shot
<sks> front-right quarter view eye-level shot wide shot
<sks> right side view eye-level shot wide shot
<sks> back-right quarter view eye-level shot wide shot
<sks> back view eye-level shot wide shot
<sks> back-left quarter view eye-level shot wide shot
<sks> left side view eye-level shot wide shot
<sks> front-left quarter view eye-level shot wide shot
```

**Elevated (30°)**
```
<sks> front view elevated shot wide shot
<sks> front-right quarter view elevated shot wide shot
<sks> right side view elevated shot wide shot
<sks> back-right quarter view elevated shot wide shot
<sks> back view elevated shot wide shot
<sks> back-left quarter view elevated shot wide shot
<sks> left side view elevated shot wide shot
<sks> front-left quarter view elevated shot wide shot
```

**High-angle (60°)**
```
<sks> front view high-angle shot wide shot
<sks> front-right quarter view high-angle shot wide shot
<sks> right side view high-angle shot wide shot
<sks> back-right quarter view high-angle shot wide shot
<sks> back view high-angle shot wide shot
<sks> back-left quarter view high-angle shot wide shot
<sks> left side view high-angle shot wide shot
<sks> front-left quarter view high-angle shot wide shot
```

</details>

---

## Files

| File | Description |
|------|-------------|
| `qwen-image-edit-2511-multiple-angles-lora.safetensors` | LoRA weights |
| `comfyui-workflow-multiple-angles.json` | ComfyUI workflow |

▶️ [**Try it live on fal.ai**](https://fal.ai/models/fal-ai/qwen-image-edit-2511-multiple-angles)

---

## Recommended Settings

- **LoRA Strength**: 0.8 - 1.0
- **Base Model**: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

---

## Training Details

| Parameter | Value |
|-----------|-------|
| **Training Platform** | [fal.ai Qwen Image Edit 2511 Trainer](https://fal.ai/models/fal-ai/qwen-image-edit-2511-trainer) |
| **Base Model** | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| **Training Data** | 3000+ Gaussian Splatting renders |
| **Camera Poses** | 96 unique positions (4×8×3) |
| **Data Source** | Synthetic 3D renders with precise camera control |
| **Dataset & Training** | Built by Lovis Odin at fal |

---

## Tips for Best Results

1. **Use the exact prompt format** - `<sks>` trigger is essential
2. **Respect the order** - `[azimuth] [elevation] [distance]`
3. **Start with LoRA strength 0.9** - Adjust based on results
4. **Try low-angle shots** - This LoRA excels at low camera positions (-30°)
5. **Input image matters** - Clear subjects with good lighting work best

---

## Related Work

- [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) - Base model
- [Flux-2-Multi-Angles-LoRA-v2](https://huggingface.co/lovis93/Flux-2-Multi-Angles-LoRA-v2) - Multi-angle for Flux (72 poses)
- [next-scene-qwen-image-lora](https://huggingface.co/lovis93/next-scene-qwen-image-lora)

---

## Author

Built by **Lovis Odin** ([@lovis93](https://huggingface.co/lovis93) • [@odinlovis](https://x.com/odinlovis)) at [fal](https://fal.ai)

Trained using [fal Qwen Image Edit 2511 Trainer](https://fal.ai/models/fal-ai/qwen-image-edit-2511-trainer)

---

**If you find this useful, please star the repo!**

**Issues?** Open a discussion in the Community tab
