import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.utils import load_image

# Load Kontext model with GGUF
MAX_SEED = np.iinfo(np.int32).max

# GGUF model paths - FIXED URLs without double resolve/main
GGUF_MODELS = {
    "Q2_K": "QuantStack/FLUX.1-Kontext-dev-GGUF",
    "Q4_K_S": "QuantStack/FLUX.1-Kontext-dev-GGUF", 
    "Q5_K_M": "QuantStack/FLUX.1-Kontext-dev-GGUF",
    "Q8_0": "QuantStack/FLUX.1-Kontext-dev-GGUF"
}

# Model filenames
GGUF_FILENAMES = {
    "Q2_K": "flux1-kontext-dev-Q2_K.gguf",
    "Q4_K_S": "flux1-kontext-dev-Q4_K_S.gguf",
    "Q5_K_M": "flux1-kontext-dev-Q5_K_M.gguf", 
    "Q8_0": "flux1-kontext-dev-Q8_0.gguf"
}

# Choose model based on your VRAM
SELECTED_MODEL = "Q5_K_M"

def load_gguf_pipeline():
    """Load the GGUF quantized Flux Kontext pipeline"""
    try:
        print(f"Loading GGUF model: {SELECTED_MODEL}")
        # Load the quantized transformer with proper repo and filename
        transformer = FluxTransformer2DModel.from_single_file(
            GGUF_MODELS[SELECTED_MODEL],
            filename=GGUF_FILENAMES[SELECTED_MODEL],
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        
        # Load the full pipeline with quantized transformer
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # Enable memory optimizations
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        return pipe.to("cuda")
        
    except Exception as e:
        print(f"Error loading GGUF pipeline: {e}")
        print("Falling back to regular pipeline with memory optimizations...")
        
        # Fallback to regular pipeline with heavy memory optimizations
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", 
            torch_dtype=torch.bfloat16
        )
        
        # Enable aggressive memory optimizations for 24GB GPU
        pipe.enable_sequential_cpu_offload()  # More aggressive than model_cpu_offload
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        return pipe

# Initialize pipeline
print("Initializing FLUX.1-Kontext pipeline...")
pipe = load_gguf_pipeline()
print("Pipeline loaded successfully!")

def concatenate_images(images, direction="horizontal"):
    """Concatenate multiple PIL images either horizontally or vertically."""
    if not images:
        return None
    
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        return None
    
    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")
    
    valid_images = [img.convert("RGB") for img in valid_images]
    
    if direction == "horizontal":
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        x_offset = 0
        for img in valid_images:
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
    else:  # vertical
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in valid_images:
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated

@spaces.GPU
def infer(input_images, prompt, seed=42, randomize_seed=False, guidance_scale=2.5, progress=gr.Progress(track_tqdm=True)):
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    if input_images is None:
        raise gr.Error("Please upload at least one image.")
    
    if not isinstance(input_images, list):
        input_images = [input_images]
    
    valid_images = [img[0] for img in input_images if img is not None]
    
    if not valid_images:
        raise gr.Error("Please upload at least one valid image.")
    
    concatenated_image = concatenate_images(valid_images, "horizontal")
    
    if concatenated_image is None:
        raise gr.Error("Failed to process the input images.")
    
    final_prompt = f"From the provided reference images, create a unified, cohesive image such that {prompt}. Maintain the identity and characteristics of each subject while adjusting their proportions, scale, and positioning to create a harmonious, naturally balanced composition. Blend and integrate all elements seamlessly with consistent lighting, perspective, and style. The final result should look like a single naturally captured scene where all subjects are properly sized and positioned relative to each other, not assembled from multiple sources."
    
    # Clear GPU memory before generation
    torch.cuda.empty_cache()
    
    try:
        # Try with original size first
        image = pipe(
            image=concatenated_image, 
            prompt=final_prompt,
            guidance_scale=guidance_scale,
            width=concatenated_image.size[0],
            height=concatenated_image.size[1],
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
        
    except torch.cuda.OutOfMemoryError:
        print("OOM with original size, trying smaller...")
        # If OOM, try with smaller dimensions
        width, height = concatenated_image.size
        new_width = min(width, 1024)
        new_height = int(height * (new_width / width))
        new_height = (new_height // 64) * 64
        
        resized_image = concatenated_image.resize((new_width, new_height), Image.LANCZOS)
        
        torch.cuda.empty_cache()
        
        image = pipe(
            image=resized_image, 
            prompt=final_prompt,
            guidance_scale=guidance_scale,
            width=new_width,
            height=new_height,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    
    return image, seed, gr.update(visible=True)

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 Kontext [dev] - Multi-Image (GGUF Optimized)
        Flux Kontext with multiple image input support using GGUF quantization for efficient memory usage.
        Current model: {SELECTED_MODEL}
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(
                    label="Upload image(s) for editing", 
                    show_label=True,
                    elem_id="gallery_input",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    file_types=['image'],
                    type='pil'
                )
                
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt for editing (e.g., 'Remove glasses', 'Add a hat')",
                        container=False,
                    )
                    run_button = gr.Button("Run", scale=0)
                    
                with gr.Accordion("Advanced Settings", open=False):
            
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=2.5,
                    )       
                    
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, interactive=False)
                reuse_button = gr.Button("Reuse this image", visible=False)
        
        
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [input_images, prompt, seed, randomize_seed, guidance_scale],
        outputs = [result, seed, reuse_button]
    )
    
    reuse_button.click(
        fn = lambda image: [image] if image is not None else [],
        inputs = [result],
        outputs = [input_images]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
