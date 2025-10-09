import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.utils import load_image

# Portrait poses for series generation
PORTRAIT_POSES = [
    "Arms raised overhead, hands clasped together, elbows bent outward",
    "Back facing the camera, head upright, arms relaxed at sides",
    "Facing forward, head turned slightly to the left, shoulders squared",
    "Three-quarter profile to the left, head turned toward the camera, shoulders angled",
    "Right profile, head turned to the right, shoulders angled away from the camera",
    "Leaning forward, resting right cheek on crossed arms atop a surface, body angled left",
    "Leaning on left elbow, left hand supporting chin, right arm relaxed on surface",
    "Three-quarter profile to the right, head turned slightly left, hand raised near collar",
    "Left profile, head turned to the left, shoulders angled away from the camera",
    "Facing forward, head upright, shoulders squared, arms relaxed at sides",
    "Facing forward, head turned slightly to the right, shoulders squared",
    "Leaning on both elbows on a surface, hands clasped together, shoulders squared"
]

# GGUF model configurations
MAX_SEED = np.iinfo(np.int32).max

GGUF_MODELS = {
    "Q2_K": "QuantStack/FLUX.1-Kontext-dev-GGUF",
    "Q4_K_S": "QuantStack/FLUX.1-Kontext-dev-GGUF", 
    "Q5_K_M": "QuantStack/FLUX.1-Kontext-dev-GGUF",
    "Q8_0": "QuantStack/FLUX.1-Kontext-dev-GGUF"
}

GGUF_FILENAMES = {
    "Q2_K": "flux1-kontext-dev-Q2_K.gguf",
    "Q4_K_S": "flux1-kontext-dev-Q4_K_S.gguf",
    "Q5_K_M": "flux1-kontext-dev-Q5_K_M.gguf", 
    "Q8_0": "flux1-kontext-dev-Q8_0.gguf"
}

# Default model selection (can be changed based on VRAM)
SELECTED_MODEL = "Q5_K_M"

def load_gguf_pipeline():
    """Load the GGUF quantized Flux Kontext pipeline"""
    try:
        print(f"Loading GGUF model: {SELECTED_MODEL}")
        transformer = FluxTransformer2DModel.from_single_file(
            GGUF_MODELS[SELECTED_MODEL],
            filename=GGUF_FILENAMES[SELECTED_MODEL],
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        
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
        print("Falling back to regular pipeline...")
        
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", 
            torch_dtype=torch.bfloat16
        )
        
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        return pipe

# Initialize pipeline
print("Initializing FLUX.1-Kontext pipeline for portrait series...")
pipe = load_gguf_pipeline()
print("Pipeline loaded successfully!")

@spaces.GPU
def generate_portrait_series(
    input_image, 
    selected_poses, 
    custom_pose,
    portrait_type,
    seed=42, 
    randomize_seed=False, 
    guidance_scale=2.5, 
    num_inference_steps=20,
    progress=gr.Progress(track_tqdm=True)
):
    """Generate a series of portrait variations with different poses"""
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    if input_image is None:
        raise gr.Error("Please upload a portrait image.")
    
    # Determine poses to generate
    poses_to_generate = []
    
    if custom_pose and custom_pose.strip():
        poses_to_generate.append(custom_pose.strip())
    
    if selected_poses:
        poses_to_generate.extend(selected_poses)
    
    if not poses_to_generate:
        raise gr.Error("Please select at least one pose or enter a custom pose.")
    
    # Remove duplicates while preserving order
    poses_to_generate = list(dict.fromkeys(poses_to_generate))
    
    generated_images = []
    used_seeds = []
    
    # Clear GPU memory before generation
    torch.cuda.empty_cache()
    
    progress_step = 1.0 / len(poses_to_generate)
    
    for i, pose in enumerate(poses_to_generate):
        current_seed = seed + i if not randomize_seed else random.randint(0, MAX_SEED)
        
        # Create the prompt based on portrait type
        if portrait_type == "Corporate Headshot":
            base_prompt = f"Change to a professional corporate headshot with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Professional studio lighting, clean background, business attire. High-resolution portrait photography."
        elif portrait_type == "Artistic Portrait":
            base_prompt = f"Transform into an artistic portrait with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Creative lighting, artistic composition, professional photography."
        elif portrait_type == "Casual Portrait":
            base_prompt = f"Change to a casual portrait with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Natural lighting, relaxed atmosphere, authentic snapshot."
        else:  # Full Body
            base_prompt = f"Change to a full body portrait with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Complete figure visible, balanced composition, professional photography."
        
        final_prompt = f"{base_prompt} Preserve identity and personality, maintaining distinctive appearance. Maximum detail and realism, HDR processing."
        
        try:
            # Try with original size first
            image = pipe(
                image=input_image, 
                prompt=final_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=input_image.size[0],
                height=input_image.size[1],
                generator=torch.Generator().manual_seed(current_seed),
            ).images[0]
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM with original size for pose {i+1}, trying smaller...")
            # If OOM, try with smaller dimensions
            width, height = input_image.size
            new_width = min(width, 1024)
            new_height = int(height * (new_width / width))
            new_height = (new_height // 64) * 64
            
            resized_input = input_image.resize((new_width, new_height), Image.LANCZOS)
            
            torch.cuda.empty_cache()
            
            image = pipe(
                image=resized_input, 
                prompt=final_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=new_width,
                height=new_height,
                generator=torch.Generator().manual_seed(current_seed),
            ).images[0]
        
        generated_images.append(image)
        used_seeds.append(current_seed)
        
        # Update progress
        progress((i + 1) * progress_step, f"Generated pose {i+1}/{len(poses_to_generate)}")
        
        # Clear cache between generations
        torch.cuda.empty_cache()
    
    return generated_images, used_seeds

def create_pose_grid(images, max_cols=3):
    """Create a grid layout of generated portraits"""
    if not images:
        return None
    
    num_images = len(images)
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols
    
    # Calculate grid dimensions
    img_width, img_height = images[0].size
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    return grid

css = """
#col-container {
    margin: 0 auto;
    max-width: 1200px;
}
.pose-gallery {
    height: 400px;
}
.generated-gallery {
    height: 600px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 Kontext Portrait Series Generator (GGUF Optimized)
        Generate multiple portrait variations with different poses while maintaining facial features and identity.
        Current model: {SELECTED_MODEL}
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Portrait Image", 
                    type="pil",
                    height=400
                )
                
                portrait_type = gr.Dropdown(
                    label="Portrait Type",
                    choices=["Corporate Headshot", "Artistic Portrait", "Casual Portrait", "Full Body"],
                    value="Corporate Headshot"
                )
                
                custom_pose = gr.Textbox(
                    label="Custom Pose (Optional)",
                    placeholder="Describe a custom pose...",
                    lines=3
                )
                
                selected_poses = gr.CheckboxGroup(
                    label="Select Poses for Series",
                    choices=PORTRAIT_POSES,
                    value=[PORTRAIT_POSES[0], PORTRAIT_POSES[2], PORTRAIT_POSES[4]]
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed for each pose", value=True)
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=2.5,
                    )
                    
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=20,
                    )
                
                generate_button = gr.Button("Generate Portrait Series", variant="primary")
            
            with gr.Column(scale=2):
                generated_gallery = gr.Gallery(
                    label="Generated Portrait Series",
                    elem_classes=["generated-gallery"],
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
                
                grid_image = gr.Image(
                    label="Portrait Grid",
                    interactive=False
                )
                
                seed_info = gr.Textbox(
                    label="Used Seeds",
                    interactive=False,
                    lines=2
                )
        
        # Event handlers
        def on_generate(input_image, selected_poses, custom_pose, portrait_type, seed, randomize_seed, guidance_scale, num_inference_steps):
            if input_image is None:
                return [], None, "Please upload an image first."
            
            images, seeds = generate_portrait_series(
                input_image, selected_poses, custom_pose, portrait_type,
                seed, randomize_seed, guidance_scale, num_inference_steps
            )
            
            grid = create_pose_grid(images)
            seed_text = f"Seeds used: {', '.join(map(str, seeds))}"
            
            return images, grid, seed_text
        
        generate_button.click(
            fn=on_generate,
            inputs=[
                input_image, selected_poses, custom_pose, portrait_type,
                seed, randomize_seed, guidance_scale, num_inference_steps
            ],
            outputs=[generated_gallery, grid_image, seed_info]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)