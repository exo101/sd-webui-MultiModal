import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
#from kontext_pipeline import FluxKontextPipeline
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# Load Kontext model
MAX_SEED = np.iinfo(np.int32).max

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")

def concatenate_images(images, direction="horizontal"):
    """
    Concatenate multiple PIL images either horizontally or vertically.
    
    Args:
        images: List of PIL Images
        direction: "horizontal" or "vertical"
    
    Returns:
        PIL Image: Concatenated image
    """
    if not images:
        return None
    
    # Filter out None images
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        return None
    
    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")
    
    # Convert all images to RGB
    valid_images = [img.convert("RGB") for img in valid_images]
    
    if direction == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        # Paste images
        x_offset = 0
        for img in valid_images:
            # Center image vertically if heights differ
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
    else:  # vertical
        # Calculate max width and total height
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in valid_images:
            # Center image horizontally if widths differ
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated

@spaces.GPU
def infer(input_images, prompt, seed=42, randomize_seed=False, guidance_scale=2.5, progress=gr.Progress(track_tqdm=True)):
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # Handle input_images - it could be a single image or a list of images
    if input_images is None:
        raise gr.Error("Please upload at least one image.")
    
    # If it's a single image (not a list), convert to list
    if not isinstance(input_images, list):
        input_images = [input_images]
    
    # Filter out None images
    valid_images = [img[0] for img in input_images if img is not None]
    
    if not valid_images:
        raise gr.Error("Please upload at least one valid image.")
    
    # Concatenate images horizontally
    concatenated_image = concatenate_images(valid_images, "horizontal")
    
    if concatenated_image is None:
        raise gr.Error("Failed to process the input images.")
    
    # original_width, original_height = concatenated_image.size
    
    # if original_width >= original_height:
    #     new_width = 1024
    #     new_height = int(original_height * (new_width / original_width))
    #     new_height = round(new_height / 64) * 64
    # else:
    #     new_height = 1024
    #     new_width = int(original_width * (new_height / original_height))
    #     new_width = round(new_width / 64) * 64
    
    #concatenated_image_resized = concatenated_image.resize((new_width, new_height), Image.LANCZOS)

    final_prompt = f"From the provided reference images, create a unified, cohesive image such that {prompt}. Maintain the identity and characteristics of each subject while adjusting their proportions, scale, and positioning to create a harmonious, naturally balanced composition. Blend and integrate all elements seamlessly with consistent lighting, perspective, and style.the final result should look like a single naturally captured scene where all subjects are properly sized and positioned relative to each other, not assembled from multiple sources."
    
    image = pipe(
        image=concatenated_image, 
        prompt=final_prompt,
        guidance_scale=guidance_scale,
        width=concatenated_image.size[0],
        height=concatenated_image.size[1],
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
        gr.Markdown(f"""# FLUX.1 Kontext [dev] - Multi-Image
        Flux Kontext with multiple image input support - compose a new image with elements from multiple images using Kontext [dev]
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
        fn = lambda image: [image] if image is not None else [],  # Convert single image to list for gallery
        inputs = [result],
        outputs = [input_images]
    )

demo.launch()