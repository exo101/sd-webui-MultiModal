#!/bin/bash

# Enhanced FLUX.1-Kontext Deployment Script
# Installs conda, creates environment, clones repo, installs dependencies
# Now includes both multi-image and portrait series applications

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to setup conda with proper error handling
setup_conda() {
    local CONDA_PATH="$HOME/miniconda/bin/conda"
    
    if command -v conda &> /dev/null; then
        print_success "Conda already installed"
        return 0
    fi
    
    print_status "Installing Miniconda..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh
    
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
    
    # Initialize conda
    $CONDA_PATH init bash
    
    # Add conda to current PATH for this session
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # Source bashrc to get conda in PATH (ignore errors if sourcing fails)
    source ~/.bashrc 2>/dev/null || true
    
    # Accept Terms of Service proactively
    print_status "Accepting conda Terms of Service..."
    $CONDA_PATH tos accept 2>/dev/null || {
        print_warning "Attempting to accept TOS for specific channels..."
        $CONDA_PATH tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
        $CONDA_PATH tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    }
    
    print_success "Miniconda installed and configured"
}

# Setup conda environment
setup_environment() {
    print_status "Setting up conda environment..."
    
    # Use conda command if available, otherwise use full path
    local CONDA_CMD="conda"
    if ! command -v conda &> /dev/null; then
        CONDA_CMD="$HOME/miniconda/bin/conda"
    fi
    
    # Initialize conda
    source $HOME/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
    
    # Remove existing environment if it exists
    if $CONDA_CMD env list | grep -q "flux-kontext"; then
        print_warning "Removing existing flux-kontext environment..."
        $CONDA_CMD env remove -n flux-kontext -y
    fi
    
    # Create new environment
    $CONDA_CMD create -n flux-kontext python=3.10 -y
    $CONDA_CMD activate flux-kontext || source $HOME/miniconda/bin/activate flux-kontext
    
    print_success "Environment created and activated"
}

# Clone repository and setup applications
clone_and_setup_repo() {
    print_status "Setting up repository and applications..."
    
    if [[ -d "$HOME/FLUX.1-Kontext-multi-image" ]]; then
        print_warning "Repository exists, updating..."
        cd $HOME/FLUX.1-Kontext-multi-image
        git pull
    else
        cd $HOME
        git clone https://github.com/nexusjuan12/FLUX.1-Kontext-multi-image.git
        cd FLUX.1-Kontext-multi-image
    fi
    
    print_success "Repository ready"
}

# Create application files
create_applications() {
    print_status "Creating application files..."
    
    # Create the portrait series app
    cat > portrait_series_app.py << 'EOF'
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

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

MAX_SEED = np.iinfo(np.int32).max
SELECTED_MODEL = "Q5_K_M"

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

def load_gguf_pipeline():
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
        
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        return pipe.to("cuda")
        
    except Exception as e:
        print(f"Error loading GGUF pipeline: {e}")
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        return pipe

print("Initializing FLUX.1-Kontext pipeline for portrait series...")
pipe = load_gguf_pipeline()
print("Pipeline loaded successfully!")

@spaces.GPU
def generate_portrait_series(input_image, selected_poses, custom_pose, portrait_type, seed=42, randomize_seed=False, guidance_scale=2.5, num_inference_steps=20, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    if input_image is None:
        raise gr.Error("Please upload a portrait image.")
    
    poses_to_generate = []
    if custom_pose and custom_pose.strip():
        poses_to_generate.append(custom_pose.strip())
    if selected_poses:
        poses_to_generate.extend(selected_poses)
    
    if not poses_to_generate:
        raise gr.Error("Please select at least one pose or enter a custom pose.")
    
    poses_to_generate = list(dict.fromkeys(poses_to_generate))
    generated_images = []
    used_seeds = []
    torch.cuda.empty_cache()
    
    for i, pose in enumerate(poses_to_generate):
        current_seed = seed + i if not randomize_seed else random.randint(0, MAX_SEED)
        
        if portrait_type == "Corporate Headshot":
            base_prompt = f"Change to a professional corporate headshot with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Professional studio lighting, clean background, business attire."
        elif portrait_type == "Artistic Portrait":
            base_prompt = f"Transform into an artistic portrait with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Creative lighting, artistic composition."
        elif portrait_type == "Casual Portrait":
            base_prompt = f"Change to a casual portrait with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Natural lighting, relaxed atmosphere."
        else:
            base_prompt = f"Change to a full body portrait with the pose: {pose}, while maintaining the same facial features, hairstyle, and expression. Complete figure visible, balanced composition."
        
        final_prompt = f"{base_prompt} Preserve identity and personality. Maximum detail and realism, HDR processing."
        
        try:
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
        progress((i + 1) / len(poses_to_generate), f"Generated pose {i+1}/{len(poses_to_generate)}")
        torch.cuda.empty_cache()
    
    return generated_images, used_seeds

css = """
#col-container {
    margin: 0 auto;
    max-width: 1200px;
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
                input_image = gr.Image(label="Upload Portrait Image", type="pil", height=400)
                portrait_type = gr.Dropdown(
                    label="Portrait Type",
                    choices=["Corporate Headshot", "Artistic Portrait", "Casual Portrait", "Full Body"],
                    value="Corporate Headshot"
                )
                custom_pose = gr.Textbox(label="Custom Pose (Optional)", placeholder="Describe a custom pose...", lines=3)
                selected_poses = gr.CheckboxGroup(
                    label="Select Poses for Series",
                    choices=PORTRAIT_POSES,
                    value=[PORTRAIT_POSES[0], PORTRAIT_POSES[2], PORTRAIT_POSES[4]]
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                    randomize_seed = gr.Checkbox(label="Randomize seed for each pose", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, step=0.1, value=2.5)
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, step=1, value=20)
                
                generate_button = gr.Button("Generate Portrait Series", variant="primary")
            
            with gr.Column(scale=2):
                generated_gallery = gr.Gallery(
                    label="Generated Portrait Series",
                    columns=3, rows=2, object_fit="contain", height="auto"
                )
                seed_info = gr.Textbox(label="Used Seeds", interactive=False, lines=2)
        
        def on_generate(input_image, selected_poses, custom_pose, portrait_type, seed, randomize_seed, guidance_scale, num_inference_steps):
            if input_image is None:
                return [], "Please upload an image first."
            
            images, seeds = generate_portrait_series(
                input_image, selected_poses, custom_pose, portrait_type,
                seed, randomize_seed, guidance_scale, num_inference_steps
            )
            seed_text = f"Seeds used: {', '.join(map(str, seeds))}"
            return images, seed_text
        
        generate_button.click(
            fn=on_generate,
            inputs=[input_image, selected_poses, custom_pose, portrait_type, seed, randomize_seed, guidance_scale, num_inference_steps],
            outputs=[generated_gallery, seed_info]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
EOF
    
    # Create a launcher script
    cat > launch_apps.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import threading
import time
import sys

def launch_app(script_name, port, app_name):
    """Launch a Gradio app in a separate process"""
    print(f"Starting {app_name} on port {port}...")
    try:
        # Modify the script to use the specified port
        with open(script_name, 'r') as f:
            content = f.read()
        
        # Replace the port in the launch command
        if 'server_port=' in content:
            content = content.replace('server_port=7860', f'server_port={port}')
            content = content.replace('server_port=7861', f'server_port={port}')
        else:
            content = content.replace('demo.launch()', f'demo.launch(server_name="0.0.0.0", server_port={port})')
        
        # Write temporary file
        temp_script = f"temp_{script_name}"
        with open(temp_script, 'w') as f:
            f.write(content)
        
        process = subprocess.Popen([sys.executable, temp_script])
        return process
    except Exception as e:
        print(f"Error launching {app_name}: {e}")
        return None

def main():
    print("FLUX.1 Kontext Multi-App Launcher")
    print("=" * 40)
    
    apps = []
    
    # Launch multi-image app
    if input("Launch Multi-Image App on port 7860? (y/n): ").lower().startswith('y'):
        proc1 = launch_app('gguf_flux_app.py', 7860, 'Multi-Image App')
        if proc1:
            apps.append(('Multi-Image App', proc1, 7860))
    
    # Launch portrait series app
    if input("Launch Portrait Series App on port 7861? (y/n): ").lower().startswith('y'):
        proc2 = launch_app('portrait_series_app.py', 7861, 'Portrait Series App')
        if proc2:
            apps.append(('Portrait Series App', proc2, 7861))
    
    if not apps:
        print("No apps selected. Exiting...")
        return
    
    print("\nRunning applications:")
    for app_name, proc, port in apps:
        print(f"- {app_name}: http://localhost:{port}")
    
    print("\nPress Ctrl+C to stop all applications...")
    
    try:
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process has died
            for app_name, proc, port in apps:
                if proc.poll() is not None:
                    print(f"{app_name} has stopped.")
    except KeyboardInterrupt:
        print("\nShutting down applications...")
        for app_name, proc, port in apps:
            print(f"Stopping {app_name}...")
            proc.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Force kill if needed
        for app_name, proc, port in apps:
            if proc.poll() is None:
                proc.kill()
        
        print("All applications stopped.")

if __name__ == "__main__":
    main()
EOF
    
    print_success "Application files created"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Activate environment
    source $HOME/miniconda/etc/profile.d/conda.sh
    conda activate flux-kontext
    
    # Install PyTorch with CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing PyTorch with CUDA support..."
        conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
    else
        print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch..."
        conda install pytorch cpuonly -c pytorch -y
    fi
    
    # Install core packages via conda
    conda install -c conda-forge numpy pillow requests opencv -y
    
    # Install Python packages via pip
    pip install \
        diffusers>=0.30.0 \
        transformers>=4.44.0 \
        accelerate>=0.33.0 \
        sentencepiece>=0.2.0 \
        gradio>=4.0.0 \
        huggingface_hub>=0.24.0 \
        safetensors>=0.4.0 \
        protobuf>=3.19.0 \
        spaces \
        scipy \
        matplotlib \
        tqdm
    
    # Install latest diffusers with GGUF support
    pip install git+https://github.com/huggingface/diffusers.git
    
    print_success "Dependencies installed"
}

# Main execution
main() {
    echo "=========================================="
    echo "FLUX.1-Kontext Enhanced Deployment"
    echo "Multi-Image + Portrait Series Apps"
    echo "=========================================="
    
    setup_conda
    setup_environment
    clone_and_setup_repo
    create_applications
    install_dependencies
    
    print_success "Deployment completed!"
    echo ""
    echo "Available Applications:"
    echo "1. Multi-Image App: python gguf_flux_app.py (port 7860)"
    echo "2. Portrait Series App: python portrait_series_app.py (port 7861)"
    echo "3. Launch both: python launch_apps.py"
    echo ""
    echo "To use:"
    echo "1. conda activate flux-kontext"
    echo "2. cd $HOME/FLUX.1-Kontext-multi-image"
    echo "3. Choose one of the options above"
    echo ""
    echo "Application features:"
    echo "- Multi-Image: Combine multiple images into cohesive scenes"
    echo "- Portrait Series: Generate portrait variations with different poses"
    echo "- Both apps support GGUF quantization for memory efficiency"
}

main "$@"
