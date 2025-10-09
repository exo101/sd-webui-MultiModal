# FLUX.1-Kontext Multi-Image & Portrait Series Applications

This repository contains two powerful applications built on FLUX.1-Kontext with GGUF quantization support for efficient memory usage:

1. **Multi-Image App**: Combine multiple images into cohesive, unified scenes
2. **Portrait Series App**: Generate portrait variations with different poses while maintaining identity

## Features

### Multi-Image Application (`gguf_flux_app.py`)
- Upload multiple images and combine them into a single cohesive scene
- Intelligent image concatenation and processing  
- Automatic fallback for memory-constrained systems
- GGUF quantization support (Q2_K, Q4_K_S, Q5_K_M, Q8_0)
- Memory optimizations with VAE slicing and tiling

### Portrait Series Application (`portrait_series_app.py`)
- Generate multiple portrait variations from a single input image
- 12 predefined professional poses + custom pose support
- Multiple portrait types: Corporate Headshot, Artistic Portrait, Casual Portrait, Full Body
- Batch generation with individual seed control
- Identity preservation across all variations
- Gallery view and grid layout options

## System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for models
- **OS**: Linux (Ubuntu/Debian recommended)

## Quick Start

### Option 1: Automated Installation

```bash
# Download and run the deployment script
wget https://raw.githubusercontent.com/nexusjuan12/FLUX.1-Kontext-multi-image/main/flux_deploy_enhanced.sh
chmod +x flux_deploy_enhanced.sh
./flux_deploy_enhanced.sh
```

### Option 2: Manual Installation

1. **Setup Environment**
```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Create and activate environment
conda create -n flux-kontext python=3.10 -y
conda activate flux-kontext
```

2. **Clone Repository**
```bash
git clone https://github.com/nexusjuan12/FLUX.1-Kontext-multi-image.git
cd FLUX.1-Kontext-multi-image
```

3. **Install Dependencies**
```bash
# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core packages
conda install -c conda-forge numpy pillow requests opencv -y

# Install Python packages
pip install diffusers>=0.30.0 transformers>=4.44.0 accelerate>=0.33.0 \
    sentencepiece>=0.2.0 gradio>=4.0.0 huggingface_hub>=0.24.0 \
    safetensors>=0.4.0 protobuf>=3.19.0 spaces scipy matplotlib tqdm

# Install latest diffusers with GGUF support
pip install git+https://github.com/huggingface/diffusers.git
```

## Usage

### Launch Individual Applications

**Multi-Image App:**
```bash
conda activate flux-kontext
cd FLUX.1-Kontext-multi-image
python gguf_flux_app.py
# Access at: http://localhost:7860
```

**Portrait Series App:**
```bash
conda activate flux-kontext
cd FLUX.1-Kontext-multi-image
python portrait_series_app.py
# Access at: http://localhost:7861
```

### Launch Both Applications
```bash
conda activate flux-kontext
cd FLUX.1-Kontext-multi-image
python launch_apps.py
```

## GGUF Model Configuration

The applications support different GGUF quantization levels. Edit the `SELECTED_MODEL` variable in the app files:

- **Q2_K**: Lowest quality, ~2GB VRAM
- **Q4_K_S**: Balanced quality/size, ~4GB VRAM  
- **Q5_K_M**: Good quality, ~5GB VRAM (default)
- **Q8_0**: Highest quality, ~8GB VRAM

```python
SELECTED_MODEL = "Q5_K_M"  # Change this to your preferred model
```

## Application Screenshots

### Multi-Image App Interface
- Upload multiple images in the gallery
- Enter descriptive prompt for the combined scene
- Adjust guidance scale and inference steps
- Get unified, cohesive output

### Portrait Series App Interface
- Upload a single portrait image
- Select from 12 predefined poses or enter custom pose
- Choose portrait type (Corporate, Artistic, Casual, Full Body)
- Generate multiple variations while preserving identity

## Tips for Best Results

### Multi-Image App
- Use images with similar lighting conditions
- Keep subjects at similar scales
- Use descriptive prompts that specify the desired interaction
- Example: "Two people having a conversation in a modern office"

### Portrait Series App
- Use high-quality portrait images with clear facial features
- Ensure good lighting in the input image
- Experiment with different portrait types for varied styles
- Use custom poses for specific requirements

## Memory Management

Both applications include automatic memory management:

- **Automatic fallback**: Reduces image size if GPU runs out of memory
- **Memory clearing**: Clears CUDA cache between generations
- **VAE optimizations**: Slicing and tiling enabled by default
- **CPU offloading**: Models moved to CPU when not in use

## Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce `SELECTED_MODEL` to a lower quantization (Q4_K_S or Q2_K)
- Lower inference steps
- Use smaller input images

**Conda TOS Error:**
```bash
conda tos accept
```

**Model Download Issues:**
- Ensure stable internet connection
- Models are downloaded automatically on first run
- Check Hugging Face access if using gated models

**Poor Quality Results:**
- Increase guidance scale (2.5-4.0 range)
- Use higher quantization model (Q8_0)
- Increase inference steps (20-30)

## Model Information

- **Base Model**: FLUX.1-Kontext-dev
- **Quantization**: GGUF format from QuantStack
- **License**: Follows FLUX.1 licensing terms
- **Size**: Varies by quantization level (2GB-16GB)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project follows the licensing terms of the underlying FLUX.1-Kontext model. Please review the official FLUX.1 license for commercial usage terms.

## Acknowledgments

- Black Forest Labs for FLUX.1-Kontext
- Hugging Face for Diffusers library
- QuantStack for GGUF quantization
- Gradio team for the web interface framework

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information about your problem

---

**Note**: This is an unofficial implementation. For official FLUX.1 models and support, visit [Black Forest Labs](https://blackforestlabs.ai/).
