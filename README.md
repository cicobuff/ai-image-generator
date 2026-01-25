# AI Image Generator

A GTK 4 desktop application for AI image generation using Stable Diffusion, built with Python and the Hugging Face Diffusers library. Designed for Linux with multi-GPU support.

![image](docs/images/main-screen.png)

## Features

### Image Generation
- **Text-to-Image**: Generate images from text prompts using Stable Diffusion models
- **Image-to-Image (Img2Img)**: Transform existing images based on text prompts with adjustable strength
- **Inpainting**: Selectively regenerate parts of an image using rectangular or painted masks
- **Outpainting**: Extend images beyond their original borders in any direction (left, right, top, bottom)
- **Batch Generation**: Generate multiple images in sequence with automatic seed variation
- **Multi-GPU Support**: Parallel batch generation across multiple GPUs (e.g., 2x RTX 3090)
- **Refiner Mode**: Detect and refine specific regions using SAM3 text-prompted segmentation

### Image Editing
- **Crop Mode**: Crop images to a selected region with preset sizes or custom dimensions
- **Remove with Mask**: Delete content under a mask and fill with blended colors from surrounding pixels
- **Mask Tools**: Draw rectangular masks, paint freehand masks, move and resize masks

### Model Support
- **SDXL and SD 1.5**: Supports both Stable Diffusion XL and SD 1.5 model architectures
- **Checkpoint Loading**: Load models from `.safetensors` or `.ckpt` files
- **VAE Support**: Use embedded VAE or load separate VAE models
- **LoRA Support**: Load and apply multiple LoRA models with adjustable weights
- **Upscaling**: 4x upscaling using Real-ESRGAN models (RealESRGAN_x4plus, RealESRGAN_x4plus_anime)

### Prompt Management
- **Prompt Lists**: Create reusable lists of words/phrases (e.g., quality tags, style modifiers)
- **Random Selection**: Automatically select random words from checked lists during generation
- **Configurable Count**: Choose how many words (1-10) to randomly pick from each list
- **Batch Variation**: Each image in a batch gets a fresh random selection for variety
- **File-Based Storage**: Lists are saved as simple text files for easy editing

### Performance Optimization
- **torch.compile**: Optional model compilation with cached kernels for ~20% faster generation
  - Warning: First compilation takes several minutes depending on GPU
  - Compiled kernels are cached and shared across all GPUs
  - Does NOT work with Inpaint/Outpaint modes
  - Each optimization is for a fixed resolution
- **xformers**: Memory-efficient attention when available
- **CUDA Optimizations**: cuDNN benchmark, TF32, channels_last memory format

### User Interface
- **Three-Panel Layout**: Model/parameters (left), image display (center), thumbnail gallery (right)
- **Resizable Panels**: Drag panel borders to customize layout
- **GPU Monitoring**: Real-time VRAM usage, GPU utilization percentage, and temperature display
- **Progress Tracking**: Step-by-step generation progress with cancel support
- **Metadata Preservation**: Generation parameters saved in PNG metadata and restored on load
- **Info System**: Click info icons next to section headers for detailed explanations; hover over labels for tooltips

## Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Width | 256-2048 | 1024 | Image width (64px steps) |
| Height | 256-2048 | 1024 | Image height (64px steps) |
| Steps | 1-150 | 20 | Denoising steps |
| CFG Scale | 1.0-30.0 | 7.0 | Classifier-free guidance scale |
| Seed | -1 or 0+ | -1 | Random seed (-1 = random) |
| Strength | 0.0-1.0 | 0.75 | Img2Img/Inpaint transformation strength |

### Supported Samplers
- euler, euler_a
- heun
- dpm_2, dpm_2_a
- lms
- dpm++_2m, dpm++_2m_karras
- dpm++_sde, dpm++_sde_karras
- ddim, ddpm
- uni_pc, pndm

### Scheduler Types
- normal, simple, karras, exponential, sgm_uniform

## Prompt Management

The Prompt Management system allows you to create lists of words or phrases that can be randomly combined with your prompts during generation. This is useful for adding variety to batch generations or consistently applying quality tags.

### Location
The Prompt Management panel is located in the center panel, below the image display. It consists of:
- **Prompts** (left): Positive and negative prompt text areas
- **List** (right): Your saved prompt lists with checkboxes
- **Words** (right): Words in the currently selected list

### Creating a Prompt List
1. Click the **+** button next to "List"
2. Enter a name for your list (e.g., "quality", "style", "lighting")
3. Click "Create"
4. The new list appears in the List panel

### Adding Words to a List
1. Click on a list to select it
2. Click the **+** button next to "Words"
3. Enter a word or phrase (e.g., "masterpiece", "best quality", "highly detailed")
4. Click "Add"
5. Repeat to add more words

### Using Prompt Lists
1. Check the checkbox next to one or more lists you want to use
2. Set the number dropdown (1-10) to control how many words to randomly select from each list
3. When you generate an image:
   - The system randomly picks N words from each checked list (where N is the dropdown value)
   - Selected words are prepended to your positive prompt with commas
   - Each word is only picked once per list (no duplicates)
   - If N exceeds the number of words in a list, all words are selected

### Batch Generation
During batch generation, each image gets a **fresh random selection** of words. This creates natural variety across your batch while maintaining your base prompt.

### Example
**List "quality"** contains: `masterpiece, best quality, highly detailed, 8k resolution`
**Dropdown set to**: `2`
**Your prompt**: `a beautiful landscape`

Each generation might produce:
- `masterpiece, 8k resolution, a beautiful landscape`
- `best quality, highly detailed, a beautiful landscape`
- `masterpiece, highly detailed, a beautiful landscape`

### File Storage
Lists are saved as text files in `models/prompt-lists/`:
```
models/prompt-lists/
├── quality.txt
├── style.txt
└── lighting.txt
```

Each file contains one word/phrase per line, making them easy to edit with any text editor.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Q` | Quit application |
| `Ctrl++` or `Ctrl+=` | Zoom in (image display) |
| `Ctrl+-` | Zoom out (image display) |
| `Ctrl+0` | Reset zoom to fit window |
| `R` | Reset zoom to fit window |
| `0` | Reset zoom to fit window |
| `Delete` or `Backspace` | Delete selected image from gallery |

## Mouse Controls

### Image Display - Zoom
| Action | Effect |
|--------|--------|
| Scroll Up | Zoom in (centered on cursor position) |
| Scroll Down | Zoom out (centered on cursor position) |

- Zoom range: 10% to 2000%
- Zoom indicator appears in the corner when zoomed

### Image Display - Pan
| Action | Effect |
|--------|--------|
| Middle Mouse Button + Drag | Pan the image when zoomed in |

### Inpaint Mode - Mask Drawing
| Action | Effect |
|--------|--------|
| Left Click + Drag (Rect Mask tool) | Draw rectangular mask |
| Left Click + Drag (Paint Mask tool) | Paint circular mask (25px brush) |

- Masks appear as semi-transparent red overlay
- White areas in mask = regions to regenerate
- Black areas in mask = regions to preserve

### Outpaint Mode - Edge Selection
| Action | Effect |
|--------|--------|
| Left Click on edge | Toggle edge mask for that side |

- Edge masks appear as pink overlays on image borders
- Multiple edges can be selected simultaneously
- Use directional buttons (Left/Right/Top/Bottom) to extend

### Crop Mode - Mask Drawing and Editing
| Action | Effect |
|--------|--------|
| Left Click + Drag | Draw crop rectangle |
| Left Click inside mask + Drag | Move the crop region |
| Left Click on corner + Drag | Resize the crop region |

- Crop mask appears as semi-transparent purple overlay
- Tooltip shows dimensions while drawing/resizing
- Use size presets dropdown for quick sizing

### Refiner Mode - Mask Selection
| Action | Effect |
|--------|--------|
| Single Left Click on mask | Toggle mask selection (selected/deselected) |
| Double Left Click on mask | Delete the mask |
| Hover over mask | Highlight mask with brighter color |

- Detected masks appear as yellow overlays
- Selected masks have solid borders and higher opacity
- Deselected masks have dashed borders and lower opacity
- Labels show the detection prompt and confidence score

## Toolbar Buttons

### Model Management
- **Load Models**: Load selected checkpoint, VAE, and apply optimizations
- **Clear Models**: Unload all models and free VRAM

### Generation
- **Generate**: Create new image from text prompt
- **Img2Img**: Transform current image based on prompt
- **Upscale**: Upscale current image using Real-ESRGAN (when enabled)

### Inpaint Mode (appears when image is loaded)
- **Inpaint Mode**: Toggle inpaint editing mode
- **Rect Mask**: Draw rectangular selection masks
- **Paint Mask**: Paint freehand masks with brush
- **Clear Masks**: Remove all drawn masks
- **Generate Inpaint**: Regenerate masked areas

### Outpaint Mode (appears when image is loaded)
- **Outpaint Mode**: Toggle outpaint editing mode
- **Edge Mask**: Click on image edges to mark for extension
- **Clear Masks**: Remove all edge masks
- **Left/Right/Top/Bottom**: Extend image in the selected direction
- Outpaint generates new content beyond the original image borders with smooth blending

### Crop Mode (appears when image is loaded)
- **Crop Mode**: Toggle crop editing mode
- **Crop Mask**: Draw a rectangular crop region
- **Clear Mask**: Remove the crop selection
- **Crop Image**: Crop the image to the selected region
- **Remove with Mask**: Fill the masked area with blended colors from surrounding pixels
- **Size Presets**: Quick selection of common crop sizes (1024x1024, 1152x896, etc.)
- Crop masks can be moved (drag inside) and resized (drag corners)

### Refiner Mode (appears when image is loaded)
- **Refiner Mode**: Toggle refiner editing mode (yellow theme)
- **Detect**: Open text prompt dialog to detect objects in the image
- **Clear Masks**: Remove all detected masks
- **Generate Refine**: Refine selected regions using diffusion inpainting

## Refiner Mode

The Refiner Mode allows you to detect specific objects or regions in an image using text prompts, then selectively refine those areas using the diffusion model. This is useful for:
- Enhancing faces, hands, or other detailed areas
- Fixing specific parts of an image without affecting the rest
- Applying different prompts to different regions

### Requirements
- SAM3 model from Meta (facebook/sam3)
- Install the sam3 package: `pip install git+https://github.com/facebookresearch/sam3.git`
- Additional dependencies: `pip install einops decord pycocotools`

### Model Setup
1. Download the SAM3 model from HuggingFace: [facebook/sam3](https://huggingface.co/facebook/sam3)
2. Place the model files in `models/sams/sam3/`
3. Required files: `config.json`, `model.safetensors`, `processor_config.json`, tokenizer files

### Using Refiner Mode

#### Step 1: Enable Refiner Mode
1. Load an image in the main display
2. Click the **Refiner Mode** toggle button (yellow)
3. The center panel will show a yellow border indicating refiner mode is active

#### Step 2: Detect Objects
1. Click the **Detect** button
2. Enter a text prompt describing what to detect (e.g., "face", "hand", "person", "eye")
3. Adjust the **Threshold** slider (0.1-1.0) to control detection sensitivity
   - Lower values = more detections (may include false positives)
   - Higher values = fewer detections (more precise)
4. Click **Detect** to run SAM3 segmentation

#### Step 3: Select Masks
- Detected regions appear as yellow overlays with labels showing confidence scores
- **Single-click** on a mask to toggle its selection (selected = brighter, deselected = dimmer with dashed border)
- **Double-click** on a mask to delete it
- Only selected masks will be refined

#### Step 4: Generate Refinement
1. Enter a refinement prompt in the main positive prompt area
2. Optionally adjust negative prompt
3. Click **Generate Refine**
4. Each selected region is processed:
   - Cropped with padding
   - Upscaled (2x minimum, scaled to at least 512x512 for quality)
   - Inpainted using the diffusion model
   - Downscaled and composited back with feathered edges

### Refinement Pipeline Details

For each selected mask, the refiner performs:
1. **Crop**: Extract bounding box with 32px padding
2. **Upscale**: Scale up using Real-ESRGAN (if available) or Lanczos interpolation
3. **Minimum Size**: Ensure region is at least 512x512 for quality inpainting
4. **Inpaint**: Run diffusion inpainting with your prompt
5. **Downscale**: Scale back to original region size
6. **Composite**: Blend into original image with Gaussian-blurred edges

### Tips
- Use specific prompts for detection: "face" works better than "person's face"
- For small regions, the refiner automatically upscales to ensure quality
- Lower the threshold if you're not detecting enough regions
- Use the strength parameter (in generation params) to control how much the region changes
- Each mask uses a different seed for variety
- The SAM3 model is unloaded before generation to free VRAM

### Troubleshooting
- **No detections**: Try lowering the threshold or using different prompt words
- **Too many detections**: Increase the threshold
- **Poor quality refinement**: The region may be too small; the refiner will auto-upscale but very tiny regions may still lack detail
- **VRAM issues**: SAM3 uses ~2.5GB VRAM; it's unloaded before diffusion runs

## Configuration

Configuration is stored in `~/.aiimagegen/config.yaml`:

```yaml
version: "1.0"
directories:
  models: "./models"
  output: "./output"
generation:
  default_width: 1024
  default_height: 1024
  default_sampler: "euler"
  default_scheduler: "normal"
  default_steps: 20
  default_cfg_scale: 7.0
  default_seed: -1
```

## Directory Structure

```
models/
├── checkpoints/     # Stable Diffusion model files (.safetensors, .ckpt)
├── vae/             # VAE model files
├── loras/           # LoRA model files
├── upscale/         # Real-ESRGAN upscaler models
├── prompt-lists/    # Prompt list text files (one word per line)
├── compiled/        # Cached torch.compile kernels
└── sams/
    └── sam3/        # SAM3 segmentation model (for Refiner Mode)
        ├── config.json
        ├── model.safetensors
        ├── processor_config.json
        ├── tokenizer.json
        ├── vocab.json
        └── merges.txt

output/              # Generated images (organized in subdirectories)
```

## Requirements

- Python 3.12+
- GTK 4 / PyGObject
- PyTorch with CUDA support
- NVIDIA GPU with CUDA capability

### Python Dependencies
- PyGObject >= 3.46.0
- torch >= 2.1.0
- diffusers >= 0.27.0
- transformers >= 4.38.0
- accelerate >= 0.27.0
- safetensors >= 0.4.0
- Pillow >= 10.0.0
- PyYAML >= 6.0
- nvidia-ml-py >= 11.5.0
- xformers (optional, for memory-efficient attention)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-image-generator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## First Launch

1. On first launch, the Setup Screen appears
2. Configure paths for models and output directories
3. Select which GPUs to use (if multiple available)
4. Click "Save & Continue" to enter the main workspace

## Usage Tips

- **Auto Model Loading**: Clicking Generate/Img2Img will automatically load models if not already loaded
- **Metadata Restoration**: Clicking a thumbnail restores all generation parameters from that image
- **Batch + Optimize**: Enable Optimize for ~20% faster batch generation. Compiles once, then all GPUs share the cached kernels
- **Gallery Navigation**: Use the directory dropdown in the gallery panel to organize outputs into subdirectories
- **LoRA Stacking**: Multiple LoRAs can be active simultaneously with individual weight controls
- **Outpaint Chaining**: You can outpaint multiple times to progressively extend an image in any direction
- **Crop Mode**: Use size presets for quick cropping to standard dimensions, or draw custom regions
- **Remove with Mask**: Great for removing unwanted objects - fills with blended colors from the edges
- **Mode Switching**: Modes are mutually exclusive - enabling one mode automatically disables others
- **Info Icons**: Click the info icon (ⓘ) next to any section header for detailed help
- **Label Tooltips**: Hover over parameter labels for 1 second to see explanations
- **Prompt Lists**: Create lists of quality tags or style modifiers, check them to randomly add variety to your generations
- **Batch Variety**: Use prompt lists with batch generation - each image gets different random words for natural variation
- **Refiner Mode**: Use text prompts like "face" or "hand" to detect and selectively enhance specific regions
- **Refiner Quality**: Small regions are automatically upscaled to at least 512x512 for better inpainting results

## License

MIT License
