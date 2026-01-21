"""Constants for the AI Image Generator application."""

from pathlib import Path

# Application info
APP_ID = "com.aiimagegenerator.app"
APP_NAME = "AI Image Generator"
APP_VERSION = "0.1.0"

# Config paths
CONFIG_DIR = Path.home() / ".aiimagegen"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Supported model extensions
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}

# Supported samplers (diffusers scheduler names)
SAMPLERS = {
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "dpm_2": "KDPM2DiscreteScheduler",
    "dpm_2_a": "KDPM2AncestralDiscreteScheduler",
    "lms": "LMSDiscreteScheduler",
    "dpm++_2m": "DPMSolverMultistepScheduler",
    "dpm++_2m_karras": "DPMSolverMultistepScheduler",
    "dpm++_sde": "DPMSolverSDEScheduler",
    "dpm++_sde_karras": "DPMSolverSDEScheduler",
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "uni_pc": "UniPCMultistepScheduler",
    "pndm": "PNDMScheduler",
}

# Samplers that use Karras sigmas
KARRAS_SAMPLERS = {"dpm++_2m_karras", "dpm++_sde_karras"}

# Scheduler types (noise schedules)
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform"]

# Default generation parameters
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "normal"
DEFAULT_STEPS = 20
DEFAULT_CFG_SCALE = 7.0
DEFAULT_SEED = -1  # -1 means random

# Image size presets (width, height)
SIZE_PRESETS = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024),
    "1024x768": (1024, 768),
    "768x1024": (768, 1024),
    "1280x720": (1280, 720),
    "720x1280": (720, 1280),
    "1536x1024": (1536, 1024),
    "1024x1536": (1024, 1536),
}

# GPU memory settings (in GB)
GPU_MEMORY_HEADROOM = 4  # Leave 4GB headroom per GPU
GPU_MEMORY_UPDATE_INTERVAL = 1000  # Update VRAM display every 1 second (ms)

# Generation settings
MAX_STEPS = 150
MIN_STEPS = 1
MAX_CFG_SCALE = 30.0
MIN_CFG_SCALE = 1.0
MIN_SIZE = 256
MAX_SIZE = 2048
SIZE_STEP = 64

# Thumbnail settings
THUMBNAIL_SIZE = 128
THUMBNAIL_COLUMNS = 2

# Output settings
OUTPUT_FORMAT = "png"
OUTPUT_QUALITY = 95  # For JPEG if ever used
