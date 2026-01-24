"""Diffusers backend for Stable Diffusion image generation."""

import os
# Disable CUDA graphs to avoid conflicts with persistent worker threads
# Must be set before importing torch
os.environ["TORCHINDUCTOR_CUDAGRAPH_TREES"] = "0"
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Keep async for performance

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
import gc
import warnings


def _log(message: str) -> None:
    """Print a log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

import torch

# Suppress harmless warnings from diffusers
warnings.filterwarnings("ignore", message=".*Pipelines loaded with.*dtype=torch.float16.*cannot run with.*cpu.*")
warnings.filterwarnings("ignore", message=".*upcast_vae.*is deprecated.*")

from PIL import Image, ImageDraw
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DDIMScheduler,
    DDPMScheduler,
    UniPCMultistepScheduler,
    PNDMScheduler,
)

from src.utils.constants import SAMPLERS, KARRAS_SAMPLERS
from src.core.gpu_manager import gpu_manager


@dataclass
class GenerationParams:
    """Parameters for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "euler"
    scheduler: str = "normal"


class DiffusersBackend:
    """Backend for image generation using Hugging Face diffusers."""

    SCHEDULER_CLASSES = {
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "HeunDiscreteScheduler": HeunDiscreteScheduler,
        "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
        "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
        "LMSDiscreteScheduler": LMSDiscreteScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "DPMSolverSDEScheduler": DPMSolverSDEScheduler,
        "DDIMScheduler": DDIMScheduler,
        "DDPMScheduler": DDPMScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "PNDMScheduler": PNDMScheduler,
    }

    def __init__(self, gpu_index: int = 0):
        """
        Initialize the backend.

        Args:
            gpu_index: The GPU index to use for this backend instance
        """
        self._pipeline: Optional[Any] = None
        self._img2img_pipeline: Optional[Any] = None
        self._inpaint_pipeline: Optional[Any] = None
        self._is_sdxl: bool = False
        self._loaded_checkpoint: Optional[str] = None
        self._loaded_vae: Optional[str] = None
        self._gpu_index: int = gpu_index
        self._gpu_indices: list[int] = [gpu_index]
        self._device: str = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        # LoRA tracking
        self._loaded_loras: list[tuple[str, float]] = []  # List of (path, weight) tuples
        # Prompt embedding cache for faster repeated generations
        self._cached_prompt: Optional[str] = None
        self._cached_negative_prompt: Optional[str] = None
        self._cached_prompt_embeds: Optional[torch.Tensor] = None
        self._cached_negative_prompt_embeds: Optional[torch.Tensor] = None
        self._cached_pooled_prompt_embeds: Optional[torch.Tensor] = None  # SDXL only
        self._cached_negative_pooled_prompt_embeds: Optional[torch.Tensor] = None  # SDXL only
        # Track if model is compiled
        self._is_compiled: bool = False
        # Track generation count and last resolution for warm-up diagnostics
        self._generation_count: int = 0
        self._last_resolution: Optional[tuple[int, int]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._pipeline is not None

    @property
    def is_sdxl(self) -> bool:
        """Check if loaded model is SDXL."""
        return self._is_sdxl

    @property
    def loaded_checkpoint(self) -> Optional[str]:
        """Get path of currently loaded checkpoint, or None if no model loaded."""
        return self._loaded_checkpoint

    @property
    def loaded_vae(self) -> Optional[str]:
        """Get path of currently loaded VAE, or None if using embedded."""
        return self._loaded_vae

    def set_gpus(self, indices: list[int]) -> None:
        """Set which GPUs to use for generation."""
        self._gpu_indices = indices if indices else [0]
        # Update primary GPU index and device
        if indices:
            self._gpu_index = indices[0]
            self._device = f"cuda:{self._gpu_index}" if torch.cuda.is_available() else "cpu"

    @property
    def gpu_index(self) -> int:
        """Get the GPU index this backend uses."""
        return self._gpu_index

    @property
    def is_compiled(self) -> bool:
        """Check if the loaded model has torch.compile applied."""
        return self._is_compiled

    def _get_compiled_cache_path(
        self,
        checkpoint_path: str,
        target_width: int,
        target_height: int,
    ) -> Path:
        """
        Get the cache directory path for a compiled model.

        Args:
            checkpoint_path: Path to the checkpoint file
            target_width: Target generation width
            target_height: Target generation height

        Returns:
            Path to the cache directory
        """
        import hashlib

        cache_base = Path(checkpoint_path).parent.parent / "compiled"
        cache_key = hashlib.md5(
            f"{checkpoint_path}_{target_width}x{target_height}".encode()
        ).hexdigest()[:12]

        return cache_base / cache_key

    def has_compiled_cache(
        self,
        checkpoint_path: str,
        target_width: int = 1024,
        target_height: int = 1024,
    ) -> bool:
        """
        Check if a compiled cache exists for the given model and resolution.

        Args:
            checkpoint_path: Path to the checkpoint file
            target_width: Target generation width
            target_height: Target generation height

        Returns:
            True if a compiled cache exists
        """
        cache_dir = self._get_compiled_cache_path(checkpoint_path, target_width, target_height)
        # Check for our marker file that indicates successful compilation
        marker_file = cache_dir / ".compile_complete"
        return marker_file.exists()

    def load_model(
        self,
        checkpoint_path: str,
        model_type: str = "sdxl",
        vae_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        use_compiled: bool = False,
        target_width: int = 1024,
        target_height: int = 1024,
    ) -> bool:
        """
        Load a Stable Diffusion model.

        Args:
            checkpoint_path: Path to the checkpoint file
            model_type: Type of model ("sdxl", "sd15", "sd")
            vae_path: Optional path to separate VAE
            clip_path: Optional path to separate CLIP (not commonly used)
            progress_callback: Callback for progress updates (message, progress 0-1)
            use_compiled: If True and compiled cache exists, apply torch.compile
            target_width: Target width for compiled cache lookup
            target_height: Target height for compiled cache lookup

        Returns:
            True if loading succeeded
        """
        import os

        print(f"[DEBUG] DiffusersBackend.load_model called: checkpoint={checkpoint_path}, use_compiled={use_compiled}")

        try:
            if progress_callback:
                progress_callback("Preparing to load model...", 0.0)

            # Check if we should use compiled cache
            should_compile = False
            cache_dir = None
            if use_compiled:
                cache_dir = self._get_compiled_cache_path(checkpoint_path, target_width, target_height)
                cache_exists = cache_dir.exists()
                has_files = any(cache_dir.glob("*")) if cache_exists else False
                _log(f"Checking compiled cache: use_compiled={use_compiled}, cache_dir={cache_dir}, exists={cache_exists}, has_files={has_files}")
                if cache_exists and has_files:
                    should_compile = True
                    self._setup_inductor_cache(cache_dir)
                    _log(f"Using compiled cache from: {cache_dir}")
                else:
                    _log(f"No compiled cache found at: {cache_dir}, loading without compilation")

            # Unload existing model first
            self.unload_model()

            self._is_sdxl = "sdxl" in model_type.lower()

            # Get max memory config for multi-GPU
            max_memory = gpu_manager.get_max_memory_config(self._gpu_indices)

            if progress_callback:
                progress_callback("Loading checkpoint...", 0.1)

            # Load the appropriate pipeline
            pipeline_class = StableDiffusionXLPipeline if self._is_sdxl else StableDiffusionPipeline

            # Use first selected GPU (multi-GPU requires more complex setup)
            self._device = f"cuda:{self._gpu_indices[0]}" if torch.cuda.is_available() else "cpu"
            _log(f"Target device: {self._device}")

            # Load the pipeline
            _log(f"Loading model from checkpoint...")
            self._pipeline = pipeline_class.from_single_file(
                checkpoint_path,
                torch_dtype=torch.float16,
                use_safetensors=checkpoint_path.endswith(".safetensors"),
            )

            # Move all components to the same GPU
            _log(f"Moving pipeline components to {self._device}...")

            if hasattr(self._pipeline, 'unet') and self._pipeline.unet is not None:
                self._pipeline.unet = self._pipeline.unet.to(self._device)
                _log(f"  UNet moved to {self._device}")

            if hasattr(self._pipeline, 'vae') and self._pipeline.vae is not None:
                self._pipeline.vae = self._pipeline.vae.to(self._device)
                _log(f"  VAE moved to {self._device}")

            if hasattr(self._pipeline, 'text_encoder') and self._pipeline.text_encoder is not None:
                self._pipeline.text_encoder = self._pipeline.text_encoder.to(self._device)
                _log(f"  Text encoder moved to {self._device}")

            if hasattr(self._pipeline, 'text_encoder_2') and self._pipeline.text_encoder_2 is not None:
                self._pipeline.text_encoder_2 = self._pipeline.text_encoder_2.to(self._device)
                _log(f"  Text encoder 2 moved to {self._device}")

            _log(f"Pipeline device: {self._pipeline.device}")

            if progress_callback:
                progress_callback("Loading VAE...", 0.5)

            # Load separate VAE if provided
            if vae_path:
                vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=torch.float16,
                )
                self._pipeline.vae = vae.to(self._pipeline.device)

            if progress_callback:
                progress_callback("Creating img2img pipeline...", 0.7)

            # Create img2img pipeline from the same components
            img2img_class = StableDiffusionXLImg2ImgPipeline if self._is_sdxl else StableDiffusionImg2ImgPipeline
            self._img2img_pipeline = img2img_class(
                vae=self._pipeline.vae,
                text_encoder=self._pipeline.text_encoder,
                text_encoder_2=self._pipeline.text_encoder_2 if self._is_sdxl else None,
                tokenizer=self._pipeline.tokenizer,
                tokenizer_2=self._pipeline.tokenizer_2 if self._is_sdxl else None,
                unet=self._pipeline.unet,
                scheduler=self._pipeline.scheduler,
            ) if self._is_sdxl else img2img_class(
                vae=self._pipeline.vae,
                text_encoder=self._pipeline.text_encoder,
                tokenizer=self._pipeline.tokenizer,
                unet=self._pipeline.unet,
                scheduler=self._pipeline.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            _log("Img2img pipeline created")

            if progress_callback:
                progress_callback("Creating inpaint pipeline...", 0.75)

            # Create inpaint pipeline from the same components
            inpaint_class = StableDiffusionXLInpaintPipeline if self._is_sdxl else StableDiffusionInpaintPipeline
            if self._is_sdxl:
                self._inpaint_pipeline = inpaint_class(
                    vae=self._pipeline.vae,
                    text_encoder=self._pipeline.text_encoder,
                    text_encoder_2=self._pipeline.text_encoder_2,
                    tokenizer=self._pipeline.tokenizer,
                    tokenizer_2=self._pipeline.tokenizer_2,
                    unet=self._pipeline.unet,
                    scheduler=self._pipeline.scheduler,
                )
            else:
                self._inpaint_pipeline = inpaint_class(
                    vae=self._pipeline.vae,
                    text_encoder=self._pipeline.text_encoder,
                    tokenizer=self._pipeline.tokenizer,
                    unet=self._pipeline.unet,
                    scheduler=self._pipeline.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
            _log("Inpaint pipeline created")

            if progress_callback:
                progress_callback("Optimizing model...", 0.8)

            # Enable optimizations
            self._enable_optimizations(progress_callback)

            # Apply torch.compile if using compiled cache
            if should_compile and cache_dir:
                # Re-apply cache settings right before compile to ensure they're active
                self._setup_inductor_cache(cache_dir)
                if progress_callback:
                    progress_callback("Applying torch.compile with cached kernels...", 0.9)
                _log("Applying torch.compile to UNet (using cached kernels)...")
                # Reset dynamo state to avoid conflicts when loading on multiple GPUs
                torch._dynamo.reset()
                # Use default mode - CUDA graphs are disabled via environment variables
                self._pipeline.unet = torch.compile(
                    self._pipeline.unet,
                    fullgraph=False,
                )
                self._is_compiled = True
                _log("torch.compile applied - cached kernels will be used")
            else:
                self._is_compiled = False

            self._loaded_checkpoint = checkpoint_path
            self._loaded_vae = vae_path

            # Sync CUDA to ensure memory is allocated
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Print VRAM usage after loading
            for i in self._gpu_indices:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    _log(f"GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

            if progress_callback:
                progress_callback("Model loaded successfully", 1.0)

            return True

        except Exception as e:
            _log(f"Error loading model: {e}")
            self.unload_model()
            if progress_callback:
                progress_callback(f"Error: {e}", 0.0)
            return False

    def _setup_inductor_cache(self, cache_dir: Path) -> None:
        """Set up PyTorch inductor cache directory for torch.compile.

        Args:
            cache_dir: Path to the cache directory
        """
        import os

        # Ensure directory exists with proper structure
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables for inductor cache
        # These are the primary way to configure the cache in PyTorch 2.9+
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

        # Also set TORCH_COMPILE_CACHE_DIR as fallback
        os.environ["TORCH_COMPILE_CACHE_DIR"] = str(cache_dir)

        _log(f"Inductor cache directory set to: {cache_dir}")
        _log(f"Directory exists: {cache_dir.exists()}")

    def compile_model(
        self,
        checkpoint_path: str,
        model_type: str = "sdxl",
        vae_path: Optional[str] = None,
        target_width: int = 1024,
        target_height: int = 1024,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        Compile a model using torch.compile and run warm-up to cache compiled kernels.

        This uses PyTorch's persistent cache to store compiled kernels, so subsequent
        loads of the same model will be faster.

        Args:
            checkpoint_path: Path to the checkpoint file
            model_type: Type of model ("sdxl", "sd15", "sd")
            vae_path: Optional path to separate VAE
            target_width: Width for warm-up generation (to compile for this size)
            target_height: Height for warm-up generation (to compile for this size)
            progress_callback: Callback for progress updates (message, progress 0-1)

        Returns:
            True if compilation succeeded
        """
        import os
        import hashlib

        try:
            # Set up persistent torch.compile cache directory
            cache_base = Path(checkpoint_path).parent.parent / "compiled"
            cache_base.mkdir(parents=True, exist_ok=True)

            # Create a unique cache key based on model path and size
            cache_key = hashlib.md5(
                f"{checkpoint_path}_{target_width}x{target_height}".encode()
            ).hexdigest()[:12]

            cache_dir = cache_base / cache_key

            # Set up inductor cache BEFORE loading model
            self._setup_inductor_cache(cache_dir)

            _log(f"Compilation cache directory: {cache_dir}")

            if progress_callback:
                progress_callback("Loading model for compilation...", 0.1)

            # Load the model first
            success = self.load_model(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                vae_path=vae_path,
                progress_callback=lambda msg, prog: (
                    progress_callback(msg, 0.1 + prog * 0.3) if progress_callback else None
                ),
            )

            if not success:
                _log("Failed to load model for compilation")
                return False

            # Re-apply inductor cache settings after model load (in case load_model reset anything)
            self._setup_inductor_cache(cache_dir)

            if progress_callback:
                progress_callback("Applying torch.compile to UNet...", 0.45)

            # Apply torch.compile to the UNet (the main computation bottleneck)
            _log("Applying torch.compile to UNet (default mode, no CUDA graphs)...")

            # Reset dynamo state to ensure clean compilation
            torch._dynamo.reset()

            # Use default mode - CUDA graphs are disabled via environment variables
            self._pipeline.unet = torch.compile(
                self._pipeline.unet,
                fullgraph=False,
            )

            _log("torch.compile applied to UNet")

            if progress_callback:
                progress_callback("Running warm-up generation (this triggers compilation)...", 0.5)

            # Run a warm-up generation to trigger compilation
            # This is where the actual compilation happens and gets cached
            _log(f"Starting warm-up generation at {target_width}x{target_height}...")

            # Use a simple prompt for warm-up
            warmup_prompt = "a test image"
            warmup_negative = ""

            # Encode prompts
            self._encode_prompts(warmup_prompt, warmup_negative)

            # Create generator for reproducibility
            generator = torch.Generator(device="cpu").manual_seed(12345)

            # Build generation kwargs
            # IMPORTANT: Use guidance_scale > 1.0 to trigger CFG which doubles batch size
            # This ensures compilation happens for the same shapes used in real generation
            gen_kwargs = {
                "width": target_width,
                "height": target_height,
                "num_inference_steps": 4,  # Minimal steps for warm-up
                "guidance_scale": 7.0,  # Use realistic CFG to trigger batch doubling
                "generator": generator,
            }

            # Use cached embeddings
            if self._is_sdxl:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds
                gen_kwargs["pooled_prompt_embeds"] = self._cached_pooled_prompt_embeds
                gen_kwargs["negative_pooled_prompt_embeds"] = self._cached_negative_pooled_prompt_embeds
            else:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                if self._cached_negative_prompt_embeds is not None:
                    gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds

            # Track compilation progress
            compilation_steps = 0
            def warmup_callback(pipeline, step, timestep, callback_kwargs):
                nonlocal compilation_steps
                compilation_steps += 1
                if progress_callback:
                    # Compilation progress from 0.5 to 0.95
                    prog = 0.5 + (compilation_steps / 4) * 0.45
                    progress_callback(f"Compiling... (step {compilation_steps}/4)", prog)
                return callback_kwargs

            gen_kwargs["callback_on_step_end"] = warmup_callback

            _log("Calling pipeline for warm-up (compilation will happen now)...")
            start_time = datetime.now()

            # This call triggers the actual compilation
            _ = self._pipeline(**gen_kwargs)

            elapsed = (datetime.now() - start_time).total_seconds()
            _log(f"Warm-up generation complete in {elapsed:.1f}s")

            # Clear the warm-up prompt cache so user prompts work correctly
            self._clear_prompt_cache()

            # Sync CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if progress_callback:
                progress_callback("Model compiled successfully!", 1.0)

            # Create marker file to indicate successful compilation
            marker_file = cache_dir / ".compile_complete"
            marker_file.write_text(f"Compiled at {datetime.now().isoformat()}\n"
                                   f"Resolution: {target_width}x{target_height}\n"
                                   f"Checkpoint: {checkpoint_path}\n")

            _log(f"Model compilation complete. Cache saved to: {cache_dir}")
            _log(f"Cache directory exists: {cache_dir.exists()}")

            # Log cache directory contents
            if cache_dir.exists():
                cache_files = list(cache_dir.iterdir())
                _log(f"Cache directory contains {len(cache_files)} items")
                for f in cache_files[:10]:  # Show first 10
                    _log(f"  - {f.name}")
                if len(cache_files) > 10:
                    _log(f"  ... and {len(cache_files) - 10} more")
            else:
                _log("WARNING: Cache directory does not exist after compilation!")

            _log("The compiled model is now loaded and ready for fast generation.")

            return True

        except Exception as e:
            _log(f"Error during model compilation: {e}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(f"Compilation error: {e}", 0.0)
            return False

    def _enable_optimizations(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """Enable performance optimizations for the pipeline."""
        if self._pipeline is None:
            return

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Enable cuDNN benchmark for faster convolutions
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster matrix operations on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _log("Enabled CUDA optimizations (cuDNN benchmark, TF32)")

        # Try to enable xformers memory efficient attention (fastest option)
        xformers_enabled = False
        try:
            self._pipeline.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
            _log("Enabled xformers memory efficient attention")
        except Exception as e:
            _log(f"xformers not available: {e}")

        # If xformers not available, try PyTorch 2.0 SDPA
        if not xformers_enabled:
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self._pipeline.unet.set_attn_processor(AttnProcessor2_0())
                _log("Enabled PyTorch 2.0 SDPA attention")
            except Exception as e:
                _log(f"Could not enable SDPA: {e}")

        # DISABLE memory optimizations that slow down generation
        # These trade speed for memory - we want speed
        try:
            self._pipeline.disable_attention_slicing()
            _log("Disabled attention slicing (for speed)")
        except Exception:
            pass

        try:
            if hasattr(self._pipeline.vae, 'disable_slicing'):
                self._pipeline.vae.disable_slicing()
            if hasattr(self._pipeline.vae, 'disable_tiling'):
                self._pipeline.vae.disable_tiling()
            _log("Disabled VAE slicing/tiling (for speed)")
        except Exception:
            pass

        # Set UNet and VAE to channels_last memory format for better GPU performance
        try:
            self._pipeline.unet.to(memory_format=torch.channels_last)
            _log("Enabled channels_last memory format for UNet")
        except Exception as e:
            _log(f"Could not set channels_last: {e}")

        # Fuse QKV projections for faster attention (if supported)
        try:
            self._pipeline.fuse_qkv_projections()
            _log("Fused QKV projections")
        except Exception:
            pass

    def unload_model(self) -> None:
        """Unload the current model and free VRAM."""

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if self._img2img_pipeline is not None:
            del self._img2img_pipeline
            self._img2img_pipeline = None

        if self._inpaint_pipeline is not None:
            del self._inpaint_pipeline
            self._inpaint_pipeline = None

        self._loaded_checkpoint = None
        self._loaded_vae = None
        self._is_sdxl = False
        self._loaded_loras = []
        self._is_compiled = False

        # Clear prompt embedding cache
        self._clear_prompt_cache()

        # Force garbage collection and CUDA cache clear
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_loras(
        self,
        loras: list[tuple[str, str, float]],  # List of (path, name, weight) tuples
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        Load and apply LoRAs to the pipeline.

        Args:
            loras: List of (path, name, weight) tuples for LoRAs to load
            progress_callback: Callback for progress updates

        Returns:
            True if successful
        """
        if not self.is_loaded:
            _log("Cannot load LoRAs: no model loaded")
            return False

        if not loras:
            # No LoRAs to load, unload any existing ones
            self.unload_loras()
            return True

        try:
            # First, unload any existing LoRAs
            self.unload_loras(notify=False)

            total = len(loras)
            adapter_names = []
            adapter_weights = []

            for i, (lora_path, lora_name, weight) in enumerate(loras):
                if progress_callback:
                    progress_callback(f"Loading LoRA {i + 1}/{total}: {lora_name}", (i / total) * 0.5)

                # Create a unique adapter name based on the LoRA file name
                adapter_name = f"lora_{i}_{Path(lora_path).stem}"

                _log(f"Loading LoRA: {lora_name} (weight: {weight}) from {lora_path}")

                # Load the LoRA weights
                self._pipeline.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name,
                )

                adapter_names.append(adapter_name)
                adapter_weights.append(weight)
                self._loaded_loras.append((lora_path, weight))

            # Set the adapter weights
            if adapter_names:
                if progress_callback:
                    progress_callback("Applying LoRA weights...", 0.8)

                # Set active adapters with weights
                self._pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

                # Also apply to img2img and inpaint pipelines if they exist
                # Note: They share the same UNet, so LoRAs are already loaded
                # But we need to set adapters on them too
                if self._img2img_pipeline is not None:
                    try:
                        self._img2img_pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    except Exception:
                        pass

                if self._inpaint_pipeline is not None:
                    try:
                        self._inpaint_pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    except Exception:
                        pass

            # Clear prompt cache since LoRAs can affect text encoding
            self._clear_prompt_cache()

            if progress_callback:
                progress_callback(f"Loaded {len(loras)} LoRA(s)", 1.0)

            _log(f"Successfully loaded {len(loras)} LoRA(s)")
            return True

        except Exception as e:
            _log(f"Error loading LoRAs: {e}")
            import traceback
            traceback.print_exc()
            # Try to unload any partially loaded LoRAs
            self.unload_loras()
            return False

    def unload_loras(self, notify: bool = True) -> None:
        """Unload all currently loaded LoRAs."""
        if not self.is_loaded or not self._loaded_loras:
            self._loaded_loras = []
            return

        try:
            # Unload all LoRA adapters
            self._pipeline.unload_lora_weights()

            if self._img2img_pipeline is not None:
                try:
                    self._img2img_pipeline.unload_lora_weights()
                except Exception:
                    pass

            if self._inpaint_pipeline is not None:
                try:
                    self._inpaint_pipeline.unload_lora_weights()
                except Exception:
                    pass

            if notify:
                _log("Unloaded all LoRAs")
        except Exception as e:
            # This can happen if no LoRAs were ever loaded, which is fine
            if "PEFT" not in str(e):
                _log(f"Error unloading LoRAs: {e}")

        self._loaded_loras = []

        # Clear prompt cache since LoRAs can affect text encoding
        self._clear_prompt_cache()

    @property
    def loaded_loras(self) -> list[tuple[str, float]]:
        """Get list of currently loaded LoRAs as (path, weight) tuples."""
        return self._loaded_loras.copy()

    def _get_scheduler(self, sampler: str, scheduler_type: str = "normal") -> Any:
        """Get the appropriate scheduler for the given sampler name and schedule type.

        Args:
            sampler: The sampler name (e.g., "euler", "dpmpp_2m")
            scheduler_type: The schedule type ("normal", "karras", "exponential", "sgm_uniform")
        """
        scheduler_name = SAMPLERS.get(sampler, "EulerDiscreteScheduler")
        scheduler_class = self.SCHEDULER_CLASSES.get(
            scheduler_name, EulerDiscreteScheduler
        )

        # Get scheduler config from pipeline
        config = self._pipeline.scheduler.config

        # Create scheduler with appropriate settings
        kwargs = {}

        # Apply schedule type
        if scheduler_type == "simple":
            kwargs["timestep_spacing"] = "linspace"
        elif scheduler_type == "karras":
            kwargs["use_karras_sigmas"] = True
        elif scheduler_type == "exponential":
            kwargs["use_exponential_sigmas"] = True
        elif scheduler_type == "sgm_uniform":
            kwargs["timestep_spacing"] = "trailing"
        # "normal" uses defaults

        return scheduler_class.from_config(config, **kwargs)

    def _encode_prompts(self, prompt: str, negative_prompt: Optional[str]) -> bool:
        """
        Encode prompts into embeddings.
        Always re-encodes to support dynamic prompt generation (e.g., random word selection).
        """
        negative_prompt = negative_prompt or ""

        _log("Encoding prompts...")

        try:
            if self._is_sdxl:
                # SDXL uses dual text encoders
                (
                    self._cached_prompt_embeds,
                    self._cached_negative_prompt_embeds,
                    self._cached_pooled_prompt_embeds,
                    self._cached_negative_pooled_prompt_embeds,
                ) = self._pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device=self._device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=None,
                )
            else:
                # SD 1.5 uses single text encoder
                # encode_prompt returns (prompt_embeds, negative_prompt_embeds) when do_classifier_free_guidance=True
                prompt_embeds, negative_prompt_embeds = self._pipeline.encode_prompt(
                    prompt=prompt,
                    device=self._device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                self._cached_prompt_embeds = prompt_embeds
                self._cached_negative_prompt_embeds = negative_prompt_embeds
                self._cached_pooled_prompt_embeds = None
                self._cached_negative_pooled_prompt_embeds = None

            # Store the prompts we encoded
            self._cached_prompt = prompt
            self._cached_negative_prompt = negative_prompt

            _log("Prompt encoding complete")
            return True

        except Exception as e:
            _log(f"Error encoding prompts: {e}")
            # Clear cache on error
            self._clear_prompt_cache()
            raise

    def _clear_prompt_cache(self) -> None:
        """Clear the cached prompt embeddings."""
        self._cached_prompt = None
        self._cached_negative_prompt = None
        self._cached_prompt_embeds = None
        self._cached_negative_prompt_embeds = None
        self._cached_pooled_prompt_embeds = None
        self._cached_negative_pooled_prompt_embeds = None

    def generate(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Image.Image]:
        """
        Generate an image using the loaded model.

        Args:
            params: Generation parameters
            progress_callback: Callback for step progress (current_step, total_steps)

        Returns:
            Generated PIL Image or None if generation failed
        """
        if not self.is_loaded:
            _log("No model loaded")
            return None

        try:
            # Set up scheduler
            self._pipeline.scheduler = self._get_scheduler(params.sampler, params.scheduler)

            # Encode prompts (uses cache if prompts haven't changed)
            self._encode_prompts(params.prompt, params.negative_prompt)

            # Handle seed
            if params.seed == -1:
                actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            else:
                actual_seed = params.seed
            generator = torch.Generator(device="cpu").manual_seed(actual_seed)

            self._generation_count += 1
            current_resolution = (params.width, params.height)
            resolution_changed = self._last_resolution != current_resolution
            self._last_resolution = current_resolution

            import threading
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name

            _log(f"Starting generation: {params.width}x{params.height}, {params.steps} steps (seed: {actual_seed})")
            _log(f"Generation #{self._generation_count}, Resolution changed: {resolution_changed}, Compiled: {self._is_compiled}")
            _log(f"Worker thread: {thread_name} (id: {thread_id})")

            # Create progress callback wrapper
            callback_fn = None
            first_step_logged = False
            if progress_callback:
                def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                    nonlocal first_step_logged
                    if not first_step_logged:
                        _log(f"First step complete (step {step + 1})")
                        first_step_logged = True
                    progress_callback(step + 1, params.steps)
                    return callback_kwargs
                callback_fn = callback_on_step_end

            # Build generation kwargs with cached embeddings
            gen_kwargs = {
                "width": params.width,
                "height": params.height,
                "num_inference_steps": params.steps,
                "guidance_scale": params.cfg_scale,
                "generator": generator,
                "callback_on_step_end": callback_fn,
            }

            # Use cached embeddings instead of text prompts
            if self._is_sdxl:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds
                gen_kwargs["pooled_prompt_embeds"] = self._cached_pooled_prompt_embeds
                gen_kwargs["negative_pooled_prompt_embeds"] = self._cached_negative_pooled_prompt_embeds
            else:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                if self._cached_negative_prompt_embeds is not None:
                    gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds

            # Generate image
            import time
            _log("Calling pipeline...")
            pipeline_start = time.time()
            result = self._pipeline(**gen_kwargs)
            pipeline_end = time.time()
            _log(f"Pipeline returned in {pipeline_end - pipeline_start:.3f}s")

            # Synchronize CUDA to ensure all GPU operations complete
            # This is important for compiled models using CUDA graphs
            if torch.cuda.is_available():
                sync_start = time.time()
                torch.cuda.synchronize()
                sync_end = time.time()
                _log(f"CUDA sync took {sync_end - sync_start:.3f}s")

            _log("Generation complete")

            if result.images:
                return result.images[0]
            return None

        except Exception as e:
            import traceback
            _log(f"Error during generation: {e}")
            traceback.print_exc()
            return None

    def generate_img2img(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        strength: float = 0.75,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Image.Image]:
        """
        Generate an image using img2img with the loaded model.

        Args:
            params: Generation parameters
            input_image: Input image to transform
            strength: How much to transform (0=no change, 1=full generation)
            progress_callback: Callback for step progress (current_step, total_steps)

        Returns:
            Generated PIL Image or None if generation failed
        """
        if not self.is_loaded or self._img2img_pipeline is None:
            _log("No model loaded for img2img")
            return None

        try:
            # Set up scheduler
            self._img2img_pipeline.scheduler = self._get_scheduler(params.sampler, params.scheduler)

            # Encode prompts (uses cache if prompts haven't changed)
            self._encode_prompts(params.prompt, params.negative_prompt)

            # Handle seed - use CPU generator for reliability
            if params.seed == -1:
                generator = None
            else:
                generator = torch.Generator(device="cpu").manual_seed(params.seed)

            # Calculate effective steps (img2img uses fewer steps based on strength)
            effective_steps = max(1, int(params.steps * strength))

            # Create progress callback wrapper (only if callback provided)
            callback_fn = None
            if progress_callback:
                def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                    progress_callback(step + 1, effective_steps)
                    return callback_kwargs
                callback_fn = callback_on_step_end

            _log(f"Starting img2img: {input_image.width}x{input_image.height} -> {params.width}x{params.height}, strength={strength}, {params.steps} steps")

            # Resize input image to target size if needed
            if input_image.width != params.width or input_image.height != params.height:
                input_image = input_image.resize((params.width, params.height), Image.Resampling.LANCZOS)
                _log(f"Resized input image to {params.width}x{params.height}")

            # Build generation kwargs with cached embeddings
            gen_kwargs = {
                "image": input_image,
                "strength": strength,
                "num_inference_steps": params.steps,
                "guidance_scale": params.cfg_scale,
                "generator": generator,
                "callback_on_step_end": callback_fn,
            }

            # Use cached embeddings instead of text prompts
            if self._is_sdxl:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds
                gen_kwargs["pooled_prompt_embeds"] = self._cached_pooled_prompt_embeds
                gen_kwargs["negative_pooled_prompt_embeds"] = self._cached_negative_pooled_prompt_embeds
            else:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                if self._cached_negative_prompt_embeds is not None:
                    gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds

            # Generate image
            result = self._img2img_pipeline(**gen_kwargs)

            # Synchronize CUDA to ensure all GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            _log("Img2img generation complete")

            if result.images:
                return result.images[0]
            return None

        except Exception as e:
            import traceback
            _log(f"Error during img2img generation: {e}")
            traceback.print_exc()
            return None

    def generate_inpaint(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        mask_image: Image.Image,
        strength: float = 0.75,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Image.Image]:
        """
        Generate an inpainted image using the loaded model.

        Args:
            params: Generation parameters
            input_image: Input image to inpaint
            mask_image: Mask image (white = inpaint, black = keep)
            strength: How much to transform masked areas (0=no change, 1=full generation)
            progress_callback: Callback for step progress (current_step, total_steps)

        Returns:
            Generated PIL Image or None if generation failed
        """
        if not self.is_loaded or self._inpaint_pipeline is None:
            _log("No model loaded for inpainting")
            return None

        try:
            # Set up scheduler
            self._inpaint_pipeline.scheduler = self._get_scheduler(params.sampler, params.scheduler)

            # Encode prompts (uses cache if prompts haven't changed)
            self._encode_prompts(params.prompt, params.negative_prompt)

            # Handle seed - use CPU generator for reliability
            if params.seed == -1:
                generator = None
            else:
                generator = torch.Generator(device="cpu").manual_seed(params.seed)

            # Calculate effective steps
            effective_steps = max(1, int(params.steps * strength))

            # Create progress callback wrapper (only if callback provided)
            callback_fn = None
            if progress_callback:
                def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                    progress_callback(step + 1, effective_steps)
                    return callback_kwargs
                callback_fn = callback_on_step_end

            # For inpainting, use the original image dimensions (not params dimensions)
            target_width = input_image.width
            target_height = input_image.height

            _log(f"Starting inpaint: {target_width}x{target_height}, strength={strength}, {params.steps} steps")

            # Ensure images are RGB
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            # Ensure mask is the right format (L mode for grayscale)
            if mask_image.mode != "L":
                mask_image = mask_image.convert("L")

            # Ensure mask matches input image size
            if mask_image.size != input_image.size:
                mask_image = mask_image.resize(input_image.size, Image.Resampling.LANCZOS)
                _log(f"Resized mask to match input image: {input_image.width}x{input_image.height}")

            # Build generation kwargs with cached embeddings
            gen_kwargs = {
                "image": input_image,
                "mask_image": mask_image,
                "width": target_width,
                "height": target_height,
                "strength": strength,
                "num_inference_steps": params.steps,
                "guidance_scale": params.cfg_scale,
                "generator": generator,
                "callback_on_step_end": callback_fn,
            }

            # Use cached embeddings instead of text prompts
            if self._is_sdxl:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds
                gen_kwargs["pooled_prompt_embeds"] = self._cached_pooled_prompt_embeds
                gen_kwargs["negative_pooled_prompt_embeds"] = self._cached_negative_pooled_prompt_embeds
            else:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                if self._cached_negative_prompt_embeds is not None:
                    gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds

            # Generate image
            result = self._inpaint_pipeline(**gen_kwargs)

            # Synchronize CUDA to ensure all GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            _log("Inpaint generation complete")

            if result.images:
                return result.images[0]
            return None

        except Exception as e:
            import traceback
            _log(f"Error during inpaint generation: {e}")
            traceback.print_exc()
            return None

    def get_actual_seed(self, params: GenerationParams) -> int:
        """Get the actual seed that will be used (resolves -1 to random)."""
        if params.seed == -1:
            return torch.randint(0, 2**32 - 1, (1,)).item()
        return params.seed

    def generate_outpaint(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        extensions: dict,
        strength: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Image.Image]:
        """
        Generate an outpainted image by extending the image in specified directions.

        The working canvas is kept approximately the same size as the original by
        cropping from the opposite side of each extension. After generation, the
        result is composited back onto the full-size output.

        Args:
            params: Generation parameters
            input_image: Input image to outpaint
            extensions: Dict with 'left', 'right', 'top', 'bottom' extension amounts in pixels
            strength: How much to transform extended areas (0=no change, 1=full generation)
            progress_callback: Callback for step progress (current_step, total_steps)

        Returns:
            Generated PIL Image (extended) or None if generation failed
        """
        if not self.is_loaded or self._inpaint_pipeline is None:
            _log("No model loaded for outpainting")
            return None

        try:
            import numpy as np

            # Extract extension amounts
            left_ext = extensions.get('left', 0)
            right_ext = extensions.get('right', 0)
            top_ext = extensions.get('top', 0)
            bottom_ext = extensions.get('bottom', 0)

            orig_width = input_image.width
            orig_height = input_image.height

            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            # Calculate how much to crop from opposite sides to keep working canvas manageable
            # We want the working canvas to be approximately the original size
            crop_left = right_ext  # If extending right, crop from left
            crop_right = left_ext  # If extending left, crop from right
            crop_top = bottom_ext  # If extending bottom, crop from top
            crop_bottom = top_ext  # If extending top, crop from bottom

            # Ensure we don't crop more than the image size minus a minimum overlap
            min_overlap = 64  # Minimum pixels of original image to keep for context
            crop_left = min(crop_left, orig_width - min_overlap)
            crop_right = min(crop_right, orig_width - min_overlap - crop_left)
            crop_top = min(crop_top, orig_height - min_overlap)
            crop_bottom = min(crop_bottom, orig_height - min_overlap - crop_top)

            # Ensure crops are non-negative
            crop_left = max(0, crop_left)
            crop_right = max(0, crop_right)
            crop_top = max(0, crop_top)
            crop_bottom = max(0, crop_bottom)

            # Crop the original image
            cropped_image = input_image.crop((
                crop_left,
                crop_top,
                orig_width - crop_right,
                orig_height - crop_bottom
            ))
            cropped_width = cropped_image.width
            cropped_height = cropped_image.height

            _log(f"Original: {orig_width}x{orig_height}, Cropped: {cropped_width}x{cropped_height}")
            _log(f"Crops: left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom}")

            # Calculate working canvas dimensions
            work_width = cropped_width + left_ext + right_ext
            work_height = cropped_height + top_ext + bottom_ext

            # Round to be divisible by 8
            work_width = ((work_width + 7) // 8) * 8
            work_height = ((work_height + 7) // 8) * 8

            # Adjust extensions to match rounded dimensions
            width_diff = work_width - (cropped_width + left_ext + right_ext)
            height_diff = work_height - (cropped_height + top_ext + bottom_ext)

            if width_diff > 0:
                if right_ext > 0:
                    right_ext += width_diff
                elif left_ext > 0:
                    left_ext += width_diff
                else:
                    right_ext = width_diff

            if height_diff > 0:
                if bottom_ext > 0:
                    bottom_ext += height_diff
                elif top_ext > 0:
                    top_ext += height_diff
                else:
                    bottom_ext = height_diff

            _log(f"Working canvas: {work_width}x{work_height}")
            _log(f"Extensions (adjusted): left={left_ext}, right={right_ext}, top={top_ext}, bottom={bottom_ext}")

            # Create working canvas
            work_image = Image.new("RGB", (work_width, work_height), (128, 128, 128))

            # Fill extension areas with edge colors for better blending
            if left_ext > 0:
                left_strip = cropped_image.crop((0, 0, min(10, cropped_width), cropped_height))
                for x in range(left_ext):
                    work_image.paste(left_strip.resize((1, cropped_height)), (x, top_ext))
            if right_ext > 0:
                right_strip = cropped_image.crop((max(0, cropped_width - 10), 0, cropped_width, cropped_height))
                for x in range(right_ext):
                    work_image.paste(right_strip.resize((1, cropped_height)), (left_ext + cropped_width + x, top_ext))
            if top_ext > 0:
                top_strip = cropped_image.crop((0, 0, cropped_width, min(10, cropped_height)))
                for y in range(top_ext):
                    work_image.paste(top_strip.resize((cropped_width, 1)), (left_ext, y))
            if bottom_ext > 0:
                bottom_strip = cropped_image.crop((0, max(0, cropped_height - 10), cropped_width, cropped_height))
                for y in range(bottom_ext):
                    work_image.paste(bottom_strip.resize((cropped_width, 1)), (left_ext, top_ext + cropped_height + y))

            # Paste cropped image onto working canvas
            work_image.paste(cropped_image, (left_ext, top_ext))

            # Create mask (white = areas to generate, black = keep original)
            mask_image = Image.new("L", (work_width, work_height), 255)
            mask_draw = ImageDraw.Draw(mask_image)
            mask_draw.rectangle(
                [left_ext, top_ext, left_ext + cropped_width, top_ext + cropped_height],
                fill=0
            )

            # Add feathering at the edges for smooth blending
            active_exts = [e for e in [left_ext, right_ext, top_ext, bottom_ext] if e > 0]
            feather_size = min(32, min(active_exts) if active_exts else 32)

            if feather_size > 0:
                mask_arr = np.array(mask_image)

                if left_ext > 0:
                    for i in range(min(feather_size, cropped_width)):
                        alpha = int(255 * (1 - i / feather_size))
                        mask_arr[top_ext:top_ext + cropped_height, left_ext + i] = alpha
                if right_ext > 0:
                    for i in range(min(feather_size, cropped_width)):
                        alpha = int(255 * (1 - i / feather_size))
                        mask_arr[top_ext:top_ext + cropped_height, left_ext + cropped_width - 1 - i] = alpha
                if top_ext > 0:
                    for i in range(min(feather_size, cropped_height)):
                        alpha = int(255 * (1 - i / feather_size))
                        mask_arr[top_ext + i, left_ext:left_ext + cropped_width] = np.maximum(
                            mask_arr[top_ext + i, left_ext:left_ext + cropped_width], alpha
                        )
                if bottom_ext > 0:
                    for i in range(min(feather_size, cropped_height)):
                        alpha = int(255 * (1 - i / feather_size))
                        mask_arr[top_ext + cropped_height - 1 - i, left_ext:left_ext + cropped_width] = np.maximum(
                            mask_arr[top_ext + cropped_height - 1 - i, left_ext:left_ext + cropped_width], alpha
                        )

                mask_image = Image.fromarray(mask_arr)

            # Set up scheduler and encode prompts
            self._inpaint_pipeline.scheduler = self._get_scheduler(params.sampler, params.scheduler)
            self._encode_prompts(params.prompt, params.negative_prompt)

            # Handle seed
            generator = None if params.seed == -1 else torch.Generator(device="cpu").manual_seed(params.seed)

            # Calculate effective steps
            effective_steps = max(1, int(params.steps * strength))

            # Create progress callback wrapper
            callback_fn = None
            if progress_callback:
                def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                    progress_callback(step + 1, effective_steps)
                    return callback_kwargs
                callback_fn = callback_on_step_end

            # Build generation kwargs
            gen_kwargs = {
                "image": work_image,
                "mask_image": mask_image,
                "width": work_width,
                "height": work_height,
                "strength": strength,
                "num_inference_steps": params.steps,
                "guidance_scale": params.cfg_scale,
                "generator": generator,
                "callback_on_step_end": callback_fn,
            }

            # Use cached embeddings
            if self._is_sdxl:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds
                gen_kwargs["pooled_prompt_embeds"] = self._cached_pooled_prompt_embeds
                gen_kwargs["negative_pooled_prompt_embeds"] = self._cached_negative_pooled_prompt_embeds
            else:
                gen_kwargs["prompt_embeds"] = self._cached_prompt_embeds
                if self._cached_negative_prompt_embeds is not None:
                    gen_kwargs["negative_prompt_embeds"] = self._cached_negative_prompt_embeds

            # Generate on working canvas
            result = self._inpaint_pipeline(**gen_kwargs)

            # Synchronize CUDA to ensure all GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if not result.images:
                return None

            generated = result.images[0]

            # Now composite the generated result back onto a full-size output
            # Final dimensions = original + all extensions
            orig_left_ext = extensions.get('left', 0)
            orig_right_ext = extensions.get('right', 0)
            orig_top_ext = extensions.get('top', 0)
            orig_bottom_ext = extensions.get('bottom', 0)

            final_width = orig_width + orig_left_ext + orig_right_ext
            final_height = orig_height + orig_top_ext + orig_bottom_ext

            # No rounding needed for final output - only the working canvas needs divisible by 8
            final_image = Image.new("RGB", (final_width, final_height), (128, 128, 128))

            # Paste original image at its position (offset by left/top extensions)
            final_image.paste(input_image, (orig_left_ext, orig_top_ext))

            # Blend size for smooth transitions
            blend_size = 48

            # Helper function to create gradient alpha blend
            def blend_regions(base_img, overlay_img, paste_x, paste_y, direction):
                """Blend overlay onto base with gradient at the edge."""
                overlay_w, overlay_h = overlay_img.size

                # Create alpha mask with gradient
                alpha = Image.new("L", (overlay_w, overlay_h), 255)
                alpha_arr = np.array(alpha)

                if direction == "left":
                    # Gradient on right edge (fades into original)
                    for i in range(min(blend_size, overlay_w)):
                        alpha_arr[:, overlay_w - 1 - i] = int(255 * i / blend_size)
                elif direction == "right":
                    # Gradient on left edge
                    for i in range(min(blend_size, overlay_w)):
                        alpha_arr[:, i] = int(255 * i / blend_size)
                elif direction == "top":
                    # Gradient on bottom edge
                    for i in range(min(blend_size, overlay_h)):
                        alpha_arr[overlay_h - 1 - i, :] = int(255 * i / blend_size)
                elif direction == "bottom":
                    # Gradient on top edge
                    for i in range(min(blend_size, overlay_h)):
                        alpha_arr[i, :] = int(255 * i / blend_size)

                alpha = Image.fromarray(alpha_arr)

                # Get the region from base that will be blended
                base_region = base_img.crop((
                    paste_x, paste_y,
                    paste_x + overlay_w, paste_y + overlay_h
                ))

                # Composite using alpha
                blended = Image.composite(overlay_img, base_region, alpha)
                base_img.paste(blended, (paste_x, paste_y))

            # Extract and blend the generated extension areas
            # Include overlap region for blending
            overlap = blend_size

            # Left extension with overlap
            if left_ext > 0:
                # Extract extension + overlap from generated
                ext_with_overlap = min(left_ext + overlap, left_ext + cropped_width)
                left_region = generated.crop((0, top_ext, ext_with_overlap, top_ext + cropped_height))
                paste_y = orig_top_ext + crop_top
                blend_regions(final_image, left_region, 0, paste_y, "left")

            # Right extension with overlap
            if right_ext > 0:
                # Extract overlap + extension from generated
                start_x = max(0, left_ext + cropped_width - overlap)
                right_region = generated.crop((
                    start_x, top_ext,
                    left_ext + cropped_width + right_ext, top_ext + cropped_height
                ))
                paste_x = orig_left_ext + orig_width - overlap
                paste_y = orig_top_ext + crop_top
                blend_regions(final_image, right_region, paste_x, paste_y, "right")

            # Top extension with overlap
            if top_ext > 0:
                ext_with_overlap = min(top_ext + overlap, top_ext + cropped_height)
                top_region = generated.crop((left_ext, 0, left_ext + cropped_width, ext_with_overlap))
                paste_x = orig_left_ext + crop_left
                blend_regions(final_image, top_region, paste_x, 0, "top")

            # Bottom extension with overlap
            if bottom_ext > 0:
                start_y = max(0, top_ext + cropped_height - overlap)
                bottom_region = generated.crop((
                    left_ext, start_y,
                    left_ext + cropped_width, top_ext + cropped_height + bottom_ext
                ))
                paste_x = orig_left_ext + crop_left
                paste_y = orig_top_ext + orig_height - overlap
                blend_regions(final_image, bottom_region, paste_x, paste_y, "bottom")

            _log(f"Outpaint complete: {final_width}x{final_height}")

            return final_image

        except Exception as e:
            import traceback
            _log(f"Error during outpaint generation: {e}")
            traceback.print_exc()
            return None


# Global backend instance
diffusers_backend = DiffusersBackend()
