"""Diffusers backend for Stable Diffusion image generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any
import gc

import torch
from PIL import Image
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

    def __init__(self):
        self._pipeline: Optional[Any] = None
        self._img2img_pipeline: Optional[Any] = None
        self._inpaint_pipeline: Optional[Any] = None
        self._is_sdxl: bool = False
        self._loaded_checkpoint: Optional[str] = None
        self._loaded_vae: Optional[str] = None
        self._gpu_indices: list[int] = [0]
        self._device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        # LoRA tracking
        self._loaded_loras: list[tuple[str, float]] = []  # List of (path, weight) tuples

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._pipeline is not None

    @property
    def is_sdxl(self) -> bool:
        """Check if loaded model is SDXL."""
        return self._is_sdxl

    def set_gpus(self, indices: list[int]) -> None:
        """Set which GPUs to use for generation."""
        self._gpu_indices = indices if indices else [0]

    def load_model(
        self,
        checkpoint_path: str,
        model_type: str = "sdxl",
        vae_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        Load a Stable Diffusion model.

        Args:
            checkpoint_path: Path to the checkpoint file
            model_type: Type of model ("sdxl", "sd15", "sd")
            vae_path: Optional path to separate VAE
            clip_path: Optional path to separate CLIP (not commonly used)
            progress_callback: Callback for progress updates (message, progress 0-1)

        Returns:
            True if loading succeeded
        """
        try:
            if progress_callback:
                progress_callback("Preparing to load model...", 0.0)

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
            print(f"Target device: {self._device}")

            # Load the pipeline
            print(f"Loading model from checkpoint...")
            self._pipeline = pipeline_class.from_single_file(
                checkpoint_path,
                torch_dtype=torch.float16,
                use_safetensors=checkpoint_path.endswith(".safetensors"),
            )

            # Move all components to the same GPU
            print(f"Moving pipeline components to {self._device}...")

            if hasattr(self._pipeline, 'unet') and self._pipeline.unet is not None:
                self._pipeline.unet = self._pipeline.unet.to(self._device)
                print(f"  UNet moved to {self._device}")

            if hasattr(self._pipeline, 'vae') and self._pipeline.vae is not None:
                self._pipeline.vae = self._pipeline.vae.to(self._device)
                print(f"  VAE moved to {self._device}")

            if hasattr(self._pipeline, 'text_encoder') and self._pipeline.text_encoder is not None:
                self._pipeline.text_encoder = self._pipeline.text_encoder.to(self._device)
                print(f"  Text encoder moved to {self._device}")

            if hasattr(self._pipeline, 'text_encoder_2') and self._pipeline.text_encoder_2 is not None:
                self._pipeline.text_encoder_2 = self._pipeline.text_encoder_2.to(self._device)
                print(f"  Text encoder 2 moved to {self._device}")

            print(f"Pipeline device: {self._pipeline.device}")

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
            print("Img2img pipeline created")

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
            print("Inpaint pipeline created")

            if progress_callback:
                progress_callback("Optimizing model...", 0.8)

            # Enable optimizations
            self._enable_optimizations()

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
                    print(f"GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

            if progress_callback:
                progress_callback("Model loaded successfully", 1.0)

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.unload_model()
            if progress_callback:
                progress_callback(f"Error: {e}", 0.0)
            return False

    def _enable_optimizations(self) -> None:
        """Enable memory optimizations for the pipeline."""
        if self._pipeline is None:
            return

        try:
            # Enable attention slicing for lower memory
            self._pipeline.enable_attention_slicing(1)
        except Exception:
            pass

        try:
            # Enable VAE slicing (use new API)
            if hasattr(self._pipeline.vae, 'enable_slicing'):
                self._pipeline.vae.enable_slicing()
            else:
                self._pipeline.enable_vae_slicing()
        except Exception:
            pass

        try:
            # Enable VAE tiling for large images (use new API)
            if hasattr(self._pipeline.vae, 'enable_tiling'):
                self._pipeline.vae.enable_tiling()
            else:
                self._pipeline.enable_vae_tiling()
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
            print("Cannot load LoRAs: no model loaded")
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

                print(f"Loading LoRA: {lora_name} (weight: {weight}) from {lora_path}")

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

            if progress_callback:
                progress_callback(f"Loaded {len(loras)} LoRA(s)", 1.0)

            print(f"Successfully loaded {len(loras)} LoRA(s)")
            return True

        except Exception as e:
            print(f"Error loading LoRAs: {e}")
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
                print("Unloaded all LoRAs")
        except Exception as e:
            # This can happen if no LoRAs were ever loaded, which is fine
            if "PEFT" not in str(e):
                print(f"Error unloading LoRAs: {e}")

        self._loaded_loras = []

    @property
    def loaded_loras(self) -> list[tuple[str, float]]:
        """Get list of currently loaded LoRAs as (path, weight) tuples."""
        return self._loaded_loras.copy()

    def _get_scheduler(self, sampler: str) -> Any:
        """Get the appropriate scheduler for the given sampler name."""
        scheduler_name = SAMPLERS.get(sampler, "EulerDiscreteScheduler")
        scheduler_class = self.SCHEDULER_CLASSES.get(
            scheduler_name, EulerDiscreteScheduler
        )

        # Get scheduler config from pipeline
        config = self._pipeline.scheduler.config

        # Create scheduler with appropriate settings
        kwargs = {}

        # Apply Karras sigmas if needed
        if sampler in KARRAS_SAMPLERS:
            kwargs["use_karras_sigmas"] = True

        return scheduler_class.from_config(config, **kwargs)

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
            print("No model loaded")
            return None

        try:
            # Set up scheduler
            self._pipeline.scheduler = self._get_scheduler(params.sampler)

            # Handle seed - use the device where UNet is located
            if params.seed == -1:
                generator = None  # Random seed
            else:
                # Use the device where UNet is (that's where latents are generated)
                gen_device = self._device
                print(f"Creating generator on device: {gen_device}")
                generator = torch.Generator(device=gen_device).manual_seed(params.seed)

            # Create progress callback wrapper
            def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                if progress_callback:
                    progress_callback(step + 1, params.steps)
                return callback_kwargs

            print(f"Starting generation: {params.width}x{params.height}, {params.steps} steps")

            # Generate image
            result = self._pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt if params.negative_prompt else None,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg_scale,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )

            print("Generation complete")

            if result.images:
                return result.images[0]
            return None

        except Exception as e:
            import traceback
            print(f"Error during generation: {e}")
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
            print("No model loaded for img2img")
            return None

        try:
            # Set up scheduler
            self._img2img_pipeline.scheduler = self._get_scheduler(params.sampler)

            # Handle seed
            if params.seed == -1:
                generator = None
            else:
                gen_device = self._device
                print(f"Creating generator on device: {gen_device}")
                generator = torch.Generator(device=gen_device).manual_seed(params.seed)

            # Calculate effective steps (img2img uses fewer steps based on strength)
            effective_steps = max(1, int(params.steps * strength))

            # Create progress callback wrapper
            def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                if progress_callback:
                    progress_callback(step + 1, effective_steps)
                return callback_kwargs

            print(f"Starting img2img: {input_image.width}x{input_image.height} -> {params.width}x{params.height}, strength={strength}, {params.steps} steps")

            # Resize input image to target size if needed
            if input_image.width != params.width or input_image.height != params.height:
                input_image = input_image.resize((params.width, params.height), Image.Resampling.LANCZOS)
                print(f"Resized input image to {params.width}x{params.height}")

            # Generate image
            result = self._img2img_pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt if params.negative_prompt else None,
                image=input_image,
                strength=strength,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg_scale,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )

            print("Img2img generation complete")

            if result.images:
                return result.images[0]
            return None

        except Exception as e:
            import traceback
            print(f"Error during img2img generation: {e}")
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
            print("No model loaded for inpainting")
            return None

        try:
            # Set up scheduler
            self._inpaint_pipeline.scheduler = self._get_scheduler(params.sampler)

            # Handle seed
            if params.seed == -1:
                generator = None
            else:
                gen_device = self._device
                print(f"Creating generator on device: {gen_device}")
                generator = torch.Generator(device=gen_device).manual_seed(params.seed)

            # Calculate effective steps
            effective_steps = max(1, int(params.steps * strength))

            # Create progress callback wrapper
            def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
                if progress_callback:
                    progress_callback(step + 1, effective_steps)
                return callback_kwargs

            # For inpainting, use the original image dimensions (not params dimensions)
            target_width = input_image.width
            target_height = input_image.height

            print(f"Starting inpaint: {target_width}x{target_height}, strength={strength}, {params.steps} steps")

            # Ensure images are RGB
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            # Ensure mask is the right format (L mode for grayscale)
            if mask_image.mode != "L":
                mask_image = mask_image.convert("L")

            # Ensure mask matches input image size
            if mask_image.size != input_image.size:
                mask_image = mask_image.resize(input_image.size, Image.Resampling.LANCZOS)
                print(f"Resized mask to match input image: {input_image.width}x{input_image.height}")

            # Generate image - explicitly pass dimensions to ensure output matches input
            result = self._inpaint_pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt if params.negative_prompt else None,
                image=input_image,
                mask_image=mask_image,
                width=target_width,
                height=target_height,
                strength=strength,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg_scale,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )

            print("Inpaint generation complete")

            if result.images:
                return result.images[0]
            return None

        except Exception as e:
            import traceback
            print(f"Error during inpaint generation: {e}")
            traceback.print_exc()
            return None

    def get_actual_seed(self, params: GenerationParams) -> int:
        """Get the actual seed that will be used (resolves -1 to random)."""
        if params.seed == -1:
            return torch.randint(0, 2**32 - 1, (1,)).item()
        return params.seed


# Global backend instance
diffusers_backend = DiffusersBackend()
