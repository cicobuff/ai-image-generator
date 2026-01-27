"""Generation service for async image generation with GTK-safe callbacks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from PIL import Image
from gi.repository import GLib

from src.core.config import config_manager
from src.core.model_manager import model_manager
from src.core.gpu_manager import gpu_manager
from src.backends.diffusers_backend import diffusers_backend, GenerationParams
from src.backends.upscale_backend import upscale_backend
from src.backends.segmentation_backend import DetectedMask
from src.utils.constants import OUTPUT_FORMAT
from src.utils.metadata import GenerationMetadata, save_image_with_metadata
import numpy as np


# Type alias for LoRA info: (path, name, weight)
LoRAInfo = tuple[str, str, float]


class GenerationState(Enum):
    """State of the generation service."""
    IDLE = "idle"
    LOADING = "loading"
    GENERATING = "generating"
    CANCELLING = "cancelling"


@dataclass
class GenerationResult:
    """Result of an image generation."""
    success: bool
    image: Optional[Image.Image] = None
    path: Optional[Path] = None
    seed: int = -1
    error: Optional[str] = None


class GenerationService:
    """Service for managing async image generation."""

    def __init__(self):
        self._state = GenerationState.IDLE
        self._cancel_requested = False
        self._current_thread: Optional[threading.Thread] = None

        # Persistent worker thread pool for CUDA warmth between generations.
        # Using a single worker keeps the thread alive, preserving cuDNN benchmark
        # cache and other CUDA state that would be lost if we created new threads.
        self._worker_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen_worker")
        self._current_future = None

        # Track loaded model info for metadata
        self._loaded_checkpoint_name: str = ""
        self._loaded_vae_name: str = ""
        self._loaded_clip_name: str = ""
        self._loaded_model_type: str = ""

        # Callbacks
        self._on_state_changed: list[Callable[[GenerationState], None]] = []
        self._on_progress: list[Callable[[str, float], None]] = []
        self._on_step_progress: list[Callable[[int, int], None]] = []
        self._on_generation_complete: list[Callable[[GenerationResult], None]] = []

    @property
    def state(self) -> GenerationState:
        """Get current generation state."""
        return self._state

    @property
    def is_busy(self) -> bool:
        """Check if service is busy (loading or generating)."""
        return self._state in (GenerationState.LOADING, GenerationState.GENERATING)

    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return diffusers_backend.is_loaded

    # Callback registration
    def add_state_changed_callback(self, callback: Callable[[GenerationState], None]) -> None:
        self._on_state_changed.append(callback)

    def add_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        self._on_progress.append(callback)

    def add_step_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        self._on_step_progress.append(callback)

    def add_generation_complete_callback(self, callback: Callable[[GenerationResult], None]) -> None:
        self._on_generation_complete.append(callback)

    def remove_state_changed_callback(self, callback: Callable[[GenerationState], None]) -> None:
        if callback in self._on_state_changed:
            self._on_state_changed.remove(callback)

    def remove_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        if callback in self._on_progress:
            self._on_progress.remove(callback)

    def remove_step_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        if callback in self._on_step_progress:
            self._on_step_progress.remove(callback)

    def remove_generation_complete_callback(self, callback: Callable[[GenerationResult], None]) -> None:
        if callback in self._on_generation_complete:
            self._on_generation_complete.remove(callback)

    def _set_state(self, state: GenerationState) -> None:
        """Set state and notify callbacks (GTK-safe)."""
        self._state = state
        for callback in self._on_state_changed:
            GLib.idle_add(callback, state)

    def _notify_progress(self, message: str, progress: float) -> None:
        """Notify progress callbacks (GTK-safe)."""
        for callback in self._on_progress:
            GLib.idle_add(callback, message, progress)

    def _notify_step_progress(self, step: int, total: int) -> None:
        """Notify step progress callbacks (GTK-safe)."""
        for callback in self._on_step_progress:
            GLib.idle_add(callback, step, total)

    def _notify_generation_complete(self, result: GenerationResult) -> None:
        """Notify generation complete callbacks (GTK-safe)."""
        for callback in self._on_generation_complete:
            GLib.idle_add(callback, result)

    def load_models(
        self,
        use_compiled: bool = False,
        target_width: int = 1024,
        target_height: int = 1024,
    ) -> None:
        """Load selected models in background thread.

        Args:
            use_compiled: If True, try to load with torch.compile using cached kernels
            target_width: Target generation width (for compiled cache lookup)
            target_height: Target generation height (for compiled cache lookup)
        """
        if self.is_busy:
            return

        load_config = model_manager.get_load_config()
        if not load_config.get("checkpoint_path"):
            self._notify_progress("No checkpoint selected", 0.0)
            return

        self._cancel_requested = False
        self._set_state(GenerationState.LOADING)

        def load_thread():
            try:
                # Set GPUs
                selected_gpus = config_manager.config.gpus.selected
                diffusers_backend.set_gpus(selected_gpus)

                # Load model
                success = diffusers_backend.load_model(
                    checkpoint_path=load_config["checkpoint_path"],
                    model_type=load_config.get("model_type", "sdxl"),
                    vae_path=load_config.get("vae_path"),
                    clip_path=load_config.get("clip_path"),
                    progress_callback=self._notify_progress,
                    use_compiled=use_compiled,
                    target_width=target_width,
                    target_height=target_height,
                )

                if success:
                    # Store model names for metadata
                    self._loaded_checkpoint_name = Path(load_config["checkpoint_path"]).stem
                    self._loaded_model_type = load_config.get("model_type", "sdxl")
                    if load_config.get("vae_path"):
                        self._loaded_vae_name = Path(load_config["vae_path"]).stem
                    else:
                        self._loaded_vae_name = ""
                    if load_config.get("clip_path"):
                        self._loaded_clip_name = Path(load_config["clip_path"]).stem
                    else:
                        self._loaded_clip_name = ""

                    self._notify_progress("Model loaded and ready", 1.0)
                else:
                    self._notify_progress("Failed to load model", 0.0)

            except Exception as e:
                self._notify_progress(f"Error: {e}", 0.0)
            finally:
                self._set_state(GenerationState.IDLE)

        self._current_thread = threading.Thread(target=load_thread, daemon=True)
        self._current_thread.start()

    def unload_models(self) -> None:
        """Unload current models."""
        if self.is_busy:
            return

        diffusers_backend.unload_model()
        self._notify_progress("Models unloaded", 0.0)

    def generate(
        self,
        params: GenerationParams,
        upscale_enabled: bool = False,
        upscale_model_path: Optional[str] = None,
        upscale_model_name: str = "",
        output_dir: Optional[Path] = None,
        loras: Optional[list[LoRAInfo]] = None,
    ) -> None:
        """Start image generation in background thread.

        Args:
            params: Generation parameters
            upscale_enabled: Whether to upscale the result
            upscale_model_path: Path to the upscale model
            upscale_model_name: Name of the upscale model (for metadata)
            output_dir: Optional output directory for saving images
            loras: Optional list of LoRAs to apply as (path, name, weight) tuples
        """
        if self.is_busy:
            return

        if not self.is_model_loaded:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No model loaded")
            )
            return

        self._cancel_requested = False
        self._set_state(GenerationState.GENERATING)

        def generate_thread():
            try:
                print(f"Generation thread started")

                # Load LoRAs if specified
                if loras:
                    self._notify_progress("Loading LoRAs...", 0.0)
                    if not diffusers_backend.load_loras(loras, self._notify_progress):
                        print("Warning: Failed to load some LoRAs")
                else:
                    # Unload any previously loaded LoRAs
                    diffusers_backend.unload_loras()

                # Get actual seed
                actual_seed = diffusers_backend.get_actual_seed(params)
                if params.seed == -1:
                    params.seed = actual_seed

                print(f"Using seed: {params.seed}")
                self._notify_progress("Preparing generation...", 0.0)
                self._notify_step_progress(0, params.steps)

                # Generate image
                print(f"Calling diffusers_backend.generate()")
                image = diffusers_backend.generate(
                    params,
                    progress_callback=self._notify_step_progress,
                )
                print(f"diffusers_backend.generate() returned: {image is not None}")

                # Show decoding progress (VAE decode happens at the end of pipeline)
                self._notify_progress("Finalizing image...", 0.95)

                if self._cancel_requested:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Generation cancelled")
                    )
                    return

                if image is None:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Generation failed")
                    )
                    return

                # Store original size before potential upscaling
                original_width = image.width
                original_height = image.height
                upscale_factor = 1

                # Upscale if enabled
                if upscale_enabled and upscale_model_path:
                    self._notify_progress("Loading upscale model...", 0.0)

                    # Load upscale model if not loaded or different
                    if not upscale_backend.is_loaded or upscale_backend._loaded_model_path != upscale_model_path:
                        upscale_backend.set_device(diffusers_backend._device)
                        if not upscale_backend.load_model(upscale_model_path, self._notify_progress):
                            print("Failed to load upscale model, skipping upscaling")
                        else:
                            upscale_factor = upscale_backend.scale

                    if upscale_backend.is_loaded:
                        self._notify_progress("Upscaling image...", 0.5)
                        upscaled = upscale_backend.upscale(image, self._notify_progress)
                        if upscaled:
                            image = upscaled
                            upscale_factor = upscale_backend.scale
                            print(f"Upscaled from {original_width}x{original_height} to {image.width}x{image.height}")
                        else:
                            print("Upscaling failed, using original image")

                # Save image with metadata
                self._notify_progress("Saving image...", 0.98)
                output_path = self._save_image(
                    image,
                    params,
                    upscale_enabled=upscale_enabled and upscale_backend.is_loaded,
                    upscale_model_name=upscale_model_name,
                    upscale_factor=upscale_factor,
                    original_width=original_width,
                    original_height=original_height,
                    output_dir=output_dir,
                )

                self._notify_progress("Complete", 1.0)
                self._notify_generation_complete(
                    GenerationResult(
                        success=True,
                        image=image,
                        path=output_path,
                        seed=params.seed,
                    )
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._notify_generation_complete(
                    GenerationResult(success=False, error=str(e))
                )
            finally:
                self._set_state(GenerationState.IDLE)

        # Submit to persistent worker pool to keep CUDA state warm between generations
        self._current_future = self._worker_pool.submit(generate_thread)

    def generate_img2img(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        strength: float = 0.75,
        upscale_enabled: bool = False,
        upscale_model_path: Optional[str] = None,
        upscale_model_name: str = "",
        output_dir: Optional[Path] = None,
        loras: Optional[list[LoRAInfo]] = None,
    ) -> None:
        """Start img2img generation in background thread.

        Args:
            params: Generation parameters
            input_image: Input image to transform
            strength: How much to transform (0=no change, 1=full generation)
            upscale_enabled: Whether to upscale the result
            upscale_model_path: Path to the upscale model
            upscale_model_name: Name of the upscale model (for metadata)
            output_dir: Optional output directory for saving images
            loras: Optional list of LoRAs to apply as (path, name, weight) tuples
        """
        if self.is_busy:
            return

        if not self.is_model_loaded:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No model loaded")
            )
            return

        if input_image is None:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No input image provided")
            )
            return

        self._cancel_requested = False
        self._set_state(GenerationState.GENERATING)

        def generate_thread():
            try:
                print(f"Img2img generation thread started")

                # Load LoRAs if specified
                if loras:
                    self._notify_progress("Loading LoRAs...", 0.0)
                    if not diffusers_backend.load_loras(loras, self._notify_progress):
                        print("Warning: Failed to load some LoRAs")
                else:
                    # Unload any previously loaded LoRAs
                    diffusers_backend.unload_loras()

                # Get actual seed
                actual_seed = diffusers_backend.get_actual_seed(params)
                if params.seed == -1:
                    params.seed = actual_seed

                print(f"Using seed: {params.seed}")
                self._notify_progress("Encoding prompts...", 0.0)

                # Calculate effective steps
                effective_steps = max(1, int(params.steps * strength))
                self._notify_step_progress(0, effective_steps)

                # Generate image
                print(f"Calling diffusers_backend.generate_img2img()")
                image = diffusers_backend.generate_img2img(
                    params,
                    input_image=input_image,
                    strength=strength,
                    progress_callback=self._notify_step_progress,
                )
                print(f"diffusers_backend.generate_img2img() returned: {image is not None}")

                self._notify_progress("Finalizing image...", 0.95)

                if self._cancel_requested:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Generation cancelled")
                    )
                    return

                if image is None:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Img2img generation failed")
                    )
                    return

                # Store original size before potential upscaling
                original_width = image.width
                original_height = image.height
                upscale_factor = 1

                # Upscale if enabled
                if upscale_enabled and upscale_model_path:
                    self._notify_progress("Loading upscale model...", 0.0)

                    if not upscale_backend.is_loaded or upscale_backend._loaded_model_path != upscale_model_path:
                        upscale_backend.set_device(diffusers_backend._device)
                        if not upscale_backend.load_model(upscale_model_path, self._notify_progress):
                            print("Failed to load upscale model, skipping upscaling")
                        else:
                            upscale_factor = upscale_backend.scale

                    if upscale_backend.is_loaded:
                        self._notify_progress("Upscaling image...", 0.5)
                        upscaled = upscale_backend.upscale(image, self._notify_progress)
                        if upscaled:
                            image = upscaled
                            upscale_factor = upscale_backend.scale
                            print(f"Upscaled from {original_width}x{original_height} to {image.width}x{image.height}")
                        else:
                            print("Upscaling failed, using original image")

                # Save image with metadata
                self._notify_progress("Saving image...", 0.98)
                output_path = self._save_image(
                    image,
                    params,
                    upscale_enabled=upscale_enabled and upscale_backend.is_loaded,
                    upscale_model_name=upscale_model_name,
                    upscale_factor=upscale_factor,
                    original_width=original_width,
                    original_height=original_height,
                    is_img2img=True,
                    img2img_strength=strength,
                    output_dir=output_dir,
                )

                self._notify_progress("Complete", 1.0)
                self._notify_generation_complete(
                    GenerationResult(
                        success=True,
                        image=image,
                        path=output_path,
                        seed=params.seed,
                    )
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._notify_generation_complete(
                    GenerationResult(success=False, error=str(e))
                )
            finally:
                self._set_state(GenerationState.IDLE)

        # Submit to persistent worker pool to keep CUDA state warm between generations
        self._current_future = self._worker_pool.submit(generate_thread)

    def generate_inpaint(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        mask_image: Image.Image,
        strength: float = 0.75,
        upscale_enabled: bool = False,
        upscale_model_path: Optional[str] = None,
        upscale_model_name: str = "",
        output_dir: Optional[Path] = None,
        loras: Optional[list[LoRAInfo]] = None,
    ) -> None:
        """Start inpaint generation in background thread.

        Args:
            params: Generation parameters
            input_image: Input image to inpaint
            mask_image: Mask image (white = inpaint, black = keep)
            strength: How much to transform masked areas (0=no change, 1=full generation)
            upscale_enabled: Whether to upscale the result
            upscale_model_path: Path to the upscale model
            upscale_model_name: Name of the upscale model (for metadata)
            output_dir: Optional output directory for saving images
            loras: Optional list of LoRAs to apply as (path, name, weight) tuples
        """
        if self.is_busy:
            return

        if not self.is_model_loaded:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No model loaded")
            )
            return

        if input_image is None:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No input image provided")
            )
            return

        if mask_image is None:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No mask provided")
            )
            return

        self._cancel_requested = False
        self._set_state(GenerationState.GENERATING)

        def generate_thread():
            try:
                print(f"Inpaint generation thread started")

                # Load LoRAs if specified
                if loras:
                    self._notify_progress("Loading LoRAs...", 0.0)
                    if not diffusers_backend.load_loras(loras, self._notify_progress):
                        print("Warning: Failed to load some LoRAs")
                else:
                    # Unload any previously loaded LoRAs
                    diffusers_backend.unload_loras()

                # Get actual seed
                actual_seed = diffusers_backend.get_actual_seed(params)
                if params.seed == -1:
                    params.seed = actual_seed

                print(f"Using seed: {params.seed}")
                self._notify_progress("Encoding prompts...", 0.0)

                # Calculate effective steps
                effective_steps = max(1, int(params.steps * strength))
                self._notify_step_progress(0, effective_steps)

                # Generate image
                print(f"Calling diffusers_backend.generate_inpaint()")
                image = diffusers_backend.generate_inpaint(
                    params,
                    input_image=input_image,
                    mask_image=mask_image,
                    strength=strength,
                    progress_callback=self._notify_step_progress,
                )
                print(f"diffusers_backend.generate_inpaint() returned: {image is not None}")

                self._notify_progress("Finalizing image...", 0.95)

                if self._cancel_requested:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Generation cancelled")
                    )
                    return

                if image is None:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Inpaint generation failed")
                    )
                    return

                # Store original size before potential upscaling
                original_width = image.width
                original_height = image.height
                upscale_factor = 1

                # Upscale if enabled
                if upscale_enabled and upscale_model_path:
                    self._notify_progress("Loading upscale model...", 0.0)

                    if not upscale_backend.is_loaded or upscale_backend._loaded_model_path != upscale_model_path:
                        upscale_backend.set_device(diffusers_backend._device)
                        if not upscale_backend.load_model(upscale_model_path, self._notify_progress):
                            print("Failed to load upscale model, skipping upscaling")
                        else:
                            upscale_factor = upscale_backend.scale

                    if upscale_backend.is_loaded:
                        self._notify_progress("Upscaling image...", 0.5)
                        upscaled = upscale_backend.upscale(image, self._notify_progress)
                        if upscaled:
                            image = upscaled
                            upscale_factor = upscale_backend.scale
                            print(f"Upscaled from {original_width}x{original_height} to {image.width}x{image.height}")
                        else:
                            print("Upscaling failed, using original image")

                # Save image with metadata
                self._notify_progress("Saving image...", 0.98)
                output_path = self._save_image(
                    image,
                    params,
                    upscale_enabled=upscale_enabled and upscale_backend.is_loaded,
                    upscale_model_name=upscale_model_name,
                    upscale_factor=upscale_factor,
                    original_width=original_width,
                    original_height=original_height,
                    is_inpaint=True,
                    inpaint_strength=strength,
                    output_dir=output_dir,
                )

                self._notify_progress("Complete", 1.0)
                self._notify_generation_complete(
                    GenerationResult(
                        success=True,
                        image=image,
                        path=output_path,
                        seed=params.seed,
                    )
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._notify_generation_complete(
                    GenerationResult(success=False, error=str(e))
                )
            finally:
                self._set_state(GenerationState.IDLE)

        # Submit to persistent worker pool to keep CUDA state warm between generations
        self._current_future = self._worker_pool.submit(generate_thread)

    def generate_outpaint(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        extensions: dict,
        strength: float = 1.0,
        upscale_enabled: bool = False,
        upscale_model_path: Optional[str] = None,
        upscale_model_name: str = "",
        output_dir: Optional[Path] = None,
        loras: Optional[list[LoRAInfo]] = None,
    ) -> None:
        """Start outpaint generation in background thread.

        Args:
            params: Generation parameters
            input_image: Input image to outpaint
            extensions: Dict with 'left', 'right', 'top', 'bottom' extension amounts in pixels
            strength: How much to transform extended areas (0=no change, 1=full generation)
            upscale_enabled: Whether to upscale the result
            upscale_model_path: Path to the upscale model
            upscale_model_name: Name of the upscale model (for metadata)
            output_dir: Optional output directory for saving images
            loras: Optional list of LoRAs to apply
        """
        if self.is_busy:
            return

        if not self.is_model_loaded:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No model loaded")
            )
            return

        if input_image is None:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No input image provided")
            )
            return

        # Check that at least one extension is specified
        total_extension = sum(extensions.values())
        if total_extension == 0:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No outpaint extensions specified")
            )
            return

        self._cancel_requested = False
        self._set_state(GenerationState.GENERATING)

        def generate_thread():
            try:
                print(f"Outpaint generation thread started")
                print(f"Extensions: {extensions}")

                # Load LoRAs if specified
                if loras:
                    self._notify_progress("Loading LoRAs...", 0.0)
                    if not diffusers_backend.load_loras(loras, self._notify_progress):
                        print("Warning: Failed to load some LoRAs")
                else:
                    # Unload any previously loaded LoRAs
                    diffusers_backend.unload_loras()

                # Get actual seed
                actual_seed = diffusers_backend.get_actual_seed(params)
                if params.seed == -1:
                    params.seed = actual_seed

                print(f"Using seed: {params.seed}")
                self._notify_progress("Preparing outpaint...", 0.0)

                # Calculate effective steps
                effective_steps = max(1, int(params.steps * strength))
                self._notify_step_progress(0, effective_steps)

                # Generate image using outpaint method
                print(f"Calling diffusers_backend.generate_outpaint()")
                image = diffusers_backend.generate_outpaint(
                    params,
                    input_image=input_image,
                    extensions=extensions,
                    strength=strength,
                    progress_callback=self._notify_step_progress,
                )
                print(f"diffusers_backend.generate_outpaint() returned: {image is not None}")

                self._notify_progress("Finalizing image...", 0.95)

                if self._cancel_requested:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Generation cancelled")
                    )
                    return

                if image is None:
                    self._notify_generation_complete(
                        GenerationResult(success=False, error="Outpaint generation failed")
                    )
                    return

                # Store original size before potential upscaling
                original_width = image.width
                original_height = image.height
                upscale_factor = 1

                # Upscale if enabled
                if upscale_enabled and upscale_model_path:
                    self._notify_progress("Loading upscale model...", 0.0)

                    if not upscale_backend.is_loaded or upscale_backend._loaded_model_path != upscale_model_path:
                        upscale_backend.set_device(diffusers_backend._device)
                        if not upscale_backend.load_model(upscale_model_path, self._notify_progress):
                            print("Failed to load upscale model, skipping upscaling")
                        else:
                            upscale_factor = upscale_backend.scale

                    if upscale_backend.is_loaded:
                        self._notify_progress("Upscaling image...", 0.5)
                        upscaled = upscale_backend.upscale(image, self._notify_progress)
                        if upscaled:
                            image = upscaled
                            upscale_factor = upscale_backend.scale
                            print(f"Upscaled from {original_width}x{original_height} to {image.width}x{image.height}")
                        else:
                            print("Upscaling failed, using original image")

                # Save image with metadata
                self._notify_progress("Saving image...", 0.98)
                output_path = self._save_image(
                    image,
                    params,
                    upscale_enabled=upscale_enabled and upscale_backend.is_loaded,
                    upscale_model_name=upscale_model_name,
                    upscale_factor=upscale_factor,
                    original_width=original_width,
                    original_height=original_height,
                    is_outpaint=True,
                    outpaint_extensions=extensions,
                    output_dir=output_dir,
                )

                self._notify_progress("Complete", 1.0)
                self._notify_generation_complete(
                    GenerationResult(
                        success=True,
                        image=image,
                        path=output_path,
                        seed=params.seed,
                    )
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._notify_generation_complete(
                    GenerationResult(success=False, error=str(e))
                )
            finally:
                self._set_state(GenerationState.IDLE)

        # Submit to persistent worker pool to keep CUDA state warm between generations
        self._current_future = self._worker_pool.submit(generate_thread)

    def generate_refine(
        self,
        params: GenerationParams,
        input_image: Image.Image,
        masks: list[DetectedMask],
        strength: float = 0.75,
        upscale_model_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        loras: Optional[list[LoRAInfo]] = None,
        padding: int = 64,
        original_prompt: Optional[str] = None,
    ) -> None:
        """Start refinement generation in background thread.

        This method refines selected regions of an image by:
        1. Cropping each region with padding
        2. Upscaling the cropped region
        3. Inpainting the upscaled region
        4. Downscaling back to original size
        5. Compositing into the original image with feathered edges

        Args:
            params: Generation parameters (prompts for generation, cfg, etc.)
            input_image: Input image to refine
            masks: List of DetectedMask objects to refine
            strength: Inpainting strength (0-1)
            upscale_model_path: Path to upscale model (optional)
            output_dir: Output directory for saving images
            loras: Optional list of LoRAs to apply
            padding: Padding around mask bbox in pixels
            original_prompt: Original positive prompt to combine with refiner prompt for metadata
        """
        if self.is_busy:
            return

        if not self.is_model_loaded:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No model loaded")
            )
            return

        if input_image is None:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No input image provided")
            )
            return

        if not masks:
            self._notify_generation_complete(
                GenerationResult(success=False, error="No masks provided")
            )
            return

        self._cancel_requested = False
        self._set_state(GenerationState.GENERATING)

        def generate_thread():
            try:
                from PIL import ImageFilter

                print(f"Refine generation thread started with {len(masks)} masks")

                # Load LoRAs if specified
                if loras:
                    self._notify_progress("Loading LoRAs...", 0.0)
                    if not diffusers_backend.load_loras(loras, self._notify_progress):
                        print("Warning: Failed to load some LoRAs")
                else:
                    diffusers_backend.unload_loras()

                # Get actual seed
                actual_seed = diffusers_backend.get_actual_seed(params)
                if params.seed == -1:
                    params.seed = actual_seed

                print(f"Using seed: {params.seed}")

                # Work on a copy of the input image
                result_image = input_image.copy()
                img_width, img_height = result_image.size

                total_masks = len(masks)

                for mask_idx, mask in enumerate(masks):
                    if self._cancel_requested:
                        self._notify_generation_complete(
                            GenerationResult(success=False, error="Generation cancelled")
                        )
                        return

                    mask_progress_base = mask_idx / total_masks
                    mask_progress_step = 1.0 / total_masks

                    self._notify_progress(
                        f"Refining region {mask_idx + 1}/{total_masks}: {mask.label}",
                        mask_progress_base
                    )

                    # Extract bounding box with padding
                    x1, y1, x2, y2 = mask.bbox
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(img_width, x2 + padding)
                    y2_padded = min(img_height, y2 + padding)

                    # Crop region from current result image
                    crop_region = result_image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
                    crop_width, crop_height = crop_region.size

                    print(f"  Cropped region: {crop_width}x{crop_height} from ({x1_padded}, {y1_padded})")

                    # Create mask for the cropped region
                    # The mask is relative to the full image, so we need to extract the relevant portion
                    mask_y_start = y1_padded
                    mask_y_end = y2_padded
                    mask_x_start = x1_padded
                    mask_x_end = x2_padded

                    # Extract mask portion
                    mask_crop = mask.mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

                    # Convert to PIL mask (white = inpaint, black = keep)
                    mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8), mode="L")

                    # Dilate the mask slightly for better edge blending
                    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))

                    # Upscale region (2x)
                    self._notify_progress(
                        f"Upscaling region {mask_idx + 1}/{total_masks}...",
                        mask_progress_base + mask_progress_step * 0.2
                    )

                    upscale_factor = 2
                    upscaled_region = None

                    # Try Real-ESRGAN first
                    if upscale_model_path:
                        try:
                            if not upscale_backend.is_loaded or upscale_backend._loaded_model_path != upscale_model_path:
                                upscale_backend.set_device(diffusers_backend._device)
                                upscale_backend.load_model(upscale_model_path, self._notify_progress)

                            if upscale_backend.is_loaded:
                                upscaled_region = upscale_backend.upscale(crop_region, self._notify_progress)
                                upscale_factor = upscale_backend.scale
                        except Exception as e:
                            print(f"  Upscale backend failed: {e}, using Lanczos fallback")

                    # Lanczos fallback
                    if upscaled_region is None:
                        upscaled_region = crop_region.resize(
                            (crop_width * 2, crop_height * 2),
                            Image.LANCZOS
                        )
                        upscale_factor = 2

                    upscaled_width, upscaled_height = upscaled_region.size
                    print(f"  Upscaled to: {upscaled_width}x{upscaled_height}")

                    # Ensure dimensions are divisible by 8 for diffusion model
                    def round_to_8(x):
                        return ((x + 7) // 8) * 8

                    # Ensure minimum size of 512x512 for proper inpainting quality
                    MIN_INPAINT_SIZE = 512

                    inpaint_width = round_to_8(max(upscaled_width, MIN_INPAINT_SIZE))
                    inpaint_height = round_to_8(max(upscaled_height, MIN_INPAINT_SIZE))

                    # If region is too small, scale up proportionally to meet minimum
                    if upscaled_width < MIN_INPAINT_SIZE or upscaled_height < MIN_INPAINT_SIZE:
                        scale_factor = max(MIN_INPAINT_SIZE / upscaled_width, MIN_INPAINT_SIZE / upscaled_height)
                        inpaint_width = round_to_8(int(upscaled_width * scale_factor))
                        inpaint_height = round_to_8(int(upscaled_height * scale_factor))
                        print(f"  Region too small, scaling up by {scale_factor:.2f}x for quality")

                    # Resize region and mask if needed
                    if inpaint_width != upscaled_width or inpaint_height != upscaled_height:
                        print(f"  Resizing for inpaint: {upscaled_width}x{upscaled_height} -> {inpaint_width}x{inpaint_height}")
                        inpaint_region = upscaled_region.resize((inpaint_width, inpaint_height), Image.LANCZOS)
                        inpaint_mask = mask_pil.resize((inpaint_width, inpaint_height), Image.NEAREST)
                    else:
                        inpaint_region = upscaled_region
                        inpaint_mask = mask_pil.resize((upscaled_width, upscaled_height), Image.NEAREST)

                    # Apply Gaussian blur to inpaint mask for better context blending
                    # This tells the diffusion model to gradually blend at the edges
                    # rather than generating with a hard boundary
                    mask_blur_radius = max(8, min(inpaint_width, inpaint_height) // 32)
                    inpaint_mask = inpaint_mask.filter(ImageFilter.GaussianBlur(radius=mask_blur_radius))
                    print(f"  Applied mask blur (radius={mask_blur_radius}) for context blending")

                    # Run inpainting on upscaled region
                    self._notify_progress(
                        f"Inpainting region {mask_idx + 1}/{total_masks}...",
                        mask_progress_base + mask_progress_step * 0.4
                    )

                    # Create params for this region
                    region_params = GenerationParams(
                        prompt=params.prompt,
                        negative_prompt=params.negative_prompt,
                        width=inpaint_width,
                        height=inpaint_height,
                        steps=params.steps,
                        cfg_scale=params.cfg_scale,
                        seed=params.seed + mask_idx,  # Different seed per mask
                        sampler=params.sampler,
                        scheduler=params.scheduler,
                    )

                    # Calculate effective steps for progress
                    effective_steps = max(1, int(params.steps * strength))
                    self._notify_step_progress(0, effective_steps)

                    inpainted_region = diffusers_backend.generate_inpaint(
                        region_params,
                        input_image=inpaint_region,
                        mask_image=inpaint_mask,
                        strength=strength,
                        progress_callback=self._notify_step_progress,
                    )

                    if inpainted_region is None:
                        print(f"  Warning: Inpainting failed for region {mask_idx + 1}")
                        continue

                    # Resize back to original upscaled dimensions if we resized for inpaint
                    if inpaint_width != upscaled_width or inpaint_height != upscaled_height:
                        inpainted_region = inpainted_region.resize(
                            (upscaled_width, upscaled_height),
                            Image.LANCZOS
                        )

                    # Downscale refined region back to original size
                    self._notify_progress(
                        f"Compositing region {mask_idx + 1}/{total_masks}...",
                        mask_progress_base + mask_progress_step * 0.8
                    )

                    refined_region = inpainted_region.resize(
                        (crop_width, crop_height),
                        Image.LANCZOS
                    )

                    # Create feathered mask for blending
                    feather_radius = max(5, padding // 4)
                    feathered_mask = mask_pil.filter(
                        ImageFilter.GaussianBlur(radius=feather_radius)
                    )

                    # Composite refined region into result image
                    # First paste the refined region
                    temp_result = result_image.copy()
                    temp_result.paste(refined_region, (x1_padded, y1_padded))

                    # Then blend using the feathered mask
                    # Convert grayscale mask to match image mode
                    if result_image.mode == "RGB":
                        # Create full-size mask
                        full_mask = Image.new("L", (img_width, img_height), 0)
                        full_mask.paste(feathered_mask, (x1_padded, y1_padded))

                        result_image = Image.composite(temp_result, result_image, full_mask)
                    else:
                        result_image = temp_result

                    print(f"  Region {mask_idx + 1} composited successfully")

                # Save final result
                self._notify_progress("Saving refined image...", 0.98)

                # Create params copy with combined prompt for metadata
                # This preserves the original prompt + refiner prompt in saved image metadata
                save_params = GenerationParams(
                    prompt=params.prompt,  # Will be modified below
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    steps=params.steps,
                    cfg_scale=params.cfg_scale,
                    seed=params.seed,
                    sampler=params.sampler,
                    scheduler=params.scheduler,
                )

                # Combine original prompt with refiner prompt for metadata
                if original_prompt and params.prompt:
                    save_params.prompt = f"{original_prompt}, [Refiner: {params.prompt}]"
                elif original_prompt:
                    save_params.prompt = original_prompt
                # else: keep params.prompt as-is (refiner prompt only)

                output_path = self._save_image(
                    result_image,
                    save_params,
                    upscale_enabled=False,
                    is_inpaint=True,  # Mark as inpaint for metadata
                    inpaint_strength=strength,
                    output_dir=output_dir,
                )

                self._notify_progress("Refinement complete", 1.0)
                self._notify_generation_complete(
                    GenerationResult(
                        success=True,
                        image=result_image,
                        path=output_path,
                        seed=params.seed,
                    )
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._notify_generation_complete(
                    GenerationResult(success=False, error=str(e))
                )
            finally:
                self._set_state(GenerationState.IDLE)

        # Submit to persistent worker pool to keep CUDA state warm between generations
        self._current_future = self._worker_pool.submit(generate_thread)

    def cancel(self) -> None:
        """Request cancellation of current operation."""
        if self._state == GenerationState.GENERATING:
            self._cancel_requested = True
            self._set_state(GenerationState.CANCELLING)

    def shutdown(self) -> None:
        """Shutdown the worker pool. Call this when the application is exiting."""
        if self._worker_pool:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    def _save_image(
        self,
        image: Image.Image,
        params: GenerationParams,
        upscale_enabled: bool = False,
        upscale_model_name: str = "",
        upscale_factor: int = 1,
        original_width: int = 0,
        original_height: int = 0,
        is_img2img: bool = False,
        img2img_strength: float = 0.0,
        is_inpaint: bool = False,
        inpaint_strength: float = 0.0,
        is_outpaint: bool = False,
        outpaint_extensions: Optional[dict] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save generated image to output directory with metadata."""
        if output_dir is None:
            output_dir = config_manager.config.get_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp (including milliseconds) and seed
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        if is_outpaint:
            prefix = "outpaint_"
        elif is_inpaint:
            prefix = "inpaint_"
        elif is_img2img:
            prefix = "img2img_"
        else:
            prefix = "gen_"
        upscale_suffix = f"_upscaled{upscale_factor}x" if upscale_enabled else ""
        filename = f"{prefix}{timestamp}_seed{params.seed}{upscale_suffix}.{OUTPUT_FORMAT}"
        output_path = output_dir / filename

        # Create metadata
        metadata = GenerationMetadata(
            checkpoint=self._loaded_checkpoint_name,
            vae=self._loaded_vae_name,
            clip=self._loaded_clip_name,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            width=image.width,  # Final image width (may be upscaled)
            height=image.height,  # Final image height (may be upscaled)
            steps=params.steps,
            cfg_scale=params.cfg_scale,
            seed=params.seed,
            sampler=params.sampler,
            scheduler=params.scheduler,
            model_type=self._loaded_model_type,
            upscale_enabled=upscale_enabled,
            upscale_model=upscale_model_name,
            upscale_factor=upscale_factor,
            original_width=original_width if upscale_enabled else 0,
            original_height=original_height if upscale_enabled else 0,
            is_img2img=is_img2img,
            img2img_strength=img2img_strength,
            is_inpaint=is_inpaint,
            inpaint_strength=inpaint_strength,
        )

        # Save with metadata
        save_image_with_metadata(image, output_path, metadata)
        return output_path


# Global generation service instance
generation_service = GenerationService()
