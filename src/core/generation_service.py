"""Generation service for async image generation with GTK-safe callbacks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import threading
from enum import Enum

from PIL import Image
from gi.repository import GLib

from src.core.config import config_manager
from src.core.model_manager import model_manager
from src.core.gpu_manager import gpu_manager
from src.backends.diffusers_backend import diffusers_backend, GenerationParams
from src.utils.constants import OUTPUT_FORMAT


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

    def load_models(self) -> None:
        """Load selected models in background thread."""
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
                )

                if success:
                    self._notify_progress("Model loaded", 1.0)
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

    def generate(self, params: GenerationParams) -> None:
        """Start image generation in background thread."""
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
                # Get actual seed
                actual_seed = diffusers_backend.get_actual_seed(params)
                if params.seed == -1:
                    params.seed = actual_seed

                print(f"Using seed: {params.seed}")
                self._notify_progress("Generating image...", 0.0)
                self._notify_step_progress(0, params.steps)

                # Generate image
                print(f"Calling diffusers_backend.generate()")
                image = diffusers_backend.generate(
                    params,
                    progress_callback=self._notify_step_progress,
                )
                print(f"diffusers_backend.generate() returned: {image is not None}")

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

                # Save image
                output_path = self._save_image(image, params)

                self._notify_generation_complete(
                    GenerationResult(
                        success=True,
                        image=image,
                        path=output_path,
                        seed=params.seed,
                    )
                )

            except Exception as e:
                self._notify_generation_complete(
                    GenerationResult(success=False, error=str(e))
                )
            finally:
                self._set_state(GenerationState.IDLE)

        self._current_thread = threading.Thread(target=generate_thread, daemon=True)
        self._current_thread.start()

    def cancel(self) -> None:
        """Request cancellation of current operation."""
        if self._state == GenerationState.GENERATING:
            self._cancel_requested = True
            self._set_state(GenerationState.CANCELLING)

    def _save_image(self, image: Image.Image, params: GenerationParams) -> Path:
        """Save generated image to output directory."""
        output_dir = config_manager.config.get_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp and seed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp}_seed{params.seed}.{OUTPUT_FORMAT}"
        output_path = output_dir / filename

        image.save(output_path, OUTPUT_FORMAT.upper())
        return output_path


# Global generation service instance
generation_service = GenerationService()
