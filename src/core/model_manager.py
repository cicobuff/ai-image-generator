"""Model management for scanning and coordinating model loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

from src.utils.constants import MODEL_EXTENSIONS
from src.core.config import config_manager
from src.core.model_inspector import model_inspector, ModelComponents


class ModelType(Enum):
    """Types of models that can be loaded."""
    CHECKPOINT = "checkpoint"
    VAE = "vae"
    CLIP = "clip"
    UPSCALE = "upscale"
    LORA = "lora"


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    path: Path
    name: str
    model_type: ModelType
    components: ModelComponents = field(default_factory=ModelComponents)
    size_bytes: int = 0

    @property
    def size_gb(self) -> float:
        """Model file size in GB."""
        return self.size_bytes / (1024 ** 3)

    @property
    def has_embedded_vae(self) -> bool:
        """Check if model has embedded VAE."""
        return self.components.has_vae

    @property
    def has_embedded_clip(self) -> bool:
        """Check if model has embedded text encoder (CLIP)."""
        return self.components.has_text_encoder

    @property
    def display_name(self) -> str:
        """Get display name with embedded indicators."""
        indicators = []
        if self.model_type == ModelType.CHECKPOINT:
            if self.has_embedded_vae:
                indicators.append("VAE")
            if self.has_embedded_clip:
                indicators.append("CLIP")

        if indicators:
            return f"{self.name} (embedded: {', '.join(indicators)})"
        return self.name


@dataclass
class LoadedModels:
    """Currently loaded models."""
    checkpoint: Optional[ModelInfo] = None
    vae: Optional[ModelInfo] = None
    clip: Optional[ModelInfo] = None

    def clear(self) -> None:
        """Clear all loaded models."""
        self.checkpoint = None
        self.vae = None
        self.clip = None


class ModelManager:
    """Manages model discovery, loading, and unloading."""

    def __init__(self):
        self._checkpoints: list[ModelInfo] = []
        self._vaes: list[ModelInfo] = []
        self._clips: list[ModelInfo] = []
        self._upscalers: list[ModelInfo] = []
        self._loras: list[ModelInfo] = []
        self._loaded = LoadedModels()
        self._on_models_changed: list[Callable[[], None]] = []

    @property
    def checkpoints(self) -> list[ModelInfo]:
        """Get list of available checkpoint models."""
        return self._checkpoints

    @property
    def vaes(self) -> list[ModelInfo]:
        """Get list of available VAE models."""
        return self._vaes

    @property
    def clips(self) -> list[ModelInfo]:
        """Get list of available CLIP models."""
        return self._clips

    @property
    def upscalers(self) -> list[ModelInfo]:
        """Get list of available upscale models."""
        return self._upscalers

    @property
    def loras(self) -> list[ModelInfo]:
        """Get list of available LoRA models."""
        return self._loras

    @property
    def loaded(self) -> LoadedModels:
        """Get currently loaded models."""
        return self._loaded

    def add_models_changed_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to be called when available models change."""
        self._on_models_changed.append(callback)

    def remove_models_changed_callback(self, callback: Callable[[], None]) -> None:
        """Remove a models changed callback."""
        if callback in self._on_models_changed:
            self._on_models_changed.remove(callback)

    def _notify_models_changed(self) -> None:
        """Notify all callbacks that models have changed."""
        for callback in self._on_models_changed:
            try:
                callback()
            except Exception as e:
                print(f"Error in models changed callback: {e}")

    # Directories to scan for each model type
    CHECKPOINT_DIRS = {"checkpoints", "checkpoint", "stable-diffusion", "sd", "sdxl"}
    VAE_DIRS = {"vae", "vaes"}
    CLIP_DIRS = {"clip", "text_encoder", "text_encoders"}
    UPSCALE_DIRS = {"upscale", "upscalers", "esrgan", "realesrgan"}
    LORA_DIRS = {"loras", "lora"}

    def scan_models(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Scan the models directory for available models.
        Only scans specific subdirectories for supported model types.

        Args:
            progress_callback: Optional callback to report scanning progress
        """
        models_path = config_manager.config.get_models_path()

        self._checkpoints.clear()
        self._vaes.clear()
        self._clips.clear()
        self._upscalers.clear()
        self._loras.clear()

        if not models_path.exists():
            if progress_callback:
                progress_callback("Models directory does not exist")
            self._notify_models_changed()
            return

        # Scan checkpoints directory and root level
        self._scan_checkpoints(models_path, progress_callback)

        # Scan VAE directories
        self._scan_vae_models(models_path, progress_callback)

        # Scan CLIP directories
        self._scan_clip_models(models_path, progress_callback)

        # Scan upscale directories
        self._scan_upscale_models(models_path, progress_callback)

        # Scan LoRA directories
        self._scan_lora_models(models_path, progress_callback)

        # Sort models by name
        self._checkpoints.sort(key=lambda m: m.name.lower())
        self._vaes.sort(key=lambda m: m.name.lower())
        self._clips.sort(key=lambda m: m.name.lower())
        self._upscalers.sort(key=lambda m: m.name.lower())
        self._loras.sort(key=lambda m: m.name.lower())

        if progress_callback:
            progress_callback(
                f"Found {len(self._checkpoints)} checkpoints, "
                f"{len(self._vaes)} VAEs, {len(self._clips)} CLIPs, "
                f"{len(self._upscalers)} upscalers, {len(self._loras)} LoRAs"
            )

        self._notify_models_changed()

    def _get_scan_directories(self, models_path: Path, dir_names: set) -> list[Path]:
        """Get list of directories to scan based on allowed names."""
        dirs_to_scan = []
        for item in models_path.iterdir():
            if item.is_dir() and item.name.lower() in dir_names:
                dirs_to_scan.append(item)
        return dirs_to_scan

    def _scan_checkpoints(self, models_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Scan for checkpoint models in specific directories and root level."""
        model_files = []

        # Scan root level for checkpoints (files directly in models/)
        for ext in MODEL_EXTENSIONS:
            for path in models_path.glob(f"*{ext}"):
                if path.is_file():
                    model_files.append(path)

        # Scan checkpoint-specific subdirectories
        for subdir in self._get_scan_directories(models_path, self.CHECKPOINT_DIRS):
            for ext in MODEL_EXTENSIONS:
                for path in subdir.rglob(f"*{ext}"):
                    model_files.append(path)

        total = len(model_files)
        for i, path in enumerate(model_files):
            if progress_callback:
                progress_callback(f"Scanning checkpoint {i + 1}/{total}: {path.name}")

            try:
                components = model_inspector.inspect(path)
                size = path.stat().st_size

                # Only add if it looks like a checkpoint (has unet)
                if components.has_unet:
                    model_info = ModelInfo(
                        path=path,
                        name=path.stem,
                        model_type=ModelType.CHECKPOINT,
                        components=components,
                        size_bytes=size,
                    )
                    self._checkpoints.append(model_info)
            except Exception as e:
                print(f"Error scanning checkpoint {path}: {e}")

    def _scan_vae_models(self, models_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Scan for VAE models in specific directories."""
        for subdir in self._get_scan_directories(models_path, self.VAE_DIRS):
            for ext in MODEL_EXTENSIONS:
                for path in subdir.rglob(f"*{ext}"):
                    if progress_callback:
                        progress_callback(f"Scanning VAE: {path.name}")

                    try:
                        components = model_inspector.inspect(path)
                        size = path.stat().st_size

                        model_info = ModelInfo(
                            path=path,
                            name=path.stem,
                            model_type=ModelType.VAE,
                            components=components,
                            size_bytes=size,
                        )
                        self._vaes.append(model_info)
                    except Exception as e:
                        print(f"Error scanning VAE {path}: {e}")

    def _scan_clip_models(self, models_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Scan for CLIP/text encoder models in specific directories."""
        for subdir in self._get_scan_directories(models_path, self.CLIP_DIRS):
            for ext in MODEL_EXTENSIONS:
                for path in subdir.rglob(f"*{ext}"):
                    if progress_callback:
                        progress_callback(f"Scanning CLIP: {path.name}")

                    try:
                        components = model_inspector.inspect(path)
                        size = path.stat().st_size

                        model_info = ModelInfo(
                            path=path,
                            name=path.stem,
                            model_type=ModelType.CLIP,
                            components=components,
                            size_bytes=size,
                        )
                        self._clips.append(model_info)
                    except Exception as e:
                        print(f"Error scanning CLIP {path}: {e}")

    def _scan_upscale_models(self, models_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Scan for upscaler models in specific directories."""
        # Upscale models can be .pth, .pt, .safetensors, .bin, .onnx
        upscale_extensions = {".pth", ".pt", ".safetensors", ".bin", ".onnx"}

        for subdir in self._get_scan_directories(models_path, self.UPSCALE_DIRS):
            for ext in upscale_extensions:
                for path in subdir.rglob(f"*{ext}"):
                    if progress_callback:
                        progress_callback(f"Scanning upscaler: {path.name}")

                    try:
                        size = path.stat().st_size
                        model_info = ModelInfo(
                            path=path,
                            name=path.stem,
                            model_type=ModelType.UPSCALE,
                            components=ModelComponents(),
                            size_bytes=size,
                        )
                        self._upscalers.append(model_info)
                    except Exception as e:
                        print(f"Error scanning upscale model {path}: {e}")

    def _scan_lora_models(self, models_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Scan for LoRA models in specific directories."""
        # LoRA models are typically .safetensors, .pt, .bin
        lora_extensions = {".safetensors", ".pt", ".bin", ".ckpt"}

        for subdir in self._get_scan_directories(models_path, self.LORA_DIRS):
            for ext in lora_extensions:
                for path in subdir.rglob(f"*{ext}"):
                    if progress_callback:
                        progress_callback(f"Scanning LoRA: {path.name}")

                    try:
                        size = path.stat().st_size
                        model_info = ModelInfo(
                            path=path,
                            name=path.stem,
                            model_type=ModelType.LORA,
                            components=ModelComponents(),
                            size_bytes=size,
                        )
                        self._loras.append(model_info)
                    except Exception as e:
                        print(f"Error scanning LoRA model {path}: {e}")

    def select_checkpoint(self, model: Optional[ModelInfo]) -> None:
        """Select a checkpoint model for loading."""
        self._loaded.checkpoint = model

    def select_vae(self, model: Optional[ModelInfo]) -> None:
        """Select a VAE model for loading."""
        self._loaded.vae = model

    def select_clip(self, model: Optional[ModelInfo]) -> None:
        """Select a CLIP model for loading."""
        self._loaded.clip = model

    def get_checkpoint_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find a checkpoint by name."""
        for model in self._checkpoints:
            if model.name == name:
                return model
        return None

    def get_vae_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find a VAE by name."""
        for model in self._vaes:
            if model.name == name:
                return model
        return None

    def get_clip_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find a CLIP by name."""
        for model in self._clips:
            if model.name == name:
                return model
        return None

    def get_upscaler_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find an upscaler by name."""
        for model in self._upscalers:
            if model.name == name:
                return model
        return None

    def get_lora_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find a LoRA by name."""
        for model in self._loras:
            if model.name == name:
                return model
        return None

    def clear_selection(self) -> None:
        """Clear all model selections."""
        self._loaded.clear()

    def get_load_config(self) -> dict:
        """
        Get configuration dict for loading the selected models.

        Returns dict with paths for checkpoint, vae, and clip.
        """
        config = {}

        if self._loaded.checkpoint:
            config["checkpoint_path"] = str(self._loaded.checkpoint.path)
            config["model_type"] = self._loaded.checkpoint.components.model_type

            # Only include separate VAE if checkpoint doesn't have embedded
            # or if user explicitly selected a different VAE
            if self._loaded.vae and not self._loaded.checkpoint.has_embedded_vae:
                config["vae_path"] = str(self._loaded.vae.path)
            elif self._loaded.vae:
                # User explicitly selected VAE, override embedded
                config["vae_path"] = str(self._loaded.vae.path)

            # Only include separate CLIP if checkpoint doesn't have embedded
            # or if user explicitly selected a different CLIP
            if self._loaded.clip and not self._loaded.checkpoint.has_embedded_clip:
                config["clip_path"] = str(self._loaded.clip.path)
            elif self._loaded.clip:
                config["clip_path"] = str(self._loaded.clip.path)

        return config


# Global model manager instance
model_manager = ModelManager()
