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

    def scan_models(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Scan the models directory for available models.

        Args:
            progress_callback: Optional callback to report scanning progress
        """
        models_path = config_manager.config.get_models_path()

        self._checkpoints.clear()
        self._vaes.clear()
        self._clips.clear()

        if not models_path.exists():
            if progress_callback:
                progress_callback("Models directory does not exist")
            self._notify_models_changed()
            return

        # Scan for all model files
        model_files = []
        for ext in MODEL_EXTENSIONS:
            model_files.extend(models_path.rglob(f"*{ext}"))

        total = len(model_files)
        for i, path in enumerate(model_files):
            if progress_callback:
                progress_callback(f"Scanning {i + 1}/{total}: {path.name}")

            model_info = self._create_model_info(path)
            if model_info:
                self._categorize_model(model_info)

        # Sort models by name
        self._checkpoints.sort(key=lambda m: m.name.lower())
        self._vaes.sort(key=lambda m: m.name.lower())
        self._clips.sort(key=lambda m: m.name.lower())

        if progress_callback:
            progress_callback(
                f"Found {len(self._checkpoints)} checkpoints, "
                f"{len(self._vaes)} VAEs, {len(self._clips)} CLIPs"
            )

        self._notify_models_changed()

    def _create_model_info(self, path: Path) -> Optional[ModelInfo]:
        """Create ModelInfo for a file path."""
        try:
            components = model_inspector.inspect(path)
            size = path.stat().st_size

            # Determine model type from path or components
            model_type = self._determine_model_type(path, components)

            return ModelInfo(
                path=path,
                name=path.stem,
                model_type=model_type,
                components=components,
                size_bytes=size,
            )
        except Exception as e:
            print(f"Error creating model info for {path}: {e}")
            return None

    def _determine_model_type(self, path: Path, components: ModelComponents) -> ModelType:
        """Determine the type of model from path and components."""
        name_lower = path.stem.lower()
        parent_lower = path.parent.name.lower()

        # Check parent directory name
        if "vae" in parent_lower:
            return ModelType.VAE
        if "clip" in parent_lower or "text_encoder" in parent_lower:
            return ModelType.CLIP

        # Check filename
        if "vae" in name_lower and not components.has_unet:
            return ModelType.VAE
        if ("clip" in name_lower or "text_encoder" in name_lower) and not components.has_unet:
            return ModelType.CLIP

        # Check components
        if components.has_unet:
            return ModelType.CHECKPOINT
        if components.has_vae and not components.has_unet:
            return ModelType.VAE
        if components.has_text_encoder and not components.has_unet:
            return ModelType.CLIP

        # Default to checkpoint if unclear
        return ModelType.CHECKPOINT

    def _categorize_model(self, model: ModelInfo) -> None:
        """Add model to the appropriate category list."""
        if model.model_type == ModelType.CHECKPOINT:
            self._checkpoints.append(model)
        elif model.model_type == ModelType.VAE:
            self._vaes.append(model)
        elif model.model_type == ModelType.CLIP:
            self._clips.append(model)

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
