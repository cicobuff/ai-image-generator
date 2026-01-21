"""Model inspection utilities for detecting embedded components."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

from safetensors import safe_open


@dataclass
class ModelComponents:
    """Detected components in a model file."""
    has_unet: bool = False
    has_vae: bool = False
    has_text_encoder: bool = False  # CLIP
    has_text_encoder_2: bool = False  # CLIP for SDXL
    is_sdxl: bool = False
    is_sd15: bool = False
    is_inpainting: bool = False
    model_type: str = "unknown"


class ModelInspector:
    """Inspects model files to detect embedded components."""

    # Key prefixes that indicate different components
    VAE_PREFIXES = (
        "first_stage_model.",
        "vae.",
    )

    TEXT_ENCODER_PREFIXES = (
        "cond_stage_model.",
        "text_encoder.",
    )

    TEXT_ENCODER_2_PREFIXES = (
        "conditioner.embedders.1.",
        "text_encoder_2.",
    )

    UNET_PREFIXES = (
        "model.diffusion_model.",
        "unet.",
    )

    # SDXL-specific indicators
    SDXL_INDICATORS = (
        "conditioner.embedders.1.",
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.",
        "model.diffusion_model.middle_block.1.transformer_blocks.4.",
    )

    def inspect_safetensors(self, path: Path) -> ModelComponents:
        """
        Inspect a safetensors file to detect embedded components.

        Args:
            path: Path to the safetensors file

        Returns:
            ModelComponents with detected features
        """
        components = ModelComponents()

        if not path.exists() or path.suffix != ".safetensors":
            return components

        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                keys = set(f.keys())

            # Check for each component
            components.has_unet = self._has_prefix(keys, self.UNET_PREFIXES)
            components.has_vae = self._has_prefix(keys, self.VAE_PREFIXES)
            components.has_text_encoder = self._has_prefix(keys, self.TEXT_ENCODER_PREFIXES)
            components.has_text_encoder_2 = self._has_prefix(keys, self.TEXT_ENCODER_2_PREFIXES)

            # Detect model type
            components.is_sdxl = self._has_any_key(keys, self.SDXL_INDICATORS)

            if components.is_sdxl:
                components.model_type = "sdxl"
            elif components.has_unet:
                # Check for SD 1.5 vs other
                if self._is_sd15(keys):
                    components.is_sd15 = True
                    components.model_type = "sd15"
                else:
                    components.model_type = "sd"

            # Check for inpainting model
            components.is_inpainting = self._is_inpainting_model(keys)
            if components.is_inpainting:
                components.model_type += "_inpainting"

        except Exception as e:
            print(f"Error inspecting {path}: {e}")

        return components

    def inspect_checkpoint(self, path: Path) -> ModelComponents:
        """
        Inspect a .ckpt or .pt file to detect embedded components.

        Note: This requires loading the file which is slower than safetensors.
        For .ckpt files, we use a minimal inspection approach.
        """
        components = ModelComponents()

        if not path.exists():
            return components

        try:
            import torch
            # Only load keys, not full tensors
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

            if "state_dict" in state_dict:
                keys = set(state_dict["state_dict"].keys())
            else:
                keys = set(state_dict.keys())

            # Same detection logic as safetensors
            components.has_unet = self._has_prefix(keys, self.UNET_PREFIXES)
            components.has_vae = self._has_prefix(keys, self.VAE_PREFIXES)
            components.has_text_encoder = self._has_prefix(keys, self.TEXT_ENCODER_PREFIXES)
            components.has_text_encoder_2 = self._has_prefix(keys, self.TEXT_ENCODER_2_PREFIXES)
            components.is_sdxl = self._has_any_key(keys, self.SDXL_INDICATORS)

            if components.is_sdxl:
                components.model_type = "sdxl"
            elif components.has_unet:
                if self._is_sd15(keys):
                    components.is_sd15 = True
                    components.model_type = "sd15"
                else:
                    components.model_type = "sd"

            components.is_inpainting = self._is_inpainting_model(keys)
            if components.is_inpainting:
                components.model_type += "_inpainting"

        except Exception as e:
            print(f"Error inspecting {path}: {e}")

        return components

    def inspect(self, path: Path) -> ModelComponents:
        """
        Inspect any supported model file.

        Args:
            path: Path to the model file

        Returns:
            ModelComponents with detected features
        """
        if path.suffix == ".safetensors":
            return self.inspect_safetensors(path)
        elif path.suffix in (".ckpt", ".pt"):
            return self.inspect_checkpoint(path)
        else:
            return ModelComponents()

    def get_safetensors_metadata(self, path: Path) -> Optional[dict]:
        """
        Get metadata from a safetensors file header.

        Some models store useful info in the metadata.
        """
        if not path.exists() or path.suffix != ".safetensors":
            return None

        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                return dict(f.metadata()) if f.metadata() else None
        except Exception:
            return None

    def _has_prefix(self, keys: set[str], prefixes: tuple[str, ...]) -> bool:
        """Check if any key starts with any of the given prefixes."""
        for key in keys:
            for prefix in prefixes:
                if key.startswith(prefix):
                    return True
        return False

    def _has_any_key(self, keys: set[str], indicators: tuple[str, ...]) -> bool:
        """Check if any of the indicator strings are substrings of any key."""
        for key in keys:
            for indicator in indicators:
                if indicator in key:
                    return True
        return False

    def _is_sd15(self, keys: set[str]) -> bool:
        """Detect if model is SD 1.5 architecture."""
        # SD 1.5 has specific layer structure
        # Check for presence of typical SD 1.5 layers but not SDXL layers
        sd15_indicator = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0."
        has_sd15 = any(sd15_indicator in k for k in keys)

        # Make sure it's not SDXL
        is_sdxl = self._has_any_key(keys, self.SDXL_INDICATORS)

        return has_sd15 and not is_sdxl

    def _is_inpainting_model(self, keys: set[str]) -> bool:
        """Detect if model is an inpainting model (has 9 input channels instead of 4)."""
        # Inpainting models have additional input channels for mask
        # Check for specific inpainting layer patterns
        for key in keys:
            if "model.diffusion_model.input_blocks.0.0.weight" in key:
                # Would need to check tensor shape, but for safetensors
                # we can't easily do that without loading
                # For now, rely on filename heuristics
                break
        return False


# Global model inspector instance
model_inspector = ModelInspector()
