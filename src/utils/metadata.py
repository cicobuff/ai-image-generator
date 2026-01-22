"""Metadata utilities for saving and loading generation parameters in PNG files."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo


@dataclass
class GenerationMetadata:
    """Metadata about how an image was generated."""
    # Model info
    checkpoint: str = ""
    vae: str = ""
    clip: str = ""

    # Generation parameters
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "euler"
    scheduler: str = "normal"

    # Additional info
    model_type: str = ""  # sdxl, sd15, etc.

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationMetadata":
        """Create metadata from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing metadata JSON: {e}")
            return cls()

    def to_png_info(self) -> PngInfo:
        """Create PngInfo object with metadata."""
        png_info = PngInfo()
        # Store as JSON in a custom chunk
        png_info.add_text("AIImageGenerator", self.to_json())
        # Also store human-readable parameters
        png_info.add_text("parameters", self._to_readable_string())
        return png_info

    def _to_readable_string(self) -> str:
        """Create human-readable parameter string (compatible with other tools)."""
        parts = [
            self.prompt,
            f"Negative prompt: {self.negative_prompt}" if self.negative_prompt else "",
            f"Steps: {self.steps}, Sampler: {self.sampler}, CFG scale: {self.cfg_scale}, Seed: {self.seed}, Size: {self.width}x{self.height}",
            f"Model: {self.checkpoint}",
        ]
        if self.vae:
            parts.append(f"VAE: {self.vae}")
        return "\n".join(p for p in parts if p)


def save_image_with_metadata(
    image: Image.Image,
    path: Path,
    metadata: GenerationMetadata,
) -> bool:
    """
    Save an image with generation metadata embedded.

    Args:
        image: PIL Image to save
        path: Path to save the image
        metadata: Generation metadata to embed

    Returns:
        True if save was successful
    """
    try:
        png_info = metadata.to_png_info()
        image.save(path, "PNG", pnginfo=png_info)
        return True
    except Exception as e:
        print(f"Error saving image with metadata: {e}")
        # Fall back to saving without metadata
        try:
            image.save(path, "PNG")
            return True
        except Exception as e2:
            print(f"Error saving image: {e2}")
            return False


def load_metadata_from_image(path: Path) -> Optional[GenerationMetadata]:
    """
    Load generation metadata from a PNG image.

    Args:
        path: Path to the PNG image

    Returns:
        GenerationMetadata if found, None otherwise
    """
    try:
        with Image.open(path) as img:
            # Try to get our custom metadata first
            if "AIImageGenerator" in img.info:
                return GenerationMetadata.from_json(img.info["AIImageGenerator"])

            # Fall back to trying to parse "parameters" field
            # (compatible with some other tools)
            if "parameters" in img.info:
                return _parse_parameters_string(img.info["parameters"])

        return None
    except Exception as e:
        print(f"Error loading metadata from {path}: {e}")
        return None


def _parse_parameters_string(params_str: str) -> Optional[GenerationMetadata]:
    """
    Try to parse a parameters string into metadata.
    This is a best-effort parser for compatibility.
    """
    try:
        metadata = GenerationMetadata()
        lines = params_str.strip().split("\n")

        if lines:
            # First line is usually the prompt
            metadata.prompt = lines[0]

        for line in lines[1:]:
            line = line.strip()

            # Check for negative prompt
            if line.startswith("Negative prompt:"):
                metadata.negative_prompt = line[16:].strip()
                continue

            # Parse key-value pairs
            if ":" in line:
                parts = line.split(",")
                for part in parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()

                        if key == "steps":
                            metadata.steps = int(value)
                        elif key == "sampler":
                            metadata.sampler = value.lower().replace(" ", "_")
                        elif key == "cfg scale":
                            metadata.cfg_scale = float(value)
                        elif key == "seed":
                            metadata.seed = int(value)
                        elif key == "size":
                            if "x" in value:
                                w, h = value.split("x")
                                metadata.width = int(w)
                                metadata.height = int(h)
                        elif key == "model":
                            metadata.checkpoint = value
                        elif key == "vae":
                            metadata.vae = value

        return metadata
    except Exception as e:
        print(f"Error parsing parameters string: {e}")
        return None
