"""Upscale backend using Real-ESRGAN models for image upscaling."""

from pathlib import Path
from typing import Optional, Callable
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block.

    Used in ESRGAN and Real-ESRGAN.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network for super-resolution.

    Used in Real-ESRGAN for fast inference.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    ):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # First conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # Activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # Body convs
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # Last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # Pixel shuffle
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # Add residual
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out = out + base
        return out


UPSCALE_AVAILABLE = True


class UpscaleBackend:
    """Backend for image upscaling using Real-ESRGAN models."""

    # Known model configurations
    MODEL_CONFIGS = {
        # RealESRGAN x4 models
        "RealESRGAN_x4plus": {
            "arch": "RRDBNet",
            "num_block": 23,
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "arch": "RRDBNet",
            "num_block": 6,
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        # RealESRGAN x2 models
        "RealESRGAN_x2plus": {
            "arch": "RRDBNet",
            "num_block": 23,
            "scale": 2,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        # RealESRNet models
        "RealESRNet_x4plus": {
            "arch": "RRDBNet",
            "num_block": 23,
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        # Compact models (faster)
        "realesr-general-x4v3": {
            "arch": "SRVGGNetCompact",
            "num_conv": 32,
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
        },
        "realesr-animevideov3": {
            "arch": "SRVGGNetCompact",
            "num_conv": 16,
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
        },
    }

    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._loaded_model_path: Optional[str] = None
        self._device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._scale: int = 4
        self._half: bool = True

    @property
    def is_available(self) -> bool:
        """Check if upscaling is available."""
        return UPSCALE_AVAILABLE

    @property
    def is_loaded(self) -> bool:
        """Check if an upscale model is loaded."""
        return self._model is not None

    @property
    def scale(self) -> int:
        """Get the current upscale factor."""
        return self._scale

    def set_device(self, device: str) -> None:
        """Set the device to use for upscaling."""
        self._device = device

    def load_model(
        self,
        model_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        Load an upscale model.

        Args:
            model_path: Path to the model file
            progress_callback: Optional progress callback

        Returns:
            True if loading succeeded
        """
        try:
            if progress_callback:
                progress_callback("Loading upscale model...", 0.1)

            # Unload existing model
            self.unload_model()

            model_name = Path(model_path).stem
            print(f"Loading upscale model: {model_name}")
            print(f"Model path: {model_path}")

            # Try to find matching config
            config = self._get_model_config(model_name, model_path)
            print(f"Using config: {config}")

            if progress_callback:
                progress_callback("Creating model architecture...", 0.3)

            # Create the model architecture
            if config["arch"] == "RRDBNet":
                model = RRDBNet(
                    num_in_ch=config.get("num_in_ch", 3),
                    num_out_ch=config.get("num_out_ch", 3),
                    num_feat=config.get("num_feat", 64),
                    num_block=config.get("num_block", 23),
                    num_grow_ch=config.get("num_grow_ch", 32),
                    scale=config.get("scale", 4),
                )
            elif config["arch"] == "SRVGGNetCompact":
                model = SRVGGNetCompact(
                    num_in_ch=config.get("num_in_ch", 3),
                    num_out_ch=config.get("num_out_ch", 3),
                    num_feat=config.get("num_feat", 64),
                    num_conv=config.get("num_conv", 32),
                    upscale=config.get("scale", 4),
                    act_type="prelu",
                )
            else:
                # Default to RRDBNet x4
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )

            self._scale = config.get("scale", 4)

            if progress_callback:
                progress_callback("Loading model weights...", 0.5)

            # Load state dict - handle safetensors separately
            if model_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    loadnet = load_file(model_path)
                except ImportError:
                    print("safetensors not available, cannot load .safetensors model")
                    return False
            else:
                # Try weights_only first, fall back to full load if needed
                try:
                    loadnet = torch.load(model_path, map_location="cpu", weights_only=True)
                except Exception:
                    print("weights_only=True failed, trying full load")
                    loadnet = torch.load(model_path, map_location="cpu")

            # Handle nested state dict
            if "params_ema" in loadnet:
                keyname = "params_ema"
                print("Found params_ema in state dict")
            elif "params" in loadnet:
                keyname = "params"
                print("Found params in state dict")
            else:
                keyname = None
                print("Using root state dict")

            state_dict = loadnet[keyname] if keyname else loadnet
            print(f"State dict keys: {list(state_dict.keys())[:5]}...")

            # Try strict loading first, fall back to non-strict
            try:
                model.load_state_dict(state_dict, strict=True)
                print("Loaded state dict with strict=True")
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Trying non-strict loading...")
                model.load_state_dict(state_dict, strict=False)
                print("Loaded state dict with strict=False")

            model.eval()
            model = model.to(torch.device(self._device))

            # Use half precision on CUDA for better performance
            self._half = "cuda" in self._device
            if self._half:
                model = model.half()

            self._model = model
            self._loaded_model_path = model_path

            if progress_callback:
                progress_callback("Upscale model loaded", 1.0)

            print(f"Upscale model loaded successfully on {self._device}")
            return True

        except Exception as e:
            print(f"Error loading upscale model: {e}")
            import traceback
            traceback.print_exc()
            self.unload_model()
            if progress_callback:
                progress_callback(f"Error: {e}", 0.0)
            return False

    def _get_model_config(self, model_name: str, model_path: str = "") -> dict:
        """Get configuration for a model based on its name or by inspecting weights."""
        # Check for exact match
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name]

        # Check for partial match
        model_name_lower = model_name.lower()
        for name, config in self.MODEL_CONFIGS.items():
            if name.lower() in model_name_lower or model_name_lower in name.lower():
                return config

        # Try to infer from name
        if "anime" in model_name_lower and "6b" in model_name_lower:
            return self.MODEL_CONFIGS["RealESRGAN_x4plus_anime_6B"]
        if "anime" in model_name_lower:
            return self.MODEL_CONFIGS["realesr-animevideov3"]
        if "x2" in model_name_lower:
            return self.MODEL_CONFIGS["RealESRGAN_x2plus"]
        if "compact" in model_name_lower or "srvgg" in model_name_lower:
            return self.MODEL_CONFIGS["realesr-general-x4v3"]

        # Try to auto-detect from weights file
        if model_path:
            detected = self._detect_model_config(model_path)
            if detected:
                print(f"Auto-detected model config: {detected}")
                return detected

        # Default configuration (RRDBNet x4)
        return {
            "arch": "RRDBNet",
            "num_block": 23,
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_grow_ch": 32,
        }

    def _detect_model_config(self, model_path: str) -> Optional[dict]:
        """Try to detect model configuration by inspecting the weights file."""
        try:
            # Load the state dict to inspect keys
            if model_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                except ImportError:
                    print("safetensors not available for model inspection")
                    return None
            else:
                try:
                    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                except Exception:
                    state_dict = torch.load(model_path, map_location="cpu")

            # Handle nested state dict (some models wrap in 'params' or 'params_ema')
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]

            keys = list(state_dict.keys())
            print(f"Model keys sample: {keys[:5]}")

            # Detect architecture based on key patterns
            # Check for SRVGGNetCompact first (has body.0, body.1, etc. with simple structure)
            if any(k.startswith("body.") for k in keys) and not any("rdb" in k.lower() for k in keys):
                # Could be SRVGGNetCompact - count layers
                max_idx = 0
                for k in keys:
                    if k.startswith("body.") and ".weight" in k:
                        try:
                            idx = int(k.split(".")[1])
                            max_idx = max(max_idx, idx)
                        except ValueError:
                            pass

                # SRVGGNetCompact has body.0 (first conv), body.1 (activation),
                # then pairs of conv+activation, then final conv
                # num_conv is roughly (max_idx - 2) / 2
                if max_idx > 0:
                    num_conv = max((max_idx - 2) // 2, 16)
                    return {
                        "arch": "SRVGGNetCompact",
                        "num_conv": num_conv,
                        "scale": 4,
                        "num_in_ch": 3,
                        "num_out_ch": 3,
                        "num_feat": 64,
                    }

            # Check for RRDBNet (has conv_first, body with RRDB blocks)
            if any("conv_first" in k for k in keys) or any("body." in k and "rdb" in k.lower() for k in keys):
                # RRDBNet architecture - count blocks
                num_block = 0
                for k in keys:
                    if "body." in k:
                        try:
                            parts = k.split(".")
                            for i, p in enumerate(parts):
                                if p == "body" and i + 1 < len(parts):
                                    block_num = int(parts[i + 1])
                                    num_block = max(num_block, block_num + 1)
                        except (ValueError, IndexError):
                            pass

                if num_block == 0:
                    num_block = 23  # Default

                # Detect scale from conv_up layers
                scale = 4
                has_up1 = any("conv_up1" in k for k in keys)
                has_up2 = any("conv_up2" in k for k in keys)
                has_up3 = any("conv_up3" in k for k in keys)

                if has_up3:
                    scale = 8
                elif has_up2:
                    scale = 4
                elif has_up1:
                    scale = 2

                return {
                    "arch": "RRDBNet",
                    "num_block": num_block,
                    "scale": scale,
                    "num_in_ch": 3,
                    "num_out_ch": 3,
                    "num_feat": 64,
                    "num_grow_ch": 32,
                }

            return None

        except Exception as e:
            print(f"Could not auto-detect model config: {e}")
            return None

    def unload_model(self) -> None:
        """Unload the current upscale model."""
        if self._model is not None:
            del self._model
            self._model = None

        self._loaded_model_path = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def upscale(
        self,
        image: Image.Image,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        tile_size: int = 0,
        tile_pad: int = 10,
    ) -> Optional[Image.Image]:
        """
        Upscale an image.

        Args:
            image: PIL Image to upscale
            progress_callback: Optional progress callback
            tile_size: Size of tiles for processing (0 = no tiling)
            tile_pad: Padding for tiles to avoid seams

        Returns:
            Upscaled PIL Image or None if upscaling failed
        """
        if not self.is_loaded:
            print("No upscale model loaded")
            return None

        try:
            if progress_callback:
                progress_callback("Preparing image...", 0.1)

            # Convert PIL to tensor
            img_np = np.array(image).astype(np.float32) / 255.0
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np] * 3, axis=2)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = img_np[:, :, :3]

            # HWC to CHW, then add batch dimension
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
            img_tensor = img_tensor.to(torch.device(self._device))

            if self._half:
                img_tensor = img_tensor.half()

            if progress_callback:
                progress_callback("Upscaling...", 0.3)

            # Process with or without tiling
            with torch.no_grad():
                if tile_size > 0:
                    output = self._tile_process(img_tensor, tile_size, tile_pad)
                else:
                    output = self._model(img_tensor)

            if progress_callback:
                progress_callback("Converting result...", 0.9)

            # Convert back to PIL
            output = output.squeeze(0).float().cpu().clamp(0, 1).numpy()
            output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            result = Image.fromarray(output)

            if progress_callback:
                progress_callback("Upscaling complete", 1.0)

            return result

        except Exception as e:
            print(f"Error upscaling image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _tile_process(
        self,
        img: torch.Tensor,
        tile_size: int,
        tile_pad: int,
    ) -> torch.Tensor:
        """Process image in tiles to reduce VRAM usage."""
        batch, channel, height, width = img.shape
        output_height = height * self._scale
        output_width = width * self._scale
        output_shape = (batch, channel, output_height, output_width)

        # Initialize output tensor
        output = img.new_zeros(output_shape)
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile bounds with padding
                x_start = x * tile_size
                y_start = y * tile_size
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)

                # Add padding
                x_start_pad = max(x_start - tile_pad, 0)
                y_start_pad = max(y_start - tile_pad, 0)
                x_end_pad = min(x_end + tile_pad, width)
                y_end_pad = min(y_end + tile_pad, height)

                # Extract tile
                tile = img[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad]

                # Process tile
                with torch.no_grad():
                    tile_output = self._model(tile)

                # Calculate output bounds
                out_x_start = x_start * self._scale
                out_y_start = y_start * self._scale
                out_x_end = x_end * self._scale
                out_y_end = y_end * self._scale

                # Calculate the region to copy (excluding padding)
                pad_left = (x_start - x_start_pad) * self._scale
                pad_top = (y_start - y_start_pad) * self._scale
                pad_right = pad_left + (x_end - x_start) * self._scale
                pad_bottom = pad_top + (y_end - y_start) * self._scale

                # Copy to output
                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                    tile_output[:, :, pad_top:pad_bottom, pad_left:pad_right]

        return output


# Global upscale backend instance
upscale_backend = UpscaleBackend()
