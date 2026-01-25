"""Segmentation backend using SAM3 for text-guided object detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import gc

import torch
import numpy as np
from PIL import Image


# Default local path for SAM3 model
SAM3_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "sams" / "sam3"


@dataclass
class DetectedMask:
    """A detected mask from SAM3 segmentation."""
    id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # Binary mask (H, W)
    selected: bool = True


class SegmentationBackend:
    """Backend for text-guided segmentation using SAM3."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._loaded: bool = False
        self._next_mask_id: int = 0
        self._model_path: Path = SAM3_MODEL_PATH

    @property
    def is_loaded(self) -> bool:
        """Check if the segmentation model is loaded."""
        return self._loaded

    def set_device(self, device: str) -> None:
        """Set the device to use for segmentation."""
        self._device = device

    def load_model(
        self,
        model_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        Load the SAM3 model.

        Args:
            model_path: Optional path to local SAM3 model directory
            progress_callback: Optional progress callback

        Returns:
            True if loading succeeded
        """
        try:
            if progress_callback:
                progress_callback("Loading SAM3 model...", 0.1)

            # Unload existing model first
            self.unload_model()

            # Determine model path
            if model_path is not None:
                self._model_path = Path(model_path)

            if progress_callback:
                progress_callback("Importing SAM3...", 0.2)

            # Import sam3 package
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            if progress_callback:
                progress_callback("Building SAM3 model...", 0.4)

            # Build the model
            self._model = build_sam3_image_model()

            if progress_callback:
                progress_callback("Creating processor...", 0.6)

            # Create processor
            self._processor = Sam3Processor(self._model)

            if progress_callback:
                progress_callback("SAM3 model loaded", 1.0)

            self._loaded = True
            print(f"SAM3 model loaded on {self._device}")
            return True

        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            import traceback
            traceback.print_exc()
            self.unload_model()
            if progress_callback:
                progress_callback(f"Error: {e}", 0.0)
            return False

    def unload_model(self) -> None:
        """Unload the SAM3 model to free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("SAM3 model unloaded")

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[DetectedMask]:
        """
        Detect objects in an image using a text prompt.

        SAM3 has native text prompt support.

        Args:
            image: PIL Image to analyze
            text_prompt: Text description of what to detect (e.g., "face", "hand")
            threshold: Confidence threshold for detection
            progress_callback: Optional progress callback

        Returns:
            List of DetectedMask objects
        """
        if not self._loaded:
            print("SAM3 model not loaded")
            return []

        try:
            if progress_callback:
                progress_callback(f"Detecting '{text_prompt}'...", 0.2)

            # Convert image to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_width, img_height = image.size

            if progress_callback:
                progress_callback("Setting image...", 0.4)

            # Set the image in the processor
            inference_state = self._processor.set_image(image)

            if progress_callback:
                progress_callback("Running text-prompted segmentation...", 0.6)

            # Run text-prompted segmentation
            output = self._processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

            if progress_callback:
                progress_callback("Processing results...", 0.8)

            # Extract masks, boxes, and scores
            masks = output.get("masks", [])
            boxes = output.get("boxes", [])
            scores = output.get("scores", [])

            masks_list = []

            for i in range(len(masks)):
                mask_data = masks[i]
                score = float(scores[i]) if i < len(scores) else 1.0

                # Skip low confidence detections
                if score < threshold:
                    continue

                # Convert mask to numpy
                if isinstance(mask_data, torch.Tensor):
                    mask_np = mask_data.cpu().numpy().astype(np.uint8)
                else:
                    mask_np = np.array(mask_data).astype(np.uint8)

                # Ensure mask is 2D
                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                # Get bounding box
                if i < len(boxes):
                    box = boxes[i]
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                else:
                    # Calculate bounding box from mask
                    ys, xs = np.where(mask_np > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = int(xs.min()), int(xs.max())
                        y1, y2 = int(ys.min()), int(ys.max())
                    else:
                        continue  # Skip empty masks

                # Skip very small masks (likely noise)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                detected = DetectedMask(
                    id=self._next_mask_id,
                    label=text_prompt,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    mask=mask_np,
                    selected=True,
                )
                masks_list.append(detected)
                self._next_mask_id += 1

            if progress_callback:
                progress_callback(f"Found {len(masks_list)} regions", 1.0)

            print(f"Detected {len(masks_list)} masks for '{text_prompt}'")
            return masks_list

        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(f"Error: {e}", 0.0)
            return []

    def reset_mask_ids(self) -> None:
        """Reset the mask ID counter."""
        self._next_mask_id = 0


# Global segmentation backend instance
segmentation_backend = SegmentationBackend()
