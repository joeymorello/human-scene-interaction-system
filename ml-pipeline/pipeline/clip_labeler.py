"""
CLIP Labeler — Uses CLIP zero-shot classification to label
segmented object masks with human-readable string labels.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .contact_projector import Contact2D
from .sam3_segmentor import SegmentationMask

# Default candidate labels for zero-shot classification
DEFAULT_LABELS = [
    "wall", "floor", "table", "chair", "desk", "box",
    "shelf", "door", "window", "couch", "bed", "monitor",
    "keyboard", "person",
]


class CLIPLabeler:
    """Labels segmented object masks using CLIP zero-shot classification."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._model = None
        self._processor = None
        self.candidate_labels = DEFAULT_LABELS

    def _load_model(self):
        """Load CLIP model from HuggingFace."""
        from transformers import CLIPProcessor, CLIPModel

        model_name = "openai/clip-vit-base-patch32"
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)

    def _classify_crop(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Classify a masked crop using CLIP zero-shot.

        Args:
            crop: RGB image crop (H, W, 3)

        Returns:
            (label, confidence) tuple
        """
        from PIL import Image

        if self._model is None:
            self._load_model()

        image = Image.fromarray(crop)
        text_prompts = [f"a photo of a {label}" for label in self.candidate_labels]

        inputs = self._processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        outputs = self._model(**inputs)
        logits = outputs.logits_per_image.softmax(dim=1)
        best_idx = logits.argmax().item()

        return self.candidate_labels[best_idx], float(logits[0, best_idx])

    def label_masks(
        self,
        video_path: Path,
        masks: list[SegmentationMask],
        contacts_2d: list[Contact2D],
    ) -> list[dict]:
        """
        Label all segmented masks and produce interaction events.

        Args:
            video_path: Source video for extracting frame crops
            masks: Segmentation masks from SAM3
            contacts_2d: Original 2D contact points

        Returns:
            List of interaction event dicts:
            [{"time": float, "frame": int, "action": "touching", "object": str, "confidence": float}]
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        events = []

        for mask_result in masks:
            # Seek to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, mask_result.frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract crop from bounding box
            x1, y1, x2, y2 = mask_result.bbox
            crop = frame_rgb[y1:y2, x1:x2]

            # Apply mask to crop for cleaner classification
            mask_crop = mask_result.mask[y1:y2, x1:x2]
            crop[~mask_crop.astype(bool)] = 0

            if crop.size == 0:
                continue

            label, confidence = self._classify_crop(crop)

            events.append({
                "time": round(mask_result.frame_index / fps, 2),
                "frame": mask_result.frame_index,
                "action": "touching",
                "object": label,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            })

        cap.release()
        return events
