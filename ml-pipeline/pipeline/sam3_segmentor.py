"""
SAM3 Segmentor — Uses projected 2D contact points as positive prompts
to segment the objects being touched in each frame.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .contact_projector import Contact2D


@dataclass
class SegmentationMask:
    """A segmentation mask for a contacted object."""
    frame_index: int
    mask: np.ndarray  # (H, W) binary mask
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    contact_point: Contact2D


class SAM3Segmentor:
    """Runs SAM3 segmentation using 2D contact points as prompts."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._model = None
        self._predictor = None

    def _load_model(self):
        """Load SAM3 model."""
        # TODO: Import from vendor/sam3
        # from vendor.sam3 import build_sam3, SamPredictor
        # self._model = build_sam3(checkpoint="sam3_default.pth")
        # self._predictor = SamPredictor(self._model)
        raise NotImplementedError(
            "SAM3 model loading requires vendor/sam3 to be cloned. "
            "Run setup_gpu.sh first."
        )

    def segment_from_points(
        self, video_path: Path, contacts_2d: list[Contact2D]
    ) -> list[SegmentationMask]:
        """
        Segment objects at contact points across video frames.

        Args:
            video_path: Path to source video (for extracting frames)
            contacts_2d: 2D contact points to use as SAM3 positive prompts

        Returns:
            List of SegmentationMask for each contacted object
        """
        if self._predictor is None:
            self._load_model()

        # TODO: Implement SAM3 inference
        # 1. Extract frame at contact_2d.frame_index from video
        # 2. Set image on predictor
        # 3. Use (contact.x, contact.y) as positive point prompt
        # 4. Get mask, score, bbox
        # 5. Return SegmentationMask

        raise NotImplementedError(
            "SAM3 segmentation not yet connected. "
            "This will be implemented once SAM3 is cloned and its API is mapped."
        )
