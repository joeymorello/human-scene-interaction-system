"""
JOSH Runner — Joint Optimization for 4D Human-Scene Reconstruction.

Wraps the JOSH model to produce:
- Scene point cloud (.ply)
- SMPL mesh sequence (.npz per frame)
- Camera intrinsics/extrinsics
- 3D contact coordinates (human-scene mesh intersections)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class JOSHOutput:
    """Output from JOSH reconstruction."""
    ply_path: Path
    smpl_frames: list[Path]
    camera_intrinsics: np.ndarray  # (3, 3)
    camera_extrinsics: np.ndarray  # (N, 4, 4) per frame
    contacts_3d: list[dict]  # [{"frame": int, "points": np.ndarray (M, 3)}]
    frame_indices: list[int] = field(default_factory=list)


class JOSHRunner:
    """Runs JOSH 4D human-scene reconstruction on input video."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._model = None

    def _load_model(self):
        """Load JOSH model and weights."""
        # TODO: Import from vendor/JOSH and initialize model
        # from vendor.JOSH.models import build_josh_model
        # self._model = build_josh_model(checkpoint_path)
        raise NotImplementedError(
            "JOSH model loading requires vendor/JOSH to be cloned. "
            "Run setup_gpu.sh first."
        )

    def run(self, video_path: Path, output_dir: Path) -> JOSHOutput:
        """
        Run JOSH on a video file.

        Args:
            video_path: Path to input .mp4
            output_dir: Directory for JOSH outputs

        Returns:
            JOSHOutput with all reconstruction results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._model is None:
            self._load_model()

        # TODO: Implement actual JOSH inference
        # 1. Extract video frames
        # 2. Run JOSH forward pass
        # 3. Export scene point cloud as .ply
        # 4. Export SMPL parameters per frame as .npz
        # 5. Extract camera parameters
        # 6. Compute human-scene contact points (mesh intersection)

        raise NotImplementedError(
            "JOSH inference pipeline not yet connected. "
            "This will be implemented once JOSH is cloned and its API is mapped."
        )
