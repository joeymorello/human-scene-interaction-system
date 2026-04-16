"""
Shared dataclasses for the HSI ML pipeline.

Kept dependency-light (only numpy) so both runners can import from here
without pulling in trimesh / torch at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class JOSHOutput:
    """Output from a reconstruction runner (JOSHRunner or JOSH3RRunner)."""
    ply_path: Path
    smpl_frames: list[Path]
    camera_intrinsics: np.ndarray  # (3, 3)
    camera_extrinsics: np.ndarray  # (N, 4, 4) per frame
    contacts_3d: list[dict]  # [{"frame": int, "points": np.ndarray (M, 3)}]
    frame_indices: list[int] = field(default_factory=list)
    fps: float = 30.0
