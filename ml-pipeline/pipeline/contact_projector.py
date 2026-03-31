"""
Contact Projector — Projects 3D contact points to 2D pixel coordinates.

Uses camera intrinsics/extrinsics from JOSH to map 3D human-scene
contact points back onto original video frames for SAM3 prompting.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Contact2D:
    """A projected 2D contact point on a video frame."""
    frame_index: int
    x: float
    y: float
    confidence: float = 1.0


class ContactProjector:
    """Projects 3D contact coordinates to 2D pixel space."""

    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray):
        """
        Args:
            intrinsics: Camera intrinsic matrix (3, 3)
            extrinsics: Camera extrinsic matrices (N, 4, 4) — world-to-camera
        """
        self.K = intrinsics
        self.extrinsics = extrinsics

    def project_point(self, point_3d: np.ndarray, extrinsic: np.ndarray) -> tuple[float, float]:
        """
        Project a single 3D point to 2D pixel coordinates.

        Args:
            point_3d: (3,) world coordinate
            extrinsic: (4, 4) world-to-camera transform

        Returns:
            (x, y) pixel coordinates
        """
        # Convert to homogeneous
        point_h = np.append(point_3d, 1.0)

        # World to camera
        point_cam = extrinsic @ point_h

        # Camera to pixel (perspective projection)
        point_proj = self.K @ point_cam[:3]
        x = point_proj[0] / point_proj[2]
        y = point_proj[1] / point_proj[2]

        return float(x), float(y)

    def project(self, contacts_3d: list[dict], frame_indices: list[int]) -> list[Contact2D]:
        """
        Project all 3D contact points to 2D.

        Args:
            contacts_3d: List of {"frame": int, "points": np.ndarray (M, 3)}
            frame_indices: Frame indices corresponding to extrinsics

        Returns:
            List of Contact2D with projected pixel coordinates
        """
        contacts_2d = []

        for contact in contacts_3d:
            frame_idx = contact["frame"]

            # Find the matching extrinsic matrix
            if frame_idx >= len(self.extrinsics):
                continue

            ext = self.extrinsics[frame_idx]
            points = contact["points"]

            for point in points:
                x, y = self.project_point(point, ext)
                contacts_2d.append(Contact2D(
                    frame_index=frame_idx,
                    x=x,
                    y=y,
                ))

        return contacts_2d
