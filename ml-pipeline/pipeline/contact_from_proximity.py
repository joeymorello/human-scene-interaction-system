"""
Post-hoc contact estimation by proximity between SMPL vertices and scene points.

Base JOSH produces per-frame contact vertex indices via DECO. JOSH3R does not —
it's a feed-forward trajectory/scene model only. To keep the downstream chain
(contact projection → SAM3 segmentation → CLIP labeling) working, we compute
contacts after the fact: for each frame, find SMPL vertices whose nearest
scene-point-cloud neighbour is within `eps_meters`, and treat those as
contact points.

The scene PC is in world coordinates, and so are the SMPL meshes exported by
JOSH3R (see josh3r_runner.py). That means proximity is just a KD-tree lookup.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def contacts_from_proximity(
    smpl_vertices_per_frame: list[np.ndarray],
    scene_points: np.ndarray,
    frame_indices: list[int],
    eps_meters: float = 0.03,
    min_contact_points: int = 3,
) -> list[dict]:
    """
    Compute per-frame 3D contact points via proximity.

    Args:
        smpl_vertices_per_frame: list of (V, 3) arrays, one per frame, in world coords.
        scene_points: (P, 3) array of scene point cloud in world coords.
        frame_indices: frame index for each entry of smpl_vertices_per_frame.
        eps_meters: distance threshold. SMPL vertices closer than this to any
            scene point are considered in contact.
        min_contact_points: frames with fewer contact points than this are dropped
            (reduces noise from single-vertex false positives).

    Returns:
        list of {"frame": int, "points": np.ndarray of shape (M, 3)} dicts, one
        per frame that has any contact. The `points` are SMPL vertex positions
        in world coordinates (same convention as JOSH's contacts_3d).
    """
    if scene_points is None or len(scene_points) == 0:
        return []

    tree = cKDTree(scene_points)
    contacts: list[dict] = []

    for frame_idx, verts in zip(frame_indices, smpl_vertices_per_frame):
        if verts is None or len(verts) == 0:
            continue
        dists, _ = tree.query(verts, k=1, distance_upper_bound=eps_meters)
        # cKDTree returns inf for points with no neighbour within the bound.
        mask = np.isfinite(dists)
        if mask.sum() < min_contact_points:
            continue
        contacts.append({
            "frame": int(frame_idx),
            "points": verts[mask].astype(np.float32),
        })

    return contacts
