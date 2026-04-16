"""
JOSH Runner — Joint Optimization for 4D Human-Scene Reconstruction.

Wraps the full JOSH pipeline (preprocessing + optimization) to produce:
- Scene point cloud (.ply)
- SMPL mesh sequence (.npz per frame)
- Camera intrinsics/extrinsics
- 3D contact coordinates (human-scene mesh intersections)

Requires: vendor/JOSH with all submodules initialized and checkpoints downloaded.
Must run from vendor/JOSH directory (or with correct PYTHONPATH/sys.path setup).
"""

import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .types import JOSHOutput


def _setup_josh_paths():
    """Add JOSH and its third-party deps to sys.path."""
    josh_root = Path(__file__).resolve().parent.parent / "vendor" / "JOSH"
    paths = [
        str(josh_root),
        str(josh_root / "third_party" / "mast3r" / "dust3r" / "croco"),
        str(josh_root / "third_party" / "mast3r" / "dust3r"),
        str(josh_root / "third_party" / "mast3r"),
        str(josh_root / "third_party" / "pi3"),
        str(josh_root / "third_party" / "tram"),
        str(josh_root / "third_party" / "deco"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Headless rendering
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return josh_root


class JOSHRunner:
    """Runs the full JOSH 4D human-scene reconstruction pipeline on input video."""

    def __init__(self, config_path: Optional[str] = None, device: str = "cuda"):
        self.config_path = config_path
        self.device = device
        self._scene_model = None
        self._josh_root = _setup_josh_paths()

    def _load_model(self):
        """Load MASt3R scene model (used by JOSH for dense reconstruction)."""
        import torch
        from mast3r.model import AsymmetricMASt3R

        self._scene_model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        ).to(self.device).eval()

    def _extract_frames(self, video_path: Path, output_dir: Path) -> float:
        """Extract video frames to rgb/ directory. Returns video FPS."""
        import cv2

        rgb_dir = output_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(rgb_dir / f"{frame_idx:06d}.jpg"), frame)
            frame_idx += 1

        cap.release()
        print(f"  Extracted {frame_idx} frames at {fps:.1f} FPS")
        return fps

    def _run_preprocessing(self, input_folder: str):
        """Run SAM3 segmentation, TRAM HMR, and DECO contact estimation."""
        cwd = str(self._josh_root)

        # SAM3 — person segmentation + tracking
        print("  Running SAM3 person segmentation...")
        subprocess.run(
            [sys.executable, "-m", "preprocess.run_sam3",
             "--input_folder", input_folder],
            cwd=cwd, check=True,
        )

        # TRAM — human mesh recovery (SMPL)
        print("  Running TRAM/VIMO HMR...")
        subprocess.run(
            [sys.executable, "-m", "preprocess.run_tram",
             "--input_folder", input_folder],
            cwd=cwd, check=True,
        )

        # DECO — contact estimation
        print("  Running DECO contact estimation...")
        subprocess.run(
            [sys.executable, "-m", "preprocess.run_deco",
             "--input_folder", input_folder],
            cwd=cwd, check=True,
        )

    def _run_inference(self, input_folder: str, num_frames: int):
        """Run JOSH joint optimization."""
        from josh.config import JOSHConfig, OptimizedResult
        from josh.inference import inference

        if self._scene_model is None:
            self._load_model()

        cfg = JOSHConfig(input_folder=input_folder)
        cfg.visualize_results = False

        if num_frames >= 200:
            # Long video — chunked processing
            from josh.inference_long_demo import main as run_long
            run_long(input_folder)
            from josh.aggregate_results import aggregate_results
            result, _ = aggregate_results(input_folder, cfg)
        else:
            result = inference(
                scene_model=self._scene_model,
                device=self.device,
                cfg=cfg,
            )

        return result

    def _export_results(
        self, result, output_dir: Path, fps: float
    ) -> JOSHOutput:
        """Extract and export results from OptimizedResult to our format."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export point cloud as .ply
        ply_path = output_dir / "scene.ply"
        if result.point_cloud is not None:
            result.point_cloud.export(str(ply_path))
        elif result.mesh is not None:
            # Fall back to mesh vertices as point cloud
            pc = trimesh.PointCloud(result.mesh.vertices)
            pc.export(str(ply_path))

        # 2. Export SMPL meshes and extract contacts per frame
        smpl_dir = output_dir / "smpl"
        smpl_dir.mkdir(exist_ok=True)
        smpl_frame_paths = []
        contacts_3d = []
        extrinsics = []
        frame_indices = []

        for frame_result in result.frame_result:
            frame_idx = frame_result["frame_idx"]
            frame_indices.append(frame_idx)
            extrinsics.append(frame_result["pred_cam"])

            # Export SMPL mesh data
            smpl_path = smpl_dir / f"frame_{frame_idx:06d}.npz"
            smpl_data = {"frame_idx": frame_idx}

            if len(frame_result["pred_smpl"]) > 0:
                all_vertices = []
                all_faces = []
                all_contacts = []

                for person_idx, mesh, contact in frame_result["pred_smpl"]:
                    all_vertices.append(mesh.vertices)
                    all_faces.append(mesh.faces)

                    # Extract 3D contact points (vertices where contact == 1)
                    if contact is not None:
                        import torch
                        if isinstance(contact, torch.Tensor):
                            contact_np = contact.cpu().numpy()
                        else:
                            contact_np = np.asarray(contact)
                        contact_vertex_ids = np.where(contact_np == 1)[0]
                        if len(contact_vertex_ids) > 0:
                            contact_points = mesh.vertices[contact_vertex_ids]
                            all_contacts.append(contact_points)

                smpl_data["vertices"] = np.stack(all_vertices)
                smpl_data["faces"] = np.stack(all_faces) if len(set(f.shape for f in all_faces)) == 1 else all_faces[0]

                if all_contacts:
                    combined_contacts = np.concatenate(all_contacts, axis=0)
                    contacts_3d.append({
                        "frame": frame_idx,
                        "points": combined_contacts,
                    })

            np.savez_compressed(str(smpl_path), **smpl_data)
            smpl_frame_paths.append(smpl_path)

        # 3. Camera parameters
        intrinsics = result.intrinsics
        if isinstance(intrinsics, np.ndarray):
            camera_intrinsics = intrinsics
        else:
            import torch
            camera_intrinsics = intrinsics.cpu().numpy() if isinstance(intrinsics, torch.Tensor) else np.array(intrinsics)

        camera_extrinsics = np.stack(extrinsics, axis=0)

        # Save camera params
        np.savez(
            str(output_dir / "camera.npz"),
            intrinsics=camera_intrinsics,
            extrinsics=camera_extrinsics,
            frame_indices=np.array(frame_indices),
        )

        return JOSHOutput(
            ply_path=ply_path,
            smpl_frames=smpl_frame_paths,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            contacts_3d=contacts_3d,
            frame_indices=frame_indices,
            fps=fps,
        )

    def run(self, video_path: Path, output_dir: Path) -> JOSHOutput:
        """
        Run the full JOSH pipeline on a video file.

        Steps:
        1. Extract video frames
        2. SAM3 person segmentation
        3. TRAM/VIMO human mesh recovery
        4. DECO contact estimation
        5. JOSH joint optimization
        6. Export results (PLY, SMPL, cameras, contacts)

        Args:
            video_path: Path to input .mp4
            output_dir: Directory for JOSH outputs

        Returns:
            JOSHOutput with all reconstruction results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        input_folder = str(output_dir)

        # Stage 1: Extract frames
        fps = self._extract_frames(video_path, output_dir)

        # Count frames
        num_frames = len(glob(str(output_dir / "rgb" / "*.jpg")))

        # Stage 2: Preprocessing (SAM3 + TRAM + DECO)
        self._run_preprocessing(input_folder)

        # Stage 3: JOSH joint optimization
        result = self._run_inference(input_folder, num_frames)

        # Stage 4: Export to our format
        return self._export_results(result, output_dir / "export", fps)
