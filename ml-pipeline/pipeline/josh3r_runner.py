"""
JOSH3R Runner — end-to-end global human trajectory + scene reconstruction.

Wraps the JOSH3R feed-forward model (from genforce/JOSH) to produce the same
output contract as `JOSHRunner`:
- Scene point cloud (.ply)
- SMPL mesh sequence (.npz per frame, world-frame vertices)
- Camera intrinsics/extrinsics
- 3D contact coordinates (computed post-hoc via proximity — JOSH3R itself does
  not estimate contact, unlike base JOSH + DECO)

Compared to JOSHRunner this runner:
- Does NOT run DECO (no learned contact stage)
- Does NOT run the JOSH optimization loop
- Runs JOSH3R's inference_josh3r() in a single feed-forward pass
- Is orders of magnitude faster than the optimization pipeline

Requires:
- `ml-pipeline/vendor/JOSH` with third-party submodules initialized
- A JOSH3R checkpoint (download from the JOSH repo's Google Drive link) —
  path provided via constructor or HSI_JOSH3R_CKPT env var.
- SMPL neutral body model at ml-pipeline/vendor/JOSH/data/smpl (as the
  JOSH3R demo expects).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np

from .types import JOSHOutput


def _setup_josh_paths() -> Path:
    """Add JOSH and its third-party deps to sys.path (same layout as JOSHRunner)."""
    josh_root = Path(__file__).resolve().parent.parent / "vendor" / "JOSH"
    paths = [
        str(josh_root),
        str(josh_root / "third_party" / "mast3r" / "dust3r" / "croco"),
        str(josh_root / "third_party" / "mast3r" / "dust3r"),
        str(josh_root / "third_party" / "mast3r"),
        str(josh_root / "third_party" / "pi3"),
        str(josh_root / "third_party" / "tram"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return josh_root


class JOSH3RRunner:
    """Runs the JOSH3R feed-forward reconstruction pipeline on an input video."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        smpl_dir: Optional[str] = None,
        device: str = "cuda",
        contact_eps_meters: float = 0.03,
    ):
        self.device = device
        self.contact_eps_meters = contact_eps_meters
        self._josh_root = _setup_josh_paths()

        self.ckpt_path = Path(
            ckpt_path or os.environ.get("HSI_JOSH3R_CKPT", "")
        )
        self.smpl_dir = Path(
            smpl_dir or os.environ.get(
                "HSI_JOSH3R_SMPL_DIR", self._josh_root / "data" / "smpl"
            )
        )
        self._model = None
        self._smpl_model = None

    # ------------------------------------------------------------------ #
    # Stage 1 — frame extraction
    # ------------------------------------------------------------------ #
    def _extract_frames(self, video_path: Path, output_dir: Path) -> float:
        """Extract frames to rgb/ dir at native fps. Returns video FPS."""
        import cv2

        rgb_dir = output_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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

    # ------------------------------------------------------------------ #
    # Stage 2 — preprocessing (SAM3 person masks + TRAM HPS)
    # JOSH3R needs TRAM .npy; TRAM in the JOSH repo consumes SAM3 masks.
    # DECO is skipped because JOSH3R has no contact stage.
    # ------------------------------------------------------------------ #
    def _run_preprocessing(self, input_folder: str) -> None:
        cwd = str(self._josh_root)

        print("  Running SAM3 person segmentation...")
        subprocess.run(
            [sys.executable, "-m", "preprocess.run_sam3",
             "--input_folder", input_folder],
            cwd=cwd, check=True,
        )

        print("  Running TRAM/VIMO HMR...")
        subprocess.run(
            [sys.executable, "-m", "preprocess.run_tram",
             "--input_folder", input_folder],
            cwd=cwd, check=True,
        )

    # ------------------------------------------------------------------ #
    # Stage 3 — JOSH3R feed-forward inference
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        import torch
        from josh.josh3r.model import JOSH3R
        from smplx import SMPL

        if not self.ckpt_path or not self.ckpt_path.exists():
            raise FileNotFoundError(
                "JOSH3R checkpoint not found. Set HSI_JOSH3R_CKPT or pass "
                f"ckpt_path to JOSH3RRunner. Looked at: {self.ckpt_path}"
            )
        if not self.smpl_dir.exists():
            raise FileNotFoundError(
                f"SMPL data dir not found at {self.smpl_dir}. Download the "
                "SMPL neutral body model (see JOSH repo README) and place it "
                "there, or set HSI_JOSH3R_SMPL_DIR."
            )

        model = JOSH3R().to(self.device)
        ckpt = torch.load(str(self.ckpt_path), map_location=self.device)
        model.load_state_dict(ckpt if "state_dict" not in ckpt else ckpt["state_dict"])
        model.eval()
        self._model = model
        self._smpl_model = SMPL(model_path=str(self.smpl_dir), gender="neutral")

    def _run_inference(self, input_folder: str):
        from josh.config import JOSHConfig
        from josh.inference_josh3r import inference_josh3r

        if self._model is None:
            self._load_model()

        cfg = JOSHConfig(input_folder=input_folder)
        cfg.visualize_results = False
        return inference_josh3r(self._model, self._smpl_model, self.device, cfg)

    # ------------------------------------------------------------------ #
    # Stage 4 — export in our pipeline's output contract
    # ------------------------------------------------------------------ #
    def _export_results(
        self, result, output_dir: Path, input_folder: Path, fps: float
    ) -> JOSHOutput:
        import trimesh

        from .contact_from_proximity import contacts_from_proximity

        output_dir.mkdir(parents=True, exist_ok=True)
        smpl_dir = output_dir / "smpl"
        smpl_dir.mkdir(exist_ok=True)

        # Recover the HPS frame indices (same order inference_josh3r walked).
        tram_files = sorted(glob(str(input_folder / "tram" / "*.npy")))
        if not tram_files:
            raise RuntimeError(f"TRAM predictions not found under {input_folder}/tram")
        pred_smpl = np.load(tram_files[0], allow_pickle=True).item()
        frame_indices = pred_smpl["frame"].numpy().astype(int).tolist()

        # Point cloud → .ply
        ply_path = output_dir / "scene.ply"
        if result.point_cloud is None:
            raise RuntimeError("JOSH3R returned no point cloud")
        result.point_cloud.export(str(ply_path))
        scene_points = np.asarray(result.point_cloud.vertices)

        # Per-frame SMPL meshes → .npz  (and collect verts for contact calc)
        smpl_frame_paths: list[Path] = []
        extrinsics: list[np.ndarray] = []
        smpl_verts_per_frame: list[np.ndarray] = []
        faces_once: Optional[np.ndarray] = None

        for frame_result, frame_idx in zip(result.frame_result, frame_indices):
            extrinsics.append(np.asarray(frame_result["pred_cam"]))
            if not frame_result["pred_smpl"]:
                continue
            _, mesh, _ = frame_result["pred_smpl"][0]  # single-person demo
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.uint32)
            if faces_once is None:
                faces_once = faces

            smpl_path = smpl_dir / f"frame_{frame_idx:06d}.npz"
            np.savez_compressed(
                str(smpl_path),
                frame_idx=frame_idx,
                vertices=verts,
                faces=faces,
            )
            smpl_frame_paths.append(smpl_path)
            smpl_verts_per_frame.append(verts)

        # Contacts — proximity between SMPL verts and scene PC
        contacts_3d = contacts_from_proximity(
            smpl_verts_per_frame,
            scene_points,
            frame_indices[: len(smpl_verts_per_frame)],
            eps_meters=self.contact_eps_meters,
        )

        # Cameras
        camera_intrinsics = np.asarray(result.intrinsics)
        camera_extrinsics = np.stack(extrinsics, axis=0)
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

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self, video_path: Path, output_dir: Path) -> JOSHOutput:
        """
        Run the JOSH3R pipeline on a video file.

        Steps:
          1. Extract video frames to rgb/
          2. Preprocessing (SAM3 + TRAM HPS)
          3. JOSH3R feed-forward inference
          4. Export PLY + SMPL npz + camera.npz; compute contacts post-hoc

        Returns a JOSHOutput identical in shape to JOSHRunner.run(), so the
        downstream ContactProjector → SAM3Segmentor → CLIPLabeler chain is
        drop-in compatible.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_folder = output_dir

        fps = self._extract_frames(video_path, output_dir)
        self._run_preprocessing(str(input_folder))
        result = self._run_inference(str(input_folder))
        return self._export_results(result, output_dir / "export", input_folder, fps)
