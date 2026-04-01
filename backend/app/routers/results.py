"""
Results endpoint — returns pipeline outputs for a completed job.
"""

import io
import json
import struct
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .stream import job_status_store

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"


@router.get("/results/{result_id}/smpl.bin")
async def get_smpl_binary(result_id: str):
    """
    Serve all SMPL frames as a single binary blob for efficient browser loading.

    Format:
        4 bytes: uint32 num_frames
        4 bytes: uint32 num_vertices (V)
        4 bytes: uint32 num_faces (F)
        F*3*4 bytes: uint32[] face indices (shared across frames)
        Then for each frame:
            V*3*4 bytes: float32[] vertex positions
    """
    smpl_dir = RESULTS_DIR / result_id / "smpl"
    if not smpl_dir.exists():
        raise HTTPException(status_code=404, detail="SMPL data not found")

    npz_files = sorted(smpl_dir.glob("*.npz"))
    if not npz_files:
        raise HTTPException(status_code=404, detail="No SMPL frames found")

    # Load first frame to get dimensions
    first = np.load(npz_files[0])
    faces = first["faces"].astype(np.uint32)
    num_verts = first["vertices"].shape[0]
    num_faces = faces.shape[0]
    num_frames = len(npz_files)

    buf = io.BytesIO()
    # Header
    buf.write(struct.pack("<III", num_frames, num_verts, num_faces))
    # Faces (shared)
    buf.write(faces.tobytes())
    # Vertices per frame
    for f in npz_files:
        data = np.load(f)
        verts = data["vertices"].astype(np.float32)
        buf.write(verts.tobytes())

    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/results/test_run")
async def get_test_results():
    """Serve the test_run pipeline results directly for development."""
    result_dir = RESULTS_DIR / "test_run"
    base_url = "/static/results/test_run"

    ply_files = list(result_dir.glob("*.ply"))
    smpl_files = sorted((result_dir / "smpl").glob("*.npz"))
    interactions_path = result_dir / "interactions.json"

    interactions = []
    if interactions_path.exists():
        with open(interactions_path) as f:
            interactions = json.load(f)

    return {
        "job_id": "test_run",
        "status": "completed",
        "ply_url": f"{base_url}/{ply_files[0].name}" if ply_files else None,
        "smpl_urls": [f"{base_url}/smpl/{f.name}" for f in smpl_files],
        "video_url": "/static/uploads/test_5s.mp4",
        "interactions": interactions,
    }


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Get pipeline results for a completed job.

    Returns:
        - ply_url: URL to download the scene point cloud
        - smpl_urls: URLs for SMPL mesh sequence (.npz files)
        - video_url: URL to the original uploaded video
        - interactions: Timestamped interaction events
    """
    if job_id not in job_status_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_status_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not yet complete. Current status: {job['status']}",
        )

    result_dir = RESULTS_DIR / job_id
    base_url = f"/static/results/{job_id}"

    # Find output files
    ply_files = list(result_dir.glob("**/*.ply"))
    smpl_files = sorted(result_dir.glob("**/*.npz"))
    interactions_path = result_dir / "interactions.json"

    interactions = []
    if interactions_path.exists():
        with open(interactions_path) as f:
            interactions = json.load(f)

    return {
        "job_id": job_id,
        "status": "completed",
        "ply_url": f"{base_url}/{ply_files[0].relative_to(result_dir)}" if ply_files else None,
        "smpl_urls": [f"{base_url}/{f.relative_to(result_dir)}" for f in smpl_files],
        "video_url": f"/static/uploads/{job_id}/{job['video_filename']}",
        "interactions": interactions,
    }
