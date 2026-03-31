"""
Results endpoint — returns pipeline outputs for a completed job.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .stream import job_status_store

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"


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
