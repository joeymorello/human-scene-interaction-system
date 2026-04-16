"""
Upload endpoint — receives video files and triggers the ML pipeline.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from .stream import job_status_store, publish_status

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
ML_PIPELINE_DIR = REPO_ROOT / "ml-pipeline"

# Default runner — "josh3r" (fast feed-forward) or "josh" (optimization). Users
# can override via env var when starting uvicorn.
DEFAULT_RUNNER = os.environ.get("HSI_RUNNER", "josh3r")
DEFAULT_DEVICE = os.environ.get("HSI_DEVICE", "cuda")


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file and start the ML pipeline.

    Returns a job_id that can be used to track progress via SSE
    and retrieve results when complete.
    """
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use .mp4, .mov, .avi, or .webm")

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    video_path = job_dir / file.filename
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    # Initialize job status
    job_status_store[job_id] = {
        "status": "queued",
        "video_path": str(video_path),
        "video_filename": file.filename,
        "progress": [],
    }

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline(job_id, video_path))

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Video '{file.filename}' uploaded. Pipeline starting.",
    }


def _invoke_pipeline(video_path: Path, output_dir: Path, runner: str, device: str):
    """
    Synchronous pipeline entry point, run in a threadpool from the async router.

    Kept at module scope so it's picklable / unit-testable and so the heavy
    imports (torch, trimesh, open3d) happen on the worker thread — not at
    backend startup.
    """
    sys.path.insert(0, str(ML_PIPELINE_DIR))
    from scripts.run_pipeline import run_pipeline

    return run_pipeline(
        video_path=str(video_path),
        output_dir=str(output_dir),
        device=device,
        runner=runner,
    )


async def _run_pipeline(job_id: str, video_path: Path):
    """Run the ML pipeline asynchronously in a worker thread."""
    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = DEFAULT_RUNNER
    device = DEFAULT_DEVICE

    try:
        await publish_status(
            job_id, "processing",
            f"Starting {runner.upper()} pipeline on {video_path.name}...",
        )

        interactions = await asyncio.to_thread(
            _invoke_pipeline, video_path, output_dir, runner, device,
        )

        await publish_status(
            job_id, "completed",
            f"Pipeline finished with {len(interactions)} interaction events.",
        )

    except Exception as e:
        # Surface the error class + message so the UI can show useful info.
        await publish_status(
            job_id, "error",
            f"Pipeline failed: {type(e).__name__}: {e}",
        )
