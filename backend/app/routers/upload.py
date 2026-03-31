"""
Upload endpoint — receives video files and triggers the ML pipeline.
"""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from .stream import job_status_store, publish_status

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"


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


async def _run_pipeline(job_id: str, video_path: Path):
    """Run the ML pipeline asynchronously."""
    output_dir = RESULTS_DIR / job_id

    try:
        await publish_status(job_id, "processing", "Extracting video frames...")

        # Stage 1: JOSH Reconstruction
        await publish_status(job_id, "processing", "Running JOSH 4D reconstruction...")
        # TODO: Call josh_runner.run() in a thread pool
        # josh_output = await asyncio.to_thread(josh.run, video_path, output_dir / "josh")
        await asyncio.sleep(0.1)  # Placeholder

        # Stage 2: Contact Projection
        await publish_status(job_id, "processing", "Projecting 3D contacts to 2D...")
        await asyncio.sleep(0.1)

        # Stage 3: SAM3 Segmentation
        await publish_status(job_id, "processing", "Running SAM3 segmentation...")
        await asyncio.sleep(0.1)

        # Stage 4: CLIP Labeling
        await publish_status(job_id, "processing", "Labeling objects with CLIP...")
        await asyncio.sleep(0.1)

        await publish_status(job_id, "completed", "Pipeline finished successfully.")

    except Exception as e:
        await publish_status(job_id, "error", f"Pipeline failed: {str(e)}")
