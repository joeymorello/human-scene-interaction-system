"""
SSE streaming endpoint — sends live processing updates to the frontend.
"""

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# In-memory job status store (swap for Redis in production)
job_status_store: dict[str, dict] = {}

# SSE event queues per job
_event_queues: dict[str, list[asyncio.Queue]] = {}


async def publish_status(job_id: str, status: str, message: str):
    """Publish a status update to all SSE listeners for a job."""
    event = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Update store
    if job_id in job_status_store:
        job_status_store[job_id]["status"] = status
        job_status_store[job_id]["progress"].append(event)

    # Push to all listeners
    if job_id in _event_queues:
        for queue in _event_queues[job_id]:
            await queue.put(event)


@router.get("/stream-status/{job_id}")
async def stream_status(job_id: str):
    """
    SSE endpoint for live pipeline progress updates.

    Sends events like:
        data: {"status": "processing", "message": "Running JOSH 4D reconstruction..."}
    """
    queue: asyncio.Queue = asyncio.Queue()

    if job_id not in _event_queues:
        _event_queues[job_id] = []
    _event_queues[job_id].append(queue)

    async def event_generator():
        try:
            # Send any existing progress first
            if job_id in job_status_store:
                for event in job_status_store[job_id]["progress"]:
                    yield f"data: {json.dumps(event)}\n\n"

            # Stream new events
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=300)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ("completed", "error"):
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'status': 'timeout', 'message': 'Connection timed out'})}\n\n"
        finally:
            if job_id in _event_queues:
                _event_queues[job_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
