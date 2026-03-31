"""
HSI Backend API — FastAPI server linking the ML pipeline to the frontend.
"""

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import upload, results, stream

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

app = FastAPI(
    title="Human-Scene Interaction System",
    version="0.1.0",
    description="Backend API for HSI ML pipeline orchestration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result files (PLY, NPZ, video) as static assets
app.mount("/static/results", StaticFiles(directory=str(DATA_DIR / "results")), name="results")
app.mount("/static/uploads", StaticFiles(directory=str(DATA_DIR / "uploads")), name="uploads")

app.include_router(upload.router)
app.include_router(results.router)
app.include_router(stream.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
