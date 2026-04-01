"""Screenshot upload endpoint for debugging."""
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "screenshots"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/screenshot")
async def upload_screenshot(file: UploadFile = File(...)):
    path = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    return {"saved": str(path), "size": len(content)}

@router.get("/screenshot", response_class=HTMLResponse)
async def screenshot_form():
    return """<html><body style="background:#111;color:#eee;font-family:system-ui;display:flex;justify-content:center;padding-top:80px">
    <div style="max-width:500px"><h2>Upload Screenshot</h2>
    <form method="POST" action="/api/screenshot" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" style="margin:16px 0"><br>
    <button type="submit" style="background:#3b82f6;color:white;border:none;padding:10px 24px;border-radius:8px;cursor:pointer">Upload</button>
    </form></div></body></html>"""
