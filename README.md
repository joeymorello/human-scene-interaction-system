# Human-Scene Interaction System

Full-stack prototype for physical-to-digital human-scene interaction analysis. Ingests single-camera video, reconstructs 3D room and human motion using JOSH, identifies contacted objects with SAM3, and displays synced results in a real-time web interface.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐
│  ML Pipeline │───▸│  Backend API │───▸│  Frontend (Next.js)      │
│  (Python)    │    │  (FastAPI)   │    │  ┌────┬────────┬───────┐ │
│              │    │              │    │  │Vid │ 3D     │ Event │ │
│  JOSH        │    │  /upload     │    │  │    │ Canvas │ Log   │ │
│  SAM3        │    │  /stream-sse │    │  │    │ (R3F)  │       │ │
│  CLIP        │    │  /results    │    │  └────┴────────┴───────┘ │
└──────────────┘    └──────────────┘    └──────────────────────────┘
```

## Quick Start

### ML Pipeline (GPU required)
```bash
cd ml-pipeline
pip install -e .
bash setup_gpu.sh   # clones JOSH + SAM3
python scripts/run_pipeline.py input.mp4 -o ../data/results
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

## Pipeline Stages

1. **JOSH** — 4D human-scene reconstruction → point cloud + SMPL mesh + camera params + 3D contacts
2. **Contact Projection** — 3D contact coords → 2D pixel coords via camera intrinsics
3. **SAM3** — 2D contact points as positive prompts → object segmentation masks
4. **CLIP** — Zero-shot classification of masked crops → object labels

## GPU Requirements

- **Recommended:** RTX 4090 (24GB VRAM) — sequential model loading
- **Optimal:** RTX A6000 (48GB VRAM) — concurrent model loading
