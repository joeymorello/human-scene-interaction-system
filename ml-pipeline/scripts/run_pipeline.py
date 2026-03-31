"""
HSI ML Pipeline — Main entry point.

Takes an .mp4 video and produces:
1. A .ply point cloud of the scene (from JOSH)
2. A sequence of .npz files for SMPL human mesh (from JOSH)
3. Camera extrinsics/intrinsics (from JOSH)
4. 3D contact coordinates (human-scene intersections from JOSH)
5. SAM3 segmentation masks of touched objects (via 2D projected contact points)
6. CLIP labels for each segmented object
"""

import argparse
import json
import os
import sys
from pathlib import Path


def run_pipeline(video_path: str, output_dir: str, config_path: str = None,
                 device: str = "cuda"):
    """Run the full HSI pipeline on a video file."""
    # Ensure headless rendering works
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from pipeline.josh_runner import JOSHRunner
    from pipeline.contact_projector import ContactProjector
    from pipeline.sam3_segmentor import SAM3Segmentor
    from pipeline.clip_labeler import CLIPLabeler

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"=== HSI Pipeline ===")
    print(f"Input:  {video_path}")
    print(f"Output: {output_dir}")

    # Stage 1: JOSH — 4D Human-Scene Reconstruction
    print("\n[1/4] Running JOSH reconstruction...")
    josh = JOSHRunner(config_path=config_path, device=device)
    josh_output = josh.run(video_path, output_dir / "josh")
    print(f"  → Scene point cloud: {josh_output.ply_path}")
    print(f"  → SMPL sequence: {len(josh_output.smpl_frames)} frames")
    print(f"  → Contact points: {len(josh_output.contacts_3d)} events")

    # Stage 2: Project 3D contacts → 2D pixel coordinates
    print("\n[2/4] Projecting 3D contacts to 2D...")
    projector = ContactProjector(josh_output.camera_intrinsics, josh_output.camera_extrinsics)
    contacts_2d = projector.project(josh_output.contacts_3d, josh_output.frame_indices)
    print(f"  → {len(contacts_2d)} 2D contact points across frames")

    # Stage 3: SAM3 — Segment touched objects
    print("\n[3/4] Running SAM3 segmentation on contact regions...")
    segmentor = SAM3Segmentor(config_path=config_path)
    masks = segmentor.segment_from_points(
        video_path, contacts_2d, output_dir=output_dir / "sam3_work"
    )
    print(f"  → {len(masks)} object masks generated")

    # Stage 4: CLIP — Label segmented objects
    print("\n[4/4] Labeling objects with CLIP...")
    labeler = CLIPLabeler(config_path=config_path)
    interaction_events = labeler.label_masks(video_path, masks, contacts_2d)

    # Save results
    results_path = output_dir / "interactions.json"
    with open(results_path, "w") as f:
        json.dump(interaction_events, f, indent=2)
    print(f"\n=== Done ===")
    print(f"Interaction events saved to: {results_path}")

    return interaction_events


def main():
    parser = argparse.ArgumentParser(description="HSI ML Pipeline")
    parser.add_argument("video", type=str, help="Path to input .mp4 video")
    parser.add_argument("--output", "-o", type=str, default="../data/results",
                        help="Output directory (default: ../data/results)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config YAML (default: configs/default.yaml)")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    args = parser.parse_args()

    run_pipeline(args.video, args.output, args.config, args.device)


if __name__ == "__main__":
    main()
