"""
SAM3 Segmentor — Uses projected 2D contact points as positive prompts
to segment the objects being touched in each frame.

Uses SAM3's video predictor API with point prompts in normalized [0,1] coords.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .contact_projector import Contact2D


@dataclass
class SegmentationMask:
    """A segmentation mask for a contacted object."""
    frame_index: int
    mask: np.ndarray  # (H, W) binary mask
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    contact_point: Contact2D


class SAM3Segmentor:
    """Runs SAM3 segmentation using 2D contact points as prompts."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._predictor = None

    def _load_model(self):
        """Load SAM3 video predictor."""
        from sam3.model_builder import build_sam3_video_predictor
        self._predictor = build_sam3_video_predictor()

    def _group_contacts_by_frame(self, contacts_2d: list[Contact2D]) -> dict[int, list[Contact2D]]:
        """Group contact points by frame index."""
        groups: dict[int, list[Contact2D]] = {}
        for c in contacts_2d:
            groups.setdefault(c.frame_index, []).append(c)
        return groups

    def _prepare_video_frames(self, video_path: Path, output_dir: Path) -> Path:
        """Extract video frames to a directory for SAM3's video predictor."""
        frames_dir = output_dir / "sam3_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame)
            idx += 1
        cap.release()
        return frames_dir

    def segment_from_points(
        self, video_path: Path, contacts_2d: list[Contact2D],
        output_dir: Optional[Path] = None,
    ) -> list[SegmentationMask]:
        """
        Segment objects at contact points across video frames.

        Uses SAM3's video predictor with point prompts. For each unique
        contact frame, adds positive point prompts and propagates masks.

        Args:
            video_path: Path to source video
            contacts_2d: 2D contact points to use as SAM3 positive prompts
            output_dir: Directory for intermediate files

        Returns:
            List of SegmentationMask for each contacted object
        """
        if not contacts_2d:
            return []

        if self._predictor is None:
            self._load_model()

        # Get video dimensions
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Prepare frames directory for SAM3
        if output_dir is None:
            output_dir = Path(video_path).parent / "sam3_work"
        frames_dir = self._prepare_video_frames(video_path, output_dir)

        # Start SAM3 session
        response = self._predictor.handle_request({
            "type": "start_session",
            "resource_path": str(frames_dir),
        })
        session_id = response["session_id"]

        masks_out = []
        grouped = self._group_contacts_by_frame(contacts_2d)

        try:
            obj_counter = 1
            for frame_idx, contacts in grouped.items():
                # Add each contact point as a separate object prompt
                for contact in contacts:
                    # SAM3 expects normalized [0,1] coordinates
                    point_norm = torch.tensor(
                        [[contact.x / width, contact.y / height]],
                        dtype=torch.float32,
                    )
                    labels = torch.tensor([1], dtype=torch.int32)  # positive

                    response = self._predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_idx,
                        "points": point_norm,
                        "point_labels": labels,
                        "obj_id": obj_counter,
                        "clear_old_points": True,
                    })

                    outputs = response.get("outputs", {})
                    if "out_binary_masks" in outputs:
                        binary_masks = outputs["out_binary_masks"]
                        boxes = outputs.get("out_boxes_xywh", None)

                        for i in range(len(binary_masks)):
                            mask_np = binary_masks[i]
                            if isinstance(mask_np, torch.Tensor):
                                mask_np = mask_np.cpu().numpy()

                            # Compute bbox from mask if not provided
                            if boxes is not None and i < len(boxes):
                                bx, by, bw, bh = boxes[i]
                                bbox = (int(bx), int(by), int(bx + bw), int(by + bh))
                            else:
                                ys, xs = np.where(mask_np)
                                if len(ys) > 0:
                                    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                                else:
                                    continue

                            masks_out.append(SegmentationMask(
                                frame_index=frame_idx,
                                mask=mask_np.astype(bool),
                                bbox=bbox,
                                score=contact.confidence,
                                contact_point=contact,
                            ))

                    obj_counter += 1

        finally:
            self._predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": True,
            })

        return masks_out
