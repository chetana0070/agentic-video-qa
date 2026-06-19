"""
Video processing utilities: loading, frame sampling, and preprocessing.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import torch


class VideoProcessor:
    """Handles video loading, sampling, and frame preprocessing."""

    def __init__(self, target_fps: float = 1.0, max_frames: int = 64):
        """
        Args:
            target_fps: Frames to sample per second of video.
            max_frames: Hard cap on total frames extracted.
        """
        self.target_fps = target_fps
        self.max_frames = max_frames

    def load_frames(self, video_path: str) -> Tuple[List[Image.Image], List[float]]:
        """
        Extract frames from a video file at target_fps.

        Returns:
            frames: List of PIL Images.
            timestamps: Corresponding timestamps in seconds.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / native_fps if native_fps > 0 else 0

        # Compute which frame indices to sample
        sample_interval = max(1, int(native_fps / self.target_fps))
        indices = list(range(0, total_frames, sample_interval))[: self.max_frames]

        frames, timestamps = [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(idx / native_fps)

        cap.release()
        return frames, timestamps

    def load_frames_from_dir(self, frame_dir: str) -> Tuple[List[Image.Image], List[float]]:
        """Load pre-extracted frames from a directory (jpg/png)."""
        frame_dir = Path(frame_dir)
        paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        paths = sorted(paths)[: self.max_frames]
        frames = [Image.open(p).convert("RGB") for p in paths]
        # Assume 1 fps if no metadata
        timestamps = [float(i) for i in range(len(frames))]
        return frames, timestamps

    @staticmethod
    def get_video_metadata(video_path: str) -> dict:
        """Return basic video metadata."""
        cap = cv2.VideoCapture(str(video_path))
        meta = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        meta["duration_sec"] = meta["total_frames"] / meta["fps"] if meta["fps"] > 0 else 0
        cap.release()
        return meta

    @staticmethod
    def uniform_sample(
        frames: List[Image.Image],
        timestamps: List[float],
        n: int,
    ) -> Tuple[List[Image.Image], List[float]]:
        """Uniformly downsample to exactly n frames."""
        if len(frames) <= n:
            return frames, timestamps
        indices = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
        return [frames[i] for i in indices], [timestamps[i] for i in indices]
