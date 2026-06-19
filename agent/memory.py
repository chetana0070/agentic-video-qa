"""
Frame memory: stores encoded frames + metadata for retrieval during agent reasoning.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import numpy as np
from PIL import Image


@dataclass
class FrameRecord:
    frame_idx: int
    timestamp: float          # seconds
    embedding: torch.Tensor   # (D,) CLIP embedding
    caption: Optional[str] = None
    scene_id: Optional[int] = None


class VideoMemory:
    """
    Stores CLIP embeddings + captions for all sampled frames.
    Supports semantic retrieval and temporal queries.
    """

    def __init__(self):
        self.records: List[FrameRecord] = []
        self._embed_matrix: Optional[torch.Tensor] = None  # (N, D) cached

    def add(self, record: FrameRecord):
        self.records.append(record)
        self._embed_matrix = None  # invalidate cache

    @property
    def embed_matrix(self) -> torch.Tensor:
        if self._embed_matrix is None and self.records:
            self._embed_matrix = torch.stack([r.embedding for r in self.records])
        return self._embed_matrix

    def retrieve_by_query(self, query_embed: torch.Tensor, top_k: int = 5) -> List[FrameRecord]:
        """Cosine-similarity retrieval against query embedding."""
        if not self.records:
            return []
        mat = self.embed_matrix  # (N, D)
        q = query_embed / (query_embed.norm() + 1e-8)
        sims = (mat @ q).numpy()
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        return [self.records[i] for i in top_k_idx]

    def retrieve_by_time_range(self, start: float, end: float) -> List[FrameRecord]:
        """Return all frames within [start, end] seconds."""
        return [r for r in self.records if start <= r.timestamp <= end]

    def get_scene_summary(self) -> str:
        """Build a concise textual summary of scenes for the LLM context."""
        if not self.records:
            return "No frames loaded."
        scenes: dict = {}
        for r in self.records:
            sid = r.scene_id if r.scene_id is not None else 0
            scenes.setdefault(sid, []).append(r)

        lines = []
        for sid, recs in sorted(scenes.items()):
            t_start = recs[0].timestamp
            t_end = recs[-1].timestamp
            caps = [r.caption for r in recs if r.caption]
            sample_cap = caps[0] if caps else "no caption"
            lines.append(
                f"Scene {sid} [{t_start:.1f}s–{t_end:.1f}s]: {len(recs)} frames. "
                f"Sample: \"{sample_cap}\""
            )
        return "\n".join(lines)

    def __len__(self):
        return len(self.records)
