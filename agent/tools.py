"""
Tool definitions for the video QA agent.
Each tool is a callable that the agent can invoke during its reasoning loop.
"""
import json
from typing import Any, Callable, Dict, List
from PIL import Image

from agent.memory import VideoMemory, FrameRecord
from models.clip_encoder import CLIPVideoEncoder


# ─── Tool registry ────────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "retrieve_relevant_frames",
        "description": (
            "Search video memory for frames semantically relevant to a text query. "
            "Returns timestamps, frame indices, and captions of the top-k frames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic search query"},
                "top_k": {"type": "integer", "description": "Number of frames to return", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_scene_overview",
        "description": (
            "Get a structured overview of the video's scenes, including time ranges "
            "and representative captions."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_frames_in_range",
        "description": "Retrieve frames occurring between start_sec and end_sec in the video.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_sec": {"type": "number"},
                "end_sec": {"type": "number"},
            },
            "required": ["start_sec", "end_sec"],
        },
    },
    {
        "name": "count_events",
        "description": (
            "Count how many frames are semantically similar to a given concept "
            "(proxy for event frequency)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {"type": "string"},
                "threshold": {"type": "number", "default": 0.25},
            },
            "required": ["concept"],
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against video memory."""

    def __init__(self, memory: VideoMemory, encoder: CLIPVideoEncoder):
        self.memory = memory
        self.encoder = encoder
        self._dispatch: Dict[str, Callable] = {
            "retrieve_relevant_frames": self._retrieve_frames,
            "get_scene_overview": self._scene_overview,
            "get_frames_in_range": self._frames_in_range,
            "count_events": self._count_events,
        }

    def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        fn = self._dispatch.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return json.dumps(fn(**tool_args), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool implementations ──────────────────────────────────────────────────

    def _retrieve_frames(self, query: str, top_k: int = 5) -> dict:
        query_embed = self.encoder.encode_texts([query])[0]
        records = self.memory.retrieve_by_query(query_embed, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "frame_idx": r.frame_idx,
                    "timestamp_sec": round(r.timestamp, 2),
                    "scene_id": r.scene_id,
                    "caption": r.caption or "N/A",
                }
                for r in records
            ],
        }

    def _scene_overview(self) -> dict:
        return {"scene_summary": self.memory.get_scene_summary()}

    def _frames_in_range(self, start_sec: float, end_sec: float) -> dict:
        records = self.memory.retrieve_by_time_range(start_sec, end_sec)
        return {
            "range": [start_sec, end_sec],
            "num_frames": len(records),
            "frames": [
                {"frame_idx": r.frame_idx, "timestamp_sec": round(r.timestamp, 2), "caption": r.caption or "N/A"}
                for r in records
            ],
        }

    def _count_events(self, concept: str, threshold: float = 0.25) -> dict:
        if not self.memory.records:
            return {"concept": concept, "count": 0}
        query_embed = self.encoder.encode_texts([concept])[0]
        mat = self.memory.embed_matrix  # (N, D)
        sims = (mat @ query_embed).numpy()
        count = int((sims > threshold).sum())
        return {"concept": concept, "threshold": threshold, "matching_frames": count}
