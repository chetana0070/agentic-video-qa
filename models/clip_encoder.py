"""
CLIP-based visual encoder for frame-level and video-level embeddings.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel


class CLIPVideoEncoder:
    """
    Wraps HuggingFace CLIP to encode video frames and compute
    text-visual similarity scores.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading CLIP model '{model_name}' on {device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_frames(self, frames: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """
        Encode a list of PIL frames into L2-normalized CLIP visual embeddings.

        Returns:
            Tensor of shape (N, D)
        """
        all_embeds = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            # Newer transformers versions may return a dataclass instead of a tensor
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
            feats = F.normalize(feats, dim=-1)
            all_embeds.append(feats.cpu())
        return torch.cat(all_embeds, dim=0)  # (N, D)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of text strings into L2-normalized CLIP text embeddings.

        Returns:
            Tensor of shape (M, D)
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        return F.normalize(feats, dim=-1).cpu()

    def score_frames(self, frame_embeds: torch.Tensor, query: str) -> np.ndarray:
        """
        Compute cosine similarity between each frame and a text query.

        Returns:
            scores: numpy array of shape (N,), values in [-1, 1]
        """
        text_embed = self.encode_texts([query])  # (1, D)
        scores = (frame_embeds @ text_embed.T).squeeze(-1).numpy()
        return scores

    def retrieve_top_k(
        self,
        frames: List[Image.Image],
        timestamps: List[float],
        query: str,
        k: int = 5,
    ) -> List[dict]:
        """
        Retrieve the top-k frames most relevant to a text query.

        Returns:
            List of dicts with keys: frame, timestamp, score, frame_idx
        """
        frame_embeds = self.encode_frames(frames)
        scores = self.score_frames(frame_embeds, query)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [
            {
                "frame": frames[i],
                "timestamp": timestamps[i],
                "score": float(scores[i]),
                "frame_idx": i,
            }
            for i in top_k_idx
        ]

    def segment_by_scene(
        self,
        frame_embeds: torch.Tensor,
        threshold: float = 0.85,
    ) -> List[List[int]]:
        """
        Simple scene segmentation based on consecutive frame similarity.
        Frames where similarity drops below threshold start a new scene.

        Returns:
            List of scenes, each scene is a list of frame indices.
        """
        scenes, current_scene = [], [0]
        for i in range(1, len(frame_embeds)):
            sim = float(frame_embeds[i - 1] @ frame_embeds[i])
            if sim < threshold:
                scenes.append(current_scene)
                current_scene = [i]
            else:
                current_scene.append(i)
        if current_scene:
            scenes.append(current_scene)
        return scenes
