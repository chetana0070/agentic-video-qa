"""
Agentic Video QA system.

Architecture:
  1. VideoProcessor extracts frames
  2. CLIPVideoEncoder encodes frames into embeddings
  3. VideoMemory stores (embedding, timestamp, caption, scene_id)
  4. VideoAgent runs an LLM with tool-use loop to answer questions
     over the memory, calling tools like retrieve_relevant_frames,
     get_scene_overview, count_events, etc.

LLM backends supported:
  - Ollama (local, free): --llm-backend ollama --llm-model llama3.2
  - OpenAI API:           --llm-backend openai  --llm-model gpt-4o-mini
  - Rule-based (no LLM):  fallback when no backend is configured
"""
import json
import os
import urllib.request
from typing import List, Optional, Tuple
from PIL import Image

from video.processor import VideoProcessor
from models.clip_encoder import CLIPVideoEncoder
from agent.memory import VideoMemory, FrameRecord
from agent.tools import TOOLS, ToolExecutor


# Optional: simple caption fallback using CLIP zero-shot labels
CAPTION_CANDIDATES = [
    "a person talking", "outdoor scene", "indoor scene", "sports activity",
    "cooking or food", "vehicle or transportation", "crowd of people",
    "nature landscape", "text or presentation", "animal",
]

OLLAMA_BASE_URL = "http://localhost:11434"


class VideoAgent:
    """
    Agentic Video QA using an LLM with function-calling.
    Supports Ollama (local/free), OpenAI, or rule-based fallback.
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        llm_model: str = "llama3.2",
        llm_backend: str = "ollama",   # "ollama" | "openai" | "none"
        target_fps: float = 1.0,
        max_frames: int = 64,
        openai_api_key: Optional[str] = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ):
        self.encoder = CLIPVideoEncoder(clip_model)
        self.processor = VideoProcessor(target_fps=target_fps, max_frames=max_frames)
        self.llm_model = llm_model
        self.llm_backend = llm_backend
        self.memory = VideoMemory()
        self._client = None

        if llm_backend == "openai":
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                print(f"[VideoAgent] Using OpenAI backend: {llm_model}")
            else:
                print("[VideoAgent] No OPENAI_API_KEY — falling back to rule-based mode.")
                self.llm_backend = "none"

        elif llm_backend == "ollama":
            self._ollama_url = ollama_base_url
            if self._check_ollama():
                print(f"[VideoAgent] Using Ollama backend: {llm_model}")
            else:
                print("[VideoAgent] Ollama not reachable — falling back to rule-based mode.")
                self.llm_backend = "none"

        else:
            print("[VideoAgent] No LLM backend — using rule-based fallback mode.")
            self.llm_backend = "none"

    def _check_ollama(self) -> bool:
        try:
            urllib.request.urlopen(f"{self._ollama_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_video(self, video_path: str) -> None:
        """Load, encode, and store all frames from a video file."""
        print(f"[VideoAgent] Indexing: {video_path}")
        frames, timestamps = self.processor.load_frames(video_path)
        self._index_frames(frames, timestamps)

    def index_frame_dir(self, frame_dir: str) -> None:
        """Index pre-extracted frames from a directory."""
        print(f"[VideoAgent] Indexing frames from: {frame_dir}")
        frames, timestamps = self.processor.load_frames_from_dir(frame_dir)
        self._index_frames(frames, timestamps)

    def _index_frames(self, frames: List[Image.Image], timestamps: List[float]) -> None:
        self.memory = VideoMemory()
        print(f"  Encoding {len(frames)} frames with CLIP...")
        embeds = self.encoder.encode_frames(frames)  # (N, D)

        # Scene segmentation
        scenes = self.encoder.segment_by_scene(embeds)
        frame_to_scene = {}
        for sid, scene_frames in enumerate(scenes):
            for fi in scene_frames:
                frame_to_scene[fi] = sid

        # Zero-shot captions via CLIP
        caption_embeds = self.encoder.encode_texts(CAPTION_CANDIDATES)
        sims = (embeds @ caption_embeds.T).numpy()  # (N, C)
        best_caption_idx = sims.argmax(axis=1)

        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            record = FrameRecord(
                frame_idx=i,
                timestamp=ts,
                embedding=embeds[i],
                caption=CAPTION_CANDIDATES[best_caption_idx[i]],
                scene_id=frame_to_scene.get(i, 0),
            )
            self.memory.add(record)

        print(f"  Indexed {len(self.memory)} frames across {len(scenes)} scenes.")

    # ── QA ────────────────────────────────────────────────────────────────────

    def answer(self, question: str, max_tool_calls: int = 5) -> str:
        """Answer a natural-language question about the indexed video."""
        if not self.memory.records:
            return "No video has been indexed yet. Call index_video() first."

        if self.llm_backend == "openai":
            return self._answer_openai(question, max_tool_calls)
        elif self.llm_backend == "ollama":
            return self._answer_ollama(question, max_tool_calls)
        else:
            return self._answer_rule_based(question)

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert video analyst. You have access to an indexed video with "
            f"{len(self.memory)} frames. Use the provided tools to retrieve relevant visual "
            "information, then answer the user's question concisely and accurately.\n\n"
            "Video overview:\n" + self.memory.get_scene_summary()
        )

    def _answer_openai(self, question: str, max_tool_calls: int) -> str:
        executor = ToolExecutor(self.memory, self.encoder)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question},
        ]
        for _ in range(max_tool_calls):
            response = self._client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=[{"type": "function", "function": t} for t in TOOLS],
                tool_choice="auto",
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = executor.execute(tc.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                return msg.content or ""
        messages.append({"role": "user", "content": "Now provide your final answer."})
        final = self._client.chat.completions.create(model=self.llm_model, messages=messages)
        return final.choices[0].message.content or ""

    def _answer_ollama(self, question: str, max_tool_calls: int) -> str:
        """
        Ollama agentic loop using the /api/chat endpoint with tool_calls support.
        Ollama supports OpenAI-compatible tool calling for models like llama3.2, mistral, qwen2.5.
        Falls back to ReAct-style prompting if the model doesn't support native tool calling.
        """
        executor = ToolExecutor(self.memory, self.encoder)

        # Try native tool-calling first (Ollama ≥0.3 + llama3.1/3.2/mistral/qwen2.5)
        try:
            return self._ollama_tool_calling(question, max_tool_calls, executor)
        except Exception:
            # Fallback: ReAct-style text prompting
            return self._ollama_react(question, max_tool_calls, executor)

    def _ollama_tool_calling(self, question: str, max_tool_calls: int, executor: ToolExecutor) -> str:
        """Native tool-calling via Ollama's OpenAI-compatible endpoint."""
        import urllib.request, urllib.error
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question},
        ]
        tools_payload = [{"type": "function", "function": t} for t in TOOLS]

        for _ in range(max_tool_calls):
            payload = json.dumps({
                "model": self.llm_model,
                "messages": messages,
                "tools": tools_payload,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self._ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())

            msg = data["message"]
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    fn = tc["function"]
                    args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
                    result = executor.execute(fn["name"], args)
                    messages.append({"role": "tool", "content": result})
            else:
                return msg.get("content", "").strip()

        return msg.get("content", "").strip()

    def _ollama_react(self, question: str, max_tool_calls: int, executor: ToolExecutor) -> str:
        """
        ReAct-style prompting fallback for models without native tool calling.
        The LLM emits: Thought / Action / Action Input lines which we parse and execute.
        """
        import urllib.request

        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in TOOLS
        )
        system = (
            self._build_system_prompt() + "\n\n"
            "You have access to the following tools:\n" + tools_desc + "\n\n"
            "To use a tool, respond EXACTLY in this format (no extra text before Action):\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <JSON args>\n\n"
            "When you have enough information, respond with:\n"
            "Final Answer: <your answer>\n"
        )
        prompt = system + f"\nQuestion: {question}\n"

        for _ in range(max_tool_calls):
            payload = json.dumps({
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self._ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            response_text = data.get("response", "").strip()
            prompt += response_text + "\n"

            if "Final Answer:" in response_text:
                return response_text.split("Final Answer:")[-1].strip()

            # Parse Action / Action Input
            if "Action:" in response_text and "Action Input:" in response_text:
                try:
                    action = response_text.split("Action:")[1].split("\n")[0].strip()
                    action_input_str = response_text.split("Action Input:")[1].split("\n")[0].strip()
                    args = json.loads(action_input_str)
                    result = executor.execute(action, args)
                    prompt += f"Observation: {result}\n"
                except Exception as e:
                    prompt += f"Observation: Error parsing action — {e}\n"

        # Ask for final answer explicitly
        payload = json.dumps({
            "model": self.llm_model,
            "prompt": prompt + "Final Answer:",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()

    def _answer_rule_based(self, question: str) -> str:
        """Simple retrieval-based fallback (no LLM)."""
        executor = ToolExecutor(self.memory, self.encoder)
        result = json.loads(executor.execute("retrieve_relevant_frames", {"query": question, "top_k": 3}))
        overview = self.memory.get_scene_summary()
        lines = [f"[Rule-based mode] Top frames for: '{question}'"]
        for r in result.get("results", []):
            lines.append(f"  • t={r['timestamp_sec']}s | scene={r['scene_id']} | caption='{r['caption']}'")
        lines.append("\nVideo overview:\n" + overview)
        return "\n".join(lines)

    def batch_eval(self, qa_pairs: List[Tuple[str, str]]) -> dict:
        """
        Evaluate on a list of (question, ground_truth_answer) pairs.
        Returns accuracy using simple substring matching.
        """
        correct = 0
        results = []
        for question, gt in qa_pairs:
            pred = self.answer(question)
            match = gt.lower() in pred.lower()
            correct += int(match)
            results.append({"question": question, "gt": gt, "pred": pred, "correct": match})
        return {"accuracy": correct / len(qa_pairs) if qa_pairs else 0, "details": results}
