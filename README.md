# Agentic Video QA

An agentic system for long-form video understanding using CLIP embeddings and LLM function-calling. The agent reasons over video frames through a structured tool-use loop retrieving semantically relevant frames, detecting scenes, counting events, and synthesizing a natural-language answer.

Supports **Ollama (local/free)**, OpenAI, or a CLIP-only retrieval fallback with no LLM required.

---

## Architecture

```
Video File
    │
    ▼
VideoProcessor          ← Frame extraction & uniform sampling (OpenCV)
    │
    ▼
CLIPVideoEncoder        ← Visual embeddings + zero-shot captions + scene segmentation
    │
    ▼
VideoMemory             ← Embedding store with semantic & temporal retrieval
    │
    ▼
VideoAgent (LLM loop)   ← Ollama (llama3.2) or OpenAI with tool-calling
    ├── retrieve_relevant_frames
    ├── get_scene_overview
    ├── get_frames_in_range
    └── count_events
    │
    ▼
Natural-language answer
```

---

## Key Concepts

| Concept | Where |
|---|---|
| Vision-Language Models (CLIP) | `models/clip_encoder.py` |
| Agentic tool-use loop | `agent/video_agent.py` |
| Ollama / OpenAI LLM backends | `agent/video_agent.py` |
| ReAct-style prompting fallback | `agent/video_agent.py:_ollama_react` |
| Frame memory & retrieval | `agent/memory.py` |
| Function/tool definitions | `agent/tools.py` |
| Scene segmentation | `models/clip_encoder.py:segment_by_scene` |
| Batch evaluation / benchmarking | `agent/video_agent.py:batch_eval` |

---

## Setup

```bash
pip install -r requirements.txt
```

**Install Ollama** (free, runs locally): download from [ollama.com](https://ollama.com/download), then pull a model:

```bash
ollama pull llama3.2
```

---

## Usage

```bash
# Ollama (local, free) — recommended
python demo.py --video path/to/video.mp4 --question "What is happening in this video?" --llm-backend ollama --llm-model llama3.2

# Smaller/faster model
python demo.py --video path/to/video.mp4 --question "How many scenes are there?" --llm-model qwen2.5:3b

# OpenAI
python demo.py --video path/to/video.mp4 --question "..." --llm-backend openai --llm-model gpt-4o-mini

# CLIP-only retrieval (no LLM, instant)
python demo.py --video path/to/video.mp4 --question "Is there any outdoor activity?" --llm-backend none
```

### Python API

```python
from agent.video_agent import VideoAgent

# Local Ollama
agent = VideoAgent(llm_backend="ollama", llm_model="llama3.2", target_fps=1.0, max_frames=64)
agent.index_video("path/to/video.mp4")

answer = agent.answer("What activities are shown in the first half of the video?")
print(answer)

# Batch evaluation
results = agent.batch_eval([
    ("What is the setting?", "outdoor"),
    ("Is there a person speaking?", "yes"),
])
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

## LLM Backends

| Backend | Model | Cost | Requires |
|---|---|---|---|
| Ollama | llama3.2, mistral, qwen2.5 | Free | [Ollama](https://ollama.com) installed |
| OpenAI | gpt-4o-mini, gpt-4o | Paid | `OPENAI_API_KEY` env var |
| None | — | Free | Nothing |

The agent tries native tool-calling first, and automatically falls back to ReAct-style prompting if the model doesn't support it.

---

## Extending

- **Swap CLIP for BLIP-2 / InternVL**: Replace `CLIPVideoEncoder` with any HuggingFace VLM.
- **Add richer captions**: Use BLIP or LLaVA to generate per-frame captions instead of zero-shot label matching.
- **Benchmark on EgoSchema / ActivityNet-QA**: Drop in a dataset loader in `eval/` and call `batch_eval`.
- **Multimodal context**: Pass top-k frame images directly to a vision-capable LLM (GPT-4o) in the tool results.

---

## File Structure

```
agentic-video-qa/
├── agent/
│   ├── memory.py          # Frame embedding store + retrieval
│   ├── tools.py           # Tool definitions + ToolExecutor
│   └── video_agent.py     # Main agent loop (Ollama + OpenAI + ReAct fallback)
├── models/
│   └── clip_encoder.py    # CLIP wrapper (encode, retrieve, segment)
├── video/
│   └── processor.py       # Frame extraction & sampling
├── demo.py
└── requirements.txt
```
