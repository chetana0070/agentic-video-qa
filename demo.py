"""
Demo: Agentic Video QA

Usage:
    # Ollama (local, free) — recommended:
    python demo.py --video video.mp4 --question "What is happening?" --llm-backend ollama --llm-model llama3.2

    # OpenAI:
    python demo.py --video video.mp4 --question "..." --llm-backend openai --llm-model gpt-4o-mini

    # CLIP-only retrieval (no LLM needed):
    python demo.py --video video.mp4 --question "..." --llm-backend none
"""
import argparse
from agent.video_agent import VideoAgent


def main():
    parser = argparse.ArgumentParser(description="Agentic Video QA Demo")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--question", default="What is happening in this video?")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames to sample per second")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--llm-backend", default="ollama", choices=["ollama", "openai", "none"],
                        help="LLM backend to use (default: ollama)")
    parser.add_argument("--llm-model", default="llama3.2",
                        help="Model name (e.g. llama3.2, mistral, gpt-4o-mini)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL")
    args = parser.parse_args()

    agent = VideoAgent(
        target_fps=args.fps,
        max_frames=args.max_frames,
        llm_model=args.llm_model,
        llm_backend=args.llm_backend,
        ollama_base_url=args.ollama_url,
    )

    agent.index_video(args.video)

    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}")
    answer = agent.answer(args.question)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
