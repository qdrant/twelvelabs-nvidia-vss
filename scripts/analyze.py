"""Ask a natural language question about a specific video using Pegasus.

Usage:
    uv run python scripts/analyze.py --video-id <pegasus_video_id> --prompt "What happens in this video?"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Video Q&A via Twelve Labs Pegasus")
    parser.add_argument("--video-id", required=True, help="Pegasus video ID")
    parser.add_argument("--prompt", required=True, help="Question to ask about the video")
    return parser.parse_args()


def main():
    args = parse_args()
    from src import twelvelabs_client

    result = twelvelabs_client.analyze_video(args.video_id, args.prompt)
    print(f"\nQ: {args.prompt}")
    print(f"A: {result.text}")
    print(f"\nLatency: {result.latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
