"""Semantic video search using Twelve Labs Marengo.

Usage:
    uv run python scripts/search.py --query "person running" --max-results 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Search indexed videos via Twelve Labs")
    parser.add_argument("--query", required=True, help="Natural language search query")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--threshold", default="medium", choices=["low", "medium", "high"])
    return parser.parse_args()


def main():
    args = parse_args()
    from src import twelvelabs_client

    print(f"Searching for: {args.query!r}")
    results = twelvelabs_client.search_videos(
        query=args.query,
        max_results=args.max_results,
        threshold=args.threshold,
    )

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] video_id={r.video_id}")
        print(f"    score={r.score:.4f}  confidence={r.confidence}")
        print(f"    time: {r.start:.1f}s - {r.end:.1f}s")


if __name__ == "__main__":
    main()
