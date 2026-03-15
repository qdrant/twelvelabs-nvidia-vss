"""Ingest a video through the full NVIDIA VSS + Twelve Labs pipeline.

Usage:
    uv run python scripts/ingest.py --video path/to/video.mp4

Steps:
    1. Chunk the video with FFmpeg
    2. Upload chunks to NVIDIA VSS
    3. Upload original to Twelve Labs Marengo (for embedding) and Pegasus (for Q&A)
    4. Print the resulting video IDs and embedding dimension
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest video via VSS + Twelve Labs")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--skip-vss", action="store_true", help="Skip NVIDIA VSS upload")
    parser.add_argument("--index-type", default="both", choices=["marengo", "pegasus", "both"])
    return parser.parse_args()


async def main():
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    from src import vss_client, twelvelabs_client

    # Step 1: NVIDIA VSS ingestion (chunk + upload)
    if not args.skip_vss:
        logger.info("Ingesting %s into NVIDIA VSS...", video_path.name)
        vss_result = await vss_client.ingest_video(video_path)
        if vss_result["status"] == "ok":
            logger.info(
                "VSS: uploaded %d/%d chunks, file IDs: %s",
                vss_result["uploaded_chunks"],
                vss_result["total_chunks"],
                vss_result["vss_file_ids"],
            )
        else:
            logger.error("VSS ingestion failed: %s", vss_result.get("message"))
    else:
        logger.info("Skipping VSS upload")

    # Step 2: Twelve Labs upload + embedding
    logger.info("Uploading to Twelve Labs (index_type=%s)...", args.index_type)
    tl_result = twelvelabs_client.upload_video(str(video_path), index_type=args.index_type)
    print("\nTwelve Labs result:", tl_result)

    marengo_id = tl_result.get("marengo_video_id")
    if marengo_id:
        embedding = twelvelabs_client.get_video_embedding(marengo_id)
        if embedding:
            print(f"Embedding: {len(embedding)}-dimensional vector")
            print(f"First 5 values: {embedding[:5]}")
        else:
            logger.warning("Could not retrieve embedding for video %s", marengo_id)


if __name__ == "__main__":
    asyncio.run(main())
