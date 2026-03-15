"""Twelve Labs API client for video embedding, search, and analysis.

Supports Marengo 3.0 for visual embeddings and semantic search,
and Pegasus 1.2 for natural language video Q&A.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from twelvelabs import TwelveLabs

load_dotenv()
logger = logging.getLogger(__name__)

TWELVE_LABS_API_KEY = os.getenv("TWELVE_LABS_API_KEY", "")
MARENGO_INDEX_NAME = os.getenv("TWELVE_LABS_MARENGO_INDEX_NAME", "my-marengo-index")
PEGASUS_INDEX_NAME = os.getenv("TWELVE_LABS_PEGASUS_INDEX_NAME", "my-pegasus-index")
MARENGO_MODEL = os.getenv("TWELVE_LABS_MARENGO_MODEL", "marengo3.0")
PEGASUS_MODEL = os.getenv("TWELVE_LABS_PEGASUS_MODEL", "pegasus1.2")
UPLOAD_TIMEOUT = int(os.getenv("TWELVE_LABS_UPLOAD_TIMEOUT", "600"))
MAX_RESULTS = int(os.getenv("TWELVE_LABS_MAX_RESULTS", "10"))


@dataclass
class SearchResult:
    video_id: str
    score: float
    start: float
    end: float
    confidence: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    text: str
    video_id: str
    latency_ms: float = 0.0


_client: Optional[TwelveLabs] = None
_marengo_index_id: Optional[str] = None
_pegasus_index_id: Optional[str] = None


def get_client() -> TwelveLabs:
    global _client
    if _client is None:
        if not TWELVE_LABS_API_KEY:
            raise RuntimeError("TWELVE_LABS_API_KEY not set")
        _client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)
    return _client


def _ensure_index(index_name: str, models: list[dict]) -> str:
    """Get existing index by name or create it. Returns index ID."""
    client = get_client()
    for idx in client.index.list():
        if idx.name == index_name:
            return idx.id
    idx = client.index.create(name=index_name, models=models)
    logger.info("Created index %s (id=%s)", index_name, idx.id)
    return idx.id


def get_marengo_index_id() -> str:
    global _marengo_index_id
    if _marengo_index_id is None:
        _marengo_index_id = _ensure_index(
            MARENGO_INDEX_NAME,
            [{"name": MARENGO_MODEL, "options": ["visual", "audio"]}],
        )
    return _marengo_index_id


def get_pegasus_index_id() -> str:
    global _pegasus_index_id
    if _pegasus_index_id is None:
        _pegasus_index_id = _ensure_index(
            PEGASUS_INDEX_NAME,
            [{"name": PEGASUS_MODEL, "options": ["visual", "audio"]}],
        )
    return _pegasus_index_id


def upload_video(file_path: str | Path, index_type: str = "both") -> dict:
    """Upload a video to Twelve Labs for indexing.

    Args:
        file_path: Local path to the video file.
        index_type: "marengo", "pegasus", or "both".

    Returns:
        Dict with video IDs, e.g. {"marengo_video_id": "...", "pegasus_video_id": "..."}
    """
    client = get_client()
    result = {}

    if index_type in ("marengo", "both"):
        task = client.task.create(index_id=get_marengo_index_id(), file=str(file_path))
        task.wait_for_done(timeout=UPLOAD_TIMEOUT)
        if task.status == "ready":
            result["marengo_video_id"] = task.video_id
            logger.info("Uploaded to Marengo: video_id=%s", task.video_id)
        else:
            logger.error("Marengo upload failed: status=%s", task.status)

    if index_type in ("pegasus", "both"):
        task = client.task.create(index_id=get_pegasus_index_id(), file=str(file_path))
        task.wait_for_done(timeout=UPLOAD_TIMEOUT)
        if task.status == "ready":
            result["pegasus_video_id"] = task.video_id
            logger.info("Uploaded to Pegasus: video_id=%s", task.video_id)
        else:
            logger.error("Pegasus upload failed: status=%s", task.status)

    return result


def search_videos(
    query: str,
    max_results: int | None = None,
    threshold: str = "medium",
) -> list[SearchResult]:
    """Semantic search across indexed videos using Marengo.

    Args:
        query: Natural language or visual description to search for.
        max_results: Maximum clips to return.
        threshold: Confidence threshold ("low", "medium", "high").
    """
    client = get_client()
    search_results = client.search.query(
        index_id=get_marengo_index_id(),
        search_options=["visual", "audio"],
        query_text=query,
        group_by="clip",
        threshold=threshold,
        page_limit=max_results or MAX_RESULTS,
        sort_option="score",
    )

    results = []
    for group in search_results.data:
        for clip in group.clips:
            results.append(SearchResult(
                video_id=clip.video_id,
                score=clip.score,
                start=clip.start,
                end=clip.end,
                confidence=clip.confidence,
                metadata={"module_type": clip.module_type} if hasattr(clip, "module_type") else {},
            ))
    return results


def analyze_video(video_id: str, prompt: str) -> AnalysisResult:
    """Ask a natural language question about a specific video using Pegasus.

    Args:
        video_id: The Pegasus video ID to query.
        prompt: The question to ask about the video.
    """
    client = get_client()
    t0 = time.perf_counter()

    response = client.generate.text(
        video_id=video_id,
        prompt=prompt,
        temperature=0.2,
    )

    return AnalysisResult(
        text=response.data if hasattr(response, "data") else str(response),
        video_id=video_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def get_video_embedding(video_id: str) -> Optional[list[float]]:
    """Get the Marengo embedding vector for an indexed video.

    Returns a float list suitable for indexing into Qdrant, or None on failure.
    """
    client = get_client()
    try:
        response = client.embed.create(model_name=MARENGO_MODEL, video_id=video_id)
        if response.video_embedding and response.video_embedding.values:
            return response.video_embedding.values
    except Exception as exc:
        logger.warning("Failed to get embedding for video %s: %s", video_id, exc)
    return None
