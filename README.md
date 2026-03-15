# NVIDIA Metropolis VSS + Twelve Labs

Standalone integration of [NVIDIA Metropolis Video Search and Summarization (VSS)](https://developer.nvidia.com/metropolis) with [Twelve Labs](https://twelvelabs.io) for video understanding.

Provides:
- Video chunking with FFmpeg and upload to NVIDIA VSS
- Marengo 3.0 embeddings for semantic search and kNN anomaly detection
- Pegasus 1.2 for natural language video Q&A
- Async upload pipeline with parallel chunk handling

This integration is used in the [Sentinel video anomaly detection platform](https://github.com/qdrant/video-anomaly-edge).

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- FFmpeg (for video chunking)
- NVIDIA GPU (for running VSS locally) or a remote VSS endpoint
- Twelve Labs API key

## Setup

```bash
git clone https://github.com/qdrant/twelvelabs-nvidia-vss
cd twelvelabs-nvidia-vss
uv sync

cp .env.example .env
# Edit .env with your Twelve Labs API key and VSS URL
```

## NVIDIA VSS

To run VSS locally (requires NVIDIA GPU and NGC access):

```bash
docker compose up
```

This starts the VSS server on port 8080 with Twelve Labs configured as the VLM backend.

For managed VSS deployment on Vultr Cloud GPUs (A100, H100, H200, L40S), see the [Vultr documentation](https://www.vultr.com/products/cloud-gpu/).

## Usage

### Ingest a video

```bash
# Full pipeline: chunk + VSS upload + Twelve Labs indexing
uv run python scripts/ingest.py --video path/to/video.mp4

# Skip VSS, only index to Twelve Labs
uv run python scripts/ingest.py --video path/to/video.mp4 --skip-vss

# Index only for Marengo (embeddings/search), skip Pegasus
uv run python scripts/ingest.py --video path/to/video.mp4 --index-type marengo
```

### Search indexed videos

```bash
uv run python scripts/search.py --query "person running near entrance" --max-results 5
```

### Ask questions about a video

```bash
uv run python scripts/analyze.py \
    --video-id <pegasus_video_id> \
    --prompt "What is happening in this video? Are there any unusual events?"
```

## Library usage

```python
import asyncio
from src import vss_client, twelvelabs_client

# Check VSS health
health = asyncio.run(vss_client.health())
print(health)

# Upload a video to Twelve Labs
result = twelvelabs_client.upload_video("clip.mp4", index_type="marengo")
video_id = result["marengo_video_id"]

# Get the embedding vector (1024-dimensional for Marengo 3.0)
embedding = twelvelabs_client.get_video_embedding(video_id)

# Semantic search
results = twelvelabs_client.search_videos("fighting or aggressive behavior", max_results=10)
for r in results:
    print(f"{r.video_id}: score={r.score:.4f}, {r.start:.1f}s-{r.end:.1f}s")

# Video Q&A
analysis = twelvelabs_client.analyze_video(video_id, "Describe what happens in this clip.")
print(analysis.text)
```

## Architecture

```
video.mp4
    |
    v
chunk_video()          # FFmpeg segment muxer -> N x ~30s chunks
    |
    +---> VSS          # NVIDIA Metropolis: VLM captioning, Graph-RAG, CV pipeline
    |
    +---> Twelve Labs  # Marengo: embedding + semantic search
                       # Pegasus: natural language Q&A
```

The Marengo embedding (1024-dim) can be indexed into [Qdrant](https://qdrant.tech) for kNN-based anomaly detection. See [qdrant/video-anomaly-edge](https://github.com/qdrant/video-anomaly-edge) for a complete production implementation.
