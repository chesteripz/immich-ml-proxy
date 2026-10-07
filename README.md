# immich-embeddinggemma

An [Immich](https://immich.app/) machine-learning endpoint that embeds photos and search text with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). It speaks the same HTTP API Immich uses for smart search (`GET /ping`, `POST /predict`) and runs on CPU. CUDA is not wired up yet.

Facial recognition and OCR are out of scope. Immich should keep using its normal ML service for those jobs, or leave them disabled. A face or OCR `/predict` request returns HTTP 501.

## What Immich sends

Immich posts multipart form data:

| Field | Role |
| --- | --- |
| `entries` | JSON task list. Smart search is `{"clip":{"visual":{"modelName":"..."}}}` or `{"clip":{"textual":{"modelName":"...","options":{"language":"en"}}}}`. |
| `image` | JPEG/PNG preview for visual embeddings. |
| `text` | Search query for textual embeddings. |

The response matches Immich: `clip` is a JSON array encoded as a string. Image responses also include `imageHeight` and `imageWidth`.

```bash
curl http://localhost:3003/ping

curl -X POST http://localhost:3003/predict \
  -F 'entries={"clip":{"textual":{"modelName":"ViT-B-32__openai"}}}' \
  -F 'text=a photo of a cat'
```

The `modelName` in `entries` is whatever Immich is configured to use. This service always embeds with EmbeddingGemma 2. The name still matters inside Immich, because Immich sizes the `smart_search.embedding` column from that name.

## Dimension must match Immich

EmbeddingGemma 2 natively emits 768-d vectors and supports Matryoshka truncation to 512, 256, or 128. The default here is **512**, which matches Immich's default CLIP model `ViT-B-32__openai`.

| `ML_TRUNCATE_DIM` | Immich CLIP model to select |
| --- | --- |
| `512` (default) | `ViT-B-32__openai` or any other 512-d model |
| `768` | `ViT-L-14__openai` or any other 768-d model |
| `256`, `128` | No built-in Immich model uses these widths |

Text queries use the `SearchQuery` prompt. Photos are encoded as images with no prompt, so both live in one vector space. Embeddings are L2-normalized. Audio weights stay unloaded (`audio_config=None`); Immich sends still previews, including video thumbnails.

## Requirements

- Linux (or another platform with a PyTorch CPU wheel)
- [uv](https://docs.astral.sh/uv/)
- Python 3.11 or 3.12 (uv installs 3.12 from `.python-version`)
- A Hugging Face account if `google/embeddinggemma-2` is gated

## Install and run

```bash
uv sync --group dev
uv run huggingface-cli login   # only if the model repo asks for a token
uv run immich-embeddinggemma
```

The first real request downloads the checkpoint into the Hugging Face cache. Set `ML_PRELOAD=true` to download and load during startup instead.

Point Immich at this process:

```yaml
MACHINE_LEARNING_URL=http://127.0.0.1:3003
```

In Administration → Machine Learning, set the CLIP model to one whose dimension matches `ML_TRUNCATE_DIM`, then run Smart Search. Face and OCR jobs need a different ML URL; this process will reject them.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port Immich calls |
| `ML_MODEL_ID` | `google/embeddinggemma-2` | Hugging Face model id |
| `ML_TRUNCATE_DIM` | `512` | `768`, `512`, `256`, or `128` |
| `ML_DEVICE` | `cpu` | Only `cpu` is accepted |
| `ML_LOAD_AUDIO` | `false` | Also load the audio encoder |
| `ML_NORMALIZE` | `true` | L2-normalize embeddings |
| `ML_PRELOAD` | `false` | Load weights at startup |
| `ML_STUB` | `false` | Synthetic vectors, no download. Used by tests |
| `ML_MAX_IMAGE_BYTES` | `52428800` | Reject larger uploads |
| `ML_REQUEST_TIMEOUT` | `120` | Seconds before HTTP 503 |
| `ML_LOG_LEVEL` | `info` | Uvicorn / app log level |
| `HF_TOKEN` | | Hugging Face token for gated weights |

## Tests

```bash
uv run pytest
```

Tests set `ML_STUB=1` and do not download the model.

## CUDA

Inference is pinned to CPU via the PyTorch CPU index in `pyproject.toml` and `device="cpu"`. A later change can add a CUDA wheel index and accept `ML_DEVICE=cuda` without changing the Immich API.
