# Agent notes

This service is a drop-in for Immich's smart-search calls, not a full Immich ML clone. The contract lives in Immich (`machine-learning/immich_ml/main.py` and `server/src/repositories/machine-learning.repository.ts`) and is mirrored by `/home/chester/code/immich-ml-metal`.

## API

- `GET /` → `{"message":"Immich ML"}`
- `GET /ping` → body `pong` (Immich health checks this)
- `POST /predict` multipart: `entries` (JSON), plus `image` and/or `text`
- Clip visual and textual results go in `response["clip"]` as a JSON array string (`orjson` + `OPT_SERIALIZE_NUMPY`), same as Immich's `serialize_np_array`
- Image replies include `imageHeight` and `imageWidth`
- `facial-recognition` and `ocr` return 501. Do not invent face boxes or OCR text

Immich chooses the vector width from the CLIP **model name** in its admin settings. This process ignores that name and always runs EmbeddingGemma 2. Keep `ML_TRUNCATE_DIM` aligned with the Immich model (512 for `ViT-B-32__openai`, 768 for `ViT-L-14__openai`).

## Model

Follow `embeddinggemma-2-the-developer-guide.md`:

- `sentence-transformers` ≥ 6.1
- Text: `encode(..., prompt_name="SearchQuery")`
- Images: `encode({"image": pil})` with no prompt
- Truncate with `truncate_dim` in `{768, 512, 256, 128}` and `normalize_embeddings=True`
- Default load passes `config_kwargs={"audio_config": None}` so audio weights are not allocated. Photos and video thumbnails only need the vision tower
- Device is CPU. `ML_DEVICE` other than `cpu` must keep failing until a CUDA path exists. `torch` and `torchvision` both come from the PyTorch CPU index in `pyproject.toml`. The PyPI `torchvision` wheel does not register `torchvision::nms` against that CPU build, so leave `torchvision` as a direct dependency on that index.

Do not download weights in tests. `tests/conftest.py` sets `ML_STUB=1` before the app is imported. `StubEncoder` must stay free of `sentence_transformers` imports.

## Layout

- `src/immich_embeddinggemma/config.py` — env settings
- `src/immich_embeddinggemma/embedder.py` — Gemma load/encode and the stub
- `src/immich_embeddinggemma/main.py` — FastAPI app. Settings are read at import
- Run with `uv run immich-embeddinggemma` (one worker)

Inference is serialized with a lock inside `GemmaEncoder`. Keep that if you add concurrency; SentenceTransformer encode is not assumed to be thread-safe.
