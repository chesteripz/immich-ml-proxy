# Agent notes

This service is a drop-in for Immich's machine-learning calls. It does not load weights.

## API

- `GET /` → `{"message":"Immich ML"}`
- `GET /ping` → body `pong` (Immich health checks this)
- `POST /predict` multipart: `entries` (JSON), plus `image` and/or `text`
- Clip visual and textual results go in `response["clip"]` as a JSON array string
- Image replies include `imageHeight` and `imageWidth`
- `facial-recognition` and `ocr` are forwarded as the original multipart body to `{IMMICH_ML_URL}/predict`. Return that service's JSON and status code. Do not invent face boxes or OCR text

The clip entry's `modelName` must identify EmbeddingGemma. Compare the last `/` segment with `-`, `_`, and `:` removed; it has to start with `embeddinggemma`. Anything else is HTTP 400.

## Upstreams

`LLAMA_SERVER_URL` and `IMMICH_ML_URL` are required comma-separated lists. One URL is a list of one. Read them from the environment (`.env`, seeded from `sample.env`). The project name is `immich-ml-proxy`; the package is `immich_ml_proxy`.

POST `{llama}/v1/embeddings` using the shape in `sample-payload.json`:

- `encoding_format` is `float`
- `model` comes from `LLAMA_MODEL` (default `embeddinggemma2`)
- Llama and Immich ML base URLs come from `LLAMA_SERVER_URL` and `IMMICH_ML_URL`
- Text: `{"input":[{"content":[{"type":"text","text":"..."}]}]}`
- Image: one `image_url` part, `url` set to `data:<media-type>;base64,...` of the original upload bytes
- Read `data[0].embedding` and return it unchanged

Each request starts at the next server (round-robin). Timeouts, connection errors, and HTTP 5xx try the remaining servers in that order. HTTP 4xx is returned from the server that produced it. If every llama attempt times out, respond 503. Other total failures are 502.

Do not add a local encoder, torch, or sentence-transformers. Tests install an `httpx` transport on `embedder` and `ml_proxy` and must not reach the network.

## Layout

- `src/immich_ml_proxy/config.py` — env settings
- `src/immich_ml_proxy/upstream.py` — llama-server client, Immich ML proxy, and round-robin
- `src/immich_ml_proxy/main.py` — FastAPI app. Settings are read at import
- Run with `uv run immich-ml-proxy` (one worker)
