# immich-ml-proxy

An [Immich](https://immich.app/) machine-learning endpoint. It speaks Immich's HTTP API (`GET /ping`, `POST /predict`) and does not load a model.

Clip embeddings are forwarded to one or more [llama-server](https://github.com/ggml-org/llama.cpp) `/v1/embeddings` endpoints.  
Facial recognition and OCR are forwarded unchanged to one or more Immich ML services.

Requests are spread round-robin. A timeout, connection error, or HTTP 5xx tries the next server. An HTTP 4xx is returned immediately.

## What Immich sends

Immich posts multipart form data:

| Field | Role |
| --- | --- |
| `entries` | JSON task list. Smart search is `{"clip":{"visual":{"modelName":"..."}}}` or `{"clip":{"textual":{"modelName":"...","options":{"language":"en"}}}}`. |
| `image` | JPEG/PNG preview for visual embeddings. |
| `text` | Search query for textual embeddings. |

The response matches Immich: `clip` is a JSON array encoded as a string. Image responses also include `imageHeight` and `imageWidth`. The vector is whatever llama-server returns (768 dimensions for EmbeddingGemma 2).

```bash
curl http://localhost:3003/ping

curl -X POST http://localhost:3003/predict \
  -F 'entries={"clip":{"textual":{"modelName":"google/embeddinggemma-2"}}}' \
  -F 'text=hello'
```

`modelName` has to identify EmbeddingGemma. The last path segment, compared without `-`, `_`, or `:`, must start with `embeddinggemma`. The forwarded request always uses `LLAMA_MODEL` (`embeddinggemma2`).

## Forwarded request

Text and images are posted to `{server}/v1/embeddings` in the shape of `sample-payload.json`:

```json
{
  "input": [
    {
      "content": [
        { "type": "text", "text": "hello" }
      ]
    }
  ],
  "encoding_format": "float",
  "model": "embeddinggemma2"
}
```

Images use an `image_url` part whose `url` is a `data:` URL of the uploaded bytes. One Immich `/predict` call becomes one content object and one embedding.

Copy `sample.env` to `.env` and set the upstreams. List several servers with commas:

```bash
LLAMA_SERVER_URL=http://127.0.0.1:8080,http://127.0.0.1:8081
IMMICH_ML_URL=http://127.0.0.1:3003
```

## Requirements

- Linux
- [uv](https://docs.astral.sh/uv/)
- Python 3.14 or newer
- llama-server already serving EmbeddingGemma 2 with embeddings enabled

## Install and run

```bash
uv sync --group dev
cp sample.env .env
uv run immich-ml-proxy
```

Point Immich at this process:

```yaml
MACHINE_LEARNING_URL=http://127.0.0.1:3003
```

Immich's smart-search column has to match the vector width llama-server returns.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port Immich calls |
| `LLAMA_SERVER_URL` | | Comma-separated llama-server base URLs. Required |
| `LLAMA_MODEL` | `embeddinggemma2` | `model` field on the embeddings request |
| `IMMICH_ML_URL` | | Comma-separated Immich ML services for facial recognition and OCR. Required |
| `ML_MAX_IMAGE_BYTES` | `52428800` | Reject larger uploads |
| `ML_REQUEST_TIMEOUT` | `120` | Seconds per upstream attempt before trying the next server |
| `ML_LOG_LEVEL` | `info` | Uvicorn / app log level |

## Tests

```bash
uv run pytest
```

Tests mock the upstreams and do not call the network.
