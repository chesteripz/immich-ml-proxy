"""Immich ML API implemented with EmbeddingGemma 2.

Speaks the same endpoints Immich uses for smart search:

- ``GET /`` and ``GET /ping``
- ``POST /predict`` with multipart ``entries`` plus ``image`` or ``text``

Facial recognition and OCR are not implemented. Those tasks return 501 so
Immich fails the job instead of storing an empty result.
"""

from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from typing import Any

import orjson
from fastapi import FastAPI, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError

from immich_embeddinggemma import __version__
from immich_embeddinggemma.config import Settings
from immich_embeddinggemma.embedder import Encoder, build_encoder

logger = logging.getLogger(__name__)

settings = Settings.from_env()
encoder: Encoder = build_encoder(settings)

SUPPORTED_TASKS = {"clip"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "immich-embeddinggemma %s listening config host=%s port=%s dim=%s device=%s stub=%s",
        __version__,
        settings.host,
        settings.port,
        settings.truncate_dim,
        settings.device,
        settings.stub_mode,
    )
    if settings.preload and not settings.stub_mode:
        await asyncio.to_thread(encoder.encode_text, "warmup")
    yield
    encoder.close()


app = FastAPI(
    title="immich-embeddinggemma",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": "Immich ML"})


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "healthy",
            "model": settings.model_id,
            "device": settings.device,
            "truncateDim": settings.truncate_dim,
            "stub": settings.stub_mode,
        }
    )


@app.post("/predict")
async def predict(
    entries: str = Form(),
    image: bytes | None = File(default=None),
    text: str | None = Form(default=None),
) -> JSONResponse:
    tasks = _parse_entries(entries)
    if image is None and (text is None or text == ""):
        raise HTTPException(400, "Either image or text must be provided")

    unknown = sorted(set(tasks) - SUPPORTED_TASKS)
    if unknown:
        raise HTTPException(
            501,
            f"Unsupported task(s): {', '.join(unknown)}. This service embeds with EmbeddingGemma 2 (clip only).",
        )

    clip = tasks.get("clip")
    if not isinstance(clip, dict) or not clip:
        raise HTTPException(422, "Invalid request format.")

    response: dict[str, Any] = {}
    if image is not None:
        if len(image) > settings.max_image_bytes:
            raise HTTPException(413, "Image too large")
        pil = _decode_image(image)
        if pil.width == 0 or pil.height == 0:
            raise HTTPException(400, "Image has zero width or height")
        response["imageHeight"] = pil.height
        response["imageWidth"] = pil.width
        if "visual" in clip:
            _log_model(clip["visual"])
            vector = await asyncio.wait_for(
                asyncio.to_thread(encoder.encode_image, pil),
                timeout=settings.request_timeout,
            )
            response["clip"] = _dump_vector(vector)
    elif "textual" in clip:
        _log_model(clip["textual"])
        vector = await asyncio.wait_for(
            asyncio.to_thread(encoder.encode_text, text or ""),
            timeout=settings.request_timeout,
        )
        response["clip"] = _dump_vector(vector)

    if "clip" not in response:
        raise HTTPException(400, "Request did not include a clip input matching the entries")

    return JSONResponse(response)


def _dump_vector(vector) -> str:
    return orjson.dumps(vector, option=orjson.OPT_SERIALIZE_NUMPY).decode()


def _parse_entries(entries: str) -> dict[str, Any]:
    try:
        parsed = orjson.loads(entries)
    except orjson.JSONDecodeError as exc:
        logger.error("Invalid entries JSON: %s", exc)
        raise HTTPException(422, "Invalid request format.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, "Invalid request format.")
    return parsed


def _decode_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Invalid image") from exc
    return image


def _log_model(entry: Any) -> None:
    if isinstance(entry, dict):
        name = entry.get("modelName")
        if name:
            logger.info("Immich requested model %s; embedding with %s", name, settings.model_id)


@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(_request: Request, _exc: asyncio.TimeoutError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Inference timed out"})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "immich_embeddinggemma.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level,
        timeout_keep_alive=settings.request_timeout,
    )


if __name__ == "__main__":
    main()
