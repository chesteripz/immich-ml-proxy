"""Immich ML API that forwards smart search, faces, and OCR.

Speaks the same endpoints Immich uses:

- ``GET /`` and ``GET /ping``
- ``POST /predict`` with multipart ``entries`` plus ``image`` or ``text``

Clip embeddings go to llama-server. Facial recognition and OCR go to an
Immich ML service. Each side can list several servers.
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

from immich_ml_proxy import __version__
from immich_ml_proxy.config import Settings
from immich_ml_proxy.upstream import ImmichMlProxy, LlamaEmbedder, UpstreamError

logger = logging.getLogger(__name__)

settings = Settings.from_env()
embedder = LlamaEmbedder(settings)
ml_proxy = ImmichMlProxy(settings)

CLIP_TASK = 'clip'
FORWARDED_TASKS = {'facial-recognition', 'ocr'}
_MEDIA_TYPES = {
    'JPEG': 'image/jpeg',
    'PNG': 'image/png',
    'WEBP': 'image/webp',
    'GIF': 'image/gif',
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    logger.info(
        'immich-ml-proxy %s listening host=%s port=%s llama=%s model=%s ml=%s',
        __version__,
        settings.host,
        settings.port,
        ','.join(settings.llama_server_urls),
        settings.model,
        ','.join(settings.immich_ml_urls),
    )
    yield


app = FastAPI(
    title='immich-ml-proxy',
    version=__version__,
    lifespan=lifespan,
)


@app.get('/')
async def root() -> JSONResponse:
    return JSONResponse({'message': 'Immich ML'})


@app.get('/ping')
def ping() -> PlainTextResponse:
    return PlainTextResponse('pong')


@app.get('/health')
async def health() -> JSONResponse:
    return JSONResponse(
        {
            'status': 'healthy',
            'llamaServers': list(settings.llama_server_urls),
            'model': settings.model,
            'immichMl': list(settings.immich_ml_urls),
        }
    )


@app.post('/predict')
async def predict(
    entries: str = Form(),
    image: bytes | None = File(default=None),
    text: str | None = Form(default=None),
) -> JSONResponse:
    tasks = _parse_entries(entries)
    if image is None and (text is None or text == ''):
        raise HTTPException(400, 'Either image or text must be provided')

    unknown = sorted(set(tasks) - FORWARDED_TASKS - {CLIP_TASK})
    if unknown:
        raise HTTPException(
            501,
            f'Unsupported task(s): {", ".join(unknown)}.',
        )
    if not (set(tasks) & (FORWARDED_TASKS | {CLIP_TASK})):
        raise HTTPException(422, 'Invalid request format.')

    forwarded = {name: tasks[name] for name in FORWARDED_TASKS if name in tasks}
    response: dict[str, Any] = {}
    if forwarded:
        if image is not None and len(image) > settings.max_image_bytes:
            raise HTTPException(413, 'Image too large')
        forwarded_entries = entries if CLIP_TASK not in tasks else orjson.dumps(forwarded).decode()
        status, body = await ml_proxy.predict(forwarded_entries, image, text)
        if status >= 400:
            return JSONResponse(status_code=status, content=body)
        response.update(body)

    clip = tasks.get(CLIP_TASK)
    if clip is None:
        return JSONResponse(response)
    if not isinstance(clip, dict) or not clip:
        raise HTTPException(422, 'Invalid request format.')

    if image is not None:
        if len(image) > settings.max_image_bytes:
            raise HTTPException(413, 'Image too large')
        pil = _decode_image(image)
        if pil.width == 0 or pil.height == 0:
            raise HTTPException(400, 'Image has zero width or height')
        response['imageHeight'] = pil.height
        response['imageWidth'] = pil.width
        if 'visual' in clip:
            media_type = _MEDIA_TYPES.get((pil.format or 'JPEG').upper(), 'image/jpeg')
            vector = await embedder.embed_image(image, media_type)
            response['clip'] = _dump_vector(vector)
    elif 'textual' in clip:
        vector = await embedder.embed_text(text or '')
        response['clip'] = _dump_vector(vector)

    if 'clip' not in response:
        raise HTTPException(400, 'Request did not include a clip input matching the entries')

    return JSONResponse(response)


def _dump_vector(vector: list[float]) -> str:
    return orjson.dumps(vector).decode()


def _parse_entries(entries: str) -> dict[str, Any]:
    try:
        parsed = orjson.loads(entries)
    except orjson.JSONDecodeError as exc:
        logger.error('Invalid entries JSON: %s', exc)
        raise HTTPException(422, 'Invalid request format.') from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, 'Invalid request format.')
    return parsed


def _decode_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, 'Invalid image') from exc
    return image


@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(_request: Request, _exc: asyncio.TimeoutError) -> JSONResponse:
    return JSONResponse(status_code=503, content={'detail': 'Inference timed out'})


@app.exception_handler(UpstreamError)
async def upstream_handler(_request: Request, exc: UpstreamError) -> JSONResponse:
    return JSONResponse(status_code=502, content={'detail': str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        'immich_ml_proxy.main:app',
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level,
        timeout_keep_alive=settings.request_timeout,
    )


if __name__ == '__main__':
    main()
