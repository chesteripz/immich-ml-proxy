"""Forward clip inputs to llama-server and face/OCR inputs to Immich ML."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from immich_ml_proxy.config import Settings

logger = logging.getLogger(__name__)


class UpstreamError(Exception):
    """An upstream could not complete the request."""


class RoundRobin:
    """Hands each call the full backend list, starting at the next server."""

    def __init__(self, urls: tuple[str, ...]) -> None:
        if not urls:
            raise ValueError('at least one upstream URL is required')
        self.urls = urls
        self._next = 0
        self._lock = asyncio.Lock()

    async def order(self) -> list[str]:
        async with self._lock:
            start = self._next
            self._next = (self._next + 1) % len(self.urls)
        return [self.urls[(start + offset) % len(self.urls)] for offset in range(len(self.urls))]


class LlamaEmbedder:
    """Builds the sample payload and returns the first embedding vector."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._pool = RoundRobin(settings.llama_server_urls)

    def set_transport(self, transport: httpx.AsyncBaseTransport | None) -> None:
        self._transport = transport

    def set_urls(self, urls: tuple[str, ...]) -> None:
        self._pool = RoundRobin(urls)

    async def embed_text(self, text: str) -> list[float]:
        return await self._embed([{'type': 'text', 'text': text}])

    async def embed_image(self, image: bytes, media_type: str) -> list[float]:
        encoded = base64.b64encode(image).decode('ascii')
        return await self._embed(
            [
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:{media_type};base64,{encoded}'},
                }
            ]
        )

    async def _embed(self, content: list[dict[str, Any]]) -> list[float]:
        payload = {
            'input': [{'content': content}],
            'encoding_format': 'float',
            'model': self._settings.model,
        }

        async def send(client: httpx.AsyncClient, base: str) -> httpx.Response:
            return await client.post(f'{base}/v1/embeddings', json=payload)

        response = await _first_healthy(
            self._pool,
            self._settings.request_timeout,
            self._transport,
            send,
            'llama-server',
        )
        if response.status_code >= 400:
            detail = response.text[:300]
            raise UpstreamError(f'llama-server returned {response.status_code}: {detail}')

        try:
            body = response.json()
            embedding = body['data'][0]['embedding']
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise UpstreamError('llama-server response did not include an embedding') from exc
        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(isinstance(value, (int, float)) for value in embedding)
        ):
            raise UpstreamError('llama-server response did not include an embedding')
        logger.info(
            'Forwarded embedding to %s model=%s dim=%s',
            response.request.url,
            self._settings.model,
            len(embedding),
        )
        return [float(value) for value in embedding]


class ImmichMlProxy:
    """Forward facial-recognition and OCR ``/predict`` calls unchanged."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._pool = RoundRobin(settings.immich_ml_urls)

    def set_transport(self, transport: httpx.AsyncBaseTransport | None) -> None:
        self._transport = transport

    def set_urls(self, urls: tuple[str, ...]) -> None:
        self._pool = RoundRobin(urls)

    async def predict(
        self,
        entries: str,
        image: bytes | None,
        text: str | None,
    ) -> tuple[int, Any]:
        data: dict[str, str] = {'entries': entries}
        files = None
        if text is not None:
            data['text'] = text
        if image is not None:
            files = {'image': ('image', image, 'application/octet-stream')}

        async def send(client: httpx.AsyncClient, base: str) -> httpx.Response:
            return await client.post(f'{base}/predict', data=data, files=files)

        response = await _first_healthy(
            self._pool,
            self._settings.request_timeout,
            self._transport,
            send,
            'immich-ml',
        )
        try:
            body = response.json()
        except ValueError as exc:
            if response.status_code >= 400:
                return response.status_code, {'detail': response.text[:300]}
            raise UpstreamError('immich-ml response was not JSON') from exc

        if response.status_code >= 400:
            if not isinstance(body, dict):
                body = {'detail': response.text[:300]}
            return response.status_code, body
        if not isinstance(body, dict):
            raise UpstreamError('immich-ml response was not a JSON object')
        logger.info('Forwarded predict to %s status=%s', response.request.url, response.status_code)
        return response.status_code, body


async def _first_healthy(
    pool: RoundRobin,
    timeout_seconds: int,
    transport: httpx.AsyncBaseTransport | None,
    send: Callable[[httpx.AsyncClient, str], Awaitable[httpx.Response]],
    name: str,
) -> httpx.Response:
    """Try backends in round-robin order. Skip timeouts, connection errors, and HTTP 5xx."""
    last_response: httpx.Response | None = None
    failures: list[str] = []
    timed_out = 0
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        for base in await pool.order():
            try:
                response = await send(client, base)
            except httpx.TimeoutException:
                timed_out += 1
                failures.append(f'{base} timed out')
                logger.warning('%s %s timed out', name, base)
                continue
            except httpx.HTTPError as exc:
                failures.append(f'{base} failed: {exc}')
                logger.warning('%s %s failed: %s', name, base, exc)
                continue
            if response.status_code >= 500:
                last_response = response
                failures.append(f'{base} returned {response.status_code}')
                logger.warning('%s %s returned %s', name, base, response.status_code)
                continue
            return response

    if last_response is not None:
        return last_response
    if timed_out == len(failures):
        raise TimeoutError(f'{name} timed out')
    raise UpstreamError(f'{name} request failed: {"; ".join(failures)}')
