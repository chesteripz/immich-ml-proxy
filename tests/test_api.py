import io
import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from immich_ml_proxy.main import app, embedder, ml_proxy, settings

VECTOR = [0.1, -0.2, 0.3]


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, status_code: int = 200, body: dict | None = None, error: Exception | None = None):
        self.status_code = status_code
        self.body = body if body is not None else {'data': [{'embedding': VECTOR}]}
        self.error = error
        self.fail_hosts: dict[str, int] = {}
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if request.url.host in self.fail_hosts:
            status = self.fail_hosts[request.url.host]
            payload = {'detail': 'down'} if status >= 500 else self.body
            return httpx.Response(status, json=payload)
        return httpx.Response(self.status_code, json=self.body)


@pytest.fixture
async def client():
    transport = RecordingTransport()
    ml_transport = RecordingTransport(body={'imageWidth': 32, 'imageHeight': 24, 'faces': []})
    embedder.set_transport(transport)
    ml_proxy.set_transport(ml_transport)
    embedder.set_urls(settings.llama_server_urls)
    ml_proxy.set_urls(settings.immich_ml_urls)
    http_transport = ASGITransport(app=app)
    async with AsyncClient(transport=http_transport, base_url='http://test') as http:
        http.upstream = transport  # type: ignore[attr-defined]
        http.ml = ml_transport  # type: ignore[attr-defined]
        yield http
    embedder.set_transport(None)
    ml_proxy.set_transport(None)
    embedder.set_urls(settings.llama_server_urls)
    ml_proxy.set_urls(settings.immich_ml_urls)


def _jpeg(width: int = 32, height: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (width, height), color=(20, 80, 140)).save(buffer, format='JPEG')
    return buffer.getvalue()


async def test_ping(client):
    response = await client.get('/ping')
    assert response.status_code == 200
    assert response.text == 'pong'


async def test_root(client):
    response = await client.get('/')
    assert response.status_code == 200
    assert response.json()['message'] == 'Immich ML'


async def test_health(client):
    response = await client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'healthy'
    assert body['llamaServers'] == ['http://127.0.0.1:8080']
    assert body['model'] == 'embeddinggemma2'
    assert body['immichMl'] == ['http://127.0.0.1:3003']


async def test_predict_text(client):
    entries = json.dumps({'clip': {'textual': {'modelName': 'google/embeddinggemma-2', 'options': {'language': 'en'}}}})
    response = await client.post('/predict', data={'entries': entries, 'text': 'hello'})
    assert response.status_code == 200
    body = response.json()
    assert json.loads(body['clip']) == VECTOR
    assert 'imageHeight' not in body

    forwarded = json.loads(client.upstream.requests[0].content)
    assert forwarded == {
        'input': [{'content': [{'type': 'text', 'text': 'hello'}]}],
        'encoding_format': 'float',
        'model': 'embeddinggemma2',
    }
    assert client.upstream.requests[0].url.path == '/v1/embeddings'


async def test_predict_image(client):
    entries = json.dumps({'clip': {'visual': {'modelName': 'google/embeddinggemma-2'}}})
    image = _jpeg(40, 30)
    response = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', image, 'image/jpeg')},
    )
    assert response.status_code == 200
    body = response.json()
    assert body['imageWidth'] == 40
    assert body['imageHeight'] == 30
    assert json.loads(body['clip']) == VECTOR

    forwarded = json.loads(client.upstream.requests[0].content)
    part = forwarded['input'][0]['content'][0]
    assert part['type'] == 'image_url'
    assert part['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert forwarded['encoding_format'] == 'float'
    assert forwarded['model'] == 'embeddinggemma2'


async def test_predict_requires_input(client):
    entries = json.dumps({'clip': {'textual': {'modelName': 'embeddinggemma-2'}}})
    response = await client.post('/predict', data={'entries': entries})
    assert response.status_code == 400
    assert client.upstream.requests == []


async def test_predict_bad_entries(client):
    response = await client.post('/predict', data={'entries': 'not-json', 'text': 'hello'})
    assert response.status_code == 422


async def test_predict_forwards_faces(client):
    entries = json.dumps(
        {'facial-recognition': {'detection': {'modelName': 'buffalo_l'}, 'recognition': {'modelName': 'buffalo_l'}}}
    )
    image = _jpeg()
    response = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', image, 'image/jpeg')},
    )
    assert response.status_code == 200
    assert response.json()['faces'] == []
    assert client.upstream.requests == []
    forwarded = client.ml.requests[0]
    assert str(forwarded.url) == 'http://127.0.0.1:3003/predict'
    assert entries.encode() in forwarded.content
    assert image in forwarded.content


async def test_predict_forwards_ocr(client):
    entries = json.dumps(
        {'ocr': {'detection': {'modelName': 'PP-OCRv5_mobile'}, 'recognition': {'modelName': 'PP-OCRv5_mobile'}}}
    )
    client.ml.body = {'imageWidth': 32, 'imageHeight': 24, 'ocr': []}
    image = _jpeg()
    response = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', image, 'image/jpeg')},
    )
    assert response.status_code == 200
    assert response.json()['ocr'] == []
    assert client.upstream.requests == []
    assert str(client.ml.requests[0].url) == 'http://127.0.0.1:3003/predict'


async def test_predict_forwards_ml_error(client):
    client.ml.status_code = 500
    client.ml.body = {'detail': 'model failed'}
    entries = json.dumps({'ocr': {'detection': {'modelName': 'PP-OCRv5_mobile'}}})
    response = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', _jpeg(), 'image/jpeg')},
    )
    assert response.status_code == 500
    assert response.json()['detail'] == 'model failed'


async def test_predict_rejects_other_models(client):
    entries = json.dumps({'clip': {'textual': {'modelName': 'ViT-B-32__openai'}}})
    response = await client.post('/predict', data={'entries': entries, 'text': 'a photo of a cat'})
    assert response.status_code == 400
    assert 'EmbeddingGemma' in response.json()['detail']


async def test_predict_invalid_image(client):
    entries = json.dumps({'clip': {'visual': {'modelName': 'google/embeddinggemma-2'}}})
    response = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', b'not-an-image', 'image/jpeg')},
    )
    assert response.status_code == 400
    assert client.upstream.requests == []


async def test_clip_round_robin(client):
    embedder.set_urls(('http://llama-a:31234', 'http://llama-b:31234'))
    entries = json.dumps({'clip': {'textual': {'modelName': 'embeddinggemma-2'}}})
    first = await client.post('/predict', data={'entries': entries, 'text': 'hello'})
    second = await client.post('/predict', data={'entries': entries, 'text': 'hello'})
    assert first.status_code == 200
    assert second.status_code == 200
    assert [request.url.host for request in client.upstream.requests] == ['llama-a', 'llama-b']


async def test_clip_fails_over(client):
    embedder.set_urls(('http://llama-a:31234', 'http://llama-b:31234'))
    client.upstream.fail_hosts['llama-a'] = 503
    entries = json.dumps({'clip': {'textual': {'modelName': 'embeddinggemma-2'}}})
    response = await client.post('/predict', data={'entries': entries, 'text': 'hello'})
    assert response.status_code == 200
    assert json.loads(response.json()['clip']) == VECTOR
    assert [request.url.host for request in client.upstream.requests] == ['llama-a', 'llama-b']


async def test_ml_round_robin_and_failover(client):
    ml_proxy.set_urls(('http://ml-a:3003', 'http://ml-b:3003'))
    client.ml.fail_hosts['ml-a'] = 503
    entries = json.dumps({'ocr': {'detection': {'modelName': 'PP-OCRv5_mobile'}}})
    image = _jpeg()
    first = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', image, 'image/jpeg')},
    )
    second = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', image, 'image/jpeg')},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert [request.url.host for request in client.ml.requests] == ['ml-a', 'ml-b', 'ml-b']


async def test_ml_client_error_is_not_retried(client):
    ml_proxy.set_urls(('http://ml-a:3003', 'http://ml-b:3003'))
    client.ml.fail_hosts['ml-a'] = 400
    entries = json.dumps({'ocr': {'detection': {'modelName': 'PP-OCRv5_mobile'}}})
    response = await client.post(
        '/predict',
        data={'entries': entries},
        files={'image': ('photo.jpg', _jpeg(), 'image/jpeg')},
    )
    assert response.status_code == 400
    assert [request.url.host for request in client.ml.requests] == ['ml-a']


async def test_predict_upstream_error(client):
    client.upstream.status_code = 500
    client.upstream.body = {'error': 'boom'}
    entries = json.dumps({'clip': {'textual': {'modelName': 'embeddinggemma-2'}}})
    response = await client.post('/predict', data={'entries': entries, 'text': 'hello'})
    assert response.status_code == 502
