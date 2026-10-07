import io
import json

import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

from immich_embeddinggemma.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def _jpeg(width: int = 32, height: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 80, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


async def test_ping(client):
    response = await client.get("/ping")
    assert response.status_code == 200
    assert response.text == "pong"


async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Immich ML"


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["truncateDim"] == 512
    assert body["device"] == "cpu"
    assert body["stub"] is True


async def test_predict_text(client):
    entries = json.dumps(
        {"clip": {"textual": {"modelName": "ViT-B-32__openai", "options": {"language": "en"}}}}
    )
    response = await client.post("/predict", data={"entries": entries, "text": "a photo of a cat"})
    assert response.status_code == 200
    body = response.json()
    embedding = json.loads(body["clip"])
    assert len(embedding) == 512
    norm = sum(value * value for value in embedding) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)
    assert "imageHeight" not in body


async def test_predict_image(client):
    entries = json.dumps({"clip": {"visual": {"modelName": "ViT-B-32__openai"}}})
    response = await client.post(
        "/predict",
        data={"entries": entries},
        files={"image": ("photo.jpg", _jpeg(40, 30), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["imageWidth"] == 40
    assert body["imageHeight"] == 30
    embedding = json.loads(body["clip"])
    assert len(embedding) == 512


async def test_predict_requires_input(client):
    entries = json.dumps({"clip": {"textual": {"modelName": "ViT-B-32__openai"}}})
    response = await client.post("/predict", data={"entries": entries})
    assert response.status_code == 400


async def test_predict_bad_entries(client):
    response = await client.post("/predict", data={"entries": "not-json", "text": "hello"})
    assert response.status_code == 422


async def test_predict_rejects_faces(client):
    entries = json.dumps(
        {"facial-recognition": {"detection": {"modelName": "buffalo_l"}, "recognition": {"modelName": "buffalo_l"}}}
    )
    response = await client.post(
        "/predict",
        data={"entries": entries},
        files={"image": ("photo.jpg", _jpeg(), "image/jpeg")},
    )
    assert response.status_code == 501


async def test_predict_invalid_image(client):
    entries = json.dumps({"clip": {"visual": {"modelName": "ViT-B-32__openai"}}})
    response = await client.post(
        "/predict",
        data={"entries": entries},
        files={"image": ("photo.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400
