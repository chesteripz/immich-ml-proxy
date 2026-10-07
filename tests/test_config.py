import pytest

from immich_embeddinggemma.config import Settings


def test_defaults_are_cpu_512(monkeypatch):
    monkeypatch.delenv("ML_STUB", raising=False)
    monkeypatch.delenv("ML_DEVICE", raising=False)
    monkeypatch.delenv("ML_TRUNCATE_DIM", raising=False)
    monkeypatch.delenv("ML_MODEL_ID", raising=False)
    settings = Settings.from_env()
    assert settings.device == "cpu"
    assert settings.truncate_dim == 512
    assert settings.model_id == "google/embeddinggemma-2"
    assert settings.stub_mode is False


def test_rejects_cuda_until_implemented(monkeypatch):
    monkeypatch.setenv("ML_DEVICE", "cuda")
    with pytest.raises(ValueError, match="CPU"):
        Settings.from_env()


def test_rejects_unknown_dim(monkeypatch):
    monkeypatch.setenv("ML_TRUNCATE_DIM", "384")
    with pytest.raises(ValueError, match="ML_TRUNCATE_DIM"):
        Settings.from_env()
