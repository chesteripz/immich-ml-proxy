import pytest

from immich_ml_proxy.config import Settings


def test_requires_upstream_urls(monkeypatch):
    monkeypatch.delenv('LLAMA_SERVER_URL', raising=False)
    monkeypatch.delenv('LLAMA_MODEL', raising=False)
    monkeypatch.delenv('IMMICH_ML_URL', raising=False)
    with pytest.raises(ValueError, match='LLAMA_SERVER_URL'):
        Settings.from_env()

    monkeypatch.setenv('LLAMA_SERVER_URL', 'http://127.0.0.1:8080')
    with pytest.raises(ValueError, match='IMMICH_ML_URL'):
        Settings.from_env()


def test_reads_upstream_overrides(monkeypatch):
    monkeypatch.setenv('LLAMA_SERVER_URL', 'http://127.0.0.1:8080/')
    monkeypatch.setenv('LLAMA_MODEL', 'embeddinggemma2')
    settings = Settings.from_env()
    assert settings.llama_server_urls == ('http://127.0.0.1:8080',)
    assert settings.model == 'embeddinggemma2'


def test_splits_comma_separated_urls(monkeypatch):
    monkeypatch.setenv(
        'LLAMA_SERVER_URL',
        'http://llama-a:31234, http://llama-b:31234/',
    )
    monkeypatch.setenv('IMMICH_ML_URL', 'http://ml-a:3003,http://ml-b:3003')
    settings = Settings.from_env()
    assert settings.llama_server_urls == ('http://llama-a:31234', 'http://llama-b:31234')
    assert settings.immich_ml_urls == ('http://ml-a:3003', 'http://ml-b:3003')


def test_rejects_empty_upstream(monkeypatch):
    monkeypatch.setenv('LLAMA_SERVER_URL', '  , ')
    with pytest.raises(ValueError, match='LLAMA_SERVER_URL'):
        Settings.from_env()
