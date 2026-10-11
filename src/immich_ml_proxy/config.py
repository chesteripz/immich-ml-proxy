"""Runtime settings loaded from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file() -> None:
    path = Path.cwd() / '.env'
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        line = line.removeprefix('export ')
        key, sep, value = line.partition('=')
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    return int(raw)


def _env_urls(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        raise ValueError(f'{name} must be set')
    urls = tuple(part.strip().rstrip('/') for part in raw.split(',') if part.strip())
    if not urls:
        raise ValueError(f'{name} must not be empty')
    return urls


@dataclass(frozen=True)
class Settings:
    llama_server_urls: tuple[str, ...]
    immich_ml_urls: tuple[str, ...]
    host: str = '0.0.0.0'
    port: int = 3003
    model: str = 'embeddinggemma2'
    max_image_bytes: int = 50 * 1024 * 1024
    request_timeout: int = 120
    log_level: str = 'info'

    @classmethod
    def from_env(cls) -> Settings:
        model = os.getenv('LLAMA_MODEL', 'embeddinggemma2').strip()
        if not model:
            raise ValueError('LLAMA_MODEL must not be empty')
        return cls(
            host=os.getenv('ML_HOST', '0.0.0.0'),
            port=_env_int('ML_PORT', 3003),
            llama_server_urls=_env_urls('LLAMA_SERVER_URL'),
            model=model,
            immich_ml_urls=_env_urls('IMMICH_ML_URL'),
            max_image_bytes=_env_int('ML_MAX_IMAGE_BYTES', 50 * 1024 * 1024),
            request_timeout=_env_int('ML_REQUEST_TIMEOUT', 120),
            log_level=os.getenv('ML_LOG_LEVEL', 'info').strip().lower(),
        )
