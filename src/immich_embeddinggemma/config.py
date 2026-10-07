"""Runtime settings loaded from the environment."""

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# EmbeddingGemma 2 Matryoshka sizes. 768 is the native width.
SUPPORTED_DIMS = (768, 512, 256, 128)


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 3003
    model_id: str = "google/embeddinggemma-2"
    # Immich's default CLIP column is 512-d (ViT-B-32__openai). 512 is also a
    # supported Matryoshka width and keeps most multimodal quality.
    truncate_dim: int = 512
    # CPU only for now. A CUDA device path can be added later.
    device: str = "cpu"
    # Photos and search text. Audio weights stay unloaded.
    load_audio: bool = False
    normalize: bool = True
    preload: bool = False
    stub_mode: bool = False
    max_image_bytes: int = 50 * 1024 * 1024
    request_timeout: int = 120
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "Settings":
        truncate_dim = _env_int("ML_TRUNCATE_DIM", 512)
        if truncate_dim not in SUPPORTED_DIMS:
            allowed = ", ".join(str(dim) for dim in SUPPORTED_DIMS)
            raise ValueError(f"ML_TRUNCATE_DIM must be one of {allowed}, got {truncate_dim}")

        device = os.getenv("ML_DEVICE", "cpu").strip().lower()
        if device != "cpu":
            raise ValueError(
                f"ML_DEVICE={device!r} is not available. This build runs on CPU; CUDA comes later."
            )

        return cls(
            host=os.getenv("ML_HOST", "0.0.0.0"),
            port=_env_int("ML_PORT", 3003),
            model_id=os.getenv("ML_MODEL_ID", "google/embeddinggemma-2"),
            truncate_dim=truncate_dim,
            device=device,
            load_audio=_env_bool("ML_LOAD_AUDIO", False),
            normalize=_env_bool("ML_NORMALIZE", True),
            preload=_env_bool("ML_PRELOAD", False),
            stub_mode=_env_bool("ML_STUB", False),
            max_image_bytes=_env_int("ML_MAX_IMAGE_BYTES", 50 * 1024 * 1024),
            request_timeout=_env_int("ML_REQUEST_TIMEOUT", 120),
            log_level=os.getenv("ML_LOG_LEVEL", "info").strip().lower(),
        )
