"""EmbeddingGemma 2 encoder used for Immich CLIP requests.

Text queries use the SearchQuery task prompt. Photos are encoded as images
with no prompt, so both land in the same vector space. Audio weights are left
unloaded unless ML_LOAD_AUDIO is set. Inference is CPU-only.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np
from PIL import Image

from immich_embeddinggemma.config import Settings

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    dim: int

    def encode_text(self, text: str) -> np.ndarray: ...

    def encode_image(self, image: Image.Image) -> np.ndarray: ...

    def close(self) -> None: ...


class GemmaEncoder:
    """Lazy SentenceTransformer wrapper around google/embeddinggemma-2."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._settings.truncate_dim

    def encode_text(self, text: str) -> np.ndarray:
        model = self._load()
        with self._lock:
            vector = model.encode(
                text,
                prompt_name="SearchQuery",
                normalize_embeddings=self._settings.normalize,
                convert_to_numpy=True,
            )
        return _as_vector(vector, self.dim)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        model = self._load()
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        with self._lock:
            vector = model.encode(
                {"image": rgb},
                normalize_embeddings=self._settings.normalize,
                convert_to_numpy=True,
            )
        return _as_vector(vector, self.dim)

    def close(self) -> None:
        with self._lock:
            self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            config_kwargs: dict = {}
            if not self._settings.load_audio:
                config_kwargs["audio_config"] = None
            # Photos only need the vision tower. Keep it loaded.
            logger.info(
                "Loading %s on %s (truncate_dim=%s, audio=%s)",
                self._settings.model_id,
                self._settings.device,
                self._settings.truncate_dim,
                self._settings.load_audio,
            )
            self._model = SentenceTransformer(
                self._settings.model_id,
                device=self._settings.device,
                truncate_dim=self._settings.truncate_dim,
                config_kwargs=config_kwargs or None,
            )
            logger.info("Model ready")
            return self._model


class StubEncoder:
    """Deterministic unit vectors for tests. Does not download weights."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode_text(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**32)
        return _unit(np.random.default_rng(seed).standard_normal(self.dim))

    def encode_image(self, image: Image.Image) -> np.ndarray:
        seed = (image.size[0] * 1_000_003 + image.size[1]) % (2**32)
        return _unit(np.random.default_rng(seed).standard_normal(self.dim))

    def close(self) -> None:
        return None


def build_encoder(settings: Settings) -> Encoder:
    if settings.stub_mode:
        logger.warning("ML_STUB is set; returning synthetic embeddings")
        return StubEncoder(settings.truncate_dim)
    return GemmaEncoder(settings)


def _as_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] != dim:
        raise RuntimeError(f"Expected a {dim}-d embedding, got {array.shape[0]}")
    return array


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm
