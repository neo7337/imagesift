"""
clip_model.py — Thin wrapper around open_clip for imagesift.

Loads ViT-B-32 once and exposes encode_image / encode_text helpers
that return L2-normalised float32 numpy vectors.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from PIL import Image

import open_clip
import torch


# Model identifier — ViT-B/32 is the smallest/fastest CLIP variant and
# runs comfortably on CPU.
_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"


@lru_cache(maxsize=1)
def _load() -> tuple[open_clip.CLIP, open_clip.transform, str]:
    """
    Load the CLIP model, preprocessing transform, and select compute device.
    Cached so the model is only loaded once per process.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, device


def _normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_image(image: Image.Image) -> np.ndarray:
    """
    Return a normalised float32 CLIP embedding for a PIL image.

    Shape: (512,) for ViT-B-32
    """
    model, preprocess, device = _load()
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vec = features.cpu().numpy().squeeze().astype(np.float32)
    return _normalise(vec)


def encode_text(description: str) -> np.ndarray:
    """
    Return a normalised float32 CLIP embedding for a text description.

    Shape: (512,) for ViT-B-32
    """
    model, _, device = _load()
    tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
    tokens = tokenizer([description]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    vec = features.cpu().numpy().squeeze().astype(np.float32)
    return _normalise(vec)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix of
    row-wise image embeddings.

    Both query and matrix rows are assumed to be L2-normalised already, so
    this reduces to a dot product.

    Args:
        query:  shape (D,)
        matrix: shape (N, D)

    Returns:
        similarities: shape (N,), values in [-1, 1]
    """
    return (matrix @ query).astype(np.float32)
