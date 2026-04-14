"""
metadata.py — Filename and EXIF metadata scoring for imagesift.

Produces a score in [0, 1] representing how well an image's filename,
parent directory names, and embedded EXIF text match the query keywords.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS


# Common English stopwords to strip before keyword matching.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "in", "on", "at", "of", "to", "for", "with",
        "and", "or", "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "that", "this", "these", "those", "from", "by",
        "as", "but", "not", "no", "so", "if", "up", "do", "did",
    }
)

# EXIF tag names we consider for text matching.
_EXIF_TEXT_TAGS = frozenset(
    {"ImageDescription", "UserComment", "Artist", "Copyright", "XPComment", "XPTitle"}
)


def _tokenise(text: str) -> list[str]:
    """
    Lower-case, split on non-alphanumeric characters, and remove stopwords.
    Returns a list of meaningful keyword tokens.
    """
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _exif_text(image_path: str) -> str:
    """
    Extract all human-readable text from an image's EXIF metadata.
    Returns an empty string if EXIF is unavailable or unreadable.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if not exif_data:
            return ""
        parts: list[str] = []
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, "")
            if tag_name in _EXIF_TEXT_TAGS and isinstance(value, (str, bytes)):
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                parts.append(value)
        return " ".join(parts)
    except Exception:
        return ""


def score(rel_path: str, abs_path: str, description: str) -> float:
    """
    Return a metadata match score in [0, 1] for a single image.

    Scoring sources (all weighted equally, then averaged):
        1. Filename stem  (e.g. "dog_snow_2024")
        2. Parent directory names along the relative path
        3. EXIF text fields

    The score is the fraction of query keywords matched across all sources.
    A score of 1.0 means every keyword was found somewhere in the metadata.
    """
    keywords = _tokenise(description)
    if not keywords:
        return 0.0

    path = Path(rel_path)

    # Collect all searchable text tokens from metadata sources.
    searchable_tokens: set[str] = set()

    # 1. Filename stem
    searchable_tokens.update(_tokenise(path.stem))

    # 2. Parent directory names (all parts of the relative path)
    for part in path.parts[:-1]:
        searchable_tokens.update(_tokenise(part))

    # 3. EXIF text
    exif = _exif_text(abs_path)
    if exif:
        searchable_tokens.update(_tokenise(exif))

    matched = sum(1 for kw in keywords if kw in searchable_tokens)
    return matched / len(keywords)
