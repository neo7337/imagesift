#!/usr/bin/env python3
"""
search_images.py — Natural language image search CLI for imagesift.

Queries the SQLite embedding cache built by index_images.py and returns
the top-N images that best match a text description, combining CLIP
semantic similarity with filename/EXIF metadata scoring.

Usage:
    python search_images.py --description "a dog playing in snow" [options]

    mise run search -- --description "a dog playing in snow" [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import clip_model
import db
import metadata as meta


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB = "embeddings.db"
DEFAULT_TOP_N = 10
DEFAULT_OUTPUT = "results.json"
DEFAULT_METADATA_WEIGHT = 0.2


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search(
    description: str,
    folder: str,
    db_path: str = DEFAULT_DB,
    top_n: int = DEFAULT_TOP_N,
    metadata_weight: float = DEFAULT_METADATA_WEIGHT,
) -> list[dict]:
    """
    Return a ranked list of image matches for *description*.

    Each result dict contains:
        relative_path, clip_score, metadata_score, final_score
    """
    conn = db.connect(db_path)
    total = db.count(conn)

    if total == 0:
        print(
            "The embedding database is empty. Run index_images.py first.",
            file=sys.stderr,
        )
        return []

    print(f"Searching {total:,} indexed image(s) ...")

    # 1. Encode the query text.
    text_embedding = clip_model.encode_text(description)

    # 2. Load all image embeddings from DB.
    rel_paths, embeddings_matrix = db.load_all_embeddings(conn)
    conn.close()

    # 3. CLIP cosine similarity (vectorised — fast even for 50k+ images).
    clip_scores: np.ndarray = clip_model.cosine_similarity(text_embedding, embeddings_matrix)

    # Normalise CLIP scores to [0, 1] using min-max over this result set.
    c_min, c_max = clip_scores.min(), clip_scores.max()
    if c_max > c_min:
        clip_scores_norm = (clip_scores - c_min) / (c_max - c_min)
    else:
        clip_scores_norm = np.zeros_like(clip_scores)

    # 4. Metadata scores — derive abs_path from folder + rel_path.
    root = Path(folder).expanduser().resolve()
    metadata_scores = np.array(
        [
            meta.score(rel, str(root / rel), description)
            for rel in rel_paths
        ],
        dtype=np.float32,
    )

    # 5. Combined score.
    w = float(np.clip(metadata_weight, 0.0, 1.0))
    final_scores = (1.0 - w) * clip_scores_norm + w * metadata_scores

    # 6. Rank and take top-N.
    top_indices = np.argsort(final_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append(
            {
                "relative_path": rel_paths[idx],
                "clip_score": round(float(clip_scores[idx]), 6),
                "metadata_score": round(float(metadata_scores[idx]), 6),
                "final_score": round(float(final_scores[idx]), 6),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(
    results: list[dict],
    description: str,
    folder: str,
    top_n: int,
    total_indexed: int,
    output_path: str,
) -> None:
    payload = {
        "description": description,
        "folder": folder,
        "top_n": top_n,
        "searched_at": datetime.now().isoformat(timespec="seconds"),
        "total_indexed": total_indexed,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to: {output_path}")


def print_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return
    print(f"\nTop {len(results)} match(es):\n")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:>3}. [{r['final_score']:.3f}] "
            f"clip={r['clip_score']:.3f}  meta={r['metadata_score']:.3f}  "
            f"{r['relative_path']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search indexed images by natural language description.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--description", "-d",
        required=True,
        help="Natural language description of the image to find.",
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Root folder that was indexed (used to resolve absolute paths for metadata scoring).",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="Path to the SQLite embeddings database.",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top results to return.",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--metadata-weight", "-w",
        type=float,
        default=DEFAULT_METADATA_WEIGHT,
        dest="metadata_weight",
        help=(
            "Weight (0.0–1.0) given to filename/EXIF metadata score vs CLIP score. "
            "0.0 = pure CLIP, 1.0 = pure metadata."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = search(
        description=args.description,
        folder=args.folder,
        db_path=args.db,
        top_n=args.top,
        metadata_weight=args.metadata_weight,
    )

    print_results(results)

    conn = db.connect(args.db)
    total = db.count(conn)
    conn.close()

    write_output(
        results=results,
        description=args.description,
        folder=args.folder,
        top_n=args.top,
        total_indexed=total,
        output_path=args.output,
    )
