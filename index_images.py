#!/usr/bin/env python3
"""
index_images.py — Incremental image indexer for imagesift.

Walks a folder tree, computes CLIP embeddings for new or modified images,
and stores them in a SQLite cache (embeddings.db).  Unchanged images are
skipped, making repeated runs fast.

Usage:
    python index_images.py --folder <path> [options]

    mise run index -- --folder <path> [options]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import db
import clip_model


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif",
}
DEFAULT_DB = "embeddings.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_images(root: Path, extensions: set[str]) -> list[Path]:
    """Recursively find all image files under *root*."""
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def _file_stat(path: Path) -> tuple[float, int]:
    """Return (mtime, file_size) for a path."""
    stat = path.stat()
    return stat.st_mtime, stat.st_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    folder: str,
    db_path: str = DEFAULT_DB,
    extensions: set[str] = DEFAULT_EXTENSIONS,
    force: bool = False,
) -> None:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {root} ...")
    images = discover_images(root, extensions)
    total = len(images)
    if total == 0:
        print("No images found.")
        return
    print(f"Found {total:,} image(s).")

    conn = db.connect(db_path)

    # Determine which images need (re-)embedding.
    to_index: list[Path] = []
    skipped = 0
    valid_rel_paths: set[str] = set()

    for img_path in images:
        rel = str(img_path.relative_to(root))
        valid_rel_paths.add(rel)
        mtime, size = _file_stat(img_path)
        if not force and db.is_cached(conn, rel, mtime, size):
            skipped += 1
        else:
            to_index.append(img_path)

    print(
        f"{len(to_index):,} new/modified image(s) to index, "
        f"{skipped:,} unchanged (skipped)."
    )

    # Embed and store.
    errors = 0
    for img_path in tqdm(to_index, desc="Indexing", unit="img", disable=len(to_index) == 0):
        rel = str(img_path.relative_to(root))
        mtime, size = _file_stat(img_path)
        try:
            image = Image.open(img_path).convert("RGB")
            embedding = clip_model.encode_image(image)
            db.upsert(conn, rel, str(img_path), mtime, size, embedding)
        except UnidentifiedImageError:
            tqdm.write(f"  [skip] Cannot read image: {rel}")
            errors += 1
        except Exception as exc:
            tqdm.write(f"  [error] {rel}: {exc}")
            errors += 1

    conn.commit()

    # Remove stale entries (files deleted from disk since last run).
    removed = db.delete_stale(conn, valid_rel_paths)
    conn.commit()
    conn.close()

    total_indexed = db.count(db.connect(db_path))
    print(f"\nDone.")
    print(f"  Newly indexed : {len(to_index) - errors:,}")
    print(f"  Errors        : {errors:,}")
    print(f"  Stale removed : {removed:,}")
    print(f"  Total in DB   : {total_indexed:,}")
    print(f"  DB path       : {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or update the imagesift embedding cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Root folder to index (searched recursively).",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="Path to the SQLite embeddings database.",
    )
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
        help="Image file extensions to include (with leading dot).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all images, ignoring the cache.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        folder=args.folder,
        db_path=args.db,
        extensions={ext.lower() for ext in args.extensions},
        force=args.force,
    )
