"""
db.py — SQLite embedding cache layer for imagesift.

Schema:
    images(id, rel_path, abs_path, mtime, file_size, embedding)

Embeddings are stored as raw float32 binary blobs for zero-overhead
serialisation/deserialisation via numpy.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional


DDL = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_path    TEXT    NOT NULL UNIQUE,
    abs_path    TEXT    NOT NULL,
    mtime       REAL    NOT NULL,
    file_size   INTEGER NOT NULL,
    embedding   BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rel_path ON images (rel_path);
"""


def connect(db_path: str = "embeddings.db") -> sqlite3.Connection:
    """Open (and initialise if new) the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def upsert(
    conn: sqlite3.Connection,
    rel_path: str,
    abs_path: str,
    mtime: float,
    file_size: int,
    embedding: np.ndarray,
) -> None:
    """Insert or replace a single image record."""
    blob = embedding.astype(np.float32).tobytes()
    conn.execute(
        """
        INSERT INTO images (rel_path, abs_path, mtime, file_size, embedding)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(rel_path) DO UPDATE SET
            abs_path  = excluded.abs_path,
            mtime     = excluded.mtime,
            file_size = excluded.file_size,
            embedding = excluded.embedding
        """,
        (rel_path, abs_path, mtime, file_size, blob),
    )


def delete_stale(conn: sqlite3.Connection, valid_rel_paths: set[str]) -> int:
    """Remove rows whose files no longer exist on disk. Returns deleted count."""
    all_paths: list[str] = [
        row["rel_path"] for row in conn.execute("SELECT rel_path FROM images")
    ]
    stale = [p for p in all_paths if p not in valid_rel_paths]
    if stale:
        conn.executemany("DELETE FROM images WHERE rel_path = ?", [(p,) for p in stale])
    return len(stale)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get(conn: sqlite3.Connection, rel_path: str) -> Optional[sqlite3.Row]:
    """Fetch a single row by relative path (or None if not found)."""
    return conn.execute(
        "SELECT * FROM images WHERE rel_path = ?", (rel_path,)
    ).fetchone()


def is_cached(
    conn: sqlite3.Connection,
    rel_path: str,
    mtime: float,
    file_size: int,
) -> bool:
    """Return True if an up-to-date embedding already exists for this file."""
    row = conn.execute(
        "SELECT mtime, file_size FROM images WHERE rel_path = ?", (rel_path,)
    ).fetchone()
    if row is None:
        return False
    return row["mtime"] == mtime and row["file_size"] == file_size


def load_all_embeddings(
    conn: sqlite3.Connection,
) -> tuple[list[str], np.ndarray]:
    """
    Return (rel_paths, embeddings_matrix) for every row in the DB.

    embeddings_matrix shape: (N, embedding_dim)
    """
    rows = conn.execute("SELECT rel_path, embedding FROM images").fetchall()
    if not rows:
        return [], np.empty((0,), dtype=np.float32)

    rel_paths = [r["rel_path"] for r in rows]
    embeddings = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    return rel_paths, embeddings


def count(conn: sqlite3.Connection) -> int:
    """Return total number of indexed images."""
    return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
