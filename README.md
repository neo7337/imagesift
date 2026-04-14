# imagesift

Search a folder of images using natural language descriptions.

Recursively walks any locally accessible folder (including Google Drive for Desktop mounts), computes [CLIP](https://github.com/mlfoundations/open_clip) semantic embeddings for every image, and returns the top-N matches for a query like `"a dog playing in snow"`. Embeddings are cached in a local SQLite database so repeated searches are instant.

---

## Features

- **Natural language search** — powered by OpenAI CLIP (`ViT-B-32`), runs fully offline, no GPU needed
- **Persistent embedding cache** — SQLite-backed; only new or changed files are re-embedded on subsequent index runs
- **Incremental indexing** — detects changes via `mtime` + file size; stale entries are pruned automatically
- **Combined scoring** — CLIP semantic similarity + filename/EXIF metadata keyword matching, with configurable weighting
- **Works on any local folder** — Google Drive for Desktop, external drives, NAS mounts, or any directory tree
- **JSON output** — ranked results with per-image `clip_score`, `metadata_score`, and `final_score`
- **Reproducible setup** — [mise](https://mise.jdx.dev) pins the Python version and manages the virtualenv

---

## Requirements

- [mise](https://mise.jdx.dev) (`curl https://mise.run | sh`)
- macOS, Linux, or WSL (Windows)
- ~500 MB disk space for the CLIP model (downloaded once on first run)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourname/imagesift.git
cd imagesift

# 2. Trust the mise config and install Python 3.12 + create .venv
mise trust
mise install

# 3. Install Python dependencies
mise run install

# 4. Index your folder (run once; re-run incrementally after adding photos)
mise run index -- --folder "/Volumes/GoogleDrive/My Drive/Photos"

# 5. Search
mise run search -- \
  --folder "/Volumes/GoogleDrive/My Drive/Photos" \
  --description "a dog playing in snow" \
  --top 10 \
  --output results.json
```

Results are printed to the terminal and written to `results.json`.

---

## Usage

### Indexing

```
python index_images.py --folder <path> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--folder`, `-f` | *(required)* | Root folder to index recursively |
| `--db` | `embeddings.db` | Path to the SQLite embeddings database |
| `--extensions`, `-e` | `.jpg .jpeg .png .webp .gif .bmp .tiff` | Image file extensions to include |
| `--force` | off | Re-embed all images, ignoring the cache |

**Examples:**

```bash
# First-time index
python index_images.py --folder ~/Pictures

# Pick up new photos added since last run
python index_images.py --folder ~/Pictures

# Rebuild everything from scratch
python index_images.py --folder ~/Pictures --force

# Use a custom DB location
python index_images.py --folder ~/Pictures --db ~/imagesift.db
```

Via mise:
```bash
mise run index  -- --folder ~/Pictures
mise run reindex -- --folder ~/Pictures   # force re-embed
```

---

### Searching

```
python search_images.py --description <text> --folder <path> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--description`, `-d` | *(required)* | Natural language description of the image |
| `--folder`, `-f` | *(required)* | Same root folder passed to the indexer |
| `--db` | `embeddings.db` | Path to the SQLite embeddings database |
| `--top`, `-n` | `10` | Number of top results to return |
| `--output`, `-o` | `results.json` | Output JSON file path |
| `--metadata-weight`, `-w` | `0.2` | Weight for filename/EXIF score (0.0 = pure CLIP, 1.0 = pure metadata) |

**Examples:**

```bash
python search_images.py \
  --folder ~/Pictures \
  --description "sunset over the ocean" \
  --top 5

python search_images.py \
  --folder ~/Pictures \
  --description "birthday cake with candles" \
  --top 20 \
  --output birthday.json \
  --metadata-weight 0.3
```

Via mise:
```bash
mise run search -- --folder ~/Pictures --description "a cat on a sofa" --top 5
```

---

## Output

### Terminal

```
Searching 4,821 indexed image(s) ...

Top 10 match(es):

    1. [0.812] clip=0.342  meta=0.800  2024/winter/dog_snow.jpg
    2. [0.743] clip=0.318  meta=0.500  trips/canada/IMG_4821.jpg
    3. [0.701] clip=0.291  meta=0.000  2023/dec/photo_0042.jpg
   ...
```

### JSON (`results.json`)

```json
{
  "description": "a dog playing in snow",
  "folder": "/Users/you/Pictures",
  "top_n": 10,
  "searched_at": "2026-04-14T10:23:00",
  "total_indexed": 4821,
  "results": [
    {
      "relative_path": "2024/winter/dog_snow.jpg",
      "clip_score": 0.342,
      "metadata_score": 0.8,
      "final_score": 0.812
    }
  ]
}
```

`relative_path` is relative to the `--folder` root passed at search time.

---

## How It Works

```
index_images.py
  └─ walks folder recursively
  └─ skips files unchanged since last run (mtime + size)
  └─ encodes each new image with CLIP ViT-B-32
  └─ stores embedding as float32 blob in SQLite

search_images.py
  └─ encodes the text description with CLIP
  └─ loads all embeddings from SQLite
  └─ computes cosine similarity (vectorised, <1s for 50k images)
  └─ computes metadata score (filename + folder names + EXIF keywords)
  └─ final_score = 0.8 × clip_score + 0.2 × metadata_score
  └─ returns top-N ranked results as JSON
```

### Scoring

| Component | Source | Weight (default) |
|---|---|---|
| CLIP semantic score | Image content vs text embedding | 80% |
| Metadata score | Filename stem, folder names, EXIF fields | 20% |

Adjust `--metadata-weight` to shift the balance.

---

## Project Structure

```
imagesift/
├── mise.toml          # Python version (3.12), venv config, task definitions
├── requirements.txt   # Python dependencies
├── db.py              # SQLite cache layer — read/write embeddings
├── clip_model.py      # CLIP model wrapper — encode_image, encode_text
├── metadata.py        # Filename + EXIF keyword scoring
├── index_images.py    # CLI: incremental folder indexer
├── search_images.py   # CLI: natural language image search
└── .gitignore
```

---

## mise Tasks

| Task | Command | Description |
|---|---|---|
| `install` | `mise run install` | Install Python dependencies |
| `index` | `mise run index -- --folder <path>` | Build/update the embedding cache |
| `reindex` | `mise run reindex -- --folder <path>` | Force re-embed all images |
| `search` | `mise run search -- --description <text> --folder <path>` | Search indexed images |

---

## Performance

| Images | First index (CPU) | Incremental re-run | Search time |
|---|---|---|---|
| 500 | ~3 min | <5 sec | <1 sec |
| 5,000 | ~25 min | <10 sec | <1 sec |
| 50,000 | ~4 hr | <30 sec | ~2 sec |

Index times are approximate for an M-series MacBook with no GPU. First-run times are dominated by CLIP inference; subsequent runs only process changed files.

---

## Limitations

- CLIP struggles with complex compositional queries (e.g. `"a woman in a red dress near a fountain at night"`). Simpler descriptions like `"red dress fountain"` tend to work better.
- Animated GIFs are indexed using only the first frame.
- EXIF metadata is absent from most smartphone screenshots and downloaded images — the metadata score will be 0 for those.
- The CLIP model (`ViT-B-32`) is downloaded once (~350 MB) to `~/.cache/huggingface` on first run.

---

## License

MIT
