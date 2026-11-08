# AI-Powered Bible Search

A FastAPI search API for the Louis Segond 1910 (LSG) Bible. Search combines semantic similarity, calculated from FastEmbed embeddings, with lexical keyword matching using BM25. This helps return relevant verses for both natural-language questions and specific terms.

## Prerequisites

- Python 3.10 or later
- An Internet connection on first run to download the embedding model

## Installation

From the project root, create and activate a virtual environment, then install the dependencies:

```cmd
python -m venv env
.\env\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Build the index

The default source file is `data/bible_fr.json`. Generate the index files before starting the API:

```cmd
python build_index.py
```

The script creates the files used by the API:

- `outputs/bible_emb.npy`: passage embeddings
- `outputs/bible_meta.json`: the corresponding texts and references

The index contains one verse per passage. Index generation may take a while; the embedding model is downloaded to `models` the first time it is used. Rebuild the index whenever the source Bible data or passage-generation settings change.

## Run the API

From the project root:

```cmd
uvicorn main:app --reload
```

The API is then available at `http://127.0.0.1:8000`. Interactive documentation is available at [`/docs`](http://127.0.0.1:8000/docs) and the alternative API reference at [`/redoc`](http://127.0.0.1:8000/redoc).

## Usage

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check and number of indexed passages |
| `GET` | `/search` | Search Bible verses |

### Health check

```cmd
curl http://127.0.0.1:8000/
```

Example response:

```json
{
  "status": "ok",
  "passages": 31102
}
```

### Search

Pass a question or keywords in the required `query` parameter. `responseSize` is optional, defaults to `8`, and is clamped to a range of `1` to `20`.

```cmd
curl "http://127.0.0.1:8000/search?query=love%20and%20forgiveness&responseSize=5"
```

The response is an array of results. For example:

```json
[
  {
    "reference": "Jean 3:16 (LSG)",
    "text": "Car Dieu a tant aimé le monde...",
    "score": 0.72,
    "mots_cles": true
  }
]
```

Each result contains one verse. Normalized semantic and BM25 keyword scores are combined for ranking. When a meaningful phrase at the end of the query appears verbatim in a verse, matching verses are ranked first and the other results follow. To get only verses containing the exact phrase, wrap the query in double quotes (for example `query="Judas Iscariot"`); if no verse contains it, the response is an empty array. The `score` field remains the semantic similarity, not the combined ranking score. `mots_cles` is `true` when at least one normalized query term appears in that verse.

The API loads the generated index at startup. If you rebuild the index while the API is running, restart the API to load the new files.

## Project structure

```text
.
├── build_index.py       # Loads Bible data and generates one-verse embeddings
├── bm25.py              # BM25 keyword scoring and rank fusion
├── main.py              # FastAPI endpoints and hybrid search
├── requirements.txt     # Python dependencies
├── data/
│   ├── bible_fr.json    # Default source Bible text
│   └── bible_fr_2.json  # Additional source data
├── outputs/             # Generated embedding and metadata files
└── models/              # Downloaded embedding model cache
```