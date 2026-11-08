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

Pass a question or keywords in French in the required `query` parameter. `responseSize` is optional, defaults to `8`, and is clamped to a range of `1` to `20`.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `query` | Yes | | Question, keywords or phrase. Wrap it in double quotes for an exact phrase search. |
| `responseSize` | No | `8` | Number of verses to return, from `1` to `20` |

Natural-language question:

```cmd
curl "http://127.0.0.1:8000/search?query=quel%20est%20le%20plus%20grand%20commandement&responseSize=5"
```

Keywords:

```cmd
curl "http://127.0.0.1:8000/search?query=Judas%20Iscariot"
```

Exact phrase only (the query is wrapped in `%22`, the URL-encoded double quote):

```cmd
curl "http://127.0.0.1:8000/search?query=%22ton%20peuple%20sera%20mon%20peuple%22"
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

| Field | Description |
|---|---|
| `reference` | Book, chapter and verse in the LSG translation |
| `text` | Verse text |
| `score` | Semantic similarity between the query and the verse (not the final ranking score) |
| `mots_cles` | `true` when at least one query word appears in the verse |

### How results are ranked

Each result is one verse, and overlapping verses are returned only once.

1. **Semantic search**: the query and every verse are converted to vectors with the `paraphrase-multilingual-MiniLM-L12-v2` model, and the closest verses are selected.
2. **Keyword search**: BM25 selects verses containing the query words, giving more weight to rare words.
3. **Fusion**: both scores are normalized to the same scale and combined, with semantic search weighted twice as much as keywords (`W_SEM` and `W_KW` in `main.py`).
4. **Exact phrase**: when a meaningful phrase at the end of the query (at least two words that are not stop words) appears verbatim in a verse, that verse is ranked first; the other results still follow.

With a query wrapped in double quotes, only verses containing the whole phrase verbatim are returned. If no verse contains it, the response is an empty array.

The API loads the generated index at startup. If you rebuild the index while the API is running, restart the API to load the new files.

## Evaluate search quality

`eval/questions.json` contains 60 French questions with the references expected for each one. The evaluation script runs them through the search engine and measures how often a correct verse is returned:

```cmd
python eval/evaluate.py
python eval/evaluate.py --details
```

`--details` shows the rank of the first correct verse for each question, or `MISS`. Current results:

| Mode | Correct verse ranked 1st | In top 3 | In top 8 | MRR |
|---|---|---|---|---|
| Semantic only | 38% | 43% | 48% | 0.42 |
| Hybrid (used by the API) | 47% | 55% | 73% | 0.54 |

Run the evaluation after changing the search settings, to make sure the change actually improves the results. To test more cases, add questions to `eval/questions.json`; book names must match those of the index (for example `Actes des Apôtres 2:1-4`, or `Daniel 6` for a whole chapter).

## Project structure

```text
.
├── build_index.py       # Loads Bible data and generates one-verse embeddings
├── bm25.py              # BM25 keyword scoring, exact phrase matching and score fusion
├── search_engine.py     # Hybrid search shared by the API and the evaluation
├── main.py              # FastAPI endpoints
├── requirements.txt     # Python dependencies
├── eval/
│   ├── questions.json   # Test questions and expected references
│   └── evaluate.py      # Search quality measurement
├── data/
│   ├── bible_fr.json    # Default source Bible text
│   └── bible_fr_2.json  # Additional source data
├── outputs/             # Generated embedding and metadata files
└── models/              # Downloaded embedding model cache
```