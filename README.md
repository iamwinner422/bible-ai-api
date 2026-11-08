# AI-Powered Bible Search

A FastAPI search API for the Louis Segond 1910 (LSG) Bible. Search combines semantic similarity, calculated with a multilingual MiniLM embedding model, with lexical keyword matching using BM25. This helps return relevant verses for both natural-language questions and specific terms.

The API uses about 400 MB of RAM, so it fits on free hosting tiers limited to 512 MB, such as Koyeb's.

## Prerequisites

- Python 3.12 (the version used for deployment, set in `.python-version`)

## Installation

From the project root, create and activate a virtual environment, then install the dependencies:

```cmd
python -m venv env
.\env\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Model and index

Everything the API needs is in the repository, so there is nothing to generate before running it:

- `model/`: the embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) with its vocabulary reduced to French
- `outputs/bible_emb.npy`: one embedding per verse
- `outputs/bible_meta.json`: the corresponding texts and references

### Why the model is reduced

The original model has a 250,000-token vocabulary covering about 50 languages, and the API then needs about 800 MB of RAM. `tools/prune_vocab.py` keeps only the tokens used by the Bible text and the most frequent Latin-script tokens (about 44,000), which brings the model down to 77 MB and the API to about 400 MB. Verse embeddings are identical and question embeddings are almost identical, so search quality does not change.

### Rebuild the model and the index

This is only needed when the source Bible data (`data/bible_fr.json` by default) or the passage settings in `build_index.py` change. Rebuild the model first, then the index:

```cmd
python -m pip install -r requirements-dev.txt
python tools/prune_vocab.py
python build_index.py
```

The original model is downloaded to `models/` (ignored by Git). If the API is running, restart it to load the new files.

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

Run the evaluation after changing the search settings, to make sure the change actually improves the results.

### Add test questions

`eval/questions.json` is a test set: adding questions does **not** change the API's answers, it only makes the measurement more reliable. With 60 questions, one question is worth about 1.7 points. To add a question, add one line with the expected references:

```json
{"q": "qui a construit l'arche", "refs": ["Genèse 6:13-22"]}
```

- `refs` lists every passage that counts as a correct answer; the question is a hit as soon as one returned verse falls inside one of them.
- A reference is `Book chapter:first-last`, `Book chapter:verse`, or `Book chapter` for a whole chapter.
- Book names must match those of the index exactly, for example `Actes des Apôtres`, `Ésaïe` or `1 Corinthiens`. The script stops with an error that lists unknown references.

Then run `python eval/evaluate.py --details` to see where the new question is ranked.

## Deploy on Koyeb

The free instance (512 MB of RAM, 0.1 vCPU) is enough for the API.

1. Push the repository to GitHub, including `model/` and `outputs/`.
2. In the Koyeb control panel, create a **Web Service**, choose **GitHub** and select the repository.
3. Keep the **Buildpack** builder. Koyeb installs `requirements.txt`, uses the Python version from `.python-version`, and starts the API with the command in `Procfile`.
4. Choose the **Free** instance and a region where it is available (Frankfurt or Washington, D.C.).
5. Expose port `8000` over HTTP, and set the health check to HTTP on path `/`.
6. Deploy. Search is available at `https://<your-app>.koyeb.app/search?query=...`.

| Setting | Value |
|---|---|
| Builder | Buildpack |
| Run command | Taken from `Procfile`: `uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}` |
| Port | `8000` (HTTP) |
| Health check | HTTP, path `/` |
| Instance | Free |

The API takes about 20 seconds to start. On the free tier, the service sleeps after a period without traffic, so the first request after that is slow.

## Project structure

```text
.
├── build_index.py       # Loads Bible data and generates one-verse embeddings
├── bm25.py              # BM25 keyword scoring, exact phrase matching and score fusion
├── search_engine.py     # Embedding model and hybrid search, shared by the API and the evaluation
├── main.py              # FastAPI endpoints
├── requirements.txt     # Dependencies of the API
├── requirements-dev.txt # Extra dependencies to rebuild the model
├── Procfile             # Start command used by Koyeb
├── .python-version      # Python version used for deployment
├── model/               # Embedding model with a French vocabulary (used by the API)
├── tools/
│   └── prune_vocab.py   # Builds model/ from the original multilingual model
├── eval/
│   ├── questions.json   # Test questions and expected references
│   └── evaluate.py      # Search quality measurement
├── data/
│   ├── bible_fr.json    # Default source Bible text
│   └── bible_fr_2.json  # Additional source data
├── outputs/             # Verse embeddings and metadata (used by the API)
└── models/              # Original model download cache (ignored by Git)
```