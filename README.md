# AI Bible Search API

A FastAPI API for searching the Louis Segond 1910 (LSG) Bible. Results combine
semantic similarity from a multilingual MiniLM model with BM25 keyword search.

## Deploy on Vercel

Vercel detects the FastAPI application exported as `app` in `main.py`. The
Python version is set to 3.12 in `.python-version`, and dependencies are listed
in `requirements.txt`.

1. Import this repository into Vercel.
2. Keep the project root as the **Root Directory** and use the default
   framework/build settings.
3. Deploy. Vercel installs the requirements and serves the API.

The model and precomputed Bible index are included in `model/` and `outputs/`.
The application loads them during startup, so the first request waits for the
index to be ready. If startup cannot load the files, the health endpoint reports
the error.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check and number of indexed passages |
| `GET` | `/search` | Search Bible verses |
| `GET` | `/docs` | Interactive API documentation |

`/search` requires a `query` parameter. `responseSize` is optional, defaults to
`8`, and is limited to `1`–`20`.

```text
/search?query=quel%20est%20le%20plus%20grand%20commandement&responseSize=5
```

Wrap the query in double quotes for an exact phrase search. Results include the
reference, verse text, semantic score, and whether a query keyword matched.
