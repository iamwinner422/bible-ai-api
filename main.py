import json
import logging
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from search_engine import SearchEngine

W_SEM = 2.0  # weight of semantic search in the fusion (tuned with eval/evaluate.py)
W_KW = 1.0   # weight of keyword search in the fusion
SEARCH_DEPTH = 60

log = logging.getLogger("uvicorn.error")

# Loading the index takes several seconds, and up to a minute or two on small
# hosting instances. It runs in the background so the server answers right away
# (health checks pass), and /search waits until the index is ready.
state = {"engine": None, "meta": None, "error": None}


def load_index():
    try:
        log.info("Loading the search index...")
        emb = np.load("outputs/bible_emb.npy").astype("float32")
        with open("outputs/bible_meta.json", encoding="utf-8") as file:
            meta = json.load(file)
        state["engine"] = SearchEngine(emb, meta, w_sem=W_SEM, w_kw=W_KW, depth=SEARCH_DEPTH)
        state["meta"] = meta
        log.info("Search index ready: %d passages", len(meta))
    except Exception as exc:  # reported by / and /search instead of crashing silently
        state["error"] = repr(exc)
        log.exception("Could not load the search index")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=load_index, daemon=True).start()
    yield


app = FastAPI(title="AI Bible Search API", description="Search the Bible using AI embeddings.",
              version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def not_ready():
    if state["error"]:
        return JSONResponse({"status": "error", "detail": state["error"]}, status_code=500)
    return JSONResponse({"status": "loading", "detail": "The search index is loading, retry in a few seconds."},
                        status_code=503, headers={"Retry-After": "10"})


@app.get("/")
def health():
    # Always 200 so the platform health check passes while the index loads
    if state["error"]:
        return {"status": "error", "detail": state["error"]}
    if state["engine"] is None:
        return {"status": "loading"}
    return {"status": "ok", "passages": len(state["meta"])}


@app.get("/search")
def search(query: str, responseSize: int = 8):
    engine, meta = state["engine"], state["meta"]
    if engine is None:
        return not_ready()
    responseSize = min(max(responseSize, 1), 20)

    results = []
    for i, sem, kw in engine.search(query, size=responseSize):
        p = meta[i]
        results.append({
            "reference": (
                f"{p['book']} {p['chapter']}:{p['start']} (LSG)"
                if p["start"] == p["end"]
                else f"{p['book']} {p['chapter']}:{p['start']}-{p['end']} (LSG)"
            ),
            "text": p["text"],
            "score": round(sem, 3),
            "mots_cles": bool(kw > 0),
        })
    return results
