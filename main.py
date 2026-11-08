import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search_engine import SearchEngine

W_SEM = 2.0  # weight of semantic search in the fusion (tuned with eval/evaluate.py)
W_KW = 1.0   # weight of keyword search in the fusion
SEARCH_DEPTH = 60

app = FastAPI(title="AI Bible Search API", description="Search the Bible using AI embeddings.", version="1.0.0")

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

emb = np.load("outputs/bible_emb.npy").astype("float32")
with open("outputs/bible_meta.json", encoding="utf-8") as file:
    meta = json.load(file)
engine = SearchEngine(emb, meta, w_sem=W_SEM, w_kw=W_KW, depth=SEARCH_DEPTH)


@app.get("/")
def health():
    return {"status": "ok", "passages": len(meta)}


@app.get("/search")
def search(query: str, responseSize: int = 8):
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
