import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from bm25 import BM25, full_phrase_length, fuse, phrase_suffix_length

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SEM_SCORE = 0.30  # below this, we consider that nothing matches
W_SEM = 0.6  # weight of semantic search in the fusion
W_KW = 1.0   # weight of keyword search in the fusion
SEARCH_DEPTH = 60

app = FastAPI(title="AI Bible Search API", description="Search the Bible using AI embeddings.", version="1.0.0")

# CORS 
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

model = TextEmbedding(MODEL, cache_dir="models")
emb = np.load("outputs/bible_emb.npy").astype("float32")
with open("outputs/bible_meta.json", encoding="utf-8") as file:
    meta = json.load(file)
keywords = BM25([p["text"] for p in meta])


def overlaps(a, b):
    return (a["book"] == b["book"] and a["chapter"] == b["chapter"]
            and a["start"] <= b["end"] and b["start"] <= a["end"])


@app.get("/")
def health():
    return {"status": "ok", "passages": len(meta)}


@app.get("/search")
def search(query: str, responseSize: int = 8):
    responseSize = min(max(responseSize, 1), 20)

    # A query wrapped in double quotes only returns verses containing it verbatim
    query = query.strip()
    exact_only = len(query) >= 2 and query[0] == query[-1] == '"'
    if exact_only:
        query = query[1:-1].strip()
    phrase_length = full_phrase_length if exact_only else phrase_suffix_length

    qv =np.array(list(model.embed([query]))[0], dtype="float32")
    qv /= np.linalg.norm(qv)

    sem_scores = emb @ qv
    kw = keywords.scores(query)

    candidate_indices = set(np.argsort(-sem_scores)[:SEARCH_DEPTH])
    candidate_indices.update(np.argsort(-kw)[:SEARCH_DEPTH])
    phrase_scores = {
        int(idx): phrase_length(query, meta[int(idx)]["text"])
        for idx in candidate_indices
    }

    results = []
    for idx in fuse(
        sem_scores,
        kw,
        w_sem=W_SEM,
        w_kw=W_KW,
        min_sem=MIN_SEM_SCORE,
        depth=SEARCH_DEPTH,
        phrase_scores=phrase_scores,
        exact_only=exact_only,
    ):
        p = meta[idx]
        if any(overlaps(p, r) for _, r in results):
            continue
        results.append((idx, p))
        if len(results) >= responseSize:
            break

    return [{
        "reference": (
            f"{p['book']} {p['chapter']}:{p['start']} (LSG)"
            if p["start"] == p["end"]
            else f"{p['book']} {p['chapter']}:{p['start']}-{p['end']} (LSG)"
        ),
        "text": p["text"],
        "score": round(float(sem_scores[i]), 3),
        "mots_cles": bool(kw[i] > 0),
    } for i, p in results]