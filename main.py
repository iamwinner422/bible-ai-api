import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SCORE = 0.30  

app = FastAPI(title="AI Bible Search API", description="Search the Bible using AI embeddings.", version="1.0.0")

# CORS 
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

model = TextEmbedding(MODEL, cache_dir="models")
emb = np.load("outputs/bible_emb.npy").astype("float32")
with open("outputs/bible_meta.json", encoding="utf-8") as file:
    meta = json.load(file)


def overlaps(a, b):
    return (a["book"] == b["book"] and a["chapter"] == b["chapter"]
            and a["start"] <= b["end"] and b["start"] <= a["end"])


@app.get("/")
def health():
    return {"status": "ok", "passages": len(meta)}


@app.get("/search")
def search(query: str, maxResponsesSize: int = 8):
    maxResponsesSize = min(max(maxResponsesSize, 1), 20)
    qv = np.array(list(model.embed([query]))[0], dtype="float32")
    qv /= np.linalg.norm(qv)

    scores = emb @ qv  # similarity with all passages at once
    results = []
    for idx in np.argsort(-scores)[: maxResponsesSize * 6]:
        if scores[idx] < MIN_SCORE:
            break
        p = meta[idx]
        if any(overlaps(p, r) for r in results):
            continue  # avoid duplicate overlapping passages
        results.append(p)
        if len(results) == maxResponsesSize:
            break

    return [{
        "reference": f"{p['book']} {p['chapter']}:{p['start']}-{p['end']}",
        "text": p["text"],
        "score": round(float(scores[meta.index(p)]), 3),
    } for p in results]