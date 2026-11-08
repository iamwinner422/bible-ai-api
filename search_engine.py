"""Hybrid Bible search (semantic + BM25 + exact phrase), shared by the API and the evaluation."""
import numpy as np
from fastembed import TextEmbedding

from bm25 import BM25, full_phrase_length, fuse, phrase_suffix_length

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SEM_SCORE = 0.30  # below this semantic score, a passage is not considered a match


def load_model():
    # cache_dir="models": the model is downloaded to the project directory
    return TextEmbedding(MODEL, cache_dir="models")


def overlaps(a, b):
    return (a["book"] == b["book"] and a["chapter"] == b["chapter"]
            and a["start"] <= b["end"] and b["start"] <= a["end"])


class SearchEngine:
    def __init__(self, emb, meta, w_sem=2.0, w_kw=1.0, depth=60, model=None):
        self.model = model or load_model()
        self.emb = emb
        self.meta = meta
        self.keywords = BM25([p["text"] for p in meta])
        self.w_sem, self.w_kw, self.depth = w_sem, w_kw, depth

    def search(self, query, size=8, semantic_only=False):
        """Return [(passage index, semantic score, keyword score)], best first."""
        query = query.strip()
        # A query wrapped in double quotes only returns verses containing it verbatim
        exact_only = len(query) >= 2 and query[0] == query[-1] == '"'
        if exact_only:
            query = query[1:-1].strip()
        phrase_length = full_phrase_length if exact_only else phrase_suffix_length

        qv = np.array(list(self.model.embed([query]))[0], dtype="float32")
        qv /= np.linalg.norm(qv)
        sem = self.emb @ qv
        kw = self.keywords.scores(query)

        if semantic_only:
            order = [int(i) for i in np.argsort(-sem)[:self.depth]]
        else:
            candidates = set(np.argsort(-sem)[:self.depth])
            candidates.update(np.argsort(-kw)[:self.depth])
            phrase_scores = {
                int(i): phrase_length(query, self.meta[int(i)]["text"]) for i in candidates
            }
            order = fuse(sem, kw, w_sem=self.w_sem, w_kw=self.w_kw,
                         min_sem=MIN_SEM_SCORE, depth=self.depth,
                         phrase_scores=phrase_scores, exact_only=exact_only)

        results = []
        for idx in order:
            p = self.meta[idx]
            if any(overlaps(p, self.meta[r]) for r, _, _ in results):
                continue
            results.append((idx, float(sem[idx]), float(kw[idx])))
            if len(results) >= size:
                break
        return results
