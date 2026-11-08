import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np


def tokenize(text):
    """Convert to lowercase, remove accents, and split into words."""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


class BM25:
    """
    BM25 gives higher scores to passages containing query terms, especially
    RARE words (Iscariot weighs much more than "who" or "is").
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean())
        post = defaultdict(list)  # word -> [(passage index, occurrence count)]
        for i, d in enumerate(docs):
            for w, c in Counter(d).items():
                post[w].append((i, c))
        self.ids = {w: np.array([i for i, _ in p], dtype="int32") for w, p in post.items()}
        self.tfs = {w: np.array([c for _, c in p], dtype="float32") for w, p in post.items()}
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for w, p in post.items()}

    def scores(self, query):
        s = np.zeros(self.n, dtype="float32")
        for w in set(tokenize(query)):
            if w not in self.ids:
                continue
            ids, tf = self.ids[w], self.tfs[w]
            norm = tf + self.k1 * (1 - self.b + self.b * self.len[ids] / self.avg)
            s[ids] += self.idf[w] * tf * (self.k1 + 1) / norm
        return s


def fuse(sem_scores, kw_scores, w_sem=0.6, w_kw=1.0, min_sem=0.30, depth=60, c=60):
    """
    Reciprocal Rank Fusion: a passage ranked highly by either method moves up
    the list, and a passage ranked highly by both moves up even further.
    Returns passage indices from best to worst.
    """
    fused = defaultdict(float)
    for rank, idx in enumerate(np.argsort(-sem_scores)[:depth]):
        if sem_scores[idx] >= min_sem:
            fused[int(idx)] += w_sem / (c + rank + 1)
    for rank, idx in enumerate(np.argsort(-kw_scores)[:depth]):
        if kw_scores[idx] > 0:
            fused[int(idx)] += w_kw / (c + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)