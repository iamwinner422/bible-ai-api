import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

STOP_WORDS = set(
    "a au aux avec ce ces dans de des d du elle en et eux il je l la le les leur lui "
    "ma mais me meme mes moi mon n ne nos notre nous on ou par pas pour qu que qui sa "
    "se ses son sur ta te tes toi ton tu un une vos votre vous y "
    # Forms of "être" and "avoir" (accents already removed by tokenize)
    "suis es est sommes etes sont etait etaient ete etre "
    "ai as avons avez ont avait avaient eu avoir".split()
)


def tokenize(text):
    """Convert to lowercase, remove accents, and split into words."""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


def _contains(text_tokens, phrase_tokens):
    width = len(phrase_tokens)
    return width > 0 and any(
        text_tokens[i:i + width] == phrase_tokens
        for i in range(len(text_tokens) - width + 1)
    )


def phrase_suffix_length(query, text):
    """Return the longest meaningful query suffix found verbatim in a passage."""
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    for start in range(max(0, len(query_tokens) - 1)):
        suffix = query_tokens[start:]
        if sum(token not in STOP_WORDS for token in suffix) < 2:
            continue
        if _contains(text_tokens, suffix):
            return len(suffix)
    return 0


def full_phrase_length(query, text):
    """Return the query length if the whole query appears verbatim in a passage, else 0."""
    query_tokens = tokenize(query)
    return len(query_tokens) if _contains(tokenize(text), query_tokens) else 0


def term_coverage(query, text):
    """Count distinct non-stopword query terms present in a passage."""
    terms = set(tokenize(query)) - STOP_WORDS
    return len(terms.intersection(tokenize(text)))


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


def fuse(sem_scores, kw_scores, w_sem=2.0, w_kw=1.0, min_sem=0.30, depth=60,
         phrase_scores=None, coverage_scores=None, exact_only=False):
    """
    Combine normalized semantic and keyword scores. Passages with a phrase
    match (phrase_scores > 0) are ranked first. Among the remaining passages,
    matches for more distinct query terms rank first.
    With exact_only=True, only phrase-matching passages are kept.
    Returns passage indices from best to worst.
    """
    semantic = {
        int(idx) for idx in np.argsort(-sem_scores)[:depth]
        if sem_scores[idx] >= min_sem
    }
    keyword = {
        int(idx) for idx in np.argsort(-kw_scores)[:depth]
        if kw_scores[idx] > 0
    }
    candidates = semantic | keyword
    if not candidates:
        return []

    phrase_matches = phrase_scores or {}
    coverage_matches = coverage_scores or {}
    if exact_only:
        candidates = {idx for idx in candidates if phrase_matches.get(idx, 0)}
        if not candidates:
            return []

    # Min-max normalization over the candidates: each score is spread over [0, 1],
    # so neither method dominates just because its raw scores are higher.
    def normalizer(scores):
        values = [float(scores[idx]) for idx in candidates]
        low, high = min(values), max(values)
        return lambda idx: (float(scores[idx]) - low) / (high - low) if high > low else 0.0

    sem_norm, kw_norm = normalizer(sem_scores), normalizer(kw_scores)

    def rank_score(idx):
        return (w_sem * sem_norm(idx) + w_kw * kw_norm(idx)) / (w_sem + w_kw)

    return sorted(
        candidates,
        key=lambda idx: (
            phrase_matches.get(idx, 0), coverage_matches.get(idx, 0), rank_score(idx), -idx
        ),
        reverse=True,
    )
