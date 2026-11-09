"""Hybrid Bible search (semantic + BM25 + exact phrase), shared by the API and the evaluation."""
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from bm25 import BM25, full_phrase_length, fuse, phrase_suffix_length

# paraphrase-multilingual-MiniLM-L12-v2 with a vocabulary reduced to French
# (Latin-script) tokens to keep the deployed API small.
MODEL = "paraphrase-multilingual-MiniLM-L12-v2 (French vocabulary)"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MAX_TOKENS = 128
MIN_SEM_SCORE = 0.30  # below this semantic score, a passage is not considered a match


class Embedder:
    """Turn texts into normalized vectors (mean of the token vectors)."""

    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False  # lower memory use between requests
        # model.onnx is saved already optimized by tools/prune_vocab.py, so skipping the
        # optimization pass makes loading ~20x cheaper. One thread is the fastest choice
        # on small hosting instances (0.1 vCPU).
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def embed(self, texts, batch_size=64):
        """Return a float32 array with one normalized vector per text."""
        parts = []
        for i in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            tokens = self.session.run(None, {
                "input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids),
            })[0].astype(np.float32)
            weights = mask[:, :, None].astype(np.float32)
            vecs = (tokens * weights).sum(1) / np.maximum(weights.sum(1), 1e-9)
            parts.append(vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
        return np.vstack(parts)


def load_model():
    return Embedder()


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

        sem = self.emb @ self.model.embed([query])[0]
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
