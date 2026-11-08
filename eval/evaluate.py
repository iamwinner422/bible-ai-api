"""
Measure search quality on eval/questions.json.

A question is a hit at rank k when one of the first k returned verses falls
inside one of its expected references.

Usage (from the project root):
    python eval/evaluate.py             # summary table
    python eval/evaluate.py --details   # also show the result of each question
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from search_engine import SearchEngine  # noqa: E402

K = 8


def parse_ref(ref):
    """'Actes des Apôtres 9:1-19' -> (book, chapter, first verse, last verse)."""
    book, loc = ref.rsplit(" ", 1)
    if ":" not in loc:
        return book, int(loc), 1, 10**6  # whole chapter
    chapter, verses = loc.split(":")
    first, _, last = verses.partition("-")
    return book, int(chapter), int(first), int(last or first)


def is_hit(p, refs):
    return any(p["book"] == b and p["chapter"] == c and p["start"] <= hi and p["end"] >= lo
               for b, c, lo, hi in refs)


def check_refs(questions, meta):
    known = {(p["book"], p["chapter"]) for p in meta}
    bad = [r for q in questions for r in q["refs"] if parse_ref(r)[:2] not in known]
    if bad:
        sys.exit(f"Unknown references in questions.json: {bad}")


def evaluate(engine, questions, meta, semantic_only=False, details=False):
    hits = {1: 0, 3: 0, K: 0}
    rr = 0.0
    start = time.time()
    for q in questions:
        refs = [parse_ref(r) for r in q["refs"]]
        results = engine.search(q["q"], size=K, semantic_only=semantic_only)
        rank = next((n for n, (i, _, _) in enumerate(results, 1) if is_hit(meta[i], refs)), None)
        for k in hits:
            hits[k] += rank is not None and rank <= k
        rr += 1 / rank if rank else 0
        if details:
            top = meta[results[0][0]] if results else None
            got = f"{top['book']} {top['chapter']}:{top['start']}" if top else "-"
            print(f"    {'#' + str(rank) if rank else 'MISS':>5}  {q['q'][:60]:<60}  top: {got}")
    n = len(questions)
    return {
        "hit@1": hits[1] / n, "hit@3": hits[3] / n, f"hit@{K}": hits[K] / n,
        "mrr": rr / n, "ms/query": 1000 * (time.time() - start) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true", help="print the result of each question")
    args = ap.parse_args()

    with open("outputs/bible_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    check_refs(questions, meta)

    emb = np.load("outputs/bible_emb.npy").astype("float32")
    engine = SearchEngine(emb, meta)

    rows = []
    for mode, semantic_only in [("semantic", True), ("hybrid", False)]:
        if args.details:
            print(f"-- {mode}")
        rows.append((mode, evaluate(engine, questions, meta, semantic_only, args.details)))

    print(f"\n{len(questions)} questions")
    print(f"{'mode':<10} {'hit@1':>6} {'hit@3':>6} {'hit@' + str(K):>6} {'MRR':>6} {'ms/q':>6}")
    for mode, m in rows:
        print(f"{mode:<10} {m['hit@1']:>6.0%} {m['hit@3']:>6.0%} "
              f"{m[f'hit@{K}']:>6.0%} {m['mrr']:>6.2f} {m['ms/query']:>6.0f}")


if __name__ == "__main__":
    main()
