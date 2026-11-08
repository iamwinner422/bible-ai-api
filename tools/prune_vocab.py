"""
One-time preparation: build a smaller copy of the embedding model for low-memory hosting.

The multilingual MiniLM model has a 250,000-token vocabulary covering ~50 languages.
Its token embedding matrix alone takes most of the model's RAM, while French text
only uses a small part of it. This script keeps:
  - every token used by the Bible text, so verse embeddings are unchanged;
  - the most frequent Latin-script tokens, for modern words found in questions;
and writes the pruned model and tokenizer to model/.

Words made of removed tokens are split into smaller kept tokens by the tokenizer.

Usage (from the project root), then rebuild the index with build_index.py:
    pip install -r requirements-dev.txt
    python tools/prune_vocab.py
"""
import json
import os
import shutil
import sys
import unicodedata

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import snapshot_download
from onnx import numpy_helper
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from build_index import BIBLE_FILE, build_passages  # noqa: E402

SOURCE_REPO = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
OUT_DIR = "model"
FREQUENT_TOKENS = 80_000  # Latin-script tokens kept among the first ids (most frequent)
EMBEDDINGS = "embeddings.word_embeddings.weight"


def is_latin(piece):
    piece = piece.replace("▁", "")  # word-start marker
    return all(c.isascii() or "LATIN" in unicodedata.name(c, "") for c in piece)


def main():
    src = snapshot_download(SOURCE_REPO, cache_dir="models")

    with open(os.path.join(src, "tokenizer.json"), encoding="utf-8") as f:
        tok_json = json.load(f)
    tokenizer = Tokenizer.from_str(json.dumps(tok_json))
    vocab = tok_json["model"]["vocab"]  # [[piece, score], ...], index = token id

    texts = [p["text"] for p in build_passages(BIBLE_FILE)]
    keep = {t["id"] for t in tok_json["added_tokens"]}  # special tokens
    for enc in tokenizer.encode_batch(texts):
        keep.update(enc.ids)
    bible_tokens = len(keep)
    keep.update(i for i, (piece, _) in enumerate(vocab[:FREQUENT_TOKENS]) if is_latin(piece))
    keep = sorted(keep)
    new_id = {old: new for new, old in enumerate(keep)}
    print(f"Keeping {len(keep)} of {len(vocab)} tokens ({bible_tokens} used by the Bible)")

    # Tokenizer: same pieces in the same order, so new ids match the pruned embedding rows
    tok_json["model"]["vocab"] = [vocab[i] for i in keep]
    tok_json["model"]["unk_id"] = new_id[tok_json["model"]["unk_id"]]
    for t in tok_json["added_tokens"]:
        t["id"] = new_id[t["id"]]
    post = tok_json.get("post_processor") or {}
    for t in post.get("special_tokens", {}).values():
        t["ids"] = [new_id[i] for i in t["ids"]]
    for key in ("cls", "sep"):
        if key in post:
            post[key][1] = new_id[post[key][1]]
    if tok_json.get("padding"):
        tok_json["padding"]["pad_id"] = new_id[tok_json["padding"]["pad_id"]]

    # Model: keep only the matching rows of the token embedding matrix
    model = onnx.load(os.path.join(src, "model_optimized.onnx"))
    for init in model.graph.initializer:
        if init.name == EMBEDDINGS:
            weights = numpy_helper.to_array(init)[keep]
            init.CopyFrom(numpy_helper.from_array(weights, EMBEDDINGS))
            break
    else:
        raise RuntimeError(f"{EMBEDDINGS} not found in the model")

    # Save the model already optimized by onnxruntime, so the API can skip this
    # costly step at startup. EXTENDED (not ALL) keeps the file hardware-independent.
    os.makedirs(OUT_DIR, exist_ok=True)
    pruned_path = os.path.join("models", "pruned.onnx")
    onnx.save(model, pruned_path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = os.path.join(OUT_DIR, "model.onnx")
    ort.InferenceSession(pruned_path, options, providers=["CPUExecutionProvider"])
    os.remove(pruned_path)
    with open(os.path.join(OUT_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok_json, f, ensure_ascii=False)
    shutil.copy(os.path.join(src, "config.json"), OUT_DIR)

    # Check: the pruned tokenizer must split the Bible text exactly as before
    pruned = Tokenizer.from_file(os.path.join(OUT_DIR, "tokenizer.json"))
    old_ids = np.array(keep)
    for a, b in zip(tokenizer.encode_batch(texts[:2000]), pruned.encode_batch(texts[:2000])):
        assert a.ids == old_ids[b.ids].tolist(), "tokenization changed for a verse"
    size = os.path.getsize(os.path.join(OUT_DIR, "model.onnx")) / 1e6
    print(f"Done: {OUT_DIR}/model.onnx ({size:.0f} MB) and {OUT_DIR}/tokenizer.json")


if __name__ == "__main__":
    main()
