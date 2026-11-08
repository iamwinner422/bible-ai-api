import json
import numpy as np
from search_engine import MODEL, load_model
import time

BIBLE_FILE = "data/bible_fr.json"
WINDOW = 1  # one verse per search result
STEP = 1    # Reading one verse at a time
BATCH_SIZE = 500

def load_books(path):
    with open(path, encoding="utf-8-sig") as file:
        data = json.load(file)

    books = []
    if "verses" in data:
        # FIRST FORMAT: {book_name, chapter, verse, text}
        index = {}
        for v in data["verses"]:
            name = v["book_name"]
            if name not in index:
                index[name] = {}
            index[name].setdefault(v["chapter"], []).append((v["verse"], v["text"]))
        for name, chapters in index.items():
            chap_list = []
            for c in sorted(chapters):
                verses = [t for _, t in sorted(chapters[c])]
                chap_list.append(verses)
            books.append((name, chap_list))
    elif "Testaments" in data:
        # SECOND FORMAT: Testaments > Books > Chapters > Verses
        for testament in data["Testaments"]:
            for book in testament["Books"]:
                chap_list = [[v["Text"] for v in ch["Verses"]] for ch in book["Chapters"]]
                books.append((book["Text"], chap_list))
    else:
        raise ValueError("Format JSON non reconnu")
    return books


def clean(text):
    """Remove the paragraph symbol and extra whitespace."""
    return " ".join(text.replace("\u00b6", " ").split())


def build_passages(path):
    passages = []
    for name, chapters in load_books(path):
        for c, verses in enumerate(chapters, start=1):
            verses = [clean(v) for v in verses]
            for i in range(0, len(verses), STEP):
                chunk = verses[i:i + WINDOW]
                passages.append({
                    "book": name,
                    "chapter": c,
                    "start": i + 1,
                    "end": i + len(chunk),
                    "text": " ".join(chunk),
                })
    return passages


if __name__ == "__main__":
    passages = build_passages(BIBLE_FILE)
    print(f"{len(passages)} passages to convert into vectors...")

    print(f"Loading {MODEL}...")
    model = load_model()
    texts = [p["text"] for p in passages]

    parts = []
    startTime = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        parts.append(np.array(list(model.embed(batch, batch_size=64)), dtype="float32"))

        done = min(i + BATCH_SIZE, len(texts))
        elapsed = time.time() - startTime
        left = elapsed / done * (len(texts) - done) if done else 0

        print(f"Processed {done}/{len(texts)} passages ({100 * done // len(texts):.1f}%), "
              f"elapsed: {elapsed / 60:.1f}min, remaining: {left / 60:.1f}min", flush=True)

        
    vecs = np.vstack(parts)  # stack all batches into a single array

    # Normalization: each vector has a length of 1.
    # This makes cosine similarity equivalent to a simple dot product.
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    # float16: half the size, with no noticeable loss of quality.
    np.save("outputs/bible_emb.npy", vecs.astype("float16"))
    with open("outputs/bible_meta.json", "w", encoding="utf-8") as file:
        json.dump(passages, file, ensure_ascii=False)
    print("Done: bible_emb.npy and bible_meta.json created.")
