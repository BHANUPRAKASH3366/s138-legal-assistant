"""Embed the filtered s.138 chunks into a local Chroma collection.

Mirrors the GeM portal's proven approach: embeddings are computed here with a
normalized BGE model and passed to Chroma explicitly, so Chroma never embeds
anything itself and no default embedding function is ever attached to the
collection.  Queries must therefore supply their own vectors -- see retrieval.py.

Model note
----------
``BAAI/bge-small-en-v1.5`` is a general-purpose model and is used because it is
fast on CPU and already proven on this machine.  A legal-domain model such as
``law-ai/InLegalBERT`` would likely retrieve better on Indian judgments and can
be swapped in with ``--model``; the collection must then be rebuilt with
``--reset`` because vectors from different models are not comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "s138_chunks.json"
DEFAULT_CHROMA_PATH = DATA_DIR / "chroma_db"
DEFAULT_COLLECTION = "s138_judgments"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Chroma rejects None and non-scalar metadata values.
def clean_metadata(metadata: dict) -> dict:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed s.138 chunks into Chroma.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset", action="store_true", help="Delete the collection before embedding.")
    arguments = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    if not arguments.input.is_file():
        raise SystemExit(f"Chunk file not found: {arguments.input}. Run build_corpus.py first.")

    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    judgments = payload.get("judgments", [])
    if not judgments:
        raise SystemExit("No judgments in the chunk file.")

    client = chromadb.PersistentClient(path=str(arguments.chroma_path))
    if arguments.reset:
        try:
            client.delete_collection(arguments.collection)
            print(f"Deleted existing collection: {arguments.collection}", flush=True)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=arguments.collection)

    print(f"Loading embedding model: {arguments.model}", flush=True)
    model = SentenceTransformer(arguments.model)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not ids:
            return
        vectors = model.encode(documents, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas,
                          embeddings=vectors.tolist())
        processed += len(ids)
        print(f"Embedded {processed:,} chunks", flush=True)
        ids.clear()
        documents.clear()
        metadatas.clear()

    for judgment in judgments:
        base = clean_metadata(judgment.get("metadata", {}))
        base["judgment_id"] = judgment["judgment_id"]
        for chunk in judgment.get("chunks", []):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            ids.append(chunk["chunk_id"])
            documents.append(text)
            metadatas.append({**base,
                              "chunk_index": chunk.get("index", 0),
                              "start_char": chunk.get("start_char", 0)})
            if len(ids) >= arguments.batch_size:
                flush()
    flush()

    print(f"Done. Collection '{arguments.collection}' contains {collection.count():,} chunks.")


if __name__ == "__main__":
    main()
