"""Hybrid dense + BM25 retrieval over the s.138 judgment collection.

Adapted from the GeM portal's retriever, which fuses Chroma's dense index with
its already-persisted FTS5 document index using reciprocal-rank fusion.  Two
properties of that design matter even more here than they did for tenders:

* **Exact strings must be able to win.**  "Section 139", "2024 INSC 735" and
  named doctrines are literal tokens that a purely semantic search mangles.
* **The retriever is allowed to return nothing.**  When neither a close vector
  neighbour nor real lexical support exists, an empty list is returned rather
  than the least-bad match.  For a litigant-facing tool that is the single most
  important safety property in this file: no result is a usable answer, a
  confidently irrelevant judgment is not.
"""

from __future__ import annotations

import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "data" / "chroma_db"
DEFAULT_COLLECTION = "s138_judgments"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Words that carry no evidential weight in a legal query.
GENERIC_TERMS = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "case", "court", "for", "from",
    "has", "have", "in", "is", "it", "law", "not", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "which", "who", "with", "my", "me", "i",
})


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


def _fts_query(query: str) -> str:
    """Quote each term so punctuation such as ``s.138`` is not FTS syntax."""
    terms = re.findall(r"[\w.]+", query, flags=re.UNICODE)
    terms = [t.replace('"', '""') for t in terms if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms)


def _query_terms(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9]+", query)
    return [t.lower() for t in raw if t.lower() not in GENERIC_TERMS and (len(t) >= 3 or t.isdigit())]


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def lexical_match_quality(query: str, passage: Passage, *,
                          term_df: Callable[[str], int] | None = None,
                          total_docs: int = 1) -> float:
    """Estimate whether a lexical hit genuinely supports the query.

    Rare terms are weighted far above common ones: matching "dishonour" in a
    corpus where every judgment says it proves nothing, while matching
    "condonation" or a specific citation proves a great deal.
    """
    terms = _query_terms(query)
    if not terms:
        return 0.0

    text = _normalise(passage.text)
    phrase = _normalise(query)
    if phrase and phrase in text:
        return 1.0

    def idf(term: str) -> float:
        if term_df is None:
            return 1.0
        return math.log((total_docs + 1) / (term_df(term) + 1)) + 1.0

    weights = {t: idf(t) for t in terms}
    total_weight = sum(weights.values()) or 1.0
    matched = sum(w for t, w in weights.items() if re.search(rf"\b{re.escape(t)}\b", text))
    coverage = matched / total_weight

    # A query of several rare terms that match nothing is not rescued by
    # incidental hits on its common words.
    if len(terms) >= 3 and matched < 0.4 * total_weight:
        return 0.0
    return coverage


def dense_confidence(distance: float) -> float:
    """Map vector distance to a conservative confidence.

    Calibrated by measurement, not by guesswork.  Over this collection the
    nearest neighbour for a genuine s.138 question falls between 0.41 and 0.68,
    while the nearest neighbour for an off-topic question ("recipe for chicken
    biryani", "best tourist places in Goa") never came closer than 0.88.  The
    0.80 cut-off sits inside that gap, so an off-topic query produces no
    confident dense hit and the caller's refusal path fires.

    An earlier version of this function started its top band at 1.20, which is
    above every distance the model actually produces here -- so every query,
    however absurd, scored maximum confidence and nothing was ever refused.
    Re-measure these bands whenever the embedding model or corpus changes.
    """
    if distance <= 0.55:
        return 1.0
    if distance <= 0.70:
        return 0.85
    if distance <= 0.80:
        return 0.55
    if distance <= 0.90:
        return 0.20
    return 0.0


class JudgmentRetriever:
    def __init__(self, chroma_path: Path = DEFAULT_CHROMA_PATH,
                 collection_name: str = DEFAULT_COLLECTION,
                 model_name: str = "BAAI/bge-small-en-v1.5"):
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.model_name = model_name
        self._model: Any | None = None
        self._collection: Any | None = None
        self._df_cache: dict[str, int] = {}

    @property
    def collection(self) -> Any:
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            # No embedding_function: vectors are always supplied by _embed.
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def _embed(self, query: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True,
                                    convert_to_numpy=True)[0]
        return vector.tolist()

    def _dense(self, query: str, candidates: int) -> list[Passage]:
        response = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=candidates,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Passage(cid, doc or "", dict(meta or {}), float(dist))
            for cid, doc, meta, dist in zip(
                response["ids"][0], response["documents"][0],
                response["metadatas"][0], response["distances"][0])
        ]

    def _term_df(self, term: str) -> int:
        if term in self._df_cache:
            return self._df_cache[term]
        source = self.chroma_path / "chroma.sqlite3"
        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as db:
                row = db.execute(
                    "SELECT COUNT(*) FROM embedding_fulltext_search "
                    "WHERE embedding_fulltext_search MATCH ?", (f'"{term.lower()}"',)).fetchone()
            count = row[0] if row else 0
        except sqlite3.Error:
            count = 0
        self._df_cache[term] = count
        return count

    def _lexical(self, query: str, limit: int) -> list[Passage]:
        """Query Chroma's own persisted FTS5 index -- no sidecar needed."""
        source = self.chroma_path / "chroma.sqlite3"
        fts = _fts_query(query)
        if not fts or not source.exists():
            return []
        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    "SELECT f.rowid AS mid, e.embedding_id, f.string_value AS text, "
                    "bm25(embedding_fulltext_search) AS score "
                    "FROM embedding_fulltext_search f "
                    "JOIN embeddings e ON e.id = f.rowid "
                    "JOIN segments s ON s.id = e.segment_id "
                    "JOIN collections c ON c.id = s.collection "
                    "WHERE embedding_fulltext_search MATCH ? AND c.name = ? AND s.scope = 'METADATA' "
                    "ORDER BY score LIMIT ?",
                    (fts, self.collection_name, max(limit * 5, 100))).fetchall()
                if not rows:
                    return []
                placeholders = ",".join("?" for _ in rows)
                meta_rows = db.execute(
                    "SELECT id, key, string_value, int_value, float_value, bool_value "
                    f"FROM embedding_metadata WHERE id IN ({placeholders})",
                    [r["mid"] for r in rows]).fetchall()
        except sqlite3.Error:
            return []

        by_id: dict[int, dict[str, Any]] = {r["mid"]: {} for r in rows}
        for row in meta_rows:
            if row["key"] == "chroma:document":
                continue
            for column in ("string_value", "int_value", "float_value", "bool_value"):
                if row[column] is not None:
                    by_id[row["id"]][row["key"]] = row[column]
                    break

        return [Passage(r["embedding_id"], r["text"] or "", by_id.get(r["mid"], {}), float(r["score"]))
                for r in rows][:limit]

    def search(self, query: str, *, limit: int = 8, rrf_k: int = 60) -> list[Passage]:
        """Return fused passages, or an empty list when nothing is well supported."""
        if not query.strip():
            return []
        candidates = max(limit * 5, 40)
        dense = self._dense(query, candidates)
        lexical = self._lexical(query, candidates)

        total_docs = self.collection.count()
        scored_lexical = [
            (p, lexical_match_quality(query, p, term_df=self._term_df, total_docs=total_docs))
            for p in lexical
        ]
        scored_lexical = [(p, q) for p, q in scored_lexical if q >= 0.20]

        # The refusal rule: no lexical support and no close vector neighbour
        # means we genuinely do not have an answer, and must say so.
        if not scored_lexical and not any(dense_confidence(p.score) >= 0.55 for p in dense):
            return []

        combined: dict[str, dict[str, Any]] = {}
        for rank, passage in enumerate(dense, start=1):
            confidence = dense_confidence(passage.score)
            if confidence == 0:
                continue
            item = combined.setdefault(passage.chunk_id,
                                       {"p": passage, "score": 0.0, "d": None, "l": None})
            item["score"] += confidence / (rrf_k + rank)
            item["d"] = rank

        for rank, (passage, quality) in enumerate(scored_lexical, start=1):
            item = combined.setdefault(passage.chunk_id,
                                       {"p": passage, "score": 0.0, "d": None, "l": None})
            item["score"] += (0.5 + 3.5 * quality) / (rrf_k + rank)
            item["l"] = rank

        ordered = sorted(combined.values(), key=lambda v: v["score"], reverse=True)
        return [Passage(v["p"].chunk_id, v["p"].text, v["p"].metadata, v["score"], v["d"], v["l"])
                for v in ordered][:limit]

    def group_by_judgment(self, passages: Sequence[Passage]) -> list[dict[str, Any]]:
        """Collapse passages to one entry per judgment, best passage first."""
        grouped: dict[str, dict[str, Any]] = {}
        for passage in passages:
            key = str(passage.metadata.get("judgment_id") or passage.chunk_id.split("::")[0])
            entry = grouped.setdefault(key, {
                "judgment_id": key,
                "metadata": passage.metadata,
                "passages": [],
                "score": 0.0,
            })
            entry["passages"].append(passage)
            entry["score"] = max(entry["score"], passage.score)
        return sorted(grouped.values(), key=lambda e: e["score"], reverse=True)
