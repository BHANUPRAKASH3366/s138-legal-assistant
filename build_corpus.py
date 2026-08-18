"""Turn downloaded judgment tars into a filtered, chunked s.138 corpus.

Pipeline::

    data/tar/year=YYYY.tar  ->  extract PDF text  ->  score for s.138 relevance
                            ->  section-aware chunks  ->  data/s138_chunks.json

Why the relevance score exists
------------------------------
An initial filter of "mentions Negotiable Instruments AND section 138" pulled in
judgments that merely cite s.138 in passing -- a third of them never used the
word "cheque".  Reasoning from a case that is not actually about s.138 is worse
than returning nothing, so a judgment must now earn its place: repeated
references to the section, plus the vocabulary a real cheque-dishonour matter
necessarily uses (drawer, payee, dishonour, demand notice).

Chunking
--------
Judgments run from 15,000 to 376,000 characters, so a single embedding per
judgment would be meaningless.  Text is split on paragraph boundaries into
overlapping windows.  Every chunk keeps its parent judgment's full metadata and
its character offset, so the UI can always show the surrounding passage rather
than an isolated fragment.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TAR_DIR = DATA_DIR / "tar"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_PATH = DATA_DIR / "s138_chunks.json"

CHUNK_TARGET_CHARS = 1_400
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 120

SECTION_138 = re.compile(r"(?:section|sec\.?|s\.|u/s)\s*138", re.IGNORECASE)
NI_ACT = re.compile(r"negotiable\s+instrument", re.IGNORECASE)

# Vocabulary a genuine cheque-dishonour judgment necessarily uses.
SUPPORTING_TERMS = {
    "cheque": re.compile(r"\bcheques?\b", re.IGNORECASE),
    "dishonour": re.compile(r"\bdishonou?r", re.IGNORECASE),
    "drawer": re.compile(r"\bdrawer\b", re.IGNORECASE),
    "payee": re.compile(r"\bpayee\b", re.IGNORECASE),
    "demand_notice": re.compile(r"\b(?:legal|demand|statutory)\s+notice\b", re.IGNORECASE),
    "insufficient_funds": re.compile(r"insufficien\w*\s+fund|funds?\s+insufficien", re.IGNORECASE),
    "complainant": re.compile(r"\bcomplainant\b", re.IGNORECASE),
    "section_139": re.compile(r"(?:section|sec\.?|s\.)\s*139", re.IGNORECASE),
    "section_142": re.compile(r"(?:section|sec\.?|s\.)\s*142", re.IGNORECASE),
}

# A judgment must mention s.138 at least this often to count as being *about* it
# rather than citing it while deciding something else.
MIN_SECTION_138_MENTIONS = 3
MIN_SUPPORTING_TERMS = 3


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_relevance(text: str) -> dict:
    """Decide whether a judgment is genuinely about s.138, and say why."""
    mentions = len(SECTION_138.findall(text))
    matched = [name for name, pattern in SUPPORTING_TERMS.items() if pattern.search(text)]
    has_ni_act = bool(NI_ACT.search(text))

    relevant = (
        has_ni_act
        and mentions >= MIN_SECTION_138_MENTIONS
        and len(matched) >= MIN_SUPPORTING_TERMS
    )
    return {
        "relevant": relevant,
        "section_138_mentions": mentions,
        "supporting_terms": matched,
        "mentions_ni_act": has_ni_act,
    }


def split_into_chunks(text: str) -> list[dict]:
    """Split on paragraph boundaries into overlapping windows.

    Overlap matters here: a legal proposition and the authority it relies on are
    often in adjacent paragraphs, and a hard cut between them would leave both
    halves unintelligible on their own.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[dict] = []
    buffer = ""
    buffer_start = 0
    cursor = 0

    for paragraph in paragraphs:
        paragraph_start = text.find(paragraph, cursor)
        if paragraph_start == -1:
            paragraph_start = cursor
        cursor = paragraph_start + len(paragraph)

        if not buffer:
            buffer, buffer_start = paragraph, paragraph_start
        elif len(buffer) + len(paragraph) + 2 <= CHUNK_TARGET_CHARS:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            chunks.append({"text": buffer, "start_char": buffer_start})
            tail = buffer[-CHUNK_OVERLAP_CHARS:] if len(buffer) > CHUNK_OVERLAP_CHARS else buffer
            buffer = f"{tail}\n\n{paragraph}"
            buffer_start = max(0, paragraph_start - len(tail))

    if buffer.strip():
        chunks.append({"text": buffer, "start_char": buffer_start})

    return [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]


def load_metadata() -> dict[str, dict]:
    """Index all per-year metadata by the ``path`` that names its PDF."""
    records: dict[str, dict] = {}
    for parquet in sorted(glob.glob(str(METADATA_DIR / "year=*.parquet"))):
        frame = pd.read_parquet(parquet)
        for row in frame.to_dict("records"):
            key = str(row.get("path") or "").strip()
            if not key:
                continue
            records[key] = {
                "title": row.get("title"),
                "petitioner": row.get("petitioner"),
                "respondent": row.get("respondent"),
                "citation": row.get("citation"),
                "neutral_citation": row.get("case_id"),
                "cnr": row.get("cnr"),
                "judge": row.get("judge"),
                "decision_date": row.get("decision_date"),
                "disposal_nature": row.get("disposal_nature"),
                "court": row.get("court") or "Supreme Court of India",
                "year": int(row["year"]) if str(row.get("year", "")).isdigit() else None,
            }
    return records


def pdf_text(raw: bytes) -> str | None:
    try:
        document = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(page.get_text() for page in document)
        document.close()
        return text
    except Exception:
        return None


def process_tar(tar_path: Path, metadata: dict[str, dict], judgments: dict[str, dict]) -> tuple[int, int]:
    """Extract, score and chunk one year's judgments. Returns (read, kept)."""
    read = kept = 0
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            text = pdf_text(handle.read())
            if not text or len(text) < 500:
                continue
            read += 1

            score = score_relevance(text)
            if not score["relevant"]:
                continue

            path_key = member.name[:-7] if member.name.endswith("_EN.pdf") else Path(member.name).stem
            meta = dict(metadata.get(path_key, {}))
            meta.update({
                "path": path_key,
                "source_pdf": member.name,
                "section_138_mentions": score["section_138_mentions"],
                "supporting_terms": ",".join(score["supporting_terms"]),
                "char_count": len(text),
            })

            judgment_id = meta.get("neutral_citation") or meta.get("cnr") or path_key
            chunks = split_into_chunks(text)
            judgments[str(judgment_id)] = {
                "judgment_id": str(judgment_id),
                "metadata": meta,
                "full_text": text,
                "chunks": [
                    {
                        "chunk_id": f"{judgment_id}::{index:04d}",
                        "index": index,
                        "start_char": chunk["start_char"],
                        "text": chunk["text"],
                    }
                    for index, chunk in enumerate(chunks)
                ],
            }
            kept += 1
    return read, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the filtered s.138 chunk corpus.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--include-full-text", action="store_true", default=True,
                        help="Keep each judgment's full text for parent-passage display.")
    arguments = parser.parse_args()

    tars = sorted(TAR_DIR.glob("year=*.tar"))
    if not tars:
        raise SystemExit(f"No judgment tars found in {TAR_DIR}. Run escr_ingest.py first.")

    print(f"Loading metadata for {len(list(METADATA_DIR.glob('year=*.parquet')))} year(s)...", flush=True)
    metadata = load_metadata()
    print(f"  {len(metadata):,} judgment metadata records indexed")

    judgments: dict[str, dict] = {}
    total_read = 0
    for tar_path in tars:
        read, kept = process_tar(tar_path, metadata, judgments)
        total_read += read
        print(f"  {tar_path.stem}: {read:,} judgments read, {kept:,} kept as s.138", flush=True)

    total_chunks = sum(len(j["chunks"]) for j in judgments.values())
    payload = {
        "generated_at": utcnow_iso(),
        "source_dataset": "balamurugan205799/Indian-Supreme-Court-Judgments (CC-BY-4.0)",
        "filter": {
            "min_section_138_mentions": MIN_SECTION_138_MENTIONS,
            "min_supporting_terms": MIN_SUPPORTING_TERMS,
        },
        "judgments": list(judgments.values()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    print()
    print(f"Read {total_read:,} judgments; kept {len(judgments):,} as genuinely about s.138 "
          f"({100 * len(judgments) / max(total_read, 1):.1f}%).")
    print(f"{total_chunks:,} chunks -> {arguments.output}")


if __name__ == "__main__":
    main()
