"""Fetch Supreme Court of India judgments into a local corpus.

Source
------
HuggingFace dataset ``balamurugan205799/Indian-Supreme-Court-Judgments``
(CC-BY-4.0), a mirror of the Supreme Court's own eSCR system -- every record
carries an ``ESCR...`` CNR confirming that provenance.

The official portals (judgments.ecourts.gov.in, sci.gov.in) are CAPTCHA
protected and are deliberately NOT scraped here.  Judicial pronouncements are
in any case exempt from copyright under s.52(1)(q) of the Copyright Act 1957;
the CC-BY-4.0 licence covers the compilation, so attribution is retained in
``corpus_manifest.json``.

Year scoping
------------
Section 138 of the Negotiable Instruments Act 1881 was inserted by the 1988
amendment and came into force on 1 April 1989.  Judgments before then cannot
contain s.138 precedent, so ``--from-year`` defaults to 1989 rather than 1950.

Layout produced::

    data/metadata/year=YYYY.parquet   per-year judgment metadata
    data/tar/year=YYYY.tar            per-year English judgment texts
    data/corpus_manifest.json         what has been fetched, with hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
TAR_DIR = DATA_DIR / "tar"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"

DATASET = "balamurugan205799/Indian-Supreme-Court-Judgments"
DATASET_LICENCE = "CC-BY-4.0"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"

# s.138 NI Act was inserted by the 1988 amendment, in force 1 April 1989.
SECTION_138_FIRST_YEAR = 1989
LATEST_YEAR = 2025

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) legal-research-ingest/0.1"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest() -> dict:
    """Remembers fetched years so a re-run only collects what is missing."""
    if not MANIFEST_PATH.exists():
        return {"dataset": DATASET, "licence": DATASET_LICENCE, "years": {}}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"[Manifest] Unreadable ({error}); starting a fresh one.")
        return {"dataset": DATASET, "licence": DATASET_LICENCE, "years": {}}
    manifest.setdefault("years", {})
    return manifest


def save_manifest(manifest: dict) -> None:
    """Atomic write, so an interrupted run cannot corrupt the manifest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = utcnow_iso()
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, MANIFEST_PATH)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, label: str, *, max_attempts: int = 5) -> bool:
    """Stream one file to disk, resuming after a dropped connection.

    These tars are hundreds of megabytes and the host will occasionally time a
    read out part-way through.  The partial file is therefore kept between
    attempts and continued with a Range request, so a failure at 189/203 MB
    costs one more chunk rather than the whole download.  The ``.part`` file is
    renamed into place only once complete, so an aborted run never leaves a
    truncated file that later looks finished.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores Range replies 200 and restarts the body;
                # only append when it actually honoured the request with a 206.
                resuming = response.status == 206 and resume_from > 0
                if resume_from and not resuming:
                    resume_from = 0
                total = int(response.headers.get("Content-Length") or 0) + resume_from
                downloaded = resume_from
                last_report = downloaded

                with partial.open("ab" if resuming else "wb") as handle:
                    while True:
                        block = response.read(1024 * 256)
                        if not block:
                            break
                        handle.write(block)
                        downloaded += len(block)
                        if total and downloaded - last_report >= 20 * 1024 * 1024:
                            last_report = downloaded
                            print(f"    {label}: {downloaded / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)

            if total and partial.stat().st_size < total:
                raise OSError(f"incomplete: {partial.stat().st_size} of {total} bytes")

            os.replace(partial, destination)
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as error:
            have = partial.stat().st_size if partial.exists() else 0
            if attempt == max_attempts:
                print(f"    {label}: FAILED after {max_attempts} attempts ({error})")
                return False
            print(f"    {label}: attempt {attempt} stopped at {have / 1e6:,.0f} MB "
                  f"({error}); resuming...", flush=True)
            time.sleep(2 * attempt)

    return False


def fetch_year(year: int, manifest: dict, *, metadata_only: bool, force: bool) -> bool:
    """Fetch one year's metadata and (optionally) its English judgment texts."""
    record = manifest["years"].get(str(year), {})
    metadata_path = METADATA_DIR / f"year={year}.parquet"
    tar_path = TAR_DIR / f"year={year}.tar"

    needs_metadata = force or not metadata_path.exists() or not record.get("metadata_sha256")
    needs_tar = (not metadata_only) and (force or not tar_path.exists() or not record.get("tar_sha256"))

    if not needs_metadata and not needs_tar:
        print(f"  {year}: already present, skipping")
        return True

    if needs_metadata:
        url = f"{BASE}/metadata/parquet/year={year}/metadata.parquet"
        if not download(url, metadata_path, f"{year} metadata"):
            return False
        record["metadata_sha256"] = file_sha256(metadata_path)
        record["metadata_bytes"] = metadata_path.stat().st_size

    if needs_tar:
        url = f"{BASE}/data/tar/year={year}/english/english.tar"
        print(f"  {year}: downloading English judgment texts...")
        if not download(url, tar_path, f"{year} texts"):
            return False
        record["tar_sha256"] = file_sha256(tar_path)
        record["tar_bytes"] = tar_path.stat().st_size

    record["fetched_at"] = utcnow_iso()
    manifest["years"][str(year)] = record
    save_manifest(manifest)

    size_note = f"{record.get('tar_bytes', 0) / 1e6:,.0f} MB texts" if needs_tar else "metadata only"
    print(f"  {year}: OK ({size_note})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Supreme Court judgments into a local corpus.")
    parser.add_argument("--from-year", type=int, default=SECTION_138_FIRST_YEAR,
                        help=f"First year to fetch (default {SECTION_138_FIRST_YEAR}, when s.138 came into force).")
    parser.add_argument("--to-year", type=int, default=LATEST_YEAR)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Fetch only the small per-year metadata, not the multi-GB judgment texts.")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present.")
    arguments = parser.parse_args()

    if arguments.from_year > arguments.to_year:
        raise SystemExit("--from-year must not be after --to-year")

    free_gb = shutil.disk_usage(PROJECT_DIR).free / 1e9
    years = list(range(arguments.from_year, arguments.to_year + 1))
    print("=" * 70)
    print("Supreme Court judgment ingestion")
    print(f"  Source : {DATASET} ({DATASET_LICENCE})")
    print(f"  Years  : {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Mode   : {'metadata only' if arguments.metadata_only else 'metadata + English texts'}")
    print(f"  Free   : {free_gb:,.1f} GB on this drive")
    print("=" * 70)

    manifest = load_manifest()
    succeeded, failed = 0, []
    for year in years:
        if fetch_year(year, manifest, metadata_only=arguments.metadata_only, force=arguments.force):
            succeeded += 1
        else:
            failed.append(year)

    save_manifest(manifest)
    print("=" * 70)
    print(f"Done. {succeeded}/{len(years)} years fetched.")
    if failed:
        print(f"Failed years (re-run to retry): {failed}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
