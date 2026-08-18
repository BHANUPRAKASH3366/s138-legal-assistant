"""One command to go from a fresh clone to a working app.

The judgment corpus is deliberately not in the git repository -- it is several
GB of third-party CC-BY data whose per-year archives exceed GitHub's 100 MB
file limit, and all of it is reproducible from the Supreme Court's published
records.  This script reproduces it.

    python setup_corpus.py                  # 2020-2025, about 1.7 GB
    python setup_corpus.py --full           # 1989-2025, about 10 GB
    python setup_corpus.py --from-year 2015 --to-year 2025

Every stage is resumable.  If the download is interrupted, run it again and it
continues from the partial file rather than starting over.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# s.138 was inserted by the 1988 amendment, in force 1 April 1989; nothing
# earlier can contain s.138 precedent.  In practice the Supreme Court's first
# judgments on it appear in the mid-1990s.
FIRST_POSSIBLE_YEAR = 1989
LATEST_YEAR = 2025
QUICK_START_YEAR = 2020


def run(step: str, arguments: list[str]) -> None:
    print()
    print("=" * 70)
    print(step)
    print("=" * 70, flush=True)
    result = subprocess.run([sys.executable, *arguments], cwd=PROJECT_DIR)
    if result.returncode != 0:
        raise SystemExit(f"\nFailed at: {step}\nFix the error above, then run this script again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the judgment corpus from scratch.")
    parser.add_argument("--from-year", type=int, default=QUICK_START_YEAR)
    parser.add_argument("--to-year", type=int, default=LATEST_YEAR)
    parser.add_argument("--full", action="store_true",
                        help=f"Fetch every year from {FIRST_POSSIBLE_YEAR} (about 10 GB).")
    arguments = parser.parse_args()

    first = FIRST_POSSIBLE_YEAR if arguments.full else arguments.from_year
    last = arguments.to_year

    if first < FIRST_POSSIBLE_YEAR:
        raise SystemExit(f"--from-year cannot precede {FIRST_POSSIBLE_YEAR}, when s.138 came into force.")
    if first > last:
        raise SystemExit("--from-year must not be after --to-year.")

    print("Building the Supreme Court s.138 corpus")
    print(f"  Years : {first}-{last}")
    print(f"  Source: eSCR judgments via a CC-BY-4.0 mirror")
    print("  This takes a while on the first run. Every stage can be resumed.")

    run("Step 1 of 3 -- downloading judgments",
        ["escr_ingest.py", "--from-year", str(first), "--to-year", str(last)])
    run("Step 2 of 3 -- filtering to s.138 and chunking",
        ["build_corpus.py"])
    run("Step 3 of 3 -- building the search index",
        ["embed_corpus.py"])

    print()
    print("=" * 70)
    print("Corpus ready. Start the app with:")
    print("    python -m streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
