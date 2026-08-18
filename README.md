# Section 138 Legal Research Assistant

A plain-language research tool about **cheque dishonour cases under Section 138
of the Negotiable Instruments Act 1881**, grounded in Supreme Court of India
judgments.

> **This is not legal advice.** It is an information tool. It cannot represent
> you, cannot account for the specifics of your case, and can be wrong. For
> advice on your situation, consult an advocate.

## Design principles

These are constraints, not preferences. They exist because the users are
litigants rather than lawyers, and a litigant cannot spot a fabricated citation.

1. **Nothing is asserted that retrieval did not return.** Case names, citations,
   section numbers and dates are inserted from corpus metadata, never generated
   by a language model.
2. **The deterministic parts stay deterministic.** Limitation periods and the
   statutory ingredients of s.138 are computed by ordinary code. No model is
   involved in any date calculation.
3. **Low confidence produces a refusal, not a guess.** When retrieval finds no
   well-supported match, the tool says so and points to how to find a lawyer.
4. **Every claim is traceable.** Each statement shown links to the judgment
   passage it came from, with its court, date and citation.

## Data source and licensing

Judgments come from the HuggingFace dataset
[`balamurugan205799/Indian-Supreme-Court-Judgments`](https://huggingface.co/datasets/balamurugan205799/Indian-Supreme-Court-Judgments),
licensed **CC-BY-4.0**, covering **1950-2025** (76 years, ~121 GB in full).

Provenance is verifiable from the data itself: every record carries a CNR
beginning `ESCR`, identifying the Supreme Court's own **eSCR** (electronic
Supreme Court Reports) system as the origin.

Two notes on the legal position:

- Judicial pronouncements are exempt from copyright under **s.52(1)(q) of the
  Copyright Act 1957**. The CC-BY-4.0 licence covers the compilation effort, and
  attribution is retained in `data/corpus_manifest.json`.
- The official portals (`judgments.ecourts.gov.in`, `sci.gov.in`) are
  **CAPTCHA protected**. That is a deliberate anti-automation control and this
  project does not attempt to defeat it.

## Year scoping

Ingestion defaults to **1989 onward**, not 1950. Section 138 was inserted into
the Negotiable Instruments Act by the 1988 amendment and came into force on
**1 April 1989**, so earlier judgments cannot contain s.138 precedent. This cuts
39 of 76 years from the corpus as a matter of law rather than convenience.

## Status

| Stage | File | State |
|---|---|---|
| Ingestion | `escr_ingest.py` | working, resumable |
| Filtering, parsing, chunking | `build_corpus.py` | working |
| Embeddings | `embed_corpus.py` | working |
| Hybrid retrieval + refusal | `retrieval.py` | working |
| Deadline / ingredients checker | `s138_rules.py` | working |
| Streamlit app | `app.py` | working |
| Plain-language layer | - | not started, deliberately last |
| Good-law / citator check | - | **not started, most important gap** |

The corpus is **not** in this repository (see `.gitignore`) — it is several GB of
third-party CC-BY data, and every byte of it is reproducible from `escr_ingest.py`.

## Setup

```powershell
py -3.11 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
```

Fetch metadata for all in-scope years (small, fast):

```powershell
.\.venv-win\Scripts\python.exe escr_ingest.py --metadata-only
```

Then fetch judgment texts. The full 1989-2025 range is roughly 10 GB; a recent
slice is enough to try it out:

```powershell
.\.venv-win\Scripts\python.exe escr_ingest.py --from-year 2015 --to-year 2025
```

Build the filtered corpus, embed it, and run the app:

```powershell
.\.venv-win\Scripts\python.exe build_corpus.py
.\.venv-win\Scripts\python.exe embed_corpus.py
.\.venv-win\Scripts\python.exe -m streamlit run app.py
```

## Known limitations

- **The statutory constants have not been verified by an advocate.** The cheque
  validity period, the 30-day notice window, the 15-day payment window and the
  one-month filing window in `s138_rules.py` were encoded from the Act during
  development. A wrong constant produces a confidently wrong deadline, given to
  someone with no way of knowing it is wrong. Verify before relying on it.
- **Supreme Court judgments only.** High Court decisions on s.138 are far more
  numerous and often more directly applicable to a litigant's own court.
- **No good-law checking.** A judgment that has been overruled, or overtaken by
  an amendment to the Act, is presented exactly like one that still binds.
- **Retrieval thresholds are corpus-specific.** `dense_confidence()` in
  `retrieval.py` is calibrated by measurement against this collection and this
  embedding model. Re-measure both bands if either changes — an uncalibrated
  threshold silently disables the refusal behaviour the design depends on.
