"""Section 138 research assistant -- Streamlit UI.

Design rules enforced in this file:

* Every legal proposition shown comes from a retrieved judgment passage, with
  its citation, court and date attached.  Nothing is asserted from a model's
  memory, because a litigant cannot tell a real citation from an invented one.
* Limitation periods come from ``s138_rules``, which is ordinary arithmetic.
* When retrieval finds nothing well supported, the app says so rather than
  showing the least-bad match.
"""

from __future__ import annotations

import json
import traceback
from datetime import date, timedelta
from html import escape
from pathlib import Path

import streamlit as st

from retrieval import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION, JudgmentRetriever
from s138_rules import CaseFacts, assess, search_terms_for

PROJECT_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_DIR / "data" / "s138_chunks.json"

st.set_page_config(page_title="Section 138 Research Assistant", page_icon="LA",
                   layout="wide", initial_sidebar_state="expanded")

# "Court paper" theme: warm off-white ground, serif headings, judicial green.
# Deliberately light -- someone reading about a deadline that could end their claim
# should not have to do it in a dark, product-looking interface, and the dark navy
# this replaced was visually indistinguishable from the unrelated GeM portal.
# Every class name below is unchanged; only its appearance moved.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Source+Sans+3:wght@400;600&display=swap">
<style>
  :root {
    --paper:#faf7f1; --panel:#f2ece0; --card:#fffdf8; --sunk:#f4f0e6;
    --ink:#23211c; --body:#33302a; --muted:#6d675a; --line:#ddd5c4;
    --green:#2f5d3f; --green-soft:#e4eee6; --green-ink:#254c33; --green-line:#c4dbcb;
    --serif:"Lora", Georgia, "Times New Roman", serif;
    --sans:"Source Sans 3", system-ui, -apple-system, sans-serif;
  }
  .stApp { background:var(--paper); color:var(--body); font-family:var(--sans); }
  [data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:2rem; }
  [data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
  [data-testid="stSidebar"] * { color:var(--body); }
  h1, h2, h3, h4 { font-family:var(--serif); color:var(--ink); font-weight:600; }
  .hero { padding:1.4rem 0 1.3rem; border-bottom:2px solid var(--green); margin-bottom:1.1rem; }
  .hero h1 { margin:0; font-family:var(--serif); font-size:2rem; font-weight:600; color:var(--ink); }
  .hero p { margin:.55rem 0 0; color:var(--muted); max-width:70ch; line-height:1.6; font-size:1rem; }
  .warn { border:1px solid #d9bd7d; border-left:5px solid #b8860b; background:#f8efd9;
          color:#5b4d2c; padding:.95rem 1.15rem; border-radius:4px;
          margin:.4rem 0 1.4rem; font-size:.93rem; line-height:1.6; }
  .warn strong { color:#3d3418; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:4px;
          padding:1.05rem 1.2rem; margin:.8rem 0; }
  .card-title { font-family:var(--serif); color:var(--ink); font-size:1.05rem;
                font-weight:600; margin-bottom:.5rem; }
  .cite { color:var(--muted); font-size:.82rem; margin-bottom:.55rem; }
  .passage { color:var(--body); font-size:.94rem; line-height:1.68; background:var(--sunk);
             padding:.9rem 1.05rem; border-radius:3px; border-left:3px solid var(--green);
             white-space:pre-wrap; }
  .badge { display:inline-block; padding:3px 9px; border-radius:3px; font-size:.73rem;
           font-weight:600; margin:0 .35rem .3rem 0; border:1px solid; }
  .b-met { background:var(--green-soft); color:var(--green-ink); border-color:var(--green-line); }
  .b-no  { background:#f7e4e4; color:#8c2f2f; border-color:#e2bfbf; }
  .b-risk{ background:#f8efd9; color:#7a5c14; border-color:#dfc79a; }
  .b-unk { background:var(--panel); color:var(--muted); border-color:var(--line); }
  .b-info{ background:var(--green-soft); color:var(--green-ink); border-color:var(--green-line); }
  .stButton>button { background:var(--green); color:#fbfaf6; border:1px solid var(--green);
                     border-radius:3px; font-weight:600; }
  .stButton>button:hover { background:#274e35; border-color:#274e35; color:#fbfaf6; }
  [data-testid="stTextInput"] input, [data-testid="stDateInput"] input,
  [data-baseweb="select"] > div { background:var(--card) !important;
     border-color:var(--line) !important; border-radius:3px !important; color:var(--body) !important; }
  [data-baseweb="tab-list"] { border-bottom:1px solid var(--line); gap:1.4rem; }
  [data-baseweb="tab"] { font-weight:600; }
  [data-testid="stExpander"] { border:1px solid var(--line); border-radius:4px; background:var(--card); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="hero">
  <h1>Section 138 Research Assistant</h1>
  <p>Cheque dishonour matters under the Negotiable Instruments Act 1881, researched
  against Supreme Court of India judgments. Describe your situation in plain English,
  or enter your dates to check the statutory deadlines.</p>
</section>
<div class="warn"><strong>This is not legal advice.</strong> It is an information tool built on
published judgments. It cannot represent you, does not know the facts of your case beyond what you
type, and can be wrong. Deadlines under Section 138 are short and missing one can end your claim —
consult an advocate about your situation.</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever(chroma_path: str, collection: str):
    return JudgmentRetriever(Path(chroma_path), collection)


@st.cache_resource
def load_full_texts() -> dict[str, str]:
    """Full judgment text, so a passage can be shown in context."""
    if not CHUNKS_PATH.exists():
        return {}
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return {j["judgment_id"]: j.get("full_text", "") for j in payload.get("judgments", [])}


st.sidebar.header("Corpus")
chroma_path = st.sidebar.text_input("Chroma path", str(DEFAULT_CHROMA_PATH))
collection_name = st.sidebar.text_input("Collection", DEFAULT_COLLECTION)

retriever = None
try:
    retriever = get_retriever(chroma_path, collection_name)
    count = retriever.collection.count()
    st.sidebar.success(f"{count:,} passages indexed")
    texts = load_full_texts()
    st.sidebar.write(f"**Judgments:** `{len(texts):,}`")
except Exception as error:
    st.sidebar.error(f"Corpus not available: {error}")
    st.sidebar.caption("Run `build_corpus.py` then `embed_corpus.py`.")
    texts = {}

st.sidebar.markdown("---")
result_limit = st.sidebar.slider("Passages to retrieve", 3, 20, 8)
st.sidebar.caption("Source: Supreme Court eSCR judgments, CC-BY-4.0. "
                   "The tool shows only what retrieval found; it does not write law from memory.")

tab_search, tab_timeline = st.tabs(["Research a question", "Check my deadlines"])


# --------------------------------------------------------------------------
# Tab 1 -- free-text research over judgments
# --------------------------------------------------------------------------
with tab_search:
    st.subheader("Describe your situation")
    st.caption("For example: *the cheque I received bounced and I sent a notice after 40 days*, "
               "or *can the accused say the cheque was only given as security*.")

    query = st.text_input("Your question or situation",
                          placeholder="e.g. cheque given as security, not for a debt")
    go = st.button("Search judgments", type="primary")

    if go:
        if not query.strip():
            st.warning("Please describe your situation or question first.")
        elif retriever is None:
            st.error("The judgment corpus is not loaded.")
        else:
            with st.spinner("Searching Supreme Court judgments..."):
                try:
                    passages = retriever.search(query, limit=result_limit)
                    st.session_state["passages"] = passages
                    st.session_state["query"] = query
                except Exception as error:
                    st.error(f"Search failed: {error}")
                    st.code(traceback.format_exc())
                    st.session_state["passages"] = []

    passages = st.session_state.get("passages")
    if passages is not None:
        if not passages:
            # The deliberate refusal path.
            st.warning(
                "**No judgment in this corpus clearly addresses that.** Rather than show you a "
                "loosely related case, the tool is telling you it does not have a good answer.\n\n"
                "This may mean the point is not covered by the judgments loaded so far, or that "
                "it turns on facts rather than a legal question. An advocate can advise on it — "
                "your District Legal Services Authority provides free legal aid if cost is a concern."
            )
        else:
            grouped = retriever.group_by_judgment(passages)
            st.success(f"{len(passages)} passage(s) from {len(grouped)} judgment(s).")

            for entry in grouped:
                meta = entry["metadata"]
                title = meta.get("title") or meta.get("judgment_id", "Judgment")
                citation = meta.get("citation") or ""
                neutral = meta.get("neutral_citation") or ""
                decided = meta.get("decision_date") or ""
                bench = meta.get("judge") or ""
                disposal = meta.get("disposal_nature") or ""

                bits = [f'<span class="badge b-info">{escape(str(citation))}</span>' if citation else "",
                        f'<span class="badge b-unk">{escape(str(neutral))}</span>' if neutral else "",
                        f'<span class="badge b-unk">Decided: {escape(str(decided))}</span>' if decided else "",
                        f'<span class="badge b-unk">{escape(str(disposal))}</span>' if disposal else ""]

                st.markdown(
                    f'<div class="card"><div class="card-title">{escape(str(title))}</div>'
                    f'<div class="cite">{"".join(bits)}</div>'
                    + (f'<div class="cite">Bench: {escape(str(bench))}</div>' if bench else "")
                    + '</div>', unsafe_allow_html=True)

                for passage in entry["passages"]:
                    st.markdown(f'<div class="passage">{escape(passage.text.strip())}</div>',
                                unsafe_allow_html=True)
                    marks = []
                    if passage.dense_rank:
                        marks.append(f"meaning-match #{passage.dense_rank}")
                    if passage.lexical_rank:
                        marks.append(f"keyword-match #{passage.lexical_rank}")
                    st.caption("Matched by " + (", ".join(marks) if marks else "fusion")
                               + f" · passage {passage.metadata.get('chunk_index', '?')}")

                full = texts.get(entry["judgment_id"], "")
                if full:
                    with st.expander("Read more of this judgment"):
                        start = min(p.metadata.get("start_char", 0) for p in entry["passages"])
                        window = full[max(0, start - 1500): start + 6000]
                        st.text(window)


# --------------------------------------------------------------------------
# Tab 2 -- deterministic deadline checker
# --------------------------------------------------------------------------
with tab_timeline:
    st.subheader("Check the statutory deadlines")
    st.caption("These are computed by ordinary arithmetic from the Act — no AI is involved "
               "in any date calculation. Leave a date blank if you do not know it.")

    left, right = st.columns(2)
    with left:
        cheque_date = st.date_input("Date on the cheque", value=None, format="DD/MM/YYYY")
        presented = st.date_input("Date the cheque was deposited", value=None, format="DD/MM/YYYY")
        intimation = st.date_input("Date the bank told you it bounced", value=None, format="DD/MM/YYYY")
        notice_sent = st.date_input("Date you sent the demand notice", value=None, format="DD/MM/YYYY")
    with right:
        notice_served = st.date_input("Date the notice reached the drawer", value=None, format="DD/MM/YYYY")
        complaint_filed = st.date_input("Date the complaint was filed", value=None, format="DD/MM/YYYY")
        paid = st.checkbox("The drawer paid the cheque amount after the notice")
        own_account = st.selectbox("Was the cheque from the drawer's own account?",
                                   ["Not sure", "Yes", "No"])
        for_debt = st.selectbox("Was the cheque for a debt actually owed to you?",
                                ["Not sure", "Yes", "No"])
        funds = st.selectbox("Did the bank return it for insufficient funds?",
                             ["Not sure", "Yes", "No"])

    def tri(value: str) -> bool | None:
        return {"Yes": True, "No": False}.get(value)

    if st.button("Check my position", type="primary"):
        facts = CaseFacts(
            cheque_date=cheque_date, presented_date=presented,
            dishonour_intimation_date=intimation, notice_sent_date=notice_sent,
            notice_served_date=notice_served, payment_received=paid,
            complaint_filed_date=complaint_filed,
            drawn_on_own_account=tri(own_account),
            legally_enforceable_debt=tri(for_debt),
            returned_for_funds=tri(funds),
        )
        result = assess(facts)

        if result.has_blocking_problem:
            st.error("**There is a serious problem with this claim.** "
                     "One or more requirements of Section 138 appear not to be met. "
                     "Take this to an advocate before spending anything further on it.")
        elif any(f.status == "at_risk" for f in result.findings):
            st.warning("**Time-sensitive.** Some deadlines are still open or already tight — "
                       "see the dates below.")
        else:
            st.info("Each requirement checked below is based only on what you entered.")

        st.markdown("#### Requirements of the offence")
        badge_for = {"met": ("b-met", "Met"), "not_met": ("b-no", "Not met"),
                     "at_risk": ("b-risk", "Needs attention"), "unknown": ("b-unk", "Not known")}
        for finding in result.findings:
            css, label = badge_for[finding.status]
            st.markdown(
                f'<div class="card"><div class="card-title">{escape(finding.label)} '
                f'<span class="badge {css}">{label}</span></div>'
                f'<div style="color:#cbd5e1;font-size:.92rem;line-height:1.55;">{escape(finding.detail)}</div>'
                f'<div class="cite" style="margin-top:.5rem;">{escape(finding.provision)}</div></div>',
                unsafe_allow_html=True)

        if result.deadlines:
            st.markdown("#### Your dates")
            today = date.today()
            for label, when in sorted(result.deadlines.items(), key=lambda kv: kv[1]):
                days = (when - today).days
                if days < 0:
                    note = f"passed {abs(days)} day(s) ago"
                elif days == 0:
                    note = "today"
                else:
                    note = f"in {days} day(s)"
                st.markdown(f"- **{label}:** {when:%d %B %Y} — {note}")

        # Retrieval targeted at whatever is actually weak in this case.
        st.markdown("#### What the judgments say about your weak points")
        if retriever is None:
            st.info("Load the corpus to see relevant judgments.")
        else:
            for suggested in search_terms_for(facts, result)[:3]:
                st.markdown(f"**Searched:** *{suggested}*")
                try:
                    found = retriever.search(suggested, limit=3)
                except Exception as error:
                    st.caption(f"Search failed: {error}")
                    continue
                if not found:
                    st.caption("No judgment in the loaded corpus clearly addresses this point.")
                    continue
                for passage in found:
                    meta = passage.metadata
                    label = meta.get("citation") or meta.get("neutral_citation") or meta.get("judgment_id", "")
                    st.markdown(f'<div class="passage">{escape(passage.text.strip()[:900])}</div>',
                                unsafe_allow_html=True)
                    st.caption(f"{meta.get('title','')} · {label} · {meta.get('decision_date','')}")

        st.markdown("---")
        st.markdown(
            "**Next steps generally available in a Section 138 matter.** These are procedural "
            "options under the Act, not advice about your case:\n\n"
            "- A written demand notice must go to the drawer within **30 days** of the bank's "
            "intimation of dishonour (s.138 proviso (b)).\n"
            "- The drawer then has **15 days** to pay (s.138 proviso (c)). If they pay in full, "
            "no offence under s.138 remains.\n"
            "- If they do not, a complaint may be filed within **one month** of the cause of "
            "action arising (s.142(1)(b)). Filing before that date is premature.\n"
            "- Free legal aid is available through your **District Legal Services Authority** "
            "under the Legal Services Authorities Act 1987 if you cannot afford an advocate."
        )
