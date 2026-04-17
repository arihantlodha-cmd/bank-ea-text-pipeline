"""
app.py — Streamlit web UI for the Bank EA Text Analysis Pipeline
Run locally:   streamlit run app.py
Deploy free:   https://streamlit.io/cloud
"""

import csv
import io
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank EA Text Analysis",
    page_icon="🏦",
    layout="wide",
)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "ea_pipeline_output"
CACHE_DIR  = SCRIPT_DIR / "ea_pdf_cache"

# ── Helpers ───────────────────────────────────────────────────────────────────

def csv_files_in_dir():
    return sorted(SCRIPT_DIR.glob("*.csv"))

def output_exists():
    return (OUTPUT_DIR / "ea_with_topics.csv").exists()

def load_topic_words():
    path = OUTPUT_DIR / "topic_word_lists.csv"
    if not path.exists():
        return {}
    topics = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t = int(row["topic"])
            topics.setdefault(t, []).append((row["word"], float(row["weight"])))
    return topics

def load_summary():
    path = OUTPUT_DIR / "pipeline_summary.txt"
    if not path.exists():
        return ""
    return path.read_text()

def run_pipeline(cmd):
    """Run pipeline subprocess, stream stdout line-by-line into a Streamlit container."""
    log_box = st.empty()
    lines = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(SCRIPT_DIR),
    )
    for line in process.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-60:]), language=None)
    process.wait()
    return process.returncode, "\n".join(lines)

# ── Sidebar — settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/US_Bank_logo.svg/1200px-US_Bank_logo.svg.png", width=60)
    st.title("Settings")

    # Input CSV
    csv_options = [f.name for f in csv_files_in_dir()]
    if csv_options:
        csv_choice = st.selectbox("Input CSV", csv_options)
    else:
        st.warning("No CSV found in the app folder.")
        csv_choice = None

    uploaded = st.file_uploader("Or upload a different CSV", type="csv")
    if uploaded:
        dest = SCRIPT_DIR / uploaded.name
        dest.write_bytes(uploaded.read())
        st.success(f"Saved: {uploaded.name}")
        csv_choice = uploaded.name

    st.divider()

    mode = st.radio("Run mode", ["Pilot (fast, sample)", "Full (all documents)"])
    sample_n = None
    if mode.startswith("Pilot"):
        sample_n = st.slider("Sample size", 50, 1000, 300, step=50)

    n_topics = st.slider("Number of topics", 3, 15, 6)
    workers  = st.slider("Download workers", 2, 16, 8)
    cache_exists = CACHE_DIR.exists() and any(CACHE_DIR.iterdir())
    skip_dl  = st.checkbox("Skip re-downloading cached docs", value=cache_exists,
                           help="Only check this if you've already run the pipeline once on this machine.")

    st.divider()
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏦 Bank Enforcement Action — Text Analysis")
st.caption(
    "Downloads OCC, OTS, and FRB enforcement action documents, extracts text, "
    "and uses LDA topic modeling to find recurring problem categories."
)

tab_run, tab_results, tab_about = st.tabs(["Run", "Results", "About"])

# ── Tab: Run ─────────────────────────────────────────────────────────────────
with tab_run:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regulators covered", "OCC · OTS · FRB")
    col2.metric("FDIC", "Skipped (auth required)")
    col3.metric("Topics", n_topics)
    col4.metric("Mode", f"Pilot ({sample_n} docs)" if sample_n else "Full run")

    st.divider()

    if run_btn:
        if not csv_choice:
            st.error("Please select or upload a CSV file first.")
        else:
            csv_path = SCRIPT_DIR / csv_choice
            OUTPUT_DIR.mkdir(exist_ok=True)
            CACHE_DIR.mkdir(exist_ok=True)

            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "ea_text_pipeline.py"),
                "--input", str(csv_path),
                "--output", "ea_with_topics.csv",
                "--n_topics", str(n_topics),
                "--workers", str(workers),
            ]
            if sample_n:
                cmd += ["--sample", str(sample_n)]
            if skip_dl:
                cmd += ["--skip_download"]

            st.info("Pipeline running — this takes a few minutes. Do not close the tab.")
            t0 = time.time()
            returncode, full_log = run_pipeline(cmd)
            elapsed = time.time() - t0

            if returncode == 0:
                st.success(f"Pipeline finished in {elapsed/60:.1f} min. See the **Results** tab.")
                st.balloons()
            else:
                st.error("Pipeline exited with an error. See the log above.")
    else:
        st.info("Configure settings in the sidebar, then click **▶ Run Pipeline**.")

        if output_exists():
            st.success("Previous results are available in the Results tab.")

# ── Tab: Results ─────────────────────────────────────────────────────────────
with tab_results:
    if not output_exists():
        st.info("No results yet — run the pipeline first.")
    else:
        # Summary
        summary = load_summary()
        if summary:
            with st.expander("Pipeline summary", expanded=True):
                st.code(summary)

        st.divider()

        # Topic word charts
        topic_words = load_topic_words()
        if topic_words:
            st.subheader("Topics discovered")
            st.caption("Top 15 words per topic — hover for weights")

            cols = st.columns(min(3, len(topic_words)))
            for t_idx, words in topic_words.items():
                col = cols[t_idx % len(cols)]
                top = words[:15]
                labels = [w for w, _ in top]
                weights = [v for _, v in top]
                with col:
                    st.markdown(f"**Topic {t_idx}**")
                    # Horizontal bar chart via st.bar_chart needs a dict
                    import pandas as pd
                    df = pd.DataFrame({"word": labels, "weight": weights}).set_index("word")
                    st.bar_chart(df, height=300)

        st.divider()

        # Download buttons
        st.subheader("Download outputs")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)

        ea_csv = OUTPUT_DIR / "ea_with_topics.csv"
        if ea_csv.exists():
            dcol1.download_button(
                "📥 ea_with_topics.csv",
                ea_csv.read_bytes(),
                file_name="ea_with_topics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        words_csv = OUTPUT_DIR / "topic_word_lists.csv"
        if words_csv.exists():
            dcol2.download_button(
                "📥 topic_word_lists.csv",
                words_csv.read_bytes(),
                file_name="topic_word_lists.csv",
                mime="text/csv",
                use_container_width=True,
            )

        summary_txt = OUTPUT_DIR / "pipeline_summary.txt"
        if summary_txt.exists():
            dcol3.download_button(
                "📥 pipeline_summary.txt",
                summary_txt.read_bytes(),
                file_name="pipeline_summary.txt",
                use_container_width=True,
            )

        failures_log = OUTPUT_DIR / "failures.log"
        if failures_log.exists():
            dcol4.download_button(
                "📥 failures.log",
                failures_log.read_bytes(),
                file_name="failures.log",
                use_container_width=True,
            )

        st.divider()

        # Preview table
        st.subheader("Data preview")
        try:
            import pandas as pd
            df = pd.read_csv(ea_csv, low_memory=False)
            topic_cols = [c for c in df.columns if "topic" in c.lower() and df[c].notna().any()]
            show_cols = ["Regulator", "Name", "StartDate", "URL"] + topic_cols
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols].dropna(subset=["topic_dominant"]), use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"Could not preview table: {e}")

# ── Tab: About ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ## What this does

    **Input:** A CSV of ~22,000 US bank enforcement actions from 4 regulators (FDIC, OCC, OTS, FRB),
    each with a URL pointing to the actual legal document (PDF or HTML press release).

    **Pipeline steps:**
    1. **Download** — fetches documents from OCC, OTS, and FRB in parallel (FDIC skipped — requires login)
    2. **Extract text** — parses HTML for Fed press releases; uses pdfplumber for PDFs; falls back to OCR for scanned docs
    3. **Clean text** — removes numbers, punctuation, legal boilerplate, and banking stopwords
    4. **LDA topic modeling** — unsupervised ML that finds clusters of co-occurring words across all documents
    5. **Output** — annotated CSV with a dominant topic and probability scores for each enforcement action

    ## Output columns added

    | Column | Meaning |
    |---|---|
    | `topic_dominant` | Which topic (0–N) best describes this document |
    | `topic_prob` | How strongly it matches (0–1) |
    | `topic_0_prob … topic_N_prob` | Score for every topic |
    | `extract_method` | How text was extracted: `direct`, `html`, `ocr`, `failed` |

    ## Regulators

    | Regulator | Documents | Notes |
    |---|---|---|
    | OCC | ✅ Included | Machine-readable PDFs |
    | OTS | ✅ Included | Older PDFs, may be scanned |
    | FRB | ✅ Included | HTML press releases |
    | FDIC | ❌ Skipped | Salesforce portal requires authentication |

    ## Tips

    - Start with a **pilot run (300 docs)** to verify topics look sensible before running the full dataset
    - Use **6–8 topics** for a first pass; increase if topics seem too broad
    - Run the full dataset locally (not in the browser) for best speed
    - All downloaded documents are cached — reruns are much faster
    """)
