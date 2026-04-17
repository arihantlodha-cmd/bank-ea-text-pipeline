# Bank Enforcement Action — Text Analysis Pipeline

Downloads ~22,000 US bank enforcement action documents from the OCC, OTS, and FRB, extracts their text, and uses LDA topic modeling to automatically discover recurring problem categories (credit risk, BSA/AML, consumer protection, capital adequacy, etc.).

---

## Requirements

- **Mac or Linux** with Python 3.9+
- Internet connection (for first-time document downloads)
- The input CSV: `enforcement_actions_round5.csv`

---

## First-time setup (run once)

Open Terminal, drag the `model` folder into the Terminal window, press Enter, then run:

```
bash setup.sh
```

This installs all Python dependencies and optionally installs Tesseract OCR.

---

## Running the pipeline

```
python3 launch.py
```

This opens an interactive menu — no command-line flags needed. It will ask you:

| Question | What to pick |
|---|---|
| Which CSV file? | Leave as default (auto-detected) |
| Pilot or full run? | `pilot` to test (fast), `full` for all data |
| Sample size? | 300 is a good pilot |
| Number of topics? | 6–10 recommended |
| Workers? | 8 (default) |
| Skip cached downloads? | Yes (saves time on reruns) |

---

## Outputs (saved to `ea_pipeline_output/`)

| File | What it contains |
|---|---|
| `ea_with_topics.csv` | Full dataset + topic scores for each enforcement action |
| `topic_word_lists.csv` | Top 20 words per topic with weights |
| `lda_model.pkl` | Saved model for reuse in Python |
| `pipeline_summary.txt` | Run statistics and top words per topic |
| `failures.log` | Any download or extraction errors |

Open `ea_with_topics.csv` in Excel to see results. Key columns added:

- `topic_dominant` — which topic (0–N) best describes this document
- `topic_prob` — how strongly it matches that topic (0–1)
- `topic_0_prob … topic_N_prob` — probability scores for every topic

---

## Running the web app (locally)

```
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Same pipeline, point-and-click UI.

---

## Deploying online (Streamlit Cloud — free)

1. Push this folder to a GitHub repo (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → pick your repo → set main file to `app.py` → Deploy

Anyone with the URL can use it instantly. No install needed on their end.

> Note: the free Streamlit Cloud tier has a 1 GB memory limit. Pilot runs (≤500 docs) work fine. For full 9k-doc runs, run locally.

---

## Advanced: run directly with flags

```bash
python3 ea_text_pipeline.py --sample 300 --n_topics 8 --workers 8
python3 ea_text_pipeline.py --skip_download          # rerun using cache
python3 ea_text_pipeline.py                          # full run, all docs
```

---

## Notes

- **FDIC documents are skipped** — they sit behind a Salesforce login portal
- **OCR** requires `tesseract` (installed by `setup.sh`). Without it, scanned PDFs are marked `ocr_unavailable` but the pipeline still runs fine
- **Reruns are fast** — all downloaded documents are cached in `ea_pdf_cache/`
- A full run (all ~9,000 non-FDIC docs) takes roughly 30–90 minutes depending on internet speed
