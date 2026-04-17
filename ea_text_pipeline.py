"""
ea_text_pipeline.py
Bank Enforcement Action text pipeline:
  1. Download OCC/OTS/FRB documents (skip FDIC)
  2. Extract text (HTML parse, PDF direct, OCR fallback)
  3. Clean text (stopwords, tokens)
  4. LDA topic modeling
  5. Output annotated CSV + topic word lists + saved model
"""

import argparse
import csv
import hashlib
import logging
import os
import pickle
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Dependency bootstrap
# ---------------------------------------------------------------------------

def install_if_missing(packages):
    import importlib
    mapping = {
        "pdfplumber": "pdfplumber",
        "pytesseract": "pytesseract",
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "sklearn": "scikit-learn",
        "PIL": "pillow",
        "tqdm": "tqdm",
    }
    for import_name, pip_name in mapping.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[bootstrap] Installing {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])

install_if_missing([])

# Now safe to import
import pdfplumber
import requests
from bs4 import BeautifulSoup
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

try:
    import pytesseract
    from PIL import Image
    import io
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Check tesseract binary
TESSERACT_AVAILABLE = False
try:
    result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        TESSERACT_AVAILABLE = True
        print(f"[info] Tesseract found: {result.stdout.splitlines()[0]}")
    else:
        print("[warning] tesseract binary returned non-zero — OCR will be skipped")
except Exception:
    print("[warning] tesseract binary not found — OCR fallback will be skipped (docs will be marked 'ocr_unavailable')")

OCR_AVAILABLE = PYTESSERACT_AVAILABLE and TESSERACT_AVAILABLE

# ---------------------------------------------------------------------------
# Banking-domain stopwords
# ---------------------------------------------------------------------------

BANKING_STOPWORDS = {
    "bank", "banking", "shall", "order", "pursuant", "board", "federal",
    "reserve", "fdic", "occ", "ots", "comptroller", "institution", "national",
    "state", "association", "financial", "services", "agreement", "following",
    "including", "thereof", "herein", "management", "section", "page", "date",
    "respondent", "consent", "issued", "effective", "terminated", "action",
    "enforcement", "director", "officer", "employee", "person", "entity",
    "also", "may", "must", "upon", "within", "each", "any", "all", "its",
    "has", "have", "had", "been", "was", "were", "are", "not", "the",
    "and", "for", "that", "this", "with", "from", "but", "one", "new",
    "said", "use", "two", "can", "see", "get", "set", "per", "due",
}

# English stopwords (minimal inline set so no NLTK needed)
ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

ALL_STOPWORDS = ENGLISH_STOPWORDS | BANKING_STOPWORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def url_to_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def is_fdic(url: str) -> bool:
    return "orders.fdic.gov" in url


def is_html(url: str) -> bool:
    return url.lower().endswith(".htm") or url.lower().endswith(".html")


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "failures.log"
    logger = logging.getLogger("ea_pipeline")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    # Also print warnings+
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def download_one(url: str, cache_dir: Path, logger: logging.Logger, timeout: int = 30) -> Path | None:
    key = url_to_cache_key(url)
    # Determine extension
    ext = ".html" if is_html(url) else ".pdf"
    cache_path = cache_dir / (key + ext)

    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path  # already cached

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return cache_path
        except Exception as exc:
            if attempt == 2:
                logger.error(f"DOWNLOAD_FAIL url={url} err={exc}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def download_all(rows: list[dict], cache_dir: Path, logger: logging.Logger, workers: int, skip_download: bool) -> dict[str, Path | None]:
    """Returns mapping url -> local path (or None on failure)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_to_path: dict[str, Path | None] = {}

    # Deduplicate URLs to avoid re-downloading
    unique_urls = list({r["URL"] for r in rows if r["URL"] and not is_fdic(r["URL"])})
    print(f"\n[download] {len(unique_urls)} unique non-FDIC URLs to fetch ({workers} workers)")

    if skip_download:
        print("[download] --skip_download set — using cache only")
        for url in unique_urls:
            key = url_to_cache_key(url)
            for ext in (".pdf", ".html"):
                p = cache_dir / (key + ext)
                if p.exists():
                    url_to_path[url] = p
                    break
            else:
                url_to_path[url] = None
        return url_to_path

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_url = {ex.submit(download_one, url, cache_dir, logger): url for url in unique_urls}
        for fut in tqdm(as_completed(future_to_url), total=len(unique_urls), desc="downloading", unit="doc"):
            url = future_to_url[fut]
            try:
                url_to_path[url] = fut.result()
            except Exception as exc:
                logger.error(f"DOWNLOAD_EXCEPTION url={url} err={exc}")
                url_to_path[url] = None

    ok = sum(1 for v in url_to_path.values() if v is not None)
    print(f"[download] {ok}/{len(unique_urls)} downloads succeeded")
    return url_to_path


# ---------------------------------------------------------------------------
# 2. Text extraction
# ---------------------------------------------------------------------------

MIN_DIRECT_CHARS = 150


def extract_html(path: Path, logger: logging.Logger) -> tuple[str, str]:
    try:
        with open(path, "rb") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # Remove nav/header/footer noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        # Prefer article body, fall back to full text
        article = soup.find("div", class_=re.compile(r"article|content|body|main", re.I))
        text = (article or soup).get_text(separator=" ", strip=True)
        return text, "html"
    except Exception as exc:
        logger.error(f"HTML_EXTRACT_FAIL path={path} err={exc}")
        return "", "failed"


def extract_pdf_direct(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_pdf_ocr(path: Path, dpi: int = 200) -> str:
    if not OCR_AVAILABLE:
        return ""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=dpi).original
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            pil_img = Image.open(buf)
            text_parts.append(pytesseract.image_to_string(pil_img))
    return "\n".join(text_parts)


def extract_pdf(path: Path, logger: logging.Logger) -> tuple[str, str]:
    try:
        text = extract_pdf_direct(path)
        if len(text.strip()) >= MIN_DIRECT_CHARS:
            return text, "direct"
        # Try OCR fallback
        if OCR_AVAILABLE:
            ocr_text = extract_pdf_ocr(path)
            if len(ocr_text.strip()) >= MIN_DIRECT_CHARS:
                return ocr_text, "ocr"
            return ocr_text or text, "ocr"
        else:
            if len(text.strip()) > 0:
                # Short but something — keep direct
                return text, "direct"
            return "", "ocr_unavailable"
    except Exception as exc:
        logger.error(f"PDF_EXTRACT_FAIL path={path} err={exc}")
        return "", "failed"


def extract_text(url: str, path: Path | None, logger: logging.Logger) -> tuple[str, str]:
    if path is None:
        return "", "failed"
    if is_html(url):
        return extract_html(path, logger)
    else:
        return extract_pdf(path, logger)


# ---------------------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove numbers & punctuation
    tokens = text.split()
    tokens = [t for t in tokens if len(t) >= 3 and t not in ALL_STOPWORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 4. LDA
# ---------------------------------------------------------------------------

def run_lda(docs: list[str], n_topics: int) -> tuple:
    """Returns (lda_model, vectorizer, doc_topic_matrix)."""
    print(f"\n[lda] Fitting CountVectorizer on {len(docs)} documents ...")
    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(docs)
    print(f"[lda] Vocabulary size: {X.shape[1]} | Matrix: {X.shape}")
    print(f"[lda] Fitting LDA with {n_topics} topics (max_iter=50) ...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    doc_topic = lda.fit_transform(X)
    print("[lda] Done.")
    return lda, vectorizer, doc_topic


def get_topic_words(lda, vectorizer, n_words: int = 20) -> list[list[tuple[str, float]]]:
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        top_idx = comp.argsort()[::-1][:n_words]
        topics.append([(feature_names[i], float(comp[i])) for i in top_idx])
    return topics


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bank EA text pipeline")
    parser.add_argument("--input", default="enforcement_actions_round5.csv")
    parser.add_argument("--output", default="ea_with_topics.csv")
    parser.add_argument("--n_topics", type=int, default=6)
    parser.add_argument("--sample", type=int, default=None, help="Process N random rows (for piloting)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else script_dir / args.input
    output_dir = script_dir / "ea_pipeline_output"
    cache_dir = script_dir / "ea_pdf_cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    print(f"\n{'='*60}")
    print("  Bank Enforcement Action Text Pipeline")
    print(f"{'='*60}")
    print(f"Input   : {input_path}")
    print(f"Topics  : {args.n_topics}")
    print(f"Workers : {args.workers}")
    print(f"Sample  : {args.sample or 'all'}")
    print(f"OCR     : {'available' if OCR_AVAILABLE else 'unavailable (tesseract not found)'}")
    print(f"{'='*60}\n")

    # ----- Load CSV -----
    print("[step 1] Loading input CSV ...")
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    print(f"  Loaded {len(all_rows)} rows")

    # Filter to rows with non-FDIC URLs
    eligible = [r for r in all_rows if r.get("URL") and not is_fdic(r.get("URL", ""))]
    print(f"  Eligible (non-FDIC with URL): {len(eligible)}")

    if args.sample:
        import random
        random.seed(42)
        eligible = random.sample(eligible, min(args.sample, len(eligible)))
        print(f"  Sampled: {len(eligible)} rows")

    # ----- Download -----
    print("\n[step 2] Downloading documents ...")
    url_to_path = download_all(eligible, cache_dir, logger, args.workers, args.skip_download)

    # ----- Extract text -----
    print("\n[step 3] Extracting text ...")
    for row in tqdm(eligible, desc="extracting", unit="doc"):
        url = row["URL"]
        path = url_to_path.get(url)
        raw_text, method = extract_text(url, path, logger)
        row["raw_text"] = raw_text
        row["extract_method"] = method

    method_counts = {}
    for row in eligible:
        m = row.get("extract_method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1
    print(f"  Extraction methods: {method_counts}")

    # ----- Clean text -----
    print("\n[step 4] Cleaning text ...")
    for row in tqdm(eligible, desc="cleaning", unit="doc"):
        row["clean_text"] = clean_text(row.get("raw_text", ""))

    # Filter out documents with too few tokens
    before = len(eligible)
    eligible_lda = [r for r in eligible if len(r["clean_text"].split()) >= 50]
    print(f"  Documents with ≥50 tokens: {len(eligible_lda)} (dropped {before - len(eligible_lda)})")

    # ----- LDA -----
    print("\n[step 5] Running LDA topic modeling ...")
    docs = [r["clean_text"] for r in eligible_lda]
    if len(docs) < args.n_topics:
        print(f"[warning] Only {len(docs)} usable docs — fewer than n_topics={args.n_topics}. Reducing topics.")
        args.n_topics = max(2, len(docs) // 2)

    lda_model, vectorizer, doc_topic = run_lda(docs, args.n_topics)

    # Assign topic columns back to rows
    for i, row in enumerate(eligible_lda):
        probs = doc_topic[i]
        dominant = int(probs.argmax())
        row["topic_dominant"] = dominant
        row["topic_prob"] = float(probs[dominant])
        for t in range(args.n_topics):
            row[f"topic_{t}_prob"] = float(probs[t])

    # For rows that were dropped (too few tokens), fill with blanks
    topic_cols = ["topic_dominant", "topic_prob"] + [f"topic_{t}_prob" for t in range(args.n_topics)]
    lda_set = set(id(r) for r in eligible_lda)
    for row in eligible:
        if id(row) not in lda_set:
            row["topic_dominant"] = ""
            row["topic_prob"] = ""
            for t in range(args.n_topics):
                row[f"topic_{t}_prob"] = ""

    # ----- Topic word lists -----
    topic_words = get_topic_words(lda_model, vectorizer, n_words=20)

    print("\n" + "="*60)
    print("  TOP WORDS PER TOPIC")
    print("="*60)
    for t_idx, words in enumerate(topic_words):
        top10 = ", ".join(w for w, _ in words[:10])
        print(f"  Topic {t_idx}: {top10}")
    print("="*60)

    # ----- Save outputs -----
    print("\n[step 6] Saving outputs ...")

    # ea_with_topics.csv — merge eligible back into all_rows
    # Build lookup by row identity (eligible rows are a subset of all_rows)
    eligible_ids = {id(r): r for r in eligible}

    # Fieldnames
    base_fields = list(all_rows[0].keys()) if all_rows else []
    extra_fields = ["raw_text", "clean_text", "extract_method"] + topic_cols
    all_fields = base_fields + [f for f in extra_fields if f not in base_fields]

    # Add empty extra cols to all_rows not in eligible
    for row in all_rows:
        for f in extra_fields:
            if f not in row:
                row[f] = ""

    out_csv = output_dir / args.output
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved: {out_csv} ({len(all_rows)} rows)")

    # topic_word_lists.csv
    word_csv = output_dir / "topic_word_lists.csv"
    with open(word_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "rank", "word", "weight"])
        for t_idx, words in enumerate(topic_words):
            for rank, (word, weight) in enumerate(words):
                writer.writerow([t_idx, rank + 1, word, f"{weight:.4f}"])
    print(f"  Saved: {word_csv}")

    # lda_model.pkl
    model_pkl = output_dir / "lda_model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump({"lda": lda_model, "vectorizer": vectorizer}, f)
    print(f"  Saved: {model_pkl}")

    # pipeline_summary.txt
    topic_dominant_counts = {}
    for row in eligible_lda:
        t = row["topic_dominant"]
        topic_dominant_counts[t] = topic_dominant_counts.get(t, 0) + 1

    summary_lines = [
        "Bank EA Pipeline Summary",
        "=" * 50,
        f"Total CSV rows        : {len(all_rows)}",
        f"Eligible (non-FDIC)   : {len(eligible)}",
        f"LDA-eligible (≥50 tok): {len(eligible_lda)}",
        f"n_topics              : {args.n_topics}",
        "",
        "Extraction method breakdown:",
        *[f"  {k}: {v}" for k, v in sorted(method_counts.items())],
        "",
        "Topic sizes (dominant topic assignment):",
        *[f"  Topic {k}: {v} docs" for k, v in sorted(topic_dominant_counts.items())],
        "",
        "Top 10 words per topic:",
    ]
    for t_idx, words in enumerate(topic_words):
        top10 = ", ".join(w for w, _ in words[:10])
        summary_lines.append(f"  Topic {t_idx}: {top10}")

    summary_path = output_dir / "pipeline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Saved: {summary_path}")

    print(f"\n[done] All outputs in: {output_dir}")
    print(f"[done] Failures log  : {output_dir / 'failures.log'}")


if __name__ == "__main__":
    main()
