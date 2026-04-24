"""
EA Text Extraction + Topic Modeling Pipeline
Author: Ari (for Pugachev research project)

Steps:
  1. Download EA documents (PDF or HTML) from URLs in EA dataset
     - Skips FDIC Salesforce URLs (auth-gated)
     - Optionally ingests locally downloaded FDIC PDFs via --fdic_dir
  2. Extract text: direct for machine-readable PDFs, OCR for scanned, BS4 for HTML
  3. Clean and normalize text (expanded stopword list)
  4. Run unsupervised LDA topic modeling on full corpus
  5. Assign topic distributions + multi-topic flags + ambiguous flag
  6. Build spot-check zip for manual TP/FP review

Usage:
  # Pilot
  python ea_text_pipeline.py --input enforcement_actions_round5.csv --sample 300 --n_topics 6

  # Full run with 10 topics
  python ea_text_pipeline.py --input enforcement_actions_round5.csv --n_topics 10 --workers 8

  # With local FDIC PDFs
  python ea_text_pipeline.py --input enforcement_actions_round5.csv --fdic_dir ./fdic_pdfs --n_topics 10

  # Build spotcheck zip from existing output (skip re-running pipeline)
  python ea_text_pipeline.py --spotcheck_only --input ea_pipeline_output/ea_with_topics.csv
"""

import argparse
import os
import re
import time
import logging
import pickle
import hashlib
import zipfile
import shutil
import random
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import pdfplumber
import pytesseract
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR     = Path('ea_pdf_cache')
OUTPUT_DIR    = Path('ea_pipeline_output')
SPOTCHECK_DIR = OUTPUT_DIR / 'spotcheck'

# ── Download config ───────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 30
RETRY_LIMIT      = 2
MIN_TEXT_CHARS   = 150   # below this → try OCR
MIN_CLEAN_TOKENS = 50    # below this → mark unusable
OCR_DPI          = 250   # higher DPI for scanned OTS docs

# ── Multi-topic config ────────────────────────────────────────────────────────
MULTI_TOPIC_THRESHOLD = 0.20  # flag topic as present if prob >= this
AMBIGUOUS_GAP         = 0.10  # flag ambiguous if top-2 prob gap < this

# ── Stopwords ─────────────────────────────────────────────────────────────────
ENGLISH_STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','must','that',
    'this','these','those','it','its','not','no','nor','so','yet','both',
    'either','neither','each','any','all','few','more','most','other','some',
    'such','into','through','during','before','after','above','below','between',
    'out','off','over','under','again','further','then','once','here','there',
    'where','why','how','which','who','whom','as','if','while','although',
    'because','since','unless','until','also','than','about','up','when',
    'what','their','they','them','our','your','his','her','we','you','he',
    'she','can','just','only','even','back','now',
}

# Regulator identity — strip so model focuses on content not issuer
REGULATOR_STOPWORDS = {
    'fdic','occ','ots','frb','federal','reserve','comptroller','thrift',
    'supervision','office','corporation','deposit','insurance','currency',
    'board','governors','frbsf','frbny',
}

# Legal boilerplate — strip order form noise
ORDER_STOPWORDS = {
    'cease','desist','order','consent','agreement','stipulation','formal',
    'informal','memorandum','understanding','directive','pursuant','section',
    'act','shall','herein','thereof','thereto','whereas','hereby','respondent',
    'issued','effective','terminated','termination','enforcement','action',
    'actions','penalty','penalties','civil','money','fine','assessment',
    'page','date','dated','signed','signature','fdia','article',
}

# Generic banking terms in every doc regardless of topic
GENERIC_BANKING_STOPWORDS = {
    'bank','banking','institution','national','state','association','financial',
    'services','management','director','officer','employee','person','entity',
    'dollar','amount','pay','payment','paid','following','including','review',
    'examiner','examination','regulator','regulatory','report','written','plan',
    'policy','policies','procedure','procedures','corrective','immediate',
    'appropriate','ensure','adequate','written','provide','within','days',
    'management','implement','maintain',
}

ALL_STOPWORDS = ENGLISH_STOPWORDS | REGULATOR_STOPWORDS | ORDER_STOPWORDS | GENERIC_BANKING_STOPWORDS


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def is_fdic_url(url):
    return isinstance(url, str) and ('orders.fdic.gov' in url or 'fdic.gov/sfc' in url)


def url_to_cache_path(url):
    key = hashlib.md5(url.encode()).hexdigest()[:16]
    ext = '.html' if any(x in url for x in ['.htm', 'pressrelease', 'newsevents']) else '.pdf'
    return CACHE_DIR / f"{key}{ext}"


def download_url(url, force=False):
    cache_path = url_to_cache_path(url)
    if not force and cache_path.exists():
        return url, cache_path.read_bytes(), None
    headers = {'User-Agent': 'Mozilla/5.0 (research; arihantlodha48@gmail.com)'}
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                cache_path.write_bytes(r.content)
                return url, r.content, None
            err = f"HTTP {r.status_code}"
        except Exception as e:
            err = str(e)
        time.sleep(1.5 * (attempt + 1))
    return url, None, err


def download_all(urls, workers=8, force=False):
    CACHE_DIR.mkdir(exist_ok=True)
    failures_log = OUTPUT_DIR / 'failures.log'
    results = {}
    todo = [u for u in urls if isinstance(u, str) and u.startswith('http') and not is_fdic_url(u)]
    skipped = sum(1 for u in urls if is_fdic_url(u))
    log.info(f"Downloading {len(todo)} docs ({workers} workers) — skipping {skipped} FDIC URLs")

    with open(failures_log, 'a') as flog:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(download_url, u, force): u for u in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                url, content, err = fut.result()
                results[url] = content
                if err:
                    log.warning(f"  [{i}/{len(todo)}] FAILED: {url[:80]} — {err}")
                    flog.write(f"DOWNLOAD_FAIL\t{url}\t{err}\n")
                elif i % 200 == 0:
                    log.info(f"  [{i}/{len(todo)}] downloaded")

    ok = sum(1 for v in results.values() if v)
    log.info(f"Download done: {ok}/{len(todo)} succeeded")
    return results


def load_fdic_local(fdic_dir, df):
    """
    Load manually downloaded FDIC PDFs from a local folder.
    Preferred filename scheme: {EA_ID}.pdf  (e.g. EA00042.pdf)
    Fallback: match by OrderID or BankID in filename.
    Returns {url: bytes}.
    """
    if not fdic_dir or not Path(fdic_dir).exists():
        return {}
    fdic_dir = Path(fdic_dir)
    fdic_rows = df[df['URL'].apply(is_fdic_url)].copy()
    pdf_files = list(fdic_dir.glob('*.pdf'))
    log.info(f"Loading local FDIC PDFs: {len(pdf_files)} files, {len(fdic_rows)} FDIC rows")

    # Build filename → path lookup (strip .pdf for EA_ID match)
    by_stem = {pf.stem: pf for pf in pdf_files}

    results = {}
    matched_ea_id = 0
    matched_other = 0
    for _, row in fdic_rows.iterrows():
        url = row['URL']
        ea_id = str(row.get('EA_ID', ''))

        # Primary: EA_ID match
        if ea_id and ea_id in by_stem:
            results[url] = by_stem[ea_id].read_bytes()
            matched_ea_id += 1
            continue

        # Fallback: OrderID / BankID in filename
        order_id = str(row.get('OrderID', '')).replace('/', '_')
        bank_id  = str(row.get('BankID', '')).split('.')[0]
        for pf in pdf_files:
            if (order_id and order_id in pf.name) or (bank_id and bank_id in pf.name):
                results[url] = pf.read_bytes()
                matched_other += 1
                break

    log.info(f"Matched {matched_ea_id} by EA_ID + {matched_other} by fallback = {matched_ea_id + matched_other}/{len(fdic_rows)}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_from_html(content):
    try:
        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script','style','nav','header','footer','aside']):
            tag.decompose()
        article = (soup.find('div', id='article') or
                   soup.find('div', class_='col-xs-12') or
                   soup.find('div', class_='article'))
        return (article or soup).get_text(separator=' ')
    except Exception as e:
        log.debug(f"HTML error: {e}")
        return ''


def extract_from_pdf_direct(content):
    import io
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return '\n'.join(p.extract_text() or '' for p in pdf.pages)
    except Exception as e:
        log.debug(f"pdfplumber error: {e}")
        return ''


def extract_from_pdf_ocr(content):
    import io
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            texts = []
            for page in pdf.pages:
                img = page.to_image(resolution=OCR_DPI).original
                texts.append(pytesseract.image_to_string(img, lang='eng'))
            return '\n'.join(texts)
    except Exception as e:
        log.debug(f"OCR error: {e}")
        return ''


def extract_text(url, content):
    if content is None:
        return '', 'failed'
    is_html = (any(x in url for x in ['.htm', 'pressrelease', 'newsevents']) or
               content[:50].lower().lstrip().startswith((b'<!doctype', b'<html')))
    if is_html:
        return extract_from_html(content), 'html'
    text = extract_from_pdf_direct(content)
    if len(text.strip()) >= MIN_TEXT_CHARS:
        return text, 'direct'
    log.debug(f"OCR fallback: {url[:60]}")
    text = extract_from_pdf_ocr(content)
    method = 'ocr' if len(text.strip()) >= MIN_TEXT_CHARS else 'failed'
    return text, method


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if len(t) >= 3 and t not in ALL_STOPWORDS]
    return ' '.join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: LDA TOPIC MODELING
# ══════════════════════════════════════════════════════════════════════════════

def fit_lda(texts, n_topics=10, n_top_words=20, max_features=8000, random_state=42):
    log.info(f"Fitting LDA: {n_topics} topics on {len(texts)} docs...")
    vectorizer = CountVectorizer(max_features=max_features, min_df=5, max_df=0.80, ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(texts)
    log.info(f"  DTM: {dtm.shape}")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state,
                                    max_iter=50, learning_method='online', n_jobs=-1)
    doc_topics = lda.fit_transform(dtm)
    log.info(f"  Perplexity: {lda.perplexity(dtm):.1f}")
    feature_names = vectorizer.get_feature_names_out()
    topic_labels = {}
    for i, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-n_top_words:][::-1]
        words = [feature_names[j] for j in top_idx]
        topic_labels[i] = words
        log.info(f"  Topic {i:2d}: {', '.join(words[:10])}")
    return lda, vectorizer, doc_topics, topic_labels


def assign_topics(doc_topics, n_topics):
    dominant      = doc_topics.argmax(axis=1)
    top_probs     = np.sort(doc_topics, axis=1)[:, ::-1]
    dominant_prob = top_probs[:, 0]
    second_prob   = top_probs[:, 1]
    ambiguous     = (dominant_prob - second_prob) < AMBIGUOUS_GAP
    flags         = (doc_topics >= MULTI_TOPIC_THRESHOLD).astype(int)
    return dominant, dominant_prob, ambiguous, flags


# ══════════════════════════════════════════════════════════════════════════════
# SPOT-CHECK ZIP
# ══════════════════════════════════════════════════════════════════════════════

def build_spotcheck_zip(df, n_per_topic=5, n_unusable=20, random_state=42):
    """
    Build spot-check zip for Leo's manual TP/FP review:
      - 20 random unusable docs
      - 5 random docs per dominant topic
      - 5 random docs per EAType-Regulator combo
    Each sample includes the cached PDF/HTML + a JSON metadata sidecar.
    """
    SPOTCHECK_DIR.mkdir(parents=True, exist_ok=True)
    zip_path  = OUTPUT_DIR / 'spotcheck_sample.zip'
    meta_rows = []

    def copy_doc(row, label, subfolder):
        url = row.get('URL', '')
        if not isinstance(url, str) or not url.startswith('http'):
            return False
        cache_path = url_to_cache_path(url)
        if not cache_path.exists():
            return False
        dest_dir = SPOTCHECK_DIR / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, dest_dir / cache_path.name)
        meta = {
            'label': label,
            'Regulator': str(row.get('Regulator','')),
            'Name': str(row.get('Name','')),
            'StartDate': str(row.get('StartDate','')),
            'OrderID': str(row.get('OrderID','')),
            'topic_dominant': str(row.get('topic_dominant','')),
            'topic_prob': str(row.get('topic_prob','')),
            'topic_ambiguous': str(row.get('topic_ambiguous','')),
            'extract_method': str(row.get('extract_method','')),
            'url': url,
            'cached_file': cache_path.name,
        }
        (dest_dir / (cache_path.stem + '_meta.json')).write_text(json.dumps(meta, indent=2))
        meta_rows.append(meta)
        return True

    added = 0

    # 1. Unusable
    unusable_mask = (df['extract_method'].isin(['failed','no_url'])) | (df['token_count'] < MIN_CLEAN_TOKENS)
    sample_u = df[unusable_mask].sample(min(n_unusable, unusable_mask.sum()), random_state=random_state)
    for _, row in sample_u.iterrows():
        added += copy_doc(row, 'unusable', 'unusable')

    # 2. Per dominant topic
    if 'topic_dominant' in df.columns:
        for t in sorted(df['topic_dominant'].dropna().unique()):
            subset = df[df['topic_dominant'] == t]
            sample = subset.sample(min(n_per_topic, len(subset)), random_state=random_state)
            for _, row in sample.iterrows():
                added += copy_doc(row, f'topic_{int(t)}', f'by_topic/topic_{int(t)}')

    # 3. Per EAType-Regulator combo
    ea_type_cols = [c for c in ['PCA','CD','FA','Personnel','CMP','Other'] if c in df.columns]
    usable = df[~df['extract_method'].isin(['failed','no_url'])]
    for col in ea_type_cols:
        for reg in sorted(df['Regulator'].dropna().unique()):
            subset = usable[(usable.get(col, 0) == 1) & (usable['Regulator'] == reg)]
            if len(subset) == 0:
                continue
            sample = subset.sample(min(n_per_topic, len(subset)), random_state=random_state)
            for _, row in sample.iterrows():
                added += copy_doc(row, f'{col}_{reg}', f'by_type/{col}_{reg}')

    # Manifest CSV
    pd.DataFrame(meta_rows).to_csv(SPOTCHECK_DIR / 'spotcheck_manifest.csv', index=False)

    # Zip
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in SPOTCHECK_DIR.rglob('*'):
            if f.is_file():
                zf.write(f, f.relative_to(SPOTCHECK_DIR))

    log.info(f"Spot-check zip: {added} docs → {zip_path}")
    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',          default='enforcement_actions_round5.csv')
    parser.add_argument('--output',         default='ea_with_topics.csv')
    parser.add_argument('--n_topics',       type=int, default=10)
    parser.add_argument('--sample',         type=int, default=None)
    parser.add_argument('--workers',        type=int, default=8)
    parser.add_argument('--fdic_dir',       type=str, default=None)
    parser.add_argument('--skip_download',  action='store_true')
    parser.add_argument('--force_download', action='store_true')
    parser.add_argument('--spotcheck_only', action='store_true')
    parser.add_argument('--n_spotcheck',    type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # ── Spotcheck-only mode ───────────────────────────────────────────────────
    if args.spotcheck_only:
        log.info(f"Spotcheck-only: loading {args.input}")
        df = pd.read_csv(args.input, low_memory=False)
        if 'token_count' not in df.columns:
            df['token_count'] = df.get('clean_text', pd.Series(dtype=str)).fillna('').str.split().str.len()
        zip_path = build_spotcheck_zip(df, n_per_topic=args.n_spotcheck)
        print(f"\nSpot-check zip → {zip_path}")
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    log.info(f"  {len(df)} rows, {df['URL'].notna().sum()} with URLs")

    if args.sample:
        df = df[df['URL'].notna()].sample(min(args.sample, df['URL'].notna().sum()), random_state=42).copy()
        log.info(f"  Sampled {len(df)} rows")

    urls = df['URL'].dropna().tolist()

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.skip_download:
        content_map = download_all(urls, workers=args.workers, force=args.force_download)
    else:
        log.info("Loading from cache...")
        content_map = {}
        for url in urls:
            if not isinstance(url, str): continue
            p = url_to_cache_path(url)
            content_map[url] = p.read_bytes() if p.exists() else None

    if args.fdic_dir:
        content_map.update(load_fdic_local(args.fdic_dir, df))

    # ── Extract ───────────────────────────────────────────────────────────────
    log.info("Extracting text...")
    url_to_text, url_to_method = {}, {}
    unique_urls = [u for u in df['URL'].dropna().unique() if isinstance(u, str) and u.startswith('http')]
    for i, url in enumerate(unique_urls, 1):
        text, method = extract_text(url, content_map.get(url))
        url_to_text[url]   = text
        url_to_method[url] = method
        if i % 500 == 0:
            log.info(f"  {i}/{len(unique_urls)}")

    df['raw_text']       = df['URL'].map(url_to_text)
    df['extract_method'] = df['URL'].map(url_to_method).fillna('no_url')
    log.info(f"  Methods: {df['extract_method'].value_counts().to_dict()}")

    # ── Clean ─────────────────────────────────────────────────────────────────
    log.info("Cleaning text...")
    df['clean_text']  = df['raw_text'].fillna('').apply(clean_text)
    df['token_count'] = df['clean_text'].str.split().str.len().fillna(0).astype(int)
    df['is_usable']   = df['token_count'] >= MIN_CLEAN_TOKENS
    log.info(f"  Usable: {df['is_usable'].sum()}/{len(df)} ({df['is_usable'].mean()*100:.1f}%)")

    # ── LDA ───────────────────────────────────────────────────────────────────
    corpus     = df.loc[df['is_usable'], 'clean_text'].tolist()
    corpus_idx = df.index[df['is_usable']]
    lda, vectorizer, doc_topics, topic_labels = fit_lda(corpus, n_topics=args.n_topics)
    dominant, dominant_prob, ambiguous, flags  = assign_topics(doc_topics, args.n_topics)

    df['topic_dominant']  = np.nan
    df['topic_prob']      = np.nan
    df['topic_ambiguous'] = np.nan
    for i in range(args.n_topics):
        df[f'topic_{i}_prob'] = np.nan
        df[f'topic_{i}_flag'] = np.nan

    df.loc[corpus_idx, 'topic_dominant']  = dominant
    df.loc[corpus_idx, 'topic_prob']      = dominant_prob
    df.loc[corpus_idx, 'topic_ambiguous'] = ambiguous.astype(int)
    for i in range(args.n_topics):
        df.loc[corpus_idx, f'topic_{i}_prob'] = doc_topics[:, i]
        df.loc[corpus_idx, f'topic_{i}_flag'] = flags[:, i]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / args.output
    df.drop(columns=['raw_text'], errors='ignore').to_csv(out_csv, index=False)
    log.info(f"Saved → {out_csv}")

    with open(OUTPUT_DIR / 'lda_model.pkl', 'wb') as f:
        pickle.dump({'lda': lda, 'vectorizer': vectorizer, 'topic_labels': topic_labels,
                     'n_topics': args.n_topics, 'multi_topic_threshold': MULTI_TOPIC_THRESHOLD,
                     'ambiguous_gap': AMBIGUOUS_GAP}, f)

    topic_rows = []
    for t_idx, words in topic_labels.items():
        n_dom  = int((dominant == t_idx).sum())
        n_flag = int(flags[:, t_idx].sum())
        for rank, word in enumerate(words):
            topic_rows.append({'topic': t_idx, 'rank': rank+1, 'word': word,
                               'n_dominant': n_dom, 'n_flagged_20pct': n_flag})
    pd.DataFrame(topic_rows).to_csv(OUTPUT_DIR / 'topic_word_lists.csv', index=False)

    lines = [
        "="*60, "EA TEXT PIPELINE — RUN SUMMARY", "="*60,
        f"Input:           {args.input}",
        f"Total docs:      {len(df)}",
        f"Usable:          {df['is_usable'].sum()} ({df['is_usable'].mean()*100:.1f}%)",
        f"N topics:        {args.n_topics}",
        f"Multi-topic thr: {MULTI_TOPIC_THRESHOLD}  |  Ambiguous gap: {AMBIGUOUS_GAP}",
        "", "Extraction methods:",
    ] + [f"  {k}: {v}" for k,v in df['extract_method'].value_counts().items()] + [
        "", "Topics (dominant | flagged ≥20%):",
    ] + [
        f"  T{t:2d} | dom={int((dominant==t).sum()):4d} | flag={int(flags[:,t].sum()):4d} | "
        f"{', '.join(topic_labels[t][:8])}"
        for t in range(args.n_topics)
    ] + [
        f"", f"Ambiguous docs: {int(ambiguous.sum())}",
        f"", f"Outputs in {OUTPUT_DIR}/",
    ]
    summary = '\n'.join(lines)
    (OUTPUT_DIR / 'pipeline_summary.txt').write_text(summary)
    print('\n' + summary)

    # ── Spot-check zip ────────────────────────────────────────────────────────
    log.info("Building spot-check zip...")
    zip_path = build_spotcheck_zip(df, n_per_topic=args.n_spotcheck)
    print(f"\nSpot-check zip → {zip_path}")


if __name__ == '__main__':
    main()
