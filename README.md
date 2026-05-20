# Amazon Reviews Intelligence Dashboard

**An end-to-end NLP analytics platform for 87,112 Google Play reviews of the Amazon Shopping app.**

Built with DistilBERT for transformer-based sentiment scoring, interactive Plotly visualisations, sarcasm detection, and TF-IDF keyword extraction — packaged as a fully deployable Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?logo=huggingface&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)

---

## Overview

Star ratings alone are a noisy signal. A 1-star review can be sarcastic praise; a 5-star review can contain genuine complaints about a secondary feature. This dashboard goes beyond ratings by applying contextual NLP to surface what users are actually saying — at scale.

**Business question this answers:** *What are customers saying about the Amazon Shopping app, how has sentiment evolved over eight years, and where are the biggest friction points by app version and complaint category?*

What makes this more than a tutorial:
- Uses **DistilBERT** (not just VADER) with a calibrated confidence-based neutral threshold
- **Sarcasm detection** pipeline using three independent signals: emoji patterns, sentiment/rating anomaly, and phrase matching
- VADER retained as a zero-config fallback — the app is always runnable, with or without the pre-computed cache
- Version-level performance analysis — useful for product teams correlating releases with rating shifts
- TF-IDF keyword extraction scoped per sentiment group — reveals not just frequent words but distinctive ones

---

## Features

| Tab | What it shows |
|---|---|
| **Overview** | Rating distribution, sentiment split donut, year-over-year summary table |
| **Trends** | Monthly volume, 3-month rolling rating & sentiment, % negative reviews over time |
| **Sentiment** | DistilBERT label distribution, avg sentiment by star rating, rating/model agreement matrix, mismatch examples |
| **Keywords** | TF-IDF distinctive keywords for positive vs. negative reviews, word clouds |
| **Complaints** | 13 auto-categorised complaint topics, sarcasm flag, positive theme breakdown, version performance |
| **Reviews** | Paginated, multi-sort review browser with sentiment tags |

**Sidebar filters:** star rating range · date range (presets + custom) · app version · sentiment label · topic / keyword

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Data source: Google Play Store                                      │
│  amazon_reviews.csv  ·  87,112 reviews  ·  2018 – 2026             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  compute_sentiment.py  (run once locally)                            │
│                                                                      │
│  DistilBERT: distilbert-base-uncased-finetuned-sst-2-english        │
│  • Batch size 32  ·  128-token truncation  ·  CPU or CUDA           │
│  • Confidence < 72 %  →  Neutral  (calibrated threshold)           │
│                                │                                     │
│                                ▼                                     │
│               analysis/sentiment_cache.parquet                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  utils/preprocessing.py                                              │
│  • Merge Parquet cache with raw CSV                                  │
│  • VADER fallback for any uncached rows                              │
│  • Sarcasm detection & sentiment correction                          │
│  • TF-IDF keyword extraction  (scikit-learn)                        │
│  • Rule-based complaint & positive theme classification              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  app.py  —  Streamlit dashboard                                      │
│  6 tabs  ·  Plotly charts  ·  Sidebar filters                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Field | Description |
|---|---|
| `reviewId` | Unique review identifier |
| `content` | Raw review text |
| `score` | Star rating (1–5) |
| `at` | Review timestamp |
| `thumbsUpCount` | Community upvotes |
| `appVersion` | App version at time of review |
| `userName` | Reviewer display name |

**Source:** Scraped from the Amazon Shopping app (Google Play Store) using [`google-play-scraper`](https://pypi.org/project/google-play-scraper/).  
**Scope:** 87,112 reviews · January 2018 – March 2026

> The dataset is not bundled in this repository due to size. See [Getting Started](#getting-started) for how to obtain it.

---

## Sentiment Analysis Methodology

### Primary model: DistilBERT

`distilbert-base-uncased-finetuned-sst-2-english` — a 66 M-parameter transformer distilled from BERT and fine-tuned on Stanford Sentiment Treebank (SST-2).

**Inference pipeline:**
1. Reviews are tokenised and truncated to 128 tokens
2. The model outputs a binary label (`POSITIVE` / `NEGATIVE`) with a confidence score in (0.5, 1.0]
3. A **calibrated neutral threshold** is applied: confidence < 72 % → **Neutral** (the model is genuinely uncertain — typically mixed-sentiment or ambiguous text)
4. Results are saved as a Parquet cache; the Streamlit app loads from cache at startup

**Why 72 %?** Binary classifiers have no native neutral class. Below this threshold, many reviews contain contradictory sentiment ("delivery was fast but the item was damaged") — forcing them into POSITIVE or NEGATIVE inflates both counts artificially. The 72 % cut-point balances coverage and precision for a three-class output.

### Fallback: VADER

When the pre-computed Parquet cache is absent (e.g. fresh clone without running `compute_sentiment.py`), the app falls back to VADER in real time. The dashboard header shows which model is active: `🤖 DistilBERT` or `⚠️ VADER`.

### VADER vs. DistilBERT

| Dimension | VADER | DistilBERT |
|---|---|---|
| Type | Lexicon / rule-based | Transformer (fine-tuned) |
| Inference speed | Real-time | ~10–20 min for 87 K reviews on CPU |
| Negation handling | Heuristic (partial) | Contextual (strong) |
| Neutral detection | Fixed compound threshold | Calibrated confidence threshold |
| Sarcasm | Fails | Partially handles via context |
| Setup | Zero config | Requires one-time pre-computation |
| Best for | Quick prototypes, social media | Production NLP, nuanced reviews |

---

## Sarcasm & Mismatch Detection

Users writing sarcastically-positive text in 1–2-star reviews is a well-known NLP failure mode. The pipeline flags these using three independent signals:

1. **Emoji signal** — laughing / eye-roll emojis (😂 🤣 🙄) in a 1–2-star review where the model scores positive sentiment
2. **Confidence anomaly** — suspiciously high positive confidence (compound > 0.45) on a 1–2-star review
3. **Phrase matching** — explicit sarcasm markers: *"what a joke", "well done Amazon", "thanks for nothing", "clown world", "oh wow"*, etc.

When a review triggers any signal, its sentiment label is overridden to **Negative** and the compound score is negated — preventing sarcastic reviews from polluting positive sentiment metrics.

The Complaints tab surfaces the top flagged reviews ranked by upvotes, making them immediately actionable.

---

## Key Findings

- **Below-benchmark satisfaction:** The Amazon Shopping app averages ~2.7 ★ across 87 K reviews — well below the 4.0 ★ benchmark typical of top shopping apps on Google Play
- **Polarised opinions:** The 3-star tier is significantly underrepresented; users react strongly in one direction with little ambivalence
- **Top recurring complaints (1–2 ★):** Delivery delays and missing orders, refund and return friction, app performance (crashes, freezing, slow loading), and customer service responsiveness dominate complaint categories
- **Version-level sensitivity:** Rating quality varies substantially across app versions — the version performance view lets product teams correlate specific releases with satisfaction shifts
- **Sentiment/rating disagreement:** A measurable share of 5 ★ reviews contain text the model reads as negative (likely habit-rating), and vice versa — the agreement matrix quantifies this gap
- **Positive drivers:** Fast delivery, product selection breadth, and Prime value are the most distinctive keywords in 4–5 ★ reviews

---

## Screenshots

> Take screenshots of each tab and drop them into the `screenshots/` folder, then the images below will render automatically.

| Overview | Trends |
|---|---|
| ![Overview](screenshots/overview.png) | ![Trends](screenshots/trends.png) |

| Sentiment | Keywords |
|---|---|
| ![Sentiment](screenshots/sentiment.png) | ![Keywords](screenshots/keywords.png) |

| Complaints | Reviews |
|---|---|
| ![Complaints](screenshots/complaints.png) | ![Reviews](screenshots/reviews.png) |

---

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- `amazon_reviews.csv` in the project root (see below)

### 1 — Obtain the dataset

Scrape reviews from Google Play using [`google-play-scraper`](https://pypi.org/project/google-play-scraper/):

```python
from google_play_scraper import reviews, Sort
import pandas as pd

result, _ = reviews(
    'com.amazon.mShop.android.shopping',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=100_000,
)
pd.DataFrame(result).to_csv('amazon_reviews.csv', index=False)
```

The CSV must contain at minimum: `reviewId`, `content`, `score`, `at`, `thumbsUpCount`, `appVersion`, `userName`.

### 2 — Install dependencies

```bash
git clone https://github.com/<your-username>/amazon-reviews-nlp-dashboard.git
cd amazon-reviews-nlp-dashboard

# Core app (for running the dashboard)
pip install -r requirements.txt

# Full local dev (adds DistilBERT + notebook tools)
pip install -r requirements-dev.txt
```

### 3 — Pre-compute DistilBERT sentiment (recommended)

Run once to score all reviews and write the Parquet cache (~67 MB model download, 10–20 min on CPU, under 2 min on GPU):

```bash
python compute_sentiment.py
```

This writes `analysis/sentiment_cache.parquet`. Skip this step to use VADER fallback instead.

### 4 — Launch the dashboard

```bash
streamlit run app.py
```

---

## Streamlit Community Cloud Deployment

1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your fork
3. Set **Main file path** to `app.py`
4. No secrets or environment variables required

The pre-computed `analysis/sentiment_cache.parquet` is committed to the repository, so the deployed app uses DistilBERT scores without needing to run inference in the cloud.

> **Scaling note:** For datasets larger than ~50 MB of Parquet, consider hosting the cache on Hugging Face Hub or S3 and loading it via URL in `preprocessing.py`.

---

## Project Structure

```
amazon-reviews-nlp-dashboard/
├── app.py                       # Streamlit dashboard — 6 tabs, sidebar filters
├── compute_sentiment.py         # One-time DistilBERT scoring script (run locally)
├── requirements.txt             # Deployment dependencies (app only)
├── requirements-dev.txt         # Local dev dependencies (DistilBERT + notebook)
├── .streamlit/
│   └── config.toml              # Theme and server settings
├── utils/
│   ├── __init__.py
│   └── preprocessing.py         # Data loading, sentiment merge, TF-IDF, filters
├── analysis/
│   ├── amazon_analysis.ipynb    # EDA notebook with statistical analysis and charts
│   └── sentiment_cache.parquet  # Pre-computed DistilBERT scores (87 K reviews)
└── screenshots/                 # Dashboard screenshots for README
```

---

## Future Improvements

- [ ] **Unsupervised topic modelling** — Replace rule-based complaint categories with BERTopic or LDA for data-driven topic discovery
- [ ] **Aspect-based sentiment** — Score sentiment per aspect (delivery, price, quality) rather than per review
- [ ] **Real-time scraping** — Scheduled ingestion of new reviews with incremental cache updates
- [ ] **Multi-app comparison** — Extend the pipeline to benchmark Amazon against Flipkart, eBay, or Shein on identical metrics
- [ ] **Alert system** — Trigger Slack / email notifications when weekly negative rate spikes beyond a threshold
- [ ] **Embedding visualisation** — UMAP projection of review embeddings coloured by sentiment or complaint cluster

---

## Tech Stack

`Python 3.11` · `Streamlit` · `DistilBERT (HuggingFace Transformers)` · `Plotly` · `scikit-learn` · `VADER` · `pandas` · `pyarrow` · `wordcloud` · `matplotlib`
