"""
One-time script to compute transformer-based sentiment for all reviews.
Saves results to analysis/sentiment_cache.parquet.

Run once from the project root:
    py compute_sentiment.py

Uses: distilbert-base-uncased-finetuned-sst-2-english (~67 MB)
Neutral rule: confidence < 0.72 → Neutral (model is uncertain)
Estimated time: 10–20 min on CPU  |  <2 min with CUDA GPU
"""

import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

MODEL        = "distilbert-base-uncased-finetuned-sst-2-english"
CACHE_PATH   = "analysis/sentiment_cache.parquet"
BATCH_SIZE   = 128
MAX_LEN      = 128
NEUTRAL_CONF = 0.72   # below this → Neutral


def label_to_sentiment(label: str, score: float) -> str:
    if score < NEUTRAL_CONF:
        return "Neutral"
    return "Positive" if label == "POSITIVE" else "Negative"


def label_to_compound(label: str, score: float) -> float:
    """Map distilbert output to a –1…+1 compound-like score."""
    if label == "POSITIVE":
        return score          # 0.5 → 1.0
    return -(score)           # -0.5 → -1.0


def main():
    os.makedirs("analysis", exist_ok=True)

    print("Loading reviews…")
    df = pd.read_csv("amazon_reviews.csv")
    df["content"] = df["content"].fillna("").astype(str)
    texts = df["content"].tolist()

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"Loading model '{MODEL}' on {device_name}…")

    nlp = pipeline(
        "sentiment-analysis",
        model=MODEL,
        device=device,
        truncation=True,
        max_length=MAX_LEN,
    )

    print(f"Processing {len(texts):,} reviews  "
          f"(batch={BATCH_SIZE}, max_len={MAX_LEN})…")

    results = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), unit="batch"):
        batch = texts[i : i + BATCH_SIZE]
        results.extend(nlp(batch, truncation=True, max_length=MAX_LEN))

    df["compound"]  = [label_to_compound(r["label"], r["score"]) for r in results]
    df["sentiment"] = [label_to_sentiment(r["label"], r["score"]) for r in results]

    out = df[["reviewId", "compound", "sentiment"]]
    out.to_parquet(CACHE_PATH, index=False)

    print(f"\nSaved → {CACHE_PATH}")
    print(df["sentiment"].value_counts().to_string())
    print("\nDone! Restart the Streamlit app to pick up the new scores.")


if __name__ == "__main__":
    main()
