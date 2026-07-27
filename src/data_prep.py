"""Create a reproducible, capped Electronics-review dataset from Amazon Reviews 2023."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect

RAW_OUTPUT = Path("data/raw/amazon_reviews_electronics.jsonl")
PROCESSED_OUTPUT = Path("data/processed/reviews_clean.csv")
DetectorFactory.seed = 42


def sentiment_from_rating(rating: float) -> str:
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def clean_text(text: str) -> str:
    """Keep negations; only remove markup and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def text_signature(text: str) -> str:
    """Exact normalized deduplication prevents duplicate text crossing the split."""
    normalized = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


def prepare(sample_size: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Shuffle a stream with a fixed seed, then build a capped, clean sample."""
    stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics",
        split="full", streaming=True, trust_remote_code=True,
    )
    rows, seen = [], set()
    for record in stream.shuffle(seed=seed, buffer_size=20_000):
        text = clean_text(record.get("text", ""))
        rating = record.get("rating")
        if (len(text.split()) < 3 or not is_english(text)
                or rating not in {1.0, 2.0, 3.0, 4.0, 5.0, 1, 2, 3, 4, 5}):
            continue
        signature = text_signature(text)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append({
            "review_text": text,
            "star_rating": int(rating),
            "product_category": "Electronics",
            "verified_purchase": bool(record.get("verified_purchase", False)),
            "helpful_votes": int(record.get("helpful_vote", 0) or 0),
            "sentiment": sentiment_from_rating(float(rating)),
        })
        if len(rows) >= sample_size:
            break
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No valid reviews were received from the dataset stream.")
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(RAW_OUTPUT, orient="records", lines=True)
    result.to_csv(PROCESSED_OUTPUT, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = prepare(args.sample_size, args.seed)
    print(f"Saved {len(data):,} Electronics reviews.\n{data.sentiment.value_counts(normalize=True).mul(100).round(2)}")
