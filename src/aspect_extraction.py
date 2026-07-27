from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path("data/processed/reviews_clean.csv")
OUTPUT = Path("data/processed/aspect_summary.csv")
ASPECTS = {
    "battery": ["battery", "charge", "charging", "power"],
    "delivery": ["delivery", "shipping", "arrived", "package", "packaging"],
    "quality": ["broken", "defective", "stopped", "quality", "cheap", "durable"],
    "compatibility": ["compatible", "connection", "bluetooth", "software", "app"],
    "price": ["price", "expensive", "cost", "refund", "money"],
}


def detect_aspects(text: str) -> list[str]:
    content = str(text).lower()
    return [aspect for aspect, keywords in ASPECTS.items() if any(word in content for word in keywords)]


def summarize() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` first.")
    negative = pd.read_csv(DATA).query("sentiment == 'Negative'").copy()
    negative["aspects"] = negative.review_text.map(detect_aspects)
    tagged = negative.explode("aspects").dropna(subset=["aspects"])
    result = (tagged.groupby("aspects", as_index=False)
              .agg(complaint_volume=("review_text", "size"), average_rating=("star_rating", "mean"))
              .rename(columns={"aspects": "aspect"})
              .sort_values("complaint_volume", ascending=False))
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    summarize()
