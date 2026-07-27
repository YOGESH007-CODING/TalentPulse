from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/reviews_clean.csv")
MODEL = Path("models/tfidf_logreg_baseline.joblib")
METRICS = Path("data/processed/baseline_metrics.csv")
AUDIT = Path("data/processed/baseline_prediction_audit.csv")
CALIBRATION = Path("data/processed/negative_calibration.csv")
LABELS = ["Negative", "Neutral", "Positive"]
NEGATIVE_REVIEW_THRESHOLD = 0.30


def make_split(data: pd.DataFrame):
    return train_test_split(data, test_size=.2, random_state=42, stratify=data.sentiment)


def bootstrap_macro_f1(y_true: pd.Series, prediction: np.ndarray, rounds: int = 1_000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    y_true, prediction = np.asarray(y_true), np.asarray(prediction)
    scores = [f1_score(y_true[index], prediction[index], average="macro")
              for index in (rng.integers(0, len(y_true), len(y_true)) for _ in range(rounds))]
    return tuple(np.quantile(scores, [.025, .975]).round(4))


def calibration_table(y_true: pd.Series, negative_probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"is_negative": (y_true == "Negative").astype(int), "probability": negative_probability})
    frame["bin"] = pd.cut(frame.probability, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return frame.groupby("bin", observed=True).agg(predicted_probability=("probability", "mean"),
                                                     observed_negative_rate=("is_negative", "mean"),
                                                     reviews=("is_negative", "size")).reset_index()


def train() -> dict:
    if not DATA.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` first.")
    data = pd.read_csv(DATA).dropna(subset=["review_text", "sentiment"])
    train_frame, test_frame = make_split(data)
    neutral_target = max((train_frame.sentiment == "Neutral").sum(), int((train_frame.sentiment == "Positive").sum() * .35))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=20_000, sublinear_tf=True)),
        ("oversample", RandomOverSampler(sampling_strategy={"Neutral": neutral_target}, random_state=42)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=1)),
    ])
    model.fit(train_frame.review_text, train_frame.sentiment)
    probability = model.predict_proba(test_frame.review_text)
    prediction = model.classes_[probability.argmax(axis=1)]
    negative_probability = probability[:, list(model.classes_).index("Negative")]
    lower, upper = bootstrap_macro_f1(test_frame.sentiment, prediction)
    metrics = {"model": "TF-IDF + Logistic Regression + Neutral RandomOverSampler", "accuracy": accuracy_score(test_frame.sentiment, prediction),
               "macro_f1": f1_score(test_frame.sentiment, prediction, average="macro"), "macro_f1_ci95_low": lower,
               "macro_f1_ci95_high": upper, "negative_recall": recall_score(test_frame.sentiment, prediction, labels=["Negative"], average=None)[0],
               "confusion_matrix": str(confusion_matrix(test_frame.sentiment, prediction, labels=LABELS).tolist()), "test_rows": len(test_frame)}
    audit = test_frame[["review_text", "star_rating", "sentiment"]].rename(columns={"sentiment": "actual_sentiment"}).copy()
    audit["predicted_sentiment"] = prediction
    audit["negative_probability"] = negative_probability
    audit["needs_complaint_review"] = negative_probability >= NEGATIVE_REVIEW_THRESHOLD
    audit["error_type"] = np.where((audit.actual_sentiment == "Negative") & (audit.predicted_sentiment == "Positive"), "negative_to_positive", "")
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    pd.DataFrame([metrics]).to_csv(METRICS, index=False)
    audit.to_csv(AUDIT, index=False)
    calibration_table(test_frame.sentiment, negative_probability).to_csv(CALIBRATION, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved {AUDIT}, {CALIBRATION}; triage threshold: P(Negative) >= {NEGATIVE_REVIEW_THRESHOLD}")
    return metrics


if __name__ == "__main__":
    train()
