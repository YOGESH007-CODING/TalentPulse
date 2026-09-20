import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Streamlit executes this file with `app/` as the import root when launched by path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aspect_extraction import detect_aspects

st.set_page_config(page_title="Review Sentiment", page_icon="★")
st.title("Product Review Sentiment & Complaint Prioritization")
st.caption("Fast TF-IDF + Logistic Regression inference; intended for product-feedback triage.")
model_path = PROJECT_ROOT / "models/tfidf_logreg_baseline.joblib"
if not model_path.exists():
    st.warning("Train the baseline first: `python -m src.train_baseline`.")
    st.stop()
model = joblib.load(model_path)

inference_tab, dashboard_tab = st.tabs(["Review inference", "Power BI executive dashboard"])

with inference_tab:
    text = st.text_area("Paste a product review", height=180, max_chars=5000, placeholder="This charger stopped working after two weeks...")
    if st.button("Analyze"):
        normalized = text.strip()
        if not normalized:
            st.info("Enter a review to analyze.")
        elif len(normalized) < 3 or not any(character.isalpha() for character in normalized):
            st.info("Please enter a short text review; emoji-only input cannot be classified reliably.")
        else:
            if len(normalized) > 2000:
                normalized = normalized[:2000]
                st.caption("The review was truncated to 2,000 characters for a fast, stable prediction.")
            probabilities = model.predict_proba([normalized])[0]
            label = model.classes_[probabilities.argmax()]
            negative_probability = probabilities[list(model.classes_).index("Negative")]
            st.metric("Predicted sentiment", label, f"{probabilities.max():.1%} confidence")
            if negative_probability >= 0.30:
                st.warning(f"Needs complaint review: negative-sentiment probability is {negative_probability:.1%} (triage threshold: 30%).")
            aspects = detect_aspects(normalized)
            st.write("Detected complaint aspects:" if aspects else "No configured complaint aspect detected.")
            if aspects:
                st.write(", ".join(aspects))

with dashboard_tab:
    st.subheader("Power BI executive reporting view")
    st.caption("This in-app view uses the same processed CSV outputs that are loaded into Power BI for executive reporting.")

    processed = PROJECT_ROOT / "data" / "processed"
    reviews_path = processed / "reviews_clean.csv"
    aspects_path = processed / "aspect_summary.csv"
    baseline_path = processed / "baseline_metrics.csv"
    transformer_path = processed / "transformer_metrics.csv"

    if not reviews_path.exists():
        st.info("Run `python -m src.data_prep` to generate dashboard data.")
    else:
        reviews = pd.read_csv(reviews_path)
        aspects = pd.read_csv(aspects_path) if aspects_path.exists() else pd.DataFrame()
        metric_files = [path for path in (baseline_path, transformer_path) if path.exists()]
        metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True) if metric_files else pd.DataFrame()

        total_reviews = len(reviews)
        negative_reviews = int((reviews["sentiment"] == "Negative").sum())
        complaint_rate = negative_reviews / total_reviews if total_reviews else 0
        top_aspect = aspects.iloc[0]["aspect"] if not aspects.empty else "—"
        top_volume = int(aspects.iloc[0]["complaint_volume"]) if not aspects.empty else 0

        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
        kpi_1.metric("Reviews analyzed", f"{total_reviews:,}")
        kpi_2.metric("Negative reviews", f"{negative_reviews:,}", f"{complaint_rate:.1%} of sample")
        kpi_3.metric("Top complaint aspect", str(top_aspect).title())
        kpi_4.metric("Top aspect volume", f"{top_volume:,}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Sentiment distribution**")
            sentiment_counts = reviews["sentiment"].value_counts().rename_axis("sentiment").to_frame("reviews")
            st.bar_chart(sentiment_counts)
        with right:
            st.markdown("**Complaint prioritization**")
            if aspects.empty:
                st.info("Run `python -m src.aspect_extraction` to generate aspect data.")
            else:
                st.scatter_chart(aspects, x="complaint_volume", y="average_rating", x_label="Complaint volume", y_label="Average star rating")

        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            st.markdown("**Complaint aspects**")
            if not aspects.empty:
                st.dataframe(aspects, hide_index=True, use_container_width=True)
        with bottom_right:
            st.markdown("**Model performance**")
            if metrics.empty:
                st.info("Run the model trainers to generate metric CSVs.")
            else:
                display_metrics = metrics[["model", "accuracy", "macro_f1", "test_rows"]].copy()
                display_metrics["accuracy"] = display_metrics["accuracy"].map(lambda value: f"{value:.1%}")
                display_metrics["macro_f1"] = display_metrics["macro_f1"].map(lambda value: f"{value:.4f}")
                st.dataframe(display_metrics, hide_index=True, use_container_width=True)
