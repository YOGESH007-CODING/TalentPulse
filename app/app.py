import sys
from pathlib import Path

import joblib
import streamlit as st

# Streamlit executes this file with `app/` as the import root when launched by path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aspect_extraction import detect_aspects

st.set_page_config(page_title="Review Sentiment", page_icon="★")
st.title("Product Review Sentiment & Complaint Prioritization")
st.caption("Fast TF-IDF + Logistic Regression inference; intended for product-feedback triage.")
model_path = Path("models/tfidf_logreg_baseline.joblib")
if not model_path.exists():
    st.warning("Train the baseline first: `python -m src.train_baseline`.")
    st.stop()
model = joblib.load(model_path)
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
