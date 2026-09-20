# Product Review Sentiment Analysis & Complaint Prioritization

An end-to-end NLP project that classifies Amazon Electronics reviews as **Negative** (1–2 stars), **Neutral** (3 stars), or **Positive** (4–5 stars), then ranks recurrent themes in negative feedback.

## Data scope

The pipeline streams a reproducible capped sample (default: 20,000 non-empty reviews) from the [Amazon Reviews 2023 Electronics subset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). It retains review text, star rating, category, verified-purchase status, and helpful-vote count. The source records are real Amazon review data; no labels or reviews are fabricated.

The checked local run materialized **5,000** reviews (a reduced cap used for CPU-friendly verification): **4,083 Positive (81.66%)**, **591 Negative (11.82%)**, and **326 Neutral (6.52%)**. This substantial positive skew is retained and addressed with class-weighted training.

## Pipeline

1. `src.data_prep` streams, validates, cleans, and writes Electronics reviews.
2. `src.train_baseline` uses TF-IDF (uni/bigrams) and class-weighted Logistic Regression on a stratified 80/20 split.
3. `src.train_transformer` fine-tunes DistilBERT on the same seed and stratification strategy.
4. `src.aspect_extraction` tags negative reviews for battery, delivery, quality, compatibility, and price issues and ranks them by volume.
5. `app/app.py` provides fast baseline inference for a lightweight Streamlit deployment.

## Dashboard and reporting

Power BI is the executive reporting layer for the batch pipeline. It reads the CSV
outputs produced by the Python scripts and supports complaint prioritization,
sentiment/rating analysis, and model monitoring. The project does not currently
ship a `.pbix` file or a live data connection, so the dashboard should be refreshed
after the pipeline regenerates the CSVs.

Recommended Power BI sources:

- `data/processed/reviews_clean.csv` — review-level sentiment, star rating, and
  verified-purchase data for sentiment/rating distributions and slicers.
- `data/processed/aspect_summary.csv` — complaint aspect volume and average rating
  for the prioritization matrix.
- `data/processed/baseline_metrics.csv` and
  `data/processed/transformer_metrics.csv` — accuracy, macro-F1, and test-row
  counts for model comparison. The transformer file is only populated after its
  training command completes.

Suggested report pages:

1. **Complaint prioritization:** scatter plot with complaint volume on the X-axis,
   average rating on the Y-axis, and aspect as the label. Lower ratings and higher
   volume indicate higher investigation priority.
2. **Sentiment and rating distribution:** sentiment share or star-rating charts
   with verified-purchase and sentiment slicers.
3. **Model performance:** cards or a clustered column chart comparing accuracy and
   macro-F1 by model.

### Power BI setup

1. Run the data-preparation and analysis commands below so the CSV outputs exist.
2. In Power BI Desktop, choose **Get data → Text/CSV** and load the files listed
   above.
3. Build the report pages using the recommended fields and refresh the data after
   each new pipeline run.

Power BI should be positioned as the decision-support layer: executives use it to
see which issues deserve attention, while the Streamlit tab remains the review-level
inference and demonstration interface. It should not be presented as the model or
as a real-time production connection until a `.pbix` report and scheduled refresh
are added.

## Metrics

The real 5,000-review baseline run uses a 4,000/1,000 stratified split:

| Model | Accuracy | Macro-F1 | Test reviews |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 84.20% | 0.6046 | 1,000 |
| DistilBERT | Run `python -m src.train_transformer` | — | — |

The transformer command is included but its result is deliberately not claimed until its fine-tune completes. Each trainer writes computed accuracy, macro-F1, confusion matrix, and test-row count to:

- `data/processed/baseline_metrics.csv`
- `data/processed/transformer_metrics.csv`

Macro-F1 is the primary score because the observed rating-derived labels are heavily skewed toward positive reviews.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_prep --sample-size 20000
python -m src.train_baseline
python -m src.aspect_extraction
streamlit run app/app.py

# Optional: GPU strongly recommended
python -m src.train_transformer
```

After training, the app accepts review text and returns baseline sentiment plus matching complaint aspects. It explicitly handles blank, emoji-only, and long inputs.

## Transformer comparison

Compare the two generated metric CSVs after training. DistilBERT commonly helps with context, negation, and mixed sentiment, while the TF-IDF baseline is far cheaper, faster, and better suited to a free-tier interactive demo. The actual numbers—not an assumed outcome—should drive the production choice.

## Complaint prioritization

`data/processed/aspect_summary.csv` ranks detected complaint aspects by negative-review volume and reports the corresponding mean star rating. The verified run found: compatibility (160, 1.39 stars), quality (131, 1.40), price (111, 1.31), battery (89, 1.38), and delivery (22, 1.27).

## Responsible use

Predictions reflect review language and the source data’s bias. They are a triage aid, not a substitute for product, support, or safety investigation.
