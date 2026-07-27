# Model Report: Review Sentiment Analysis & Complaint Prioritization

## 1. Executive summary

This project classifies Amazon Electronics reviews into three rating-derived sentiment classes:

| Star rating | Sentiment label |
| --- | --- |
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

The verified baseline is a class-weighted Logistic Regression model over 20,000 TF-IDF unigram and bigram features. It was trained and evaluated on a stratified 80/20 split of **5,000 real Amazon Electronics reviews**.

| Metric | Result |
| --- | ---: |
| Accuracy | 84.20% |
| Macro-F1 | 0.6046 |
| Training reviews | 4,000 |
| Test reviews | 1,000 |

Accuracy is noticeably higher than macro-F1 because positive reviews dominate the data. The model is useful as a quick triage baseline, but it is not equally reliable for all three sentiment groups—especially the neutral class.

## 2. Dataset and label distribution

The dataset is a CPU-friendly, reproducible 5,000-review stream from the Amazon Reviews 2023 Electronics category. The retained fields are review text, star rating, category, verified-purchase flag, and helpful-vote count.

| Sentiment | Reviews | Share |
| --- | ---: | ---: |
| Positive | 4,083 | 81.66% |
| Negative | 591 | 11.82% |
| Neutral | 326 | 6.52% |

The class imbalance is material. A simplistic positive-only model would already obtain about 81.7% accuracy, so accuracy alone would overstate model value. Macro-F1 gives equal weight to Negative, Neutral, and Positive performance and is the more informative headline metric.

## 3. Baseline model design

The baseline uses the following approach:

- Text representation: lowercased TF-IDF with unigrams and bigrams, capped at 20,000 terms.
- Classifier: multinomial Logistic Regression with `class_weight="balanced"`.
- Evaluation: stratified random 80/20 split with `random_state=42`.
- Text preparation: HTML removal and whitespace normalization. Negation words are retained rather than treated as stop words.

Class weighting was chosen instead of discarding positive reviews or oversampling minority classes. This retains real review language and avoids creating duplicate synthetic documents, while making errors on minority classes more costly during fitting.

## 4. Evaluation results and interpretation

### Confusion matrix

Rows are true labels; columns are predicted labels. Class order is Negative, Neutral, Positive.

| Actual \ Predicted | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: |
| Negative | 77 | 15 | 26 |
| Neutral | 15 | 17 | 33 |
| Positive | 29 | 40 | 748 |

### What works well

- The baseline correctly identifies **748 of 817** positive reviews in the test set. This matches the dataset’s dominant class and supports fast routing of clearly positive feedback.
- It identifies **77 of 118** negative reviews. That is meaningful for a lightweight triage system because it surfaces many dissatisfied customers without needing a GPU-hosted model.
- TF-IDF bigrams can capture short local phrases such as “not working,” “waste of,” or “highly recommend,” which are common review signals.

### Where the model struggles

- Only **17 of 65** neutral reviews are correctly classified. Most are shifted toward Positive, reflecting both the small neutral class and the ambiguity of 3-star language.
- **26 negative reviews** are predicted Positive. In a complaint-prioritization workflow, these are costly misses because serious complaints could be routed incorrectly.
- **40 positive reviews** are predicted Neutral. This is less operationally harmful than missing a complaint, but it creates unnecessary review work.
- The gap between 84.2% accuracy and 0.6046 macro-F1 is direct evidence that performance is uneven across classes.

## 5. Complaint-aspect findings

Aspect tags are keyword rules applied only to negative reviews. They are intended for transparent prioritization, not as a claim that each matched word proves product causality.

| Rank | Aspect | Negative-review mentions | Mean star rating |
| ---: | --- | ---: | ---: |
| 1 | Compatibility | 160 | 1.39 |
| 2 | Quality | 131 | 1.40 |
| 3 | Price | 111 | 1.31 |
| 4 | Battery | 89 | 1.38 |
| 5 | Delivery | 22 | 1.27 |

The most actionable first investigation is compatibility: it has the largest mention volume. Delivery has the lowest mean rating, but the smaller sample means it should be monitored rather than treated as the top business priority without additional evidence.

## 6. Current shortcomings

### Data and sampling

1. **Small development sample.** Five thousand reviews are sufficient for a functional baseline, but small for a diverse category such as Electronics. Rare product types, languages, and failure modes may be absent.
2. **Sequential stream selection.** The current loader takes the first valid streamed records rather than a randomized sample over the entire category. It is reproducible but can introduce ordering or temporal bias.
3. **Single category.** Results cannot be assumed to transfer to Home & Kitchen, Beauty, or other categories, whose vocabulary and rating behavior differ.
4. **Rating is an imperfect sentiment label.** A 3-star review may be balanced, mildly positive, or a serious complaint with one redeeming feature. Likewise, a 5-star review can mention a delivery issue.
5. **No deduplication or language filtering.** Near-duplicate reviews, non-English text, and very short text can distort metrics.

### Modeling

1. **Bag-of-words context limitations.** TF-IDF does not truly understand long-distance negation, sarcasm, or mixed sentiment. “The battery is great, but the app makes the product unusable” is difficult to reduce to a single document-level score.
2. **One fixed decision rule.** The baseline chooses the class with the largest probability. It does not use a complaint-sensitive threshold or an abstention path for low-confidence predictions.
3. **Probability calibration is untested.** A displayed 80% confidence score should not be interpreted as an 80% observed likelihood without calibration testing.
4. **No temporal validation.** The random split may overestimate future performance if reviews for the same product or period share repeated language.
5. **Transformer results are not yet available.** The DistilBERT fine-tuning pipeline exists, but no transformer metric is reported until the run completes. It would be incorrect to claim transformer improvement beforehand.

### Aspect extraction

1. **Keyword ambiguity.** “Power” can refer to battery, device capability, or electrical compatibility. Keyword rules cannot distinguish all meanings.
2. **Limited vocabulary.** The five aspect dictionaries miss synonyms and new complaint types, such as screen defects, sound quality, warranty, or customer support.
3. **No aspect sentiment linkage.** A review can say “battery is excellent but delivery was terrible”; the current logic tags both but does not attribute the negative sentiment to the correct aspect.
4. **No product-level normalization.** High-volume products may dominate aspect counts even when their complaint rate is low.

### Deployment and monitoring

1. **The Streamlit demo is local only.** It has not been deployed to Hugging Face Spaces or Render.
2. **No feedback loop.** There is no interface for a human reviewer to correct a prediction and use that correction in future training.
3. **No production drift monitoring.** The project does not yet track changes in vocabulary, class distribution, confidence, or aspect prevalence after release.

## 7. Recommended improvements

### Highest-priority improvements

| Priority | Improvement | Why it matters | Success measure |
| --- | --- | --- | --- |
| P0 | Increase to a randomized 50k–100k sample | Reduces sampling error and representation gaps | Stable macro-F1 across three random seeds |
| P0 | Add per-class precision, recall, and F1 | Macro-F1 alone hides which class is failing | Neutral and Negative recall tracked explicitly |
| P0 | Use grouped or time-based validation | Produces a more realistic generalization estimate | Test performance remains stable on later reviews/products |
| P0 | Train and evaluate DistilBERT | Tests whether contextual modeling fixes minority-class errors | Macro-F1 and negative recall improve over baseline |
| P1 | Calibrate probabilities | Makes displayed confidence suitable for triage thresholds | Lower Brier score / improved calibration plot |
| P1 | Add a low-confidence “Needs review” state | Prevents forced labels on ambiguous reviews | Error rate on accepted predictions falls |
| P1 | Expand and validate aspect taxonomy | Makes complaint ranking more complete and accurate | Human-checked aspect precision/recall |

### Data improvements

- Replace “first streamed records” with a deterministic random reservoir sample or a shuffled streaming sample.
- Filter unsupported languages or use `distilbert-base-multilingual-cased` when multilingual reviews are intentionally included.
- Remove duplicate text and separately flag extremely short reviews.
- Add product ID and timestamp to enable product-group and temporal holdouts.
- Audit performance by verified-purchase status, review length band, and major product family when sufficient metadata is available.

### Model improvements

- Fine-tune `distilbert-base-uncased` with the same split; use weighted loss or carefully sampled batches if minority recall remains weak.
- Compare against a stronger sparse baseline such as linear SVM or Complement Naive Bayes before assuming a transformer is necessary.
- Tune the baseline’s regularization strength, TF-IDF minimum document frequency, and n-gram range with cross-validation on the training partition only.
- Optimize for the actual operational objective. For example, if missing a complaint is worse than sending a review to manual triage, maximize Negative recall at a defined minimum precision instead of optimizing macro-F1 alone.
- Calibrate class probabilities using a validation partition (`CalibratedClassifierCV`) and select thresholds after calibration.

### Aspect-extraction improvements

- Replace static substrings with lemmatized keyword matching and phrase patterns, then review false matches.
- Add a zero-shot or fine-tuned multi-label aspect classifier after assembling a human-labeled sample.
- Use noun-phrase extraction and topic discovery to discover new themes, but validate topics with manual review before naming them.
- Report both volume and rate per product or product family so popular items do not dominate the dashboard.
- Extract aspect-level sentiment, not only document-level sentiment, for mixed reviews.

## 8. Suggested next experiment plan

1. Create a randomly sampled 50,000-review Electronics dataset and save the sampling seed and source revision.
2. Establish a train/validation/test split that groups reviews by product where possible.
3. Run baseline hyperparameter search on train/validation only; freeze the test set.
4. Train DistilBERT with identical labels and compare accuracy, macro-F1, per-class F1, negative recall, latency, model size, and inference cost.
5. Manually label 300–500 negative reviews with one or more aspects; use this as a held-out benchmark for aspect extraction.
6. Add confidence-based triage and capture reviewer corrections for continuous evaluation.
7. Deploy the compact baseline only if it meets the chosen complaint-recall requirement; retain the transformer as a batch-analysis option if its quality gain justifies the cost.

## 9. Conclusion

The current baseline is a credible, fast starting point: it materially improves on a positive-only accuracy baseline and produces transparent complaint themes from real reviews. Its central limitation is uneven class performance, particularly for neutral reviews and for negative reviews incorrectly sent to the positive class. The next work should prioritize representative sampling, class-specific evaluation, contextual-model comparison, probability calibration, and validated aspect labels before treating it as a production decision system.
