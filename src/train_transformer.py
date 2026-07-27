"""Fine-tune DistilBERT on exactly the baseline's stratified split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

DATA = Path("data/processed/reviews_clean.csv")
OUT = Path("models/distilbert_sentiment")
METRICS = Path("data/processed/transformer_metrics.csv")
LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def train() -> dict:
    if not DATA.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` first.")
    data = pd.read_csv(DATA).dropna(subset=["review_text", "sentiment"])
    train_frame, test_frame = train_test_split(data, test_size=.2, random_state=42, stratify=data.sentiment)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["review_text"], truncation=True, padding="max_length", max_length=192)

    train_set = Dataset.from_pandas(train_frame[["review_text", "sentiment"]].rename(columns={"sentiment": "label"}))
    test_set = Dataset.from_pandas(test_frame[["review_text", "sentiment"]].rename(columns={"sentiment": "label"}))
    train_set = train_set.map(lambda row: {"label": LABEL_TO_ID[row["label"]]}).map(tokenize, batched=True)
    test_set = test_set.map(lambda row: {"label": LABEL_TO_ID[row["label"]]}).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, predictions), "macro_f1": f1_score(labels, predictions, average="macro")}

    trainer = Trainer(model=model, args=TrainingArguments(output_dir=".trainer", num_train_epochs=2,
                      per_device_train_batch_size=16, per_device_eval_batch_size=32, eval_strategy="epoch",
                      save_strategy="no", report_to="none", seed=42), train_dataset=train_set, eval_dataset=test_set,
                      compute_metrics=compute_metrics)
    trainer.train()
    output = trainer.predict(test_set)
    prediction = np.argmax(output.predictions, axis=1)
    result = {"model": "distilbert-base-uncased", "accuracy": accuracy_score(output.label_ids, prediction),
              "macro_f1": f1_score(output.label_ids, prediction, average="macro"),
              "confusion_matrix": str(confusion_matrix(output.label_ids, prediction, labels=[0, 1, 2]).tolist()),
              "test_rows": len(prediction)}
    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    pd.DataFrame([result]).to_csv(METRICS, index=False)
    print(pd.DataFrame([result]).to_string(index=False))
    return result


if __name__ == "__main__":
    train()
