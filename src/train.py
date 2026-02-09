print("🔥 train.py started (FAST MODE - STABLE VERSION)")

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

# ---------------- CONFIG ----------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 1                  # fast + stable
DATA_PATH = "data/news.json"
SAMPLE_SIZE = 20000
# --------------------------------------


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    return pd.DataFrame(data)


def preprocess(df):
    if df.empty:
        raise ValueError("Dataset is empty")

    df.columns = [c.strip().lower() for c in df.columns]
    print("🧾 Available columns:", df.columns.tolist())

    df["text"] = (
        df["headline"].fillna("") + " " + df["short_description"].fillna("")
    )

    return df[["text", "category"]]


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


if __name__ == "__main__":

    df = load_data(DATA_PATH)
    df = preprocess(df)

    # FAST MODE
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"🚀 Using {len(df)} samples for training")

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["category"])

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )

    args = TrainingArguments(
        output_dir="results",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="logs"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    print("✅ Training complete.")
    print("📊 Evaluation Metrics:", metrics)
