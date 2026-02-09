print("🔥 train.py started (FAST MODE)")

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

# ---------------- FAST CONFIG ----------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64              # reduced for speed
BATCH_SIZE = 32           # increased for CPU efficiency
EPOCHS = 1                # single epoch
DATA_PATH = "data/news.json"
SAMPLE_SIZE = 20000       # use subset for fast training
# ---------------------------------------------


def load_data(path):
    """
    Robust loader:
    - Supports JSON array
    - Supports JSON Lines
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

        if content.startswith("["):
            data = json.loads(content)
        else:
            data = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    return pd.DataFrame(data)


def preprocess(df):
    """
    Robust preprocessing:
    - Normalizes column names
    - Automatically detects text columns
    """
    if df.empty:
        raise ValueError("❌ Dataset is empty")

    df.columns = [c.strip().lower() for c in df.columns]
    print("🧾 Available columns:", df.columns.tolist())

    text_cols = []
    for col in ["headline", "title", "short_description", "description"]:
        if col in df.columns:
            text_cols.append(col)

    if not text_cols:
        raise ValueError("❌ No valid text columns found")

    df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    if "category" not in df.columns:
        raise ValueError("❌ 'category' column not found")

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

    # Load dataset
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess(df)

    # -------- SPEED UP TRAINING --------
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"🚀 Using subset of {len(df)} samples for fast training")
    # ----------------------------------

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["category"])

    # Train / validation split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )

    # Training arguments (compatible with older transformers)
    args = TrainingArguments(
        output_dir="results",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="logs"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    # Train & evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # Save model
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    print("✅ Training complete (FAST MODE). Model saved.")
    print("📊 Evaluation Metrics:", metrics)
