print("🔥 predict.py started")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    return pred_class.item(), confidence.item()


if __name__ == "__main__":
    text = input("Enter news text: ")
    label, confidence = predict(text)
    print(f"Predicted Label ID: {label}")
    print(f"Confidence: {confidence:.4f}")
