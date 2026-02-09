import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "model"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


text = input("Enter news text: ")

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=64
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

pred_class = probs.argmax().item()
confidence = probs.max().item()

print("Predicted class index:", pred_class)
print("Confidence:", round(confidence, 4))
