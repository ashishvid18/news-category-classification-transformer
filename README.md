# News Category Classification using Transformers

## Project Overview
This project implements a multi-class news category classifier using a pretrained
Transformer model fine-tuned on the HuffPost News Category Dataset.

## Dataset
- **Name:** News Category Dataset (HuffPost)
- **Source:** https://www.kaggle.com/datasets/rmisra/news-category-dataset
- ~210,000 labeled news headlines and descriptions

## Model Used
- **distilbert-base-uncased**
- Fine-tuned using Hugging Face Transformers

## Training Details
- Tokenization: Padding & truncation
- Max sequence length: 64
- Optimizer: AdamW
- Epochs: 1
- Training strategy: Subset training for faster experimentation

## Evaluation Results
- **Accuracy:** ~0.85  
- **F1-score (weighted):** ~0.84  

## Inference
Run:
```bash
python src/predict.py
