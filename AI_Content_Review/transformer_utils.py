import torch
import streamlit as st

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

import config

# ==========================================
# Device
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.write(config.BERT_MODEL_PATH)

st.write(os.path.exists(config.BERT_MODEL_PATH))

# ==========================================
# Load BERT
# ==========================================

@st.cache_resource
def load_bert():

    tokenizer = BertTokenizer.from_pretrained(
        config.BERT_MODEL_PATH
    )

    model = BertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_PATH
    )

    model.to(DEVICE)

    model.eval()

    return tokenizer, model

# ==========================================
# Load DistilBERT
# ==========================================

@st.cache_resource
def load_distilbert():

    tokenizer = DistilBertTokenizer.from_pretrained(
        config.DISTILBERT_MODEL_PATH
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        config.DISTILBERT_MODEL_PATH
    )

    model.to(DEVICE)

    model.eval()

    return tokenizer, model

# ==========================================
# Internal Prediction
# ==========================================

def _predict(model, tokenizer, text):

    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt"
    )

    encoding = {
        k: v.to(DEVICE)
        for k, v in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**encoding)

        probs = torch.sigmoid(outputs.logits)

        preds = (probs >= 0.5).int()

    return (
        preds.cpu().numpy()[0],
        probs.cpu().numpy()[0]
    )

# ==========================================
# BERT Prediction
# ==========================================

def predict_bert(text):

    tokenizer, model = load_bert()

    return _predict(
        model,
        tokenizer,
        text
    )

# ==========================================
# DistilBERT Prediction
# ==========================================

def predict_distilbert(text):

    tokenizer, model = load_distilbert()

    return _predict(
        model,
        tokenizer,
        text
    )

# ==========================================
# Generic Transformer Prediction
# ==========================================

def predict_transformer(model_name, text):

    if model_name == "BERT":
        return predict_bert(text)

    elif model_name == "DistilBERT":
        return predict_distilbert(text)

    else:
        raise ValueError("Unknown transformer model")
