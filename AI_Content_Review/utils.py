
# ============================================================
# utils.py (Version 3)
# ============================================================

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config

# ============================================================
# Download NLTK Resources
# ============================================================

NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4"
]

for resource in NLTK_RESOURCES:

    try:
        nltk.data.find(resource)

    except LookupError:
        nltk.download(resource)

# ============================================================
# Global Objects
# ============================================================

STOPWORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

ENGINEERED_FEATURES = [
    "char_count",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "stopword_count",
    "punctuation_count",
    "uppercase_ratio",
    "digit_count",
    "url_count",
    "exclamation_count",
    "question_count"
]

# ============================================================
# Cached Resource Loading
# ============================================================

# -------------------------
# TF-IDF Vectorizer
# -------------------------

@st.cache_resource
def load_vectorizer():

    return joblib.load(config.TFIDF_PATH)


# -------------------------
# MinMax Scaler
# -------------------------

@st.cache_resource
def load_scaler():

    return joblib.load(config.SCALER_PATH)


# -------------------------
# Traditional ML Models
# -------------------------

@st.cache_resource
def load_model(model_name):

    if model_name not in config.MODEL_PATHS:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_path = config.MODEL_PATHS[model_name]

    model = joblib.load(model_path)

    return model


# ============================================================
# Load Shared Resources
# ============================================================

VECTORIZER = load_vectorizer()

SCALER = load_scaler()

# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text):
    """
    Clean and normalize text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():

        if word not in STOPWORDS:

            word = LEMMATIZER.lemmatize(word)

            words.append(word)

    return " ".join(words)


# ============================================================
# Engineered Features
# ============================================================

def create_numeric_features(text):
    """
    Creates the same engineered features
    that were used during model training.
    """

    if pd.isna(text):
        text = ""

    original_text = str(text)

    cleaned = clean_text(original_text)

    words = cleaned.split()

    char_count = len(original_text)

    word_count = len(words)

    sentence_count = max(
        1,
        len(re.findall(r"[.!?]", original_text))
    )

    avg_word_length = (
        np.mean([len(word) for word in words])
        if len(words) > 0
        else 0
    )

    stopword_count = sum(
        word.lower() in STOPWORDS
        for word in original_text.split()
    )

    punctuation_count = sum(
        ch in string.punctuation
        for ch in original_text
    )

    uppercase_ratio = (
        sum(ch.isupper() for ch in original_text)
        / max(len(original_text), 1)
    )

    digit_count = sum(
        ch.isdigit()
        for ch in original_text
    )

    url_count = len(
        re.findall(
            r"http[s]?://\S+|www\.\S+",
            original_text
        )
    )

    exclamation_count = original_text.count("!")

    question_count = original_text.count("?")

    features = np.array([[
        char_count,
        word_count,
        sentence_count,
        avg_word_length,
        stopword_count,
        punctuation_count,
        uppercase_ratio,
        digit_count,
        url_count,
        exclamation_count,
        question_count
    ]])

    return features


# ============================================================
# Complete Preprocessing Pipeline
# ============================================================

def preprocess_text(text):
    """
    Returns the exact feature matrix
    used during ML model training.
    """

    cleaned_text = clean_text(text)

    tfidf_features = VECTORIZER.transform(
        [cleaned_text]
    )

    numeric_features = create_numeric_features(text)

    numeric_features = SCALER.transform(
        numeric_features
    )

    final_features = hstack(
        [
            tfidf_features,
            numeric_features
        ]
    )

    return final_features

# ============================================================
# Traditional ML Prediction
# ============================================================

def predict_ml(comment, model_name):
    """
    Predict using traditional Machine Learning models.

    Returns
    -------
    prediction : ndarray (6,)
    probability : ndarray (6,) or None
    """

    model = load_model(model_name)

    X = preprocess_text(comment)

    prediction = model.predict(X)

    prediction = np.asarray(prediction).reshape(-1)

    probability = None

    try:

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(X)

            # OneVsRestClassifier
            if isinstance(probs, list):

                probability = np.array([
                    p[:, 1][0]
                    for p in probs
                ])

            # MultiOutputClassifier
            elif isinstance(probs, np.ndarray):

                if probs.ndim == 2:

                    probability = probs[0]

        elif hasattr(model, "decision_function"):

            probs = model.decision_function(X)

            scores = model.decision_function(X)

            scores = np.asarray(scores)

            # ---------------------------------
            # OneVsRestClassifier
            # Shape: (1, 6)
            # ---------------------------------
            if scores.ndim == 2:

                scores = scores[0]

            # ---------------------------------
            # Convert decision scores to
            # probability-like confidence
            # ---------------------------------
            scores = np.clip(scores, -20, 20)
            probability = 1 / (1 + np.exp(-scores))
    except Exception:

        probability = None

    return prediction, probability


# ============================================================
# Unified Prediction Function
# ============================================================

def predict(comment, model_name):
    """
    Universal prediction function.

    Supports

    • Logistic Regression
    • Naive Bayes
    • Linear SVM
    • Random Forest
    • XGBoost
    • BERT
    • DistilBERT
    """

    # --------------------------------------------
    # Transformer Models
    # --------------------------------------------

    if model_name in ["BERT", "DistilBERT"]:

        from transformer_utils import predict_transformer

        prediction, probability = predict_transformer( model_name, comment)

        prediction = np.asarray(prediction).reshape(-1)

        probability = np.asarray(probability).reshape(-1)

        return prediction, probability

    # --------------------------------------------
    # Traditional ML Models
    # --------------------------------------------

    prediction, probability = predict_ml( comment, model_name)

    return prediction, probability

