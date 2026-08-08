
# ============================
# Project Configuration File
# ============================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================
# Transformer Models
# ================================

BERT_MODEL_PATH = "models/bert"
DISTILBERT_MODEL_PATH = "models/distilbert"

MAX_LENGTH = 128


MODEL_PATHS = {

    "Logistic Regression":
        os.path.join(BASE_DIR, "models", "LogisticRegressionModel.pkl"),

    "Naive Bayes":
        os.path.join(BASE_DIR, "models", "NaiveBayesModel.pkl"),

    "Linear SVM":
        os.path.join(BASE_DIR, "models", "LinearSVMModel.pkl"),

    "Random Forest":
        os.path.join(BASE_DIR, "models", "RandomForestModel.pkl"),

    "XGBoost":
        os.path.join(BASE_DIR, "models", "XGBoostModel.pkl")
}


TFIDF_PATH = os.path.join(BASE_DIR, "data", "tfidf_vectorizer.pkl")

SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")

TRAIN_DATA = os.path.join(BASE_DIR, "data", "train.csv")

PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed_train.csv")

MODEL_RESULTS = os.path.join(BASE_DIR, "data", "ModelComparison.csv")  #comparison_results.csv # /content/drive/MyDrive/AI_Content_Review/data/ModelComparison.xlsx

import pandas as pd

def load_results():

    return pd.read_csv(MODEL_RESULTS)
