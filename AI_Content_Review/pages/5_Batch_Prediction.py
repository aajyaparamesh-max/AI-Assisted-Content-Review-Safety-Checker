
import time
import pandas as pd
import streamlit as st
import config
from utils import predict,LABELS
from transformer_utils import predict_transformer
import numpy as np

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Toxic Comment Prediction")
st.markdown(
    """
Upload a CSV file containing a **comment_text** column to predict
toxicity labels using the selected machine learning model.
"""
)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------

MODEL_OPTIONS = [
    "Logistic Regression",
    "Naive Bayes",
    "Linear SVM",
    "Random Forest",
    "XGBoost",
    "BERT",
    "DistilBERT"
]

# # ============================
# # Model Selection
# # ============================
model_name = st.selectbox("Select Prediction Model",MODEL_OPTIONS)

# ---------------------------------------------------------
# Upload CSV
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------------

    if "comment_text" not in df.columns:

        st.error(
            "The uploaded CSV must contain a column named 'comment_text'."
        )

        st.stop()

    # -----------------------------------------

    st.info(f"Selected Model : **{model_name}**")

    if st.button("🚀 Analyze CSV"):

        progress_bar = st.progress(0)

        status_text = st.empty()

        start = time.time()

        # predictions = []
        predictions = []
        confidences = []

        total_rows = len(df)

        for index, comment in enumerate(df["comment_text"]):

            pred, prob = predict( comment, model_name)

            predictions.append(pred)

            confidences.append(prob)

            progress = (index + 1) / total_rows

            progress_bar.progress(progress)

            status_text.text(f"Processing {index+1} of {total_rows} comments...")

        end = time.time()

        prediction_df = pd.DataFrame(
            predictions,
            columns=LABELS
        )

        confidence_df = pd.DataFrame(
        confidences,
        columns=[f"{label}_confidence" for label in LABELS]
        )

        result_df = pd.concat(
        [
            df.reset_index(drop=True),
            prediction_df,
            confidence_df
        ],
        axis=1)

        st.success(f"Prediction completed successfully in {end-start:.2f} seconds.")

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        st.subheader("Prediction Summary")

        total_comments = len(result_df)

        toxic_comments = int(result_df["toxic"].sum())

        safe_comments = total_comments - toxic_comments

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Comments",
            total_comments
        )

        col2.metric(
            "Toxic Comments",
            toxic_comments
        )

        col3.metric(
            "Safe Comments",
            safe_comments
        )

        # ---------------------------------------------------------
        # Toxicity Distribution
        # ---------------------------------------------------------

        st.subheader("Toxicity Distribution")

        toxicity_counts = result_df[LABELS].sum()

        st.bar_chart(toxicity_counts)

        # ---------------------------------------------------------
        # Prediction Table
        # ---------------------------------------------------------

        st.subheader("Prediction Results")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # ---------------------------------------------------------
        # Download CSV
        # ---------------------------------------------------------

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="Predicted_Toxicity_Results.csv",
            mime="text/csv"
        )
