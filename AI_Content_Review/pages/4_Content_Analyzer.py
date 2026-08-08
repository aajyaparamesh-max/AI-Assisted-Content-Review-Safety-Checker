
import numpy as np
import streamlit as st
from utils import predict,LABELS #load_model, predict_comment, predict_probability, LABELS
import config
from transformer_utils import predict_transformer

MODEL_OPTIONS = [
    "Logistic Regression",
    "Naive Bayes",
    "Linear SVM",
    "Random Forest",
    "XGBoost",
    "BERT",
    "DistilBERT"
]

st.set_page_config(
    page_title="Content Analyzer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Content Analyzer")

st.markdown("Analyze user comments for potential toxic content.")

st.markdown("---")

# ============================
# Model Selection
# ============================
selected_model = st.selectbox("Select Prediction Model", MODEL_OPTIONS)

# ============================
# User Input
# ============================
import traceback
comment = st.text_area(

    "Enter Comment",

    height=180,

    placeholder="Type or paste a comment here..."

)

# ============================
# Analyze
# ============================
if st.button("🔍 Analyze Content", use_container_width=True):

    if comment.strip() == "":
        st.warning("Please enter a comment.")

    else:

        with st.spinner("Analyzing..."):

            try:
                prediction, probability = predict( comment, selected_model)

                st.markdown("---")
                st.subheader("Prediction Results")

                toxic_labels = []

                for label, value in zip(LABELS, prediction):

                    if value == 1:
                        toxic_labels.append(label)

                # Overall Result
                if len(toxic_labels) == 0:
                    st.success("✅ SAFE COMMENT")
                else:
                    st.error("🚨 TOXIC COMMENT DETECTED")

                    # Detected Labels
                    st.subheader("Detected Labels")
    
                    if toxic_labels:
    
                        cols = st.columns(3)
    
                        for i, label in enumerate(toxic_labels):
                            cols[i % 3].error(label.upper())
    
                    else:
                        st.info("No toxic labels detected.")
    
                    # Probability Scores
                    if probability is not None:
    
                        st.markdown("---")
                        if selected_model in ["BERT", "DistilBERT"]:
                            st.subheader("Transformer Confidence Scores")
                        else:
                            st.subheader("ML Confidence Scores")
    
                        probability = np.asarray(probability)
    
                        if probability.ndim == 2: probability = probability[0]
    
                        for label, score in zip(LABELS, probability):
                            st.write(f"**{label.title()}**")
                            st.progress(float(score))
                            st.write(f"{score:.2%}")
                    else:
                        st.info("Confidence scores are not available for this model.")

            except Exception as e:
                st.error(f"Error while predicting:\n\n{str(e)}")

                with st.expander("Full Error Traceback"):
                    st.code(traceback.format_exc())
