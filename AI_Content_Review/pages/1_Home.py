
import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# ======================
# Title
# ======================

st.title("🛡️ AI-Assisted Content Review & Safety Checker")

st.markdown("---")

st.markdown("""
### Welcome!

This application uses **Machine Learning** and **Deep Learning** models to detect toxic online comments.

The project helps automate content moderation by identifying harmful comments across six toxicity categories.
""")

st.divider()

# Project Overview
st.subheader("Project Objective")

st.info("""
Develop an intelligent content moderation system capable of
identifying multiple forms of toxic comments such as:

• Toxic

• Severe Toxic

• Obscene

• Threat

• Insult

• Identity Hate
""")

# ======================
# Project Statistics
# ======================

st.subheader("📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1: st.metric("Dataset", "159,571 Rows")

with col2: st.metric("Labels", "6")

with col3: st.metric("Models", "7")

with col4: st.metric("Best Model", "BERT")

st.markdown("---")

# ======================
# Workflow
# ======================

st.subheader("🔄 Project Workflow")

st.markdown("""
1. Data Collection

2. Data Preprocessing

3. Exploratory Data Analysis

4. Feature Engineering

5. Model Development

6. Model Evaluation

7. Toxic Comment Prediction
""")

st.markdown("---")

# Models Used
st.subheader("Models Implemented")

models = [
    "Logistic Regression",
    "Naïve Bayes",
    "Linear SVM",
    "Random Forest",
    "XGBoost",
    "BERT",
    "DistilBERT"
]

st.table(models)


# ======================
# Technologies
# ======================

st.subheader("🛠️ Technologies Used")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.info("""
- Python
- Pandas
- NumPy
- Scikit-learn
""")

with tech2:
    st.info("""
- Streamlit
- Plotly
- Matplotlib
- Seaborn
""")

with tech3:
    st.info("""
- NLTK
- TF-IDF
- Joblib
- Transformers
""")

st.markdown("---")

# ======================
# Footer
# ======================

st.caption("Developed as part of the AI-Assisted Content Review & Safety Checker Project.")
