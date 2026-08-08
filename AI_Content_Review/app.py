
import streamlit as st
from download_models import download_models

# Download models only once
@st.cache_resource
def initialize_models():
    download_models()

initialize_models()

st.set_page_config(
    page_title="AI Content Review",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Assisted Content Review & Safety Checker")

st.markdown("""
Welcome to the AI-Assisted Content Review application.

This application detects harmful online comments using Machine Learning
and Deep Learning models trained on the Jigsaw Toxic Comment dataset.

👈 Use the **sidebar** to navigate through the application pages.
""")

