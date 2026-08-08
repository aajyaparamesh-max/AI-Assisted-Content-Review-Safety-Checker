import os
from pathlib import Path
import gdown
import streamlit as st

# -------------------------------------------------------
# Project directories
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
BERT_DIR = MODELS_DIR / "bert"
DISTILBERT_DIR = MODELS_DIR / "distilbert"

MODELS_DIR.mkdir(exist_ok=True)
BERT_DIR.mkdir(exist_ok=True)
DISTILBERT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Google Drive Folder URLs
# -------------------------------------------------------

# https://drive.google.com/drive/folders/1-uauPsy_0YO-975Wq_rTkRvhNM6hSR58?usp=sharing
BERT_FOLDER_URL = "https://drive.google.com/drive/folders/1-uauPsy_0YO-975Wq_rTkRvhNM6hSR58"

# https://drive.google.com/drive/folders/1EKZKc228ud7ImnIu4NcaLdXLwBmW1wHv?usp=drive_link
DISTILBERT_FOLDER_URL = "https://drive.google.com/drive/folders/1EKZKc228ud7ImnIu4NcaLdXLwBmW1wHv"

# https://drive.google.com/file/d/1vL65T3OIZzhkLmOaSEbIMjsy5iNqj09E/view?usp=drive_link
FILES = {
    "data/processed_train.csv": "1vL65T3OIZzhkLmOaSEbIMjsy5iNqj09E",
}

# -------------------------------------------------------
# Download Function
# -------------------------------------------------------

def download_models():

    # -------------------------------
    # BERT
    # -------------------------------

    bert_file = BERT_DIR / "model.safetensors"

    if not bert_file.exists():
        try:    
            with st.spinner("Downloading BERT model (first run only)..."):
                gdown.download_folder(
                    url=BERT_FOLDER_URL,
                    output=str(BERT_DIR), 
                    quiet=False,
                    use_cookies=False
                )    
            # Verify download
            if not bert_file.exists():
            #     st.success("✅ BERT downloaded successfully.")
            # else:
                st.error("❌ BERT download completed, but model.safetensors was not found.")
                st.stop()
    
        except Exception as e:    
            st.error("❌ Failed to download BERT model.")    
            st.exception(e)          # Shows full traceback in Streamlit    
            st.stop()    
    else:    
        st.info("✅ BERT already exists.")

    # -------------------------------
    # DistilBERT
    # -------------------------------

    distilbert_file = DISTILBERT_DIR / "model.safetensors"

    if not distilbert_file.exists():
        try:    
            with st.spinner("Downloading DistilBERT model (first run only)..."):    
                gdown.download_folder(
                    url=DISTILBERT_FOLDER_URL,
                    output=str(DISTILBERT_DIR),
                    quiet=False,
    
                    
                    use_cookies=False
                )    
            if not distilbert_file.exists():
            #     st.success("✅ DistilBERT downloaded successfully.")
            # else:
                st.error("❌ DistilBERT download completed, but model.safetensors was not found.")
                st.stop()    
        except Exception as e:    
            st.error("❌ Failed to download DistilBERT model.")    
            st.exception(e)    
            st.stop()
    else:    
        st.info("✅ DistilBERT already exists.")

    # -------------------------------
    # processed_train.csv
    # -------------------------------
       
    for relative_path, file_id in FILES.items():
        file_path = BASE_DIR / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                str(file_path),
                quiet=False
            )
            # st.success(str(BASE_DIR / relative_path))

    st.success("✅ Models downloaded successfully.")

    # ============================= DEBUG =============================
    # st.write("========== DEBUG ==========")
    
    # st.write("MODELS_DIR:", MODELS_DIR)
    
    # st.write("MODELS exists:", MODELS_DIR.exists())
    
    # st.write("BERT exists:", BERT_DIR.exists())
    
    # st.write("DISTILBERT exists:", DISTILBERT_DIR.exists())
    
    # if BERT_DIR.exists():
    #     st.write("BERT files:", os.listdir(BERT_DIR))
    
    # if DISTILBERT_DIR.exists():
    #     st.write("DISTILBERT files:", os.listdir(DISTILBERT_DIR))
