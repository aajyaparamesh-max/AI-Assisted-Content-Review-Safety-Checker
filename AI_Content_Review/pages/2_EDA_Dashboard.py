
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown(
    "Interactive analysis of the processed Jigsaw Toxic Comment dataset."
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed_train.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

possible_features = [
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

ENGINEERED_FEATURES = [
    feature
    for feature in possible_features
    if feature in df.columns
]

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("EDA Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Dataset Preview",
        "Target Analysis",
        "Feature Distributions",
        "Correlation Analysis",
        "Word Analysis"
    ]
)

# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

if section == "Dataset Overview":

    st.header("Dataset Overview")

    rows = df.shape[0]
    cols = df.shape[1]

    duplicates = df.duplicated().sum()

    missing = df.isnull().sum().sum()

    toxic_comments = int((df[LABELS].sum(axis=1) > 0).sum())

    safe_comments = rows - toxic_comments

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", cols)
    col3.metric("Duplicate Rows", duplicates)

    col4, col5, col6 = st.columns(3)

    col4.metric("Missing Values", missing)
    col5.metric("Toxic Comments", toxic_comments)
    col6.metric("Safe Comments", safe_comments)

    st.divider()

    st.subheader("Dataset Information")

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing": df.isnull().sum().values
        }
    )

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

# ---------------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------------

elif section == "Dataset Preview":

    st.header("Dataset Preview")

    rows = st.slider(
        "Number of rows",
        5,
        100,
        10
    )

    st.dataframe(
        df.head(rows),
        use_container_width=True
    )

# ---------------------------------------------------------
# TARGET ANALYSIS
# ---------------------------------------------------------

elif section == "Target Analysis":

    st.header("Target Variable Analysis")

    label_counts = df[LABELS].sum().sort_values(ascending=False)

    fig = px.bar(
        x=label_counts.index,
        y=label_counts.values,
        color=label_counts.values,
        labels={
            "x": "Label",
            "y": "Count"
        },
        title="Distribution of Toxicity Labels"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    toxic = (df[LABELS].sum(axis=1) > 0).sum()

    safe = len(df) - toxic

    pie = px.pie(
        names=["Toxic", "Safe"],
        values=[toxic, safe],
        title="Toxic vs Safe Comments"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    st.divider()

    st.subheader("Multi-label Distribution")

    label_distribution = df["label_count"].value_counts().sort_index()

    fig = px.bar(
        x=label_distribution.index,
        y=label_distribution.values,
        labels={
            "x": "Number of Labels",
            "y": "Comments"
        },
        title="Comments vs Number of Toxic Labels"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# FEATURE DISTRIBUTIONS
# ---------------------------------------------------------

elif section == "Feature Distributions":

    st.header("Engineered Feature Analysis")

    feature = st.selectbox(
        "Select Feature",
        ENGINEERED_FEATURES
    )

    fig = px.histogram(
        df,
        x=feature,
        nbins=40,
        title=f"{feature} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    box = px.box(
        df,
        y=feature,
        title=f"{feature} Box Plot"
    )

    st.plotly_chart(
        box,
        use_container_width=True
    )

    st.divider()

    st.subheader("Feature Statistics")

    stats = df[ENGINEERED_FEATURES].describe().T

    st.dataframe(
        stats,
        use_container_width=True
    )

# ---------------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------------

elif section == "Correlation Analysis":

    st.header("Feature Correlation")

    corr = df[ENGINEERED_FEATURES].corr()

    heatmap = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Engineered Feature Correlation Matrix"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.divider()

    st.subheader("Label Correlation")

    label_corr = df[LABELS].corr()

    label_heat = px.imshow(
        label_corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Toxic Label Correlation"
    )

    st.plotly_chart(
        label_heat,
        use_container_width=True
    )

# ---------------------------------------------------------
# PART A ENDS HERE
# WORD ANALYSIS SECTION CONTINUES IN PART B
# ---------------------------------------------------------

# ---------------------------------------------------------
# WORD ANALYSIS
# ---------------------------------------------------------

elif section == "Word Analysis":

    st.header("Word Analysis")

    option = st.selectbox(
        "Select Text Category",
        [
            "All Comments",
            "Toxic Comments",
            "Non-Toxic Comments"
        ]
    )

    if option == "All Comments":

        text = " ".join(df["clean_text"].fillna("").astype(str))

    elif option == "Toxic Comments":

        toxic_df = df[df[LABELS].sum(axis=1) > 0]

        text = " ".join(
            toxic_df["clean_text"].fillna("").astype(str)
        )

    else:

        safe_df = df[df[LABELS].sum(axis=1) == 0]

        text = " ".join(
            safe_df["clean_text"].fillna("").astype(str)
        )

    # -------------------------------------------------

    st.subheader("Word Cloud")

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(16,8))

    ax.imshow(wordcloud, interpolation="bilinear")

    ax.axis("off")

    st.pyplot(fig)

    # -------------------------------------------------

    st.divider()

    st.subheader("Top 20 Frequent Words")

    words = text.split()

    counter = Counter(words)

    common_words = counter.most_common(20)

    words_df = pd.DataFrame(
        common_words,
        columns=[
            "Word",
            "Frequency"
        ]
    )

    fig = px.bar(
        words_df,
        x="Frequency",
        y="Word",
        orientation="h",
        color="Frequency",
        title="Top 20 Most Frequent Words"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------------------

    # st.divider()

    # st.subheader("Top 20 Frequent Tokens")

    # token_text = " ".join(
    #     df["tokens"].fillna("").astype(str)
    # )

    # token_text = (
    #     token_text
    #     .replace("[", "")
    #     .replace("]", "")
    #     .replace("'", "")
    #     .replace(",", " ")
    # )

    # token_counter = Counter(
    #     token_text.split()
    # )

    # token_df = pd.DataFrame(
    #     token_counter.most_common(20),
    #     columns=[
    #         "Token",
    #         "Frequency"
    #     ]
    # )

    # fig = px.bar(
    #     token_df,
    #     x="Frequency",
    #     y="Token",
    #     orientation="h",
    #     color="Frequency",
    #     title="Top 20 Tokens"
    # )

    # fig.update_layout(
    #     yaxis=dict(
    #         categoryorder="total ascending"
    #     )
    # )

    # st.plotly_chart(
    #     fig,
    #     use_container_width=True
    # )

    # -------------------------------------------------

    st.divider()

    st.subheader("Comment Length Analysis")

    fig = px.scatter(
        df,
        x="word_count",
        y="char_count",
        color="label_count",
        title="Word Count vs Character Count",
        opacity=0.7
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------------------

    st.divider()

    st.subheader("Average Engineered Feature Values")

    averages = (
        df[ENGINEERED_FEATURES]
        .mean()
        .sort_values(ascending=False)
    )

    fig = px.bar(
        x=averages.index,
        y=averages.values,
        labels={
            "x":"Feature",
            "y":"Average Value"
        },
        title="Average Engineered Feature Values"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------
# DOWNLOAD DATASET
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="📥 Download Processed Dataset",
    data=csv,
    file_name="processed_train.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.success(
    "EDA Dashboard Loaded Successfully"
)

st.sidebar.caption(
    "AI-Assisted Content Review & Safety Checker"
)
