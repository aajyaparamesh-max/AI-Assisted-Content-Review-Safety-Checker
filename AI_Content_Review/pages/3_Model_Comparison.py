
import streamlit as st
import pandas as pd
import plotly.express as px
import config

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance Comparison")

st.markdown("---")

# ============================
# Load comparison results
# ============================

df = pd.read_csv(config.MODEL_RESULTS)

# Display dataframe
st.subheader("Performance Summary")

st.dataframe(df, use_container_width=True)

st.markdown("---")

# ============================
# Best Model
# ============================

best_model = df.loc[df["Micro F1"].idxmax()]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🏆 Best Model", best_model["Model"])

with col2:
    st.metric("Micro F1", f"{best_model['Micro F1']:.4f}")

with col3:
    st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")

st.markdown("---")

# ============================
# Accuracy Comparison
# ============================
st.subheader("Accuracy Comparison")

fig = px.bar(
    df,
    x="Model",
    y="Accuracy",
    color="Accuracy",
    text="Accuracy"
)

fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

st.plotly_chart(fig, use_container_width=True)

# ============================
# Micro F1 Score
# ============================
st.subheader("Micro F1 Score")

fig = px.bar(
    df,
    x="Model",
    y="Micro F1",
    color="Micro F1",
    text="Micro F1"
)

fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

st.plotly_chart(fig, use_container_width=True)

# ============================
# Precision vs Recall
# ============================
st.subheader("Precision vs Recall")

fig = px.scatter(
    df,
    x="Micro Precision",
    y="Micro Recall",
    size="Micro F1",
    color="Model",
    hover_name="Model"
)

st.plotly_chart(fig, use_container_width=True)

import plotly.graph_objects as go

# ============================
# Radar Comparison
# ============================
st.subheader("Radar Comparison")

metrics = ["Accuracy", "Micro Precision", "Micro Recall", "Micro F1"]

fig = go.Figure()

for _, row in df.iterrows():

    fig.add_trace(go.Scatterpolar(
        r=[
            row["Accuracy"],
            row["Micro Precision"],
            row["Micro Recall"],
            row["Micro F1"]
        ],
        theta=metrics,
        fill='toself',
        name=row["Model"]
    ))

st.plotly_chart(fig, use_container_width=True)
