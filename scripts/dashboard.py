import os 
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import json
from wordcloud import WordCloud
from lime.lime_text import LimeTextExplainer
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard")

DB_PATH = "db/experiments.db"
REQUEST_LOG_DB = "db/request_logs.db"

def load_tokenizer():
    with open("models/nn_tokenizer.json") as f:
      tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

def explain_baseline(text):
    model = joblib.load("models/baseline_model.pkl")
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    return explainer.explain_instance(text, model.predict_proba, num_features=6).as_html()

def explain_keras(text):
    tokenizer = load_tokenizer()
    model = load_model("models/nn_sentiment_model.h5")
    
    def keras_predict(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=200, padding='post', truncating='post')
        preds = model.predict(padded)
        return np.hstack([1 - preds, preds])  

    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    return explainer.explain_instance(text, keras_predict, num_features=6).as_html()

st.subheader("LIME Explanation")
user_input = st.text_input("Enter a review to explain")
model_type = st.radio("Select model:", ["Baseline", "Keras Neural Network"], horizontal=True)

if user_input:
    with st.spinner("Generating explanation..."):
        if model_type == "Baseline":
            html_explanation = explain_baseline(user_input)
        else:
            html_explanation = explain_keras(user_input)

        st.components.v1.html(
            f"""
            <div style="background-color:white; padding: 10px; border-radius: 10px;">
                {html_explanation}
            </div>
            """,
            height=600,
            scrolling=True
        )

   

@st.cache_data
def load_experiment_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM model_results", conn)
    conn.close()
    return df

@st.cache_data
def load_request_data():
    if not os.path.exists(REQUEST_LOG_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(REQUEST_LOG_DB)
    df = pd.read_sql("SELECT * FROM prediction_logs", conn)
    conn.close()
    return df

experiments_df = load_experiment_data()
requests_df = load_request_data()

st.subheader("Model Performance History")

if experiments_df.empty:
    st.info("No experiments found. Train and log a model to view performance.")
else:
    experiments_df["trained_at"] = pd.to_datetime(experiments_df["trained_at"])
    experiments_df = experiments_df.sort_values("trained_at")

    fig, ax = plt.subplots(figsize=(10, 4))
    for model in experiments_df["model_type"].unique():
        data = experiments_df[experiments_df["model_type"] == model]
        ax.plot(data["trained_at"], data["accuracy"], marker='o', label=model)
    ax.set_title("Accuracy Over Time")
    ax.set_xlabel("Training Date")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(experiments_df[["model_type", "accuracy", "train_time", "trained_at"]].sort_values("trained_at", ascending=False))

st.subheader("Incoming Predictions")
if requests_df.empty:
    st.info("No prediction requests logged yet.")
else:
    requests_df["timestamp"] = pd.to_datetime(requests_df["timestamp"])
    requests_df = requests_df.sort_values("timestamp")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Requests", len(requests_df))
    with col2:
        counts = requests_df["predicted_label"].value_counts()
        st.metric("Positive / Negative", f"{counts.get('Positive', 0)} / {counts.get('Negative', 0)}")

    fig2, ax2 = plt.subplots()
    sns.countplot(data=requests_df, x="predicted_label", palette="pastel", ax=ax2)
    ax2.set_title("Prediction Class Distribution")
    st.pyplot(fig2)

    all_text = " ".join(requests_df["input_text"].dropna().astype(str).tolist())
    if all_text:
        st.subheader("☁️ Word Cloud from Recent Text Inputs")
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(all_text)
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis("off")
        st.pyplot(fig3)

st.markdown("---")
st.caption("This dashboard visualizes logged experiments and live predictions. Connect your API to the same database to keep this updated.")
