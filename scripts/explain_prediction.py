import joblib
import json
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

baseline_model: Pipeline = joblib.load("models/baseline_model.pkl")

class_names = ["Negative", "Positive"]

with open("models/nn_model_metadata.json") as f:
    nn_meta = json.load(f)

tokenizer = tokenizer_from_json(nn_meta['tokenizer_config'])
nn_model = load_model("models/nn_sentiment_model.h5")


def explain_baseline_prediction(text):
    print("\n[Baseline Explanation]")
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, baseline_model.predict_proba, num_features=6)
    exp.show_in_notebook(text=True)



def predict_proba_nn(texts):
    seq = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    preds = nn_model.predict(padded)
    return np.hstack((1 - preds, preds))

def explain_nn_prediction(text):
    print("\n[Neural Network Explanation]")
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba_nn, num_features=6)
    exp.show_in_notebook(text=True)

sample_text = "I absolutely loved this movie. The story and acting were great!"

explain_baseline_prediction(sample_text)
explain_nn_prediction(sample_text)
