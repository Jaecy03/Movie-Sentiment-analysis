import json
import sqlite3
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import joblib

with open("models/baseline_model_metadata.json") as f:
    baseline_meta = json.load(f)

with open("models/nn_model_metadata.json") as f:
    nn_meta = json.load(f)

print("\n Model Comparison")
print("-" * 50)
print(f"Baseline Model Accuracy: {baseline_meta['test_accuracy']:.4f}")
print(f"NN Model Accuracy:       {nn_meta['test_accuracy']:.4f}\n")

print(f"Baseline Training Time:  ~not logged~")
print(f"NN Training Time:        {nn_meta['train_time_secs']} seconds\n")

print("Hyperparameters:")
print(f"  Baseline: {baseline_meta['hyperparameters']}")
print(f"  NN Model: {nn_meta['hyperparameters']}")

if os.path.exists("plots/nn_accuracy.png"):
    print("\nAccuracy and Loss curves already saved in 'plots/' folder.")
else:
    print("\n⚠️ NN training plot not found. Run the NN training script to generate them.")

db_path = "db/experiments.db"
os.makedirs("db", exist_ok=True)
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT,
    accuracy REAL,
    train_time REAL,
    trained_at TEXT,
    dataset_hash TEXT,
    hyperparams TEXT
)
""")

cur.execute("""
INSERT INTO model_results (model_type, accuracy, train_time, trained_at, dataset_hash, hyperparams)
VALUES (?, ?, ?, ?, ?, ?)
""", (
    "Baseline Logistic Regression",
    baseline_meta['test_accuracy'],
    None,
    baseline_meta['trained_at'],
    baseline_meta['dataset_hash'],
    json.dumps(baseline_meta['hyperparameters'])
))

cur.execute("""
INSERT INTO model_results (model_type, accuracy, train_time, trained_at, dataset_hash, hyperparams)
VALUES (?, ?, ?, ?, ?, ?)
""", (
    "Neural Network (Keras)",
    nn_meta['test_accuracy'],
    nn_meta['train_time_secs'],
    nn_meta['trained_at'],
    nn_meta['dataset_hash'],
    json.dumps(nn_meta['hyperparameters'])
))

conn.commit()
conn.close()
print("\n Metadata exported to db/experiments.db")

print("\n Running sample inference for both models:")

def predict_sentiment_baseline(text):
    model = joblib.load("models/baseline_model.pkl")
    pred = model.predict([text])[0]
    return "Positive" if pred == 1 else "Negative"

print("Baseline Prediction:", predict_sentiment_baseline("The movie was absolutely fantastic!"))

tokenizer = tokenizer_from_json(nn_meta['tokenizer_config'])
nn_model = load_model("models/nn_sentiment_model.h5")

def predict_sentiment_nn(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    pred = nn_model.predict(padded)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

print("NN Model Prediction:", predict_sentiment_nn("The movie was absolutely fantastic!"))
