from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import uuid
import time
import json
import sqlite3
from datetime import datetime
import os

with open("models/nn_model_metadata.json") as f:
    metadata = json.load(f)

tokenizer = tokenizer_from_json(metadata["tokenizer_config"])
model = load_model("models/nn_sentiment_model.h5")

app = Flask(__name__)
print("Keras NN model loaded.")

os.makedirs("db", exist_ok=True)
conn = sqlite3.connect("db/requests.db")
conn.execute("""
CREATE TABLE IF NOT EXISTS nn_requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    input_text TEXT,
    predicted_label TEXT,
    confidence REAL
)
""")
conn.commit()
conn.close()

def log_request(func):
    def wrapper(*args, **kwargs):
        print(f"[REQUEST] {datetime.now().isoformat()} - {request.method} {request.path}")
        return func(*args, **kwargs)
    return wrapper

@app.route("/")
def home():
    return "🎉 Neural Network Sentiment API is live!"

@app.route("/predict", methods=["POST"])
@log_request
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request."}), 400

    text = data["text"]
    if not text.strip():
        return jsonify({"error": "Input text is empty."}), 400

    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    confidence = float(model.predict(padded)[0][0])
    label = "Positive" if confidence > 0.5 else "Negative"

    conn = sqlite3.connect("db/requests.db")
    conn.execute("""
    INSERT INTO nn_requests (id, timestamp, input_text, predicted_label, confidence)
    VALUES (?, ?, ?, ?, ?)
    """, (request_id, timestamp, text, label, confidence))
    conn.commit()
    conn.close()

    return jsonify({
        "request_id": request_id,
        "timestamp": timestamp,
        "input_text": text,
        "prediction": label,
        "confidence": round(confidence, 4)
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
