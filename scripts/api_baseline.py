from flask import Flask, request, jsonify
import joblib
import sqlite3
import os
from datetime import datetime
import uuid
from functools import wraps

model = joblib.load("models/baseline_model.pkl")
print("Baseline model loaded.")

app = Flask(__name__)

os.makedirs("db", exist_ok=True)
conn = sqlite3.connect("db/predictions.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS prediction_logs (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT,
    input_text TEXT,
    predicted_label TEXT,
    confidence REAL
)
""")
conn.commit()
conn.close()

def log_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[REQUEST] {datetime.now().isoformat()} - {request.method} {request.path}")
        return func(*args, **kwargs)
    return wrapper

@app.route("/", methods=["GET"])
@log_request
def home():
    return jsonify({"message": "Sentiment API is running."}), 200

@app.route("/predict", methods=["POST"])
@log_request
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON payload"}), 400

    text = data["text"]
    prob = model.predict_proba([text])[0]
    label = int(prob[1] >= 0.5)
    confidence = round(float(prob[1]), 4)

    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect("db/predictions.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_logs (request_id, timestamp, input_text, predicted_label, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (request_id, timestamp, text, "Positive" if label == 1 else "Negative", confidence))
    conn.commit()
    conn.close()

    return jsonify({
        "request_id": request_id,
        "timestamp": timestamp,
        "input_text": text,
        "prediction": "Positive" if label == 1 else "Negative",
        "confidence": confidence
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
