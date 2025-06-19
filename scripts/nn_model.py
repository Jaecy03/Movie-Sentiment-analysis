import pandas as pd
import numpy as np
import os
import time
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import hashlib
import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_reviews.csv')
X = df['clean_text'].astype(str)
y = df['label']

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=200, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=200),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2,
                    callbacks=[EarlyStopping(patience=2)], batch_size=32)
end = time.time()

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Classification Report:\n", classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Validation")
plt.title("Loss")
plt.legend()

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/nn_accuracy.png")
plt.show()

os.makedirs("models", exist_ok=True)
model.save("models/nn_sentiment_model.h5")
print("Model saved to models/nn_sentiment_model.h5")

metadata = {
    "trained_at": datetime.now().isoformat(),
    "test_accuracy": acc,
    "dataset_hash": hashlib.md5(df.to_csv(index=False).encode()).hexdigest(),
    "model_type": "Keras Sequential",
    "tokenizer_config": tokenizer.to_json(),
    "hyperparameters": {
        "vocab_size": 5000,
        "embedding_dim": 64,
        "maxlen": 200,
        "epochs": len(history.history['loss']),
        "batch_size": 32
    },
    "train_time_secs": round(end - start, 2)
}

with open("models/nn_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Metadata saved to models/nn_model_metadata.json")

