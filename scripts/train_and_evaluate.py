import pandas as pd
import joblib
import os
import time
import json
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from functools import wraps
from datetime import datetime



df = pd.read_csv('data/cleaned_reviews.csv')
X = df['clean_text']
y = df['label']


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[START] {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[END] {func.__name__} - Took {time.time() - start:.2f}s")
        return result
    return wrapper

@log_time
def train_and_evaluate(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, 'models/baseline_model.pkl')
    print("Baseline model saved to models/baseline_model.pkl")

    
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_type": "Logistic Regression",
        "test_accuracy": accuracy,
        "dataset_hash": hashlib.md5(df.to_csv(index=False).encode()).hexdigest(),
        "hyperparameters": {
            "ngram_range": (1, 2),
            "max_features": 5000,
            "C": 1.0
        }
    }

    with open("models/baseline_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Baseline metadata saved to models/baseline_model_metadata.json")
  
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confusion_matrix.png")
    plt.show()

@log_time
def tune_hyperparameters(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LogisticRegression())
    ])

    param_grid = {
        'tfidf__max_features': [1000, 3000, 5000],
        'clf__C': [0.1, 1, 10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print(f"Validation Accuracy: {grid.best_score_:.4f}")

    
    joblib.dump(grid.best_estimator_, 'models/grid_tuned_model.pkl')

    
    results = pd.DataFrame(grid.cv_results_)
    os.makedirs("data", exist_ok=True)
    results.to_csv("data/grid_search_results.csv", index=False)
    print("Grid search results saved to data/grid_search_results.csv")

if __name__ == "__main__":
   
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_and_evaluate(X_train, X_test, y_train, y_test)
   

