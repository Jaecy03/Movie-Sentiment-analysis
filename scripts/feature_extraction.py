import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from functools import wraps
import time
import os
import joblib
from math import log

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
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression())
    ])

    
    pipeline.fit(X_train, y_train)

 
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")


    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, 'models/sentiment_pipeline.pkl')
    print("Pipeline saved to models/sentiment_pipeline.pkl")

    return pipeline

def compute_tf(text):
    words = text.split()
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    return {k: v / len(words) for k, v in tf.items()}

def compute_idf(corpus):
    N = len(corpus)
    idf = {}
    for doc in corpus:
        for word in set(doc.split()):
            idf[word] = idf.get(word, 0) + 1
    return {k: log(N / v) for k, v in idf.items()}

def compute_tfidf(tf, idf):
    return {k: tf.get(k, 0) * idf.get(k, 0) for k in tf}

@log_time
def run_custom_tfidf():
    print("\nFrom-scratch TF-IDF on first 5 samples:")
subset = df['clean_text'].head(5).tolist()
idf = compute_idf(subset)
for i, text in enumerate(subset):
    tf = compute_tf(text)
    tfidf = compute_tfidf(tf, idf)
    top_words = sorted(tfidf.items(), key=lambda x: -x[1])[:5]
    print(f"Top words for doc {i+1}:", top_words)
     

if __name__ == "__main__":
    train_model(X, y)
    run_custom_tfidf()
    