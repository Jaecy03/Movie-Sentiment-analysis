import pandas as pd
import sqlite3
import string
import time
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import wraps
from collections import Counter
import re

conn = sqlite3.connect('db/reviews.db')

df = pd.read_sql_query("SELECT * FROM reviews LIMIT 1000", conn)
conn.close()

print(df.head())

print("Class distribution:\n", df['label'].value_counts())

df['text_length'] = df['text'].apply(len)
print("\nText length stats:\n", df['text_length'].describe())

df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
print(df['word_count'].describe())

all_words = " ".join(df['text'].astype(str)).lower()
vocab = set(re.findall(r'\b\w+\b', all_words))
print("Vocabulary size:", len(vocab))

print("Missing text:", df['text'].isnull().sum())
print("Empty text:", (df['text'].str.strip() == '').sum())

def get_top_words(texts, n=20):
    words = " ".join(texts).lower()
    words = re.findall(r'\b\w+\b', words)
    return Counter(words).most_common(n)

print(get_top_words(df['text']))

stop_words = set(stopwords.words('english'))

def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[RUNNING] {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[DONE] {func.__name__} in {time.time() - start:.2f}s")
        return result
    return wrapper

@log_step
def lowercase(text):
    """Convert text to lowercase."""
    return text.lower()

@log_step
def remove_punctuation(text):
    """Remove punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

@log_step
def remove_stopwords(text):
    """Remove stopwords from text."""
    words = word_tokenize(text)
    return " ".join([word for word in words if word.lower() not in stop_words])


@log_step
def clean_text_column(df):
    """Clean the text column of the dataframe."""
    df = df.dropna(subset=['text']) 

    df['clean_text'] = (
        df['text']
        .apply(lowercase)
        .apply(remove_punctuation)
        .apply(remove_stopwords)
    )
    return df
if __name__ == "__main__":
    
    conn = sqlite3.connect('db/reviews.db')
    df = pd.read_sql_query("SELECT * FROM reviews LIMIT 1000", conn)
    conn.close()

    df_clean = clean_text_column(df)

    df_clean.to_csv('data/cleaned_reviews.csv', index=False)
    print("Cleaned data saved to data/cleaned_reviews.csv")
