import pandas as pd
import sqlite3
from functools import wraps
import time
from tqdm import tqdm

def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[START] {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[END] {func.__name__} - Took {end - start:.2f}s")
        return result
    return wrapper

def connect_db(db_path='db/reviews.db'):
    return sqlite3.connect(db_path)

@log_time
def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label INTEGER NOT NULL
        )
    ''')
    conn.commit()

@log_time
def ingest_data(conn, csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    cursor = conn.cursor()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cursor.execute(
            "INSERT INTO reviews (text, label) VALUES (?, ?)",
            (row['review'], row['label'])
        )
    conn.commit()

@log_time
def main():
    conn = connect_db()
    create_table(conn)
    ingest_data(conn, 'data/IMDB Dataset.csv')
    conn.close()

if __name__ == "__main__":
    main()
