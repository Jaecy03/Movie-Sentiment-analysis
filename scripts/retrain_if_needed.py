import pandas as pd
import sqlite3
import os
from train_and_evaluate import train_model
from datetime import datetime

THRESHOLD = 500
DATA_PATH = "db/new_data.db"
ARCHIVE_DIR = "db/archive"

def archive_data(df: pd.DataFrame):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file = os.path.join(ARCHIVE_DIR, f"labeled_{timestamp}.csv")
    df.to_csv(archive_file, index=False)
    print(f"Archived {len(df)} labeled tweets to {archive_file}")

def delete_archived_from_db(conn: sqlite3.Connection, ids: list):
    if not ids:
        return
    placeholder = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM tweets WHERE id IN ({placeholder})", ids)
    conn.commit() 

def retrain_if_ready():
    conn = sqlite3.connect(DATA_PATH)
    df = pd.read_sql("SELECT * FROM tweets WHERE keras_label IS NOT NULL", conn)

    if len(df) >= THRESHOLD:
        print(f"Retraining triggered: {len(df)} new labeled entries")
        df = df.rename(columns={"keras_label": "label"})
        
        from sklearn.model_selection import train_test_split
        X = df['text']
        y = df['label']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        train_model(X_train, X_test, y_train, y_test)
        archive_data(df)
        delete_archived_from_db(conn, df['id'].tolist())
    else:
        print(f"Not enough new data yet: {len(df)} labeled entries")
        conn.close()
        print("Model retrained and saved! Baseline model updated.")

if __name__ == "__main__":
    retrain_if_ready()
