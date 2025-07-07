import sqlite3
import joblib
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json
import os
from datetime import datetime
import logging
from typing import List, Tuple, Union, Optional
from pathlib import Path



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('label_tweets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


DB_PATH = "db/new_data.db"
ARCHIVE_PATH = "db/archive"
THRESHOLD = 0.5 
MODEL_PATHS = {
    'keras_model': "models/nn_sentiment_model.h5",
    'baseline_model': "models/baseline_model.pkl",
    'tokenizer': "models/nn_tokenizer.json"
}

def load_tokenizer() -> Optional[tokenizer_from_json]:
    """Load and return the Keras tokenizer from JSON file."""
    try:
        with open(MODEL_PATHS['tokenizer'], 'r', encoding='utf-8') as f:
            return tokenizer_from_json(f.read())
    except FileNotFoundError:
        logger.error(f"Tokenizer file not found at {MODEL_PATHS['tokenizer']}")
        return None
    except json.JSONDecodeError:
        logger.error("Failed to decode tokenizer JSON file")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading tokenizer: {str(e)}")
        return None

def keras_predict(texts: List[str], threshold: float = THRESHOLD) -> Optional[List[float]]:
    """Make predictions using the Keras model."""
    try:
        model = load_model(MODEL_PATHS['keras_model'])
        tokenizer = load_tokenizer()
        if tokenizer is None:
            return None
            
        seqs = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=200, padding='post', truncating='post')
        preds = model.predict(padded)
        return preds.flatten()
    except Exception as e:
        logger.error(f"Error in Keras prediction: {str(e)}")
        return None

def baseline_predict(texts: List[str]) -> Optional[List[float]]:
    """Make predictions using the baseline model."""
    try:
        model = joblib.load(MODEL_PATHS['baseline_model'])
        return model.predict_proba(texts)[:, 1]  
    except FileNotFoundError:
        logger.error(f"Baseline model file not found at {MODEL_PATHS['baseline_model']}")
        return None
    except Exception as e:
        logger.error(f"Error in baseline prediction: {str(e)}")
        return None

def get_db_connection() -> Optional[sqlite3.Connection]:
    """Create and return a database connection."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        return None

def ensure_table_exists(conn: sqlite3.Connection) -> bool:
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    baseline_label TEXT,
                    keras_label TEXT,
                    baseline_prob REAL,
                    keras_prob REAL,
                    last_updated TIMESTAMP
                )
            """)

        
            cursor = conn.execute("PRAGMA table_info(tweets)")
            columns = [col[1] for col in cursor.fetchall()]

            if "baseline_prob" not in columns:
                conn.execute("ALTER TABLE tweets ADD COLUMN baseline_prob REAL")
            if "keras_prob" not in columns:
                conn.execute("ALTER TABLE tweets ADD COLUMN keras_prob REAL")
            if "last_updated" not in columns:
                conn.execute("ALTER TABLE tweets ADD COLUMN last_updated TIMESTAMP")

        return True
    except sqlite3.Error as e:
        logger.error(f"Table creation error: {str(e)}")
        return False


def label_tweets(threshold: float = THRESHOLD) -> None:
    """Main function to label unlabeled tweets."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        if not ensure_table_exists(conn):
            return

        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, text FROM tweets WHERE baseline_label IS NULL OR keras_label IS NULL"
            )
            rows = cursor.fetchall()

            if not rows:
                logger.info("No new tweets to label.")
                return

            ids, texts = zip(*rows)
            texts_list = list(texts)

            baseline_preds = baseline_predict(texts_list)
            keras_preds = keras_predict(texts_list)

            if baseline_preds is None or keras_preds is None:
                logger.error("Prediction failed - aborting update")
                return

            for i, tweet_id in enumerate(ids):
                baseline_label = "Positive" if baseline_preds[i] >= threshold else "Negative"
                keras_label = "Positive" if keras_preds[i] >= threshold else "Negative"
                
                cursor.execute(
                    """UPDATE tweets 
                    SET baseline_label=?, keras_label=?, 
                        baseline_prob=?, keras_prob=?,
                        last_updated=CURRENT_TIMESTAMP
                    WHERE id=?""",
                    (baseline_label, keras_label, baseline_preds[i], keras_preds[i], tweet_id)
                )

            logger.info(f"Successfully labeled {len(rows)} tweets.")

    except sqlite3.Error as e:
        logger.error(f"Database error during labeling: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during labeling: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":

    label_tweets(threshold=THRESHOLD)
