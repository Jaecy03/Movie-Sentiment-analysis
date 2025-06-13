# Sentiment Analysis Pipeline – Step 1: Ingestion

## Dataset
- IMDB Movie Reviews dataset from Kaggle
- 50,000 movie reviews labeled as `positive` or `negative`

## Project Structure
- sentimentpipeline/
- data/ # Contains IMDB CSV
- db/ # SQLite database (reviews.db)
- scripts/ # Ingestion script (ingest.py)
- requirements.txt
- README.md

## What This Script Does
- Reads `IMDB Dataset.csv` using `pandas`
- Maps sentiment labels to 1 (positive) / 0 (negative)
- Creates `reviews` table in `reviews.db`
- Inserts rows into the table
- Logs function execution time using `@log_time` decorator

## How to Run
```bash
pip install -r requirements.txt
python scripts/ingest.py
