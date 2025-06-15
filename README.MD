# Movie Sentiment Analysis  
This project implements an end-to-end sentiment analysis pipeline using movie reviews from the IMDB dataset. The pipeline covers data ingestion, preprocessing, feature extraction, model training, and evaluation.

### 1. Ingestion
## Dataset
- IMDB Movie Reviews dataset from Kaggle
- 50,000 movie reviews labeled as `positive` or `negative`

## How to Run
```bash
pip install -r requirements.txt
python scripts/ingest.py

## What This Script Does
- Reads `IMDB Dataset.csv` using `pandas`
- Maps sentiment labels to 1 (positive) / 0 (negative)
- Creates `reviews` table in `reviews.db`
- Inserts rows into the table
- Logs function execution time using `@log_time` decorator


### 2. Data Exploration & Preprocessing
- **Script**: `scripts/preprocess.py`
    - Loads data from the database
    - Explores:
        - Class distribution
        - Text length distribution
        - Common words (top-N)
        - Vocabulary size
        - Word coun
        - Most common words
        - Missing and empty entries checkt
    - Preprocessing steps:
        - Lowercasing
        - Punctuation removal
        - Stopword removal (using NLTK)
        - Handling nulls
    - All steps use modular functions and logging decorators
    - Saves preprocessed output to: `data/cleaned_reviews.csv`

### 3. Feature Extraction & Model Training
- **Script**: `scripts/feature_extraction.py`
    - Uses `TfidfVectorizer` with unigrams & bigrams
    - Trains a `LogisticRegression` model
    - Wraps training in a `Pipeline` for clean structure
    - Saves model as `models/sentiment_pipeline.pkl`
    - Accuracy on test data: **~84%**
    - Also includes a custom TF-IDF implementation (for 5 samples)

## Project Structure
- sentimentpipeline/
- data/ Contains IMDB CSV, cleaned_reviews.csv 
- db/ SQLite database (reviews.db)
- models/ sentiment_pipeline.pkl
- plots/ class_distribution.png, text_length_distribution.png
- scripts/ explore.py, feature_extraction.py, ingest.py, preprocess.py
- requirements.txt
- README.md


