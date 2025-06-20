# Movie Sentiment Analysis  
This project implements an end-to-end sentiment analysis pipeline using movie reviews from the IMDB dataset. The pipeline covers data ingestion, preprocessing, feature extraction, model training, and evaluation.

### 1. Ingestion
## Dataset
- IMDB Movie Reviews dataset from Kaggle
- 50,000 movie reviews labeled as `positive` or `negative`

## How to Run
bash
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

### 4. Model Training & Evaluation

#### Baseline Model (Logistic Regression)
- **Script**: `scripts/train_and_evaluate.py`
- **Pipeline**: `TfidfVectorizer(ngram_range=(1,2), max_features=5000)` → `LogisticRegression()`
- **Data Split**: 70% Train / 15% Validation / 15% Test
- **Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-score
  - Confusion Matrix: plotted and saved to `plots/confusion_matrix.png`
- **Model Saving**:
  - Pickled model: `models/baseline_model.pkl`
  - Metadata JSON: `models/baseline_model_metadata.json`  
    Includes: training date, dataset hash, hyperparameters, test accuracy
- **Tuning**: Hyperparameter tuning using `GridSearchCV`
  - Results saved to: `data/grid_search_results.csv`
  - Best model saved to: `models/grid_tuned_model.pkl`
- **Other**: Decorator used to log training time for tracking


#### Neural Network Model (Keras)
- **Script**: `scripts/train_nn_model.py`
- **Architecture**:  
  `Embedding → GlobalAveragePooling1D → Dense → Dense(sigmoid)`
- **Tokenization**: `Tokenizer(num_words=5000)` with padding to `maxlen=200`
- **Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-score
  - Training/validation accuracy and loss plotted and saved to `plots/nn_accuracy.png`
- **Model Saving**:
  - Keras `.h5` model: `models/nn_sentiment_model.h5`
  - Metadata JSON: `models/nn_model_metadata.json`  
    Includes: tokenizer config, training time, hyperparameters, dataset hash, test accuracy


#### Comparison, Inference & SQLite Logging
- **Script**: `scripts/evaluate_models.py`
- **Functions**:
  - Loads metadata for both models
  - Prints side-by-side metrics and hyperparameters
  - Loads both models and runs **sample inference** on new review text
  - Saves experiment metadata into SQLite:
    - `db/experiments.db → model_results` table


### Outputs from Step 4
- `models/`: `.pkl`, `.h5`, metadata `.json`
- `plots/`: confusion matrix and NN training curve
- `db/experiments.db`: experiment logs (accuracy, hyperparameters, training time)
- `data/grid_search_results.csv`: tuning metrics 
-  Sample inference works directly from `evaluate_models.py`


## Project Structure
- sentimentpipeline/
- data/ Contains IMDB CSV, cleaned_reviews.csv 
- db/ SQLite database (reviews.db), experiments.db
- models/ baseline_model.pkl, nn_model_metadata.json, nn_sentiment_model.h5, sentiment_pipeline.pkl
- plots/ confusion_matrix.png, nn_accuracy.png
- scripts/ evaluate_models.py, explore.py, feature_extraction.py, ingest.py, nn_model.py, preprocess.py, train_and_evaluate.py
- requirements.txt
- README.md


