# Movie Sentiment Analysis Pipeline 

This project implements a **complete end-to-end sentiment analysis pipeline** using movie reviews from the IMDB dataset. The pipeline covers data ingestion, preprocessing, feature extraction, model training, evaluation, serving, automation, and advanced NLP with transformers.

## Features

- **Multiple Model Types**: Logistic Regression, Neural Networks, Fine-tuned DistilBERT
- **REST API Serving**: Flask APIs with JSON endpoints
- **Real-time Dashboard**: Streamlit dashboard for monitoring and visualization
- **Automated Data Collection**: Twitter API integration with cron jobs
- **Automated Retraining**: Pipeline triggers retraining when new data is available
- **Model Explainability**: LIME integration for prediction explanations
- **Comprehensive Testing**: Unit tests for preprocessing and API endpoints
- **Advanced NLP**: Hugging Face Transformers integration

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

### Step 4: Model Training & Evaluation

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

---

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

---

#### Comparison, Inference & SQLite Logging
- **Script**: `scripts/evaluate_models.py`
- **Functions**:
  - Loads metadata for both models
  - Prints side-by-side metrics and hyperparameters
  - Loads both models and runs **sample inference** on new review text
  - Saves experiment metadata into SQLite:
    - `db/experiments.db → model_results` table

---

### Outputs from Step 4
- `models/`: `.pkl`, `.h5`, metadata `.json`
- `plots/`: confusion matrix and NN training curve
- `db/experiments.db`: experiment logs (accuracy, hyperparameters, training time)
- `data/grid_search_results.csv`: tuning metrics (if tuning was run)
-  Sample inference works directly from `evaluate_models.py`


## 5. API Serving 

### Flask REST APIs
- **Baseline API**: `scripts/api_baseline.py` - Serves logistic regression model
- **Neural Network API**: `scripts/api_nn.py` - Serves Keras model
- **Transformer API**: `scripts/predict_transformers.py` - Serves DistilBERT model

### API Endpoints
```bash
# Health check
GET /health

# Predict sentiment
POST /predict
Content-Type: application/json
{
  "text": "This movie was absolutely fantastic!"
}

# Response
{
  "sentiment": "positive",
  "confidence": 0.89,
  "model_version": "baseline_v1.0"
}
```

### Running the APIs Locally
```bash
# Start baseline model API (port 5000)
python scripts/api_baseline.py

# Start neural network API (port 5001)
python scripts/api_nn.py

# Test with curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

## 6. Automated Data Collection & Retraining

### Twitter Data Collection
- **Script**: `scripts/fetch_data.py`
- **Automation**: `fetch_tweets.sh` (cron job)
- **Features**:
  - Fetches daily tweets using Twitter API
  - Stores in `db/new_data.db`
  - Configurable search terms and limits

### Automated Labeling
- **Script**: `scripts/label_tweets.py`
- **Automation**: `label_tweets.sh` (cron job)
- **Features**:
  - Labels new tweets using existing models
  - Stores predictions in database
  - Tracks confidence scores

### Automated Retraining
- **Script**: `scripts/retrain_if_needed.py`
- **Features**:
  - Monitors new labeled data volume
  - Triggers retraining when threshold is met
  - Updates model versions automatically
  - Maintains model performance logs

### Cron Job Setup
```bash
# Add to crontab (crontab -e)
0 9 * * * /path/to/fetch_tweets.sh
0 10 * * * /path/to/label_tweets.sh
0 11 * * * cd /path/to/project && python scripts/retrain_if_needed.py
```

## 7. Dashboard & Monitoring 

### Streamlit Dashboard
- **Script**: `scripts/dashboard.py`
- **Features**:
  - Real-time model performance monitoring
  - Class distribution visualization
  - Request volume and response time analytics
  - Model comparison metrics
  - Interactive prediction interface

### Dashboard Components
- **Performance Metrics**: Accuracy, precision, recall over time
- **Data Distribution**: Incoming request sentiment distribution
- **Model Comparison**: Side-by-side model performance
- **Live Predictions**: Interactive text input for real-time predictions
- **System Health**: API status and response times

### Running the Dashboard
```bash
streamlit run scripts/dashboard.py
# Access at http://localhost:8501
```

## 8. Model Explainability 

### LIME Integration
- **Scripts**:
  - `scripts/explain_prediction.py` - Command-line explanations
  - `scripts/lime_explainer.ipynb` - Interactive Jupyter notebook
  - `scripts/nn_lime_explainer.ipynb` - Neural network explanations

### Features
- **Word-level explanations**: Highlights important words for predictions
- **Model-agnostic**: Works with all model types (baseline, NN, transformer)
- **Interactive visualizations**: HTML outputs with highlighted text
- **Dashboard integration**: Explanations available in Streamlit dashboard

### Usage Examples
```bash
# Explain a prediction
python scripts/explain_prediction.py --text "This movie was terrible" --model baseline

# Generate explanation HTML
python scripts/explain_prediction.py --text "Amazing film!" --model nn --output explanation.html
```

## 9. Advanced NLP with Transformers

### DistilBERT Fine-tuning
- **Script**: `scripts/train_transformer.py`
- **Model**: Fine-tuned DistilBERT on IMDB sentiment dataset
- **Features**:
  - Hugging Face Transformers integration
  - Custom training loop with validation
  - Model checkpointing and saving
  - Performance evaluation and comparison

### Transformer Pipeline
- **Preprocessing**: Tokenization with DistilBERT tokenizer
- **Training**: Fine-tuning with custom classification head
- **Evaluation**: Comprehensive metrics and comparison with baseline models
- **Serving**: Integration with API endpoints

### Usage
```bash
# Train DistilBERT model
python scripts/train_transformer.py

# Use transformer for predictions
python scripts/predict_transformers.py --text "This movie is incredible!"
```

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 84% | 0.84 | 0.84 | 0.84 |
| Neural Network | 86% | 0.86 | 0.86 | 0.86 |
| DistilBERT | 92% | 0.92 | 0.91 | 0.92 |

## 10. Testing & Quality Assurance

### Unit Tests
- **Location**: `tests/`
- **Coverage**:
  - Preprocessing functions (`test_preprocessing.py`)
  - API endpoints (`test_api.py`)
  - Model predictions (`test_models.py`)

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=scripts

# Run specific test file
python -m pytest tests/test_api.py -v
```

### Test Features
- **API Testing**: Validates endpoint responses and error handling
- **Data Processing**: Tests preprocessing pipeline integrity
- **Model Loading**: Ensures models load and predict correctly
- **Integration Tests**: End-to-end pipeline validation

## Complete Project Structure

```
sentimentpipeline/
├── data/
│   ├── IMDB Dataset.csv              # Original dataset
│   └── cleaned_reviews.csv           # Preprocessed data
├── db/
│   ├── reviews.db                    # Main database
│   ├── experiments.db                # Model experiment logs
│   ├── new_data.db                   # Collected Twitter data
│   ├── predictions.db                # Prediction logs
│   └── requests.db                   # API request logs
├── models/
│   ├── baseline_model.pkl            # Logistic regression model
│   ├── baseline_model_metadata.json # Model metadata
│   ├── nn_sentiment_model.h5         # Neural network model
│   ├── nn_model_metadata.json       # NN metadata
│   ├── nn_tokenizer.json            # NN tokenizer
│   ├── sentiment_pipeline.pkl        # Original pipeline
│   └── distilbert_sentiment_model/   # Fine-tuned DistilBERT
│       ├── config.json
│       ├── tokenizer.json
│       └── vocab.txt
├── plots/
│   ├── class_distribution.png        # Data distribution plots
│   ├── confusion_matrix.png          # Model evaluation plots
│   ├── nn_accuracy.png              # Training curves
│   └── text_length_distribution.png  # Data analysis plots
├── scripts/
│   ├── ingest.py                     # Data ingestion
│   ├── preprocess.py                 # Data preprocessing
│   ├── explore.py                    # Data exploration
│   ├── feature_extraction.py         # Feature engineering
│   ├── train_and_evaluate.py         # Model training
│   ├── nn_model.py                   # Neural network training
│   ├── evaluate_models.py            # Model comparison
│   ├── train_transformer.py          # Transformer fine-tuning
│   ├── api_baseline.py               # Baseline model API
│   ├── api_nn.py                     # Neural network API
│   ├── predict_transformers.py       # Transformer predictions
│   ├── dashboard.py                  # Streamlit dashboard
│   ├── fetch_data.py                 # Twitter data collection
│   ├── label_tweets.py               # Automated labeling
│   ├── retrain_if_needed.py          # Automated retraining
│   ├── explain_prediction.py         # LIME explanations
│   ├── lime_explainer.ipynb          # Interactive explanations
│   └── nn_lime_explainer.ipynb       # NN explanations
├── tests/
│   ├── test_api.py                   # API endpoint tests
│   ├── test_preprocessing.py         # Data processing tests
│   └── test_models.py                # Model functionality tests
├── fetch_tweets.sh                   # Cron job script
├── label_tweets.sh                   # Labeling automation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/Jaecy03/Movie-Sentiment-analysis.git
cd Movie-Sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# 1. Ingest data
python scripts/ingest.py

# 2. Preprocess data
python scripts/preprocess.py

# 3. Train models
python scripts/train_and_evaluate.py
python scripts/nn_model.py

# 4. Start API server
python scripts/api_baseline.py &
python scripts/api_nn.py &

# 5. Launch dashboard
streamlit run scripts/dashboard.py
```

### 3. Test the System
```bash
# Run tests
python -m pytest tests/

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!"}'
```

## Model Performance

| Model | Accuracy | Training Time | Model Size | Use Case |
|-------|----------|---------------|------------|----------|
| Logistic Regression | 84% | 2 min | 1.2 MB | Fast inference, baseline |
| Neural Network | 86% | 15 min | 3.7 MB | Balanced performance |
| DistilBERT | 92% | 2 hours | 255 MB* | Highest accuracy |

*Note: Large DistilBERT model files are excluded from repository due to GitHub size limits. Model can be retrained using the provided scripts.

## Configuration

### Environment Variables
```bash
export TWITTER_API_KEY="your_api_key"
export TWITTER_API_SECRET="your_api_secret"
export MODEL_VERSION="v1.0"
export LOG_LEVEL="INFO"
```

### API Configuration
- **Baseline API**: Port 5000
- **Neural Network API**: Port 5001
- **Dashboard**: Port 8501
- **Request Timeout**: 30 seconds
- **Max Request Size**: 1MB

## Monitoring & Logging

- **API Requests**: Logged to `db/requests.db`
- **Predictions**: Stored in `db/predictions.db`
- **Model Performance**: Tracked in `db/experiments.db`
- **System Logs**: Available in dashboard and log files
- **Cron Job Logs**: `cron_log.txt`, `label_log.txt`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- IMDB Dataset from Kaggle
- Hugging Face Transformers library
- LIME explainability framework
- Streamlit for dashboard framework

