# Market Sentiment-Based Decision Making (MSBD)

## Project Overview
A research project analyzing market sentiment from social media to predict stock market movements.

## Directory Structure
project/
|-- data/                   # Raw & processed datasets
|   |-- raw/                # Unprocessed stock & social media data
|   |-- processed/          # Cleaned & formatted data
|-- src/                    # Source code for data processing & analysis
|   |-- data_collection/    # Scripts for fetching data
|   |-- preprocessing/      # Data cleaning & feature engineering 
|   |-- sentiment_analysis/ # NLP models for social media sentiment
|   |-- prediction_model/   # Machine learning model for predictions
|   |-- utils/             # Helper functions, config settings
|-- notebooks/             # Jupyter Notebooks for testing & analysis
|-- models/                # Saved ML models
|-- results/               # Model evaluation results, charts, logs
|-- scripts/               # Execution scripts
|-- requirements.txt       # Dependencies

## Setup Instructions
1. Create virtual environment:
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Key Components
- Data Collection: Yahoo Finance API, Twitter/Reddit APIs
- Sentiment Analysis: VADER sentiment analysis
- Machine Learning: Stock movement prediction
- Evaluation: Performance metrics and visualization

## Development Timeline
1. Data Collection & Preprocessing (Weeks 2-3)
2. Model Development & Testing (Weeks 4-6)
3. Evaluation & Documentation (Weeks 7-8)

## Dependencies
- pandas
- numpy
- scikit-learn
- yfinance
- tweepy
- vaderSentiment
- transformers
- torch
- matplotlib
- seaborn

Refresh API : curl -X POST https://www.reddit.com/api/v1/access_token \
     -u "MmoK5DoETZThmmUPZLATLg:rp4T-ejrwoww6_0MOPCGIfAg0HqdlA" \
     -d "grant_type=refresh_token" \
     -d "refresh_token=158310501943578-JQZsKWFFrBTE7RsU0d__CH3WW7TPNQ"

