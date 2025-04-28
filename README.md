# Market Sentiment-Based Decision Making (MSBD)

## Project Overview
MSBD is a sophisticated data pipeline that analyzes social media sentiment to predict stock market movements. The system combines real-time Reddit data collection, natural language processing for sentiment analysis, and machine learning to generate trading insights.

### Key Features
- **Reddit Data Collection**: Automatically scrapes and processes posts from finance-related subreddits (e.g., r/wallstreetbets, r/stocks, r/investing) to identify trending stock tickers and associated sentiment.
- **Sentiment Analysis**: Employs VADER sentiment analysis to evaluate the emotional tone of social media discussions about specific stocks.
- **Technical Analysis**: Integrates stock price data (OHLCV) and calculates technical indicators (RSI, moving averages, etc.) to enrich the feature set.
- **Machine Learning Pipeline**: 
  - Combines sentiment scores, technical indicators, and engagement metrics
  - Performs time-series cross-validation
  - Trains ticker-specific prediction models (XGBoost/Random Forest)
  - Generates performance visualizations and metrics
- **Real-time Processing**: Updates predictions daily with fresh Reddit data and market information
- **Performance Tracking**: Maintains detailed logs of model performance and generates visualizations for trend analysis

### Workflow
1. Daily collection of Reddit posts and comments from financial subreddits
2. Extraction and validation of stock ticker mentions
3. Sentiment scoring of relevant posts and comments
4. Integration with historical price data and technical indicators
5. Feature engineering and model training/updating
6. Generation of trading signals and performance metrics

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

## Dependencies

### Data Collection & Processing
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- praw: Reddit API wrapper
- yfinance: Yahoo Finance API wrapper
- requests: HTTP library for API calls
- beautifulsoup4: Web scraping and HTML parsing

### Machine Learning & Modeling
- scikit-learn: Machine learning tools
- xgboost: Gradient boosting framework
- joblib: Model persistence

### Natural Language Processing
- vaderSentiment: Sentiment analysis
- transformers: Hugging Face transformers library
- torch: PyTorch deep learning framework

### Visualization
- matplotlib: Basic plotting library
- seaborn: Statistical data visualization
- colorama: Terminal text formatting

### Utilities
- tqdm: Progress bar
- pathlib: Object-oriented filesystem paths
- logging: Logging facility
- typing: Type hints support
- datetime: Basic date and time types
- json: JSON encoder and decoder
- re: Regular expression operations

### Development Tools
- pytest: Testing framework
- black: Code formatting
- flake8: Style guide enforcement
- mypy: Static type checking

## Key Components
- Data Collection: Yahoo Finance API, Reddit APIs
- Sentiment Analysis: VADER sentiment analysis
- Machine Learning: Stock movement prediction
- Evaluation: Performance metrics and visualization

## Development Timeline
1. Data Collection & Preprocessing (Weeks 2-3)
2. Model Development & Testing (Weeks 4-6)
3. Evaluation & Documentation (Weeks 7-8)
```

