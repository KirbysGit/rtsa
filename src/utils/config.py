# src/utils/config.py

# Imports.
import os
from datetime import datetime, timedelta

# Stock Settings.
STOCK_SETTINGS = {
    'default_period': '1y',
    'default_interval': '1d',
    'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Example tickers
}

# Twitter Settings.
TWITTER_SETTINGS = {
    'max_results': 100,
    'search_queries': [
        'AAPL stock',
        'Apple stock',
        'GOOGL stock',
        'Google stock'
    ]
}

# Directory Settings.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Create Directories if They Don't Exist.
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Credentials.
TWITTER_BEARER_TOKEN = "your_bearer_token" 