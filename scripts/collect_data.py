# scripts/collect_data.py

# Imports.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Classes.
from src.data_collection.finance_data import StockDataCollector
from src.data_collection.social_media import SocialMediaCollector
from src.utils.config import (
    STOCK_SETTINGS,
    TWITTER_SETTINGS,
    TWITTER_BEARER_TOKEN
)

# Set Up Logging.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Main Function.
def main():
    # Initialize Collectors.
    stock_collector = StockDataCollector()
    social_collector = SocialMediaCollector()
    
    # Set Up Twitter Authentication.
    social_collector.setup_twitter_auth(TWITTER_BEARER_TOKEN)
    
    # Collect Stock Data.
    logger.info("Collecting stock data...")
    for ticker in STOCK_SETTINGS['tickers']:
        stock_collector.fetch_stock_data(
            ticker,
            period=STOCK_SETTINGS['default_period']
        )
    
    # Collect Social Media Data.
    logger.info("Collecting social media data...")
    for query in TWITTER_SETTINGS['search_queries']:
        social_collector.fetch_tweets(
            query,
            max_results=TWITTER_SETTINGS['max_results']
        )

if __name__ == "__main__":
    main() 