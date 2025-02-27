import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stock_processor():
    """Test stock data processor with sample data"""
    from src.data_processing.stock_processor import StockDataProcessor
    from src.data_collection.finance_data import StockDataCollector
    
    # Initialize processors
    stock_processor = StockDataProcessor()
    stock_collector = StockDataCollector()
    
    # Test stock processing
    logger.info("Testing stock data processing...")
    stock_data = stock_collector.fetch_stock_data('NVDA', period='1mo')
    
    if stock_data is not None:
        processed_stock = stock_processor.process_stock_data(stock_data)
        logger.info("\nProcessed stock data columns:")
        logger.info(processed_stock.columns.tolist())
        
        logger.info("\nSample technical indicators:")
        indicators = ['Close', 'MA5', 'MA20', 'Returns', 'Volatility', 'Volume_Change', 'Price_Momentum']
        logger.info(processed_stock[indicators].tail())
        
        # Check for missing values
        missing = processed_stock.isnull().sum()
        if missing.any():
            logger.warning("\nMissing values found:")
            logger.warning(missing[missing > 0])
        else:
            logger.info("\nNo missing values found in processed data")
    else:
        logger.error("Failed to fetch stock data")

def test_reddit_processor():
    """Test Reddit data processor with sample data"""
    from src.data_processing.reddit_processor import RedditDataProcessor
    from src.data_collection.reddit_collector import RedditDataCollector
    
    # Initialize processors
    reddit_processor = RedditDataProcessor()
    reddit_collector = RedditDataCollector()
    
    # Test Reddit processing
    logger.info("Testing Reddit data processing...")
    subreddits = ['wallstreetbets', 'stocks']
    
    for subreddit in subreddits:
        logger.info(f"\nProcessing data from r/{subreddit}")
        reddit_data = reddit_collector.fetch_subreddit_posts(subreddit, limit=5)
        
        if reddit_data is not None:
            processed_reddit = reddit_processor.process_reddit_data(reddit_data)
            
            logger.info("Processed Reddit data columns:")
            logger.info(processed_reddit.columns.tolist())
            
            logger.info("\nSample processed data:")
            display_cols = ['title', 'cleaned_title', 'overall_sentiment', 
                          'engagement_score', 'title_sentiment', 'text_sentiment']
            logger.info(processed_reddit[display_cols].head(2))
            
            # Check for missing values
            missing = processed_reddit.isnull().sum()
            if missing.any():
                logger.warning("\nMissing values found:")
                logger.warning(missing[missing > 0])
            else:
                logger.info("\nNo missing values found in processed data")
        else:
            logger.error(f"Failed to fetch data from r/{subreddit}")

if __name__ == "__main__":
    test_stock_processor()
    print("\n" + "="*80 + "\n")
    test_reddit_processor() 