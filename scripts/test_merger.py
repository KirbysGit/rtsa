import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_merger():
    """Test the data merger with stock and Reddit data"""
    from src.data_collection.finance_data import StockDataCollector
    from src.data_collection.reddit_collector import RedditDataCollector
    from src.data_processing.stock_processor import StockDataProcessor
    from src.data_processing.reddit_data_processor import RedditDataProcessor
    from src.data_processing.data_merger import DataMerger
    
    # Initialize collectors and processors
    stock_collector = StockDataCollector()
    reddit_collector = RedditDataCollector()
    stock_processor = StockDataProcessor()
    reddit_processor = RedditDataProcessor()
    merger = DataMerger()
    
    # Test with NVIDIA (NVDA) data
    ticker = 'NVDA'
    
    # Collect and process stock data
    logger.info(f"Collecting stock data for {ticker}")
    stock_data = stock_collector.fetch_stock_data(ticker, period='1mo')
    if stock_data is not None:
        processed_stock = stock_processor.process_stock_data(stock_data)
        
        # Collect and process Reddit data
        logger.info(f"Collecting Reddit data for {ticker}")
        search_query = f"{ticker} stock"
        reddit_data = reddit_collector.fetch_subreddit_posts('wallstreetbets', limit=50)
        
        if reddit_data is not None:
            processed_reddit = reddit_processor.process_reddit_data(reddit_data)
            
            # Merge the data
            logger.info("Merging stock and Reddit data")
            merged_data = merger.merge_data(processed_stock, processed_reddit, ticker)
            
            if merged_data is not None:
                logger.info("\nMerged data columns:")
                logger.info(merged_data.columns.tolist())
                
                logger.info("\nSample merged data:")
                display_cols = ['Close', 'Returns', 'avg_sentiment', 
                              'total_posts', 'total_comments', 'avg_engagement']
                logger.info(merged_data[display_cols].tail())
                
                # Save merged data
                output_dir = "data/processed"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ticker}_merged_data.csv")
                merged_data.to_csv(output_file)
                logger.info(f"\nSaved merged data to {output_file}")
            else:
                logger.error("Failed to merge data")
        else:
            logger.error("Failed to collect Reddit data")
    else:
        logger.error("Failed to collect stock data")

if __name__ == "__main__":
    load_dotenv()
    test_data_merger() 