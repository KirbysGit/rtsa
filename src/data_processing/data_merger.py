# src / data_processing / data_merger.py

# Description : This file contains the DataMerger class, which is used to merge the processed stock and Reddit data.

# Imports.
import logging
import pandas as pd
from datetime import datetime, timedelta

# Local Imports.
from .stock_data_processor import StockDataProcessor
from .reddit_data_processor import RedditDataProcessor

# Setup Logging.
logger = logging.getLogger(__name__)

# Data Merger Class.
class DataMerger:

    def __init__(self):
        """Initialize the Data Merger with processors."""
        self.stock_processor = StockDataProcessor()
        self.reddit_processor = RedditDataProcessor()
    
    def merge_data(self, raw_stock_data, raw_reddit_data, ticker):
        """
        Process and merge stock and Reddit data
        
        Args:
            raw_stock_data (pd.DataFrame): Raw stock data
            raw_reddit_data (pd.DataFrame): Raw Reddit data
            ticker (str): Stock ticker symbol
        """
        try:
            # First Process the Individual Datasets.
            logger.info(f"Processing Stock Data for {ticker}")
            stock_data = self.stock_processor.process_stock_data(raw_stock_data)
            
            logger.info(f"Processing Reddit Data for {ticker}")
            reddit_data = self.reddit_processor.process_reddit_data(raw_reddit_data)
            
            if stock_data is None or reddit_data is None:
                raise ValueError("Failed to process either stock or Reddit data")

            # Convert Timestamps to Datetime if Needed.
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)
            
            # Convert Reddit Timestamps to Datetime if Needed.
            reddit_data['created_utc'] = pd.to_datetime(reddit_data['created_utc'])
            
            # Create Daily Sentiment Aggregations
            daily_sentiment = self._aggregate_daily_sentiment(reddit_data)
            
            # Merge Stock Data with Sentiment
            merged_data = pd.merge(
                stock_data,
                daily_sentiment,
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Fill Missing Sentiment with 0 (Days with No Reddit Posts)
            sentiment_cols = ['avg_sentiment', 'total_posts', 'total_comments', 'avg_engagement']
            merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(0)
            
            # Log Success and Data Stats
            logger.info(f"Successfully Merged Data for {ticker}")
            logger.info(f"Shape of Merged Data: {merged_data.shape}")
            logger.info(f"Date Range: {merged_data.index.min()} to {merged_data.index.max()}")
            logger.info(f"Total Trading Days: {len(merged_data)}")
            logger.info(f"Days with Reddit Activity: {(merged_data['total_posts'] > 0).sum()}")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error Merging Data: {str(e)}")
            return None
    
    def _aggregate_daily_sentiment(self, reddit_data):
        """Aggregate Reddit Sentiment Data to Daily Values."""
        
        daily_sentiment = reddit_data.groupby(
            reddit_data['created_utc'].dt.date
        ).agg({
            'overall_sentiment': 'mean',
            'id': 'count',
            'num_comments': 'sum',
            'engagement_score': 'mean'
        }).rename(columns={
            'overall_sentiment': 'avg_sentiment',
            'id': 'total_posts',
            'num_comments': 'total_comments',
            'engagement_score': 'avg_engagement'
        })
        
        # Convert Index to Datetime
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        return daily_sentiment 