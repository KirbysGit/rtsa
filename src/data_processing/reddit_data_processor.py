# src / data_processing / reddit_processor.py

# Description : This file contains the RedditDataProcessor class, which is used to process the Reddit data.

# Imports.
import re
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.path_config import RAW_DIR, PROCESSED_DIR

# Setup Logging.
logger = logging.getLogger(__name__)

# Constants
TICKERS = ['NVDA', 'NVIDIA', 'AMD', 'INTC', 'TSMC']  # Add more as needed
MAX_TEXT_LENGTH = 500  # Maximum words in text
SUMMARY_LENGTH = 100  # Words in summary

# Subreddit to ticker mapping
SUBREDDIT_TICKERS = {
    'nvidia': 'NVDA',
    'amd': 'AMD',
    'intel': 'INTC',
    'tsmc': 'TSMC'
}

# Reddit Data Processor Class.
class RedditDataProcessor:

    # -----------------------------------------------------------------------------------------------

    def __init__(self):
        """Initialize the Reddit Data Processor."""
        # Set up paths using path_config
        self.raw_path = RAW_DIR / "reddit_data"
        self.processed_path = PROCESSED_DIR / "reddit_data"
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentiment analyzers
        self.textblob = TextBlob
        self.vader = SentimentIntensityAnalyzer()
        
        # Compile regex patterns for ticker matching
        self.ticker_patterns = [re.compile(r'\b' + re.escape(ticker.lower()) + r'\b') for ticker in TICKERS]
    
    # -----------------------------------------------------------------------------------------------
        
    def process_reddit_data(self, df):
        """
        Process raw Reddit data with enhanced features:
        - Clean and summarize text
        - Calculate multiple sentiment scores
        - Extract relevant features
        - Tag posts by relevance
        """
        try:
            # Log initial data size
            logger.info(f"Initial post count: {len(df)}")
            
            # Copy DataFrame
            processed_df = df.copy()
            
            # Basic text cleaning
            processed_df['cleaned_title'] = processed_df['title'].apply(self._clean_text)
            processed_df['cleaned_text'] = processed_df['text'].apply(self._clean_text)
            
            # Filter by text length
            processed_df['text_length'] = processed_df['cleaned_text'].apply(lambda x: len(str(x).split()))
            processed_df = processed_df[processed_df['text_length'] <= MAX_TEXT_LENGTH]
            logger.info(f"After text length filter: {len(processed_df)}")
            
            # Create text summaries
            processed_df['text_summary'] = processed_df['cleaned_text'].apply(self._summarize_text)
            
            # Check ticker mentions and subreddit relevance
            processed_df['is_relevant'] = processed_df.apply(
                lambda row: self._mentions_ticker(row['cleaned_text'], row.get('subreddit')), 
                axis=1
            )
            logger.info(f"Posts with ticker mentions: {processed_df['is_relevant'].sum()}")
            
            # Calculate multiple sentiment scores
            processed_df = self._calculate_sentiment_scores(processed_df)
            
            # Process comments if available
            if 'top_comments' in processed_df.columns:
                processed_df = self._process_comments(processed_df)
            
            # Add derived features
            processed_df = self._add_derived_features(processed_df)
            
            # Convert date
            processed_df['date'] = pd.to_datetime(processed_df['created_utc']).dt.date
            
            # Aggregate daily metrics
            daily_metrics = self._aggregate_daily_metrics(processed_df)
            logger.info(f"Daily sentiment shape: {daily_metrics.shape}")
            
            # Log success
            logger.info(f"Successfully processed {len(processed_df)} Reddit posts")
            logger.info(f"Generated {len(daily_metrics)} daily sentiment records")
            
            return processed_df, daily_metrics
            
        except Exception as e:
            logger.error(f"Error Processing Reddit Data: {str(e)}")
            return None, None
    
    # -----------------------------------------------------------------------------------------------
    
    def _clean_text(self, text):
        """Clean and Preprocess Text."""
        if pd.isna(text):
            return ""
            
        # Convert to Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove Special Characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    # -----------------------------------------------------------------------------------------------
    
    def _summarize_text(self, text):
        """Summarize text if it's too long."""
        try:
            words = text.split()
            if len(words) > SUMMARY_LENGTH:
                return ' '.join(words[:SUMMARY_LENGTH]) + '...'
            return text
        except:
            return text
    
    # -----------------------------------------------------------------------------------------------
    
    def _mentions_ticker(self, text, subreddit=None):
        """Check if text mentions any relevant tickers or is from a relevant subreddit."""
        if pd.isna(text):
            return False
            
        text = text.lower()
        
        # Check for ticker mentions using regex
        ticker_match = any(pattern.search(text) for pattern in self.ticker_patterns)
        
        # Check subreddit context if available
        if subreddit and not pd.isna(subreddit):
            subreddit = subreddit.lower()
            if subreddit in SUBREDDIT_TICKERS:
                return True
        
        return ticker_match
    
    # -----------------------------------------------------------------------------------------------
    
    def _calculate_sentiment_scores(self, df):
        """Calculate multiple sentiment scores using different methods."""
        # TextBlob sentiment
        df['textblob_sentiment'] = df['cleaned_text'].apply(
            lambda x: self.textblob(x).sentiment.polarity if x else 0
        )
        
        # VADER sentiment
        df['vader_sentiment'] = df['cleaned_text'].apply(
            lambda x: self.vader.polarity_scores(x)['compound'] if x else 0
        )
        
        # Combined sentiment score (weighted average)
        df['overall_sentiment'] = (
            df['textblob_sentiment'] * 0.4 + 
            df['vader_sentiment'] * 0.6
        )
        
        return df
    
    # -----------------------------------------------------------------------------------------------
    
    def _process_comments(self, df):
        """Process and analyze top comments."""
        def analyze_comments(comments):
            if not isinstance(comments, list):
                return 0, 0
            
            sentiments = []
            for comment in comments:
                if comment:
                    # Use VADER for comment sentiment (better for short text)
                    sentiments.append(self.vader.polarity_scores(comment)['compound'])
            
            if sentiments:
                return np.mean(sentiments), len(sentiments)
            return 0, 0
        
        # Apply comment analysis
        df[['comment_sentiment', 'comment_count']] = pd.DataFrame(
            df['top_comments'].apply(analyze_comments).tolist(),
            index=df.index
        )
        
        return df
    
    # -----------------------------------------------------------------------------------------------
        
    def _add_derived_features(self, df):
        """Add derived features from Reddit Data."""
        # Engagement Score (weighted)
        df['engagement_score'] = (
            df['score'] * 0.4 + 
            df['num_comments'] * 0.3 +
            df['comment_count'] * 0.3
        )
        
        # Normalize Engagement Score
        df['engagement_score'] = (df['engagement_score'] - df['engagement_score'].mean()) / df['engagement_score'].std()
        
        return df
    
    # -----------------------------------------------------------------------------------------------
    
    def _aggregate_daily_metrics(self, df):
        """Aggregate metrics by date for pipeline integration."""
        # Group by date and calculate metrics
        daily = df.groupby('date').agg({
            'overall_sentiment': ['mean', 'std', 'count'],
            'vader_sentiment': 'mean',
            'textblob_sentiment': 'mean',
            'comment_sentiment': 'mean',
            'engagement_score': 'mean',
            'score': 'sum',
            'num_comments': 'sum',
            'is_relevant': 'sum'  # Count of relevant posts per day
        }).round(4)
        
        # Flatten column names
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        
        # Reset index and rename date column
        daily.reset_index(inplace=True)
        daily.rename(columns={'date': 'Date'}, inplace=True)
        
        # Convert Date to datetime
        daily['Date'] = pd.to_datetime(daily['Date'])
        
        return daily

def main():
    """Main function to process Reddit data files."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Initialize processor
    processor = RedditDataProcessor()

    # Process each CSV file in the raw reddit data directory
    for file in os.listdir(processor.raw_path):
        if file.endswith(".csv"):
            try:
                filepath = processor.raw_path / file
                logger.info(f"Processing file: {filepath}")
                
                # Read the CSV file
                df = pd.read_csv(filepath)
                
                # Process the data
                processed_df, daily_metrics = processor.process_reddit_data(df)
                
                if processed_df is not None and daily_metrics is not None:
                    # Map common company names to tickers
                    name_to_ticker = {
                        'NVIDIA': 'NVDA',
                        'AMD': 'AMD',
                        'INTEL': 'INTC',
                        'TSMC': 'TSM'
                    }
                    
                    # Extract name from filename and convert to ticker
                    input_name = file.split("_")[0].upper()
                    output_name = name_to_ticker.get(input_name, input_name)
                    
                    # Save detailed processed data
                    detailed_file = processor.processed_path / f"{output_name}_detailed_sentiment.csv"
                    processed_df.to_csv(detailed_file, index=False)
                    
                    # Save daily aggregated data
                    daily_file = processor.processed_path / f"{output_name}_daily_sentiment.csv"
                    daily_metrics.to_csv(daily_file, index=False)
                    
                    logger.info(f"Saved detailed sentiment data to {detailed_file}")
                    logger.info(f"Saved daily sentiment data to {daily_file}")
                else:
                    logger.error(f"Failed to process data from {file}")
                    
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main() 