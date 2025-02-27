import pandas as pd
from datetime import datetime
import logging
from textblob import TextBlob
import re

logger = logging.getLogger(__name__)

class RedditDataProcessor:
    def __init__(self):
        """Initialize the Reddit data processor"""
        pass
        
    def process_reddit_data(self, df):
        """
        Process raw Reddit data:
        - Clean text
        - Calculate sentiment scores
        - Extract relevant features
        """
        try:
            processed_df = df.copy()
            
            # Clean text
            processed_df['cleaned_title'] = processed_df['title'].apply(self._clean_text)
            processed_df['cleaned_text'] = processed_df['text'].apply(self._clean_text)
            
            # Add sentiment scores
            processed_df = self._add_sentiment_scores(processed_df)
            
            # Add derived features
            processed_df = self._add_derived_features(processed_df)
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing Reddit data: {str(e)}")
            return None
            
    def _clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
        
    def _add_sentiment_scores(self, df):
        """Calculate sentiment scores for text content"""
        # Add sentiment for titles
        df['title_sentiment'] = df['cleaned_title'].apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        
        # Add sentiment for text content
        df['text_sentiment'] = df['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        
        # Combined sentiment
        df['overall_sentiment'] = (df['title_sentiment'] + df['text_sentiment']) / 2
        
        return df
        
    def _add_derived_features(self, df):
        """Add derived features from Reddit data"""
        # Engagement score
        df['engagement_score'] = (
            df['score'] * 0.6 + 
            df['num_comments'] * 0.4
        )
        
        # Normalize engagement score
        df['engagement_score'] = (df['engagement_score'] - df['engagement_score'].mean()) / df['engagement_score'].std()
        
        return df 