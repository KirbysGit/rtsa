# src / data_processing / reddit_processor.py

# Description : This file contains the RedditDataProcessor class, which is used to process the Reddit data.

# Imports.
import re
import logging
import pandas as pd
from datetime import datetime
from textblob import TextBlob

# Setup Logging.
logger = logging.getLogger(__name__)

# Reddit Data Processor Class.
class RedditDataProcessor:

    # -----------------------------------------------------------------------------------------------

    def __init__(self):
        """Initialize the Reddit Data Processor."""
        pass
    
    # -----------------------------------------------------------------------------------------------
        
    def process_reddit_data(self, df):
        """
        Process raw Reddit data:
        - Clean text
        - Calculate sentiment scores
        - Extract relevant features
        """
        try:
            # Copy DataFrame.
            processed_df = df.copy()
            
            # Clean Text.
            processed_df['cleaned_title'] = processed_df['title'].apply(self._clean_text)
            processed_df['cleaned_text'] = processed_df['text'].apply(self._clean_text)
            
            # Add Sentiment Scores.
            processed_df = self._add_sentiment_scores(processed_df)
            
            # Add Derived Features.
            processed_df = self._add_derived_features(processed_df)
            
            # Log Success.
            logger.info(f"Successfully Processed Reddit Data for {len(processed_df)} rows")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error Processing Reddit Data: {str(e)}")
            return None
    
    # -----------------------------------------------------------------------------------------------
    
    def _clean_text(self, text):
        """Clean and Preprocess Text."""
        if pd.isna(text):
            return ""
            
        # Convert to Lowercase.
        text = text.lower()
        
        # Remove URLs.
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove Special Characters.
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    # -----------------------------------------------------------------------------------------------
    def _add_sentiment_scores(self, df):
        """Calculate Sentiment Scores for Text Content."""
        # Add Sentiment for Titles.
        df['title_sentiment'] = df['cleaned_title'].apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        
        # Add Sentiment for Text Content.
        df['text_sentiment'] = df['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        
        # Combined Sentiment.
        df['overall_sentiment'] = (df['title_sentiment'] + df['text_sentiment']) / 2
        
        return df
    
    # -----------------------------------------------------------------------------------------------
        
    def _add_derived_features(self, df):
        """Add Derived Features from Reddit Data."""
        # Engagement Score.
        df['engagement_score'] = (
            df['score'] * 0.6 + 
            df['num_comments'] * 0.4
        )
        
        # Normalize Engagement Score.
        df['engagement_score'] = (df['engagement_score'] - df['engagement_score'].mean()) / df['engagement_score'].std()
        
        return df 