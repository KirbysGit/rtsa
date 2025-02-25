# src/data_collection/social_media.py

# Imports.
import tweepy
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# Set Up Logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Social Media Collector Class.
class SocialMediaCollector:
    # Initialize Social Media Collector.
    def __init__(self, data_dir="data/raw/sentiment_data"):
        """Initialize the social media data collector."""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    # Setup Twitter Authentication.
    def setup_twitter_auth(self, bearer_token):
        """
        Set up Twitter API authentication.
        
        Args:
            bearer_token (str): Twitter API bearer token
        """
        try:
            # Initialize Twitter Client.
            self.client = tweepy.Client(bearer_token=bearer_token)
            
            # Log Success.
            logger.info("Twitter authentication successful")
            
        except Exception as e:
            logger.error(f"Twitter authentication failed: {str(e)}")
            
    # Fetch Tweets.
    def fetch_tweets(self, query, max_results=100):
        """
        Fetch tweets based on search query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of tweets to fetch
        """
        try:
            # Fetch Tweets.
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'text', 'public_metrics']
            )
            
            # Check if No Tweets Were Found.
            if not tweets.data:
                logger.warning(f"No tweets found for query: {query}")
                return None
                
            # Convert to DataFrame.
            tweet_data = []
            for tweet in tweets.data:
                tweet_data.append({
                    'created_at': tweet.created_at,
                    'text': tweet.text,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count']
                })
                
            df = pd.DataFrame(tweet_data)
            
            # Save Raw Data.
            filename = f"tweets_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Successfully downloaded tweets to {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            return None 