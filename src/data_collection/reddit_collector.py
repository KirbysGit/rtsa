# src / data_collection / reddit_collector.py

# Description : This file contains the RedditDataCollector class, which is used to collect data from Reddit.

# Imports.
import os
import logging
import pandas as pd
from praw import Reddit
from datetime import datetime
from src.utils.path_config import RAW_DIR

# Setup Logging.
logger = logging.getLogger(__name__)

# Reddit Data Collector Class.
class RedditDataCollector:

    # -----------------------------------------------------------------------------------------------

    # Initialize Reddit Data Collector.
    def __init__(self, data_dir=None):
        """Initialize the Reddit Data Collector."""
        self.data_dir = data_dir or (RAW_DIR / "reddit_data")
        self.subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'nvidia',
            'AMD_Stock',
            'StockMarket'
        ]
        os.makedirs(self.data_dir, exist_ok=True)
        self.reddit = self._init_reddit()
        
    # -----------------------------------------------------------------------------------------------

    # Initialize Reddit Client.
    def _init_reddit(self):
        """Initialize Reddit Client with Refresh Token."""
        
        return Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="script:Market Sentiment Analysis:v1.0 (by /u/kiiiiiiiiirb)",
            refresh_token=os.getenv('REFRESH_TOKEN')
        )
    
    # -----------------------------------------------------------------------------------------------
    
    def fetch_subreddit_posts(self, subreddit_name, limit=100, sort='hot'):
        """
        Fetch Posts from a Subreddit.
        
        Args:
            subreddit_name (str): Name of Subreddit.
            limit (int): Number of Posts to Fetch.
            sort (str): Sort Method ('hot', 'new', 'top').
        """
        try:
            # Initialize Subreddit.
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []
            
            # Get Posts Based on Sort Method.
            if sort == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort == 'new':
                posts = subreddit.new(limit=limit)
            elif sort == 'top':
                posts = subreddit.top(limit=limit)
            
            # Get Comments for Each Post.
            for post in posts:
                # Replace More Comments.
                post.comments.replace_more(limit=0)
                
                # Get Top 5 Comments.
                top_comments = list(post.comments)[:5]
                
                # Create Post Data Dictionary.
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': post.url,
                    'upvote_ratio': post.upvote_ratio,
                    'top_comments': [c.body for c in top_comments]
                }

                # Append Post Data to List.
                posts_data.append(post_data)
            
            # Save to CSV.
            df = pd.DataFrame(posts_data)
            filename = f"{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Successfully Collected {len(posts_data)} Posts from r/{subreddit_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error Fetching Data from r/{subreddit_name}: {str(e)}")
            return None
        
    # -----------------------------------------------------------------------------------------------

    # Fetch Post Comments.
    def fetch_post_comments(self, post_id, limit=None):
        """Fetch Comments for a Specific Post."""
        try:

            # Initialize Submission.
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=None)
            
            # Initialize Comments Data List.
            comments_data = []

            # Get Comments for Each Post.
            for comment in submission.comments.list():
                comment_data = {
                    'id': comment.id,
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc)
                }
                comments_data.append(comment_data)
                
                if limit and len(comments_data) >= limit:
                    break
            
            # Return Comments Data as DataFrame.
            return pd.DataFrame(comments_data)
            
        except Exception as e:
            logger.error(f"Error Fetching Comments for Post {post_id}: {str(e)}")
            return None

    # -----------------------------------------------------------------------------------------------

    def fetch_all_subreddits(self, ticker, limit=50):
        """Fetch Posts from All Monitored Subreddits."""

        # Initialize List to Store All Posts.
        all_posts = []

        # Initialize Search Terms.
        search_terms = [
            f"{ticker} stock",
            f"{ticker} price",
            f"{ticker} analysis",
            f"{ticker} DD"
        ]

        # Fetch Posts from All Subreddits.
        for subreddit_name in self.subreddits:
            for term in search_terms:
                posts = self.fetch_subreddit_posts(subreddit_name, query=term, limit=limit)
                if posts is not None:
                    all_posts.append(posts)
        
        # Return All Posts as DataFrame.
        return pd.concat(all_posts) if all_posts else None 