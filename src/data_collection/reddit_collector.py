# src / data_collection / reddit_collector.py

# Description : This file contains the RedditDataCollector class, which is used to collect data from Reddit.

# Imports.
import os
import logging
import pandas as pd
from praw import Reddit
from datetime import datetime, timedelta
from src.utils.path_config import RAW_DIR

# Setup Logging.
logger = logging.getLogger(__name__)

# Constants
TICKERS = ['NVDA', 'NVIDIA', 'AMD', 'INTC', 'TSMC']
SUBREDDITS = {
    'wallstreetbets': ['DD', 'Discussion', 'News'],
    'stocks': ['DD', 'Discussion', 'News'],
    'investing': ['Discussion', 'News'],
    'nvidia': ['Discussion', 'News', 'Rumor'],
    'AMD_Stock': ['Discussion', 'News'],
    'StockMarket': ['Discussion', 'News']
}

# Reddit Data Collector Class.
class RedditDataCollector:

    # -----------------------------------------------------------------------------------------------

    # Initialize Reddit Data Collector.
    def __init__(self, data_dir=None):
        """Initialize the Reddit Data Collector."""
        self.data_dir = data_dir or (RAW_DIR / "reddit_data")
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
    
    def fetch_subreddit_posts(self, subreddit_name, limit=100, sort='hot', time_filter='day'):
        """
        Fetch Posts from a Subreddit with enhanced filtering.
        
        Args:
            subreddit_name (str): Name of Subreddit
            limit (int): Number of Posts to Fetch
            sort (str): Sort Method ('hot', 'new', 'top', 'relevance')
            time_filter (str): Time period to search ('day', 'week', 'month', 'year', 'all')
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []
            
            # Get posts based on sort method
            if sort == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort == 'new':
                posts = subreddit.new(limit=limit)
            elif sort == 'top':
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            
            # Process each post
            for post in posts:
                # Skip if post is too old
                post_date = datetime.fromtimestamp(post.created_utc)
                if post_date < datetime.now() - timedelta(days=30):
                    continue
                
                # Replace More Comments
                post.comments.replace_more(limit=0)
                
                # Get top comments
                top_comments = list(post.comments)[:5]
                
                # Create post data dictionary
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post_date,
                    'url': post.url,
                    'upvote_ratio': post.upvote_ratio,
                    'top_comments': [c.body for c in top_comments],
                    'subreddit': subreddit_name,
                    'flair': post.link_flair_text
                }
                
                posts_data.append(post_data)
            
            # Save to CSV
            if posts_data:
                df = pd.DataFrame(posts_data)
                filename = f"{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(posts_data)} posts from r/{subreddit_name} to {filename}")
            
            return pd.DataFrame(posts_data) if posts_data else None
            
        except Exception as e:
            logger.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")
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

    def fetch_all_subreddits(self, limit=50):
        """Fetch recent posts from all monitored subreddits."""
        all_posts = []
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        
        # Fetch from each subreddit
        for subreddit_name in subreddits:
            logger.info(f"Fetching posts from r/{subreddit_name}")
            
            # Try different sort methods
            for sort in ['hot', 'new', 'top']:
                posts = self.fetch_subreddit_posts(
                    subreddit_name,
                    limit=limit,
                    sort=sort,
                    time_filter='day'
                )
                if posts is not None:
                    all_posts.append(posts)
        
        # Combine all posts
        if all_posts:
            combined_df = pd.concat(all_posts, ignore_index=True)
            logger.info(f"Collected {len(combined_df)} total posts")
            return combined_df
        else:
            logger.warning("No posts collected")
            return None

def main():
    """Main function to test the RedditDataCollector."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Initialize collector
    collector = RedditDataCollector()

    # Collect posts from all subreddits
    df = collector.fetch_all_subreddits(limit=25)

    if df is not None:
        print(f"Collected {len(df)} total posts.")
        print(f"Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    else:
        print("No posts were collected.")

if __name__ == "__main__":
    main() 