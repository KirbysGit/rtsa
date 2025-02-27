import os
import sys
import logging
from dotenv import load_dotenv
import praw
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_reddit_client():
    """Initialize Reddit client with access token"""
    load_dotenv()
    
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent="script:Market Sentiment Analysis:v1.0 (by /u/kiiiiiiiiirb)",
        refresh_token=os.getenv('REFRESH_TOKEN')
    )
    return reddit

def test_subreddit_data():
    """Test collecting data from finance-related subreddits"""
    reddit = get_reddit_client()
    
    # List of subreddits to monitor
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']
    
    for subreddit_name in subreddits:
        try:
            logger.info(f"Fetching posts from r/{subreddit_name}")
            subreddit = reddit.subreddit(subreddit_name)
            
            # Get top posts from last day
            for post in subreddit.hot(limit=5):
                logger.info(f"\nTitle: {post.title}")
                logger.info(f"Score: {post.score}")
                logger.info(f"Comments: {post.num_comments}")
                logger.info(f"Created: {datetime.fromtimestamp(post.created_utc)}")
                
                # Get top comments
                post.comments.replace_more(limit=0)  # Remove CommentForest instances
                for comment in list(post.comments)[:3]:  # Get top 3 comments
                    logger.info(f"\tTop comment score: {comment.score}")
                    logger.info(f"\tComment text: {comment.body[:100]}...")  # First 100 chars
                
        except Exception as e:
            logger.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")

if __name__ == "__main__":
    test_subreddit_data() 