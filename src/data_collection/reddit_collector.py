import pandas as pd
import logging
from datetime import datetime
import os
from praw import Reddit

logger = logging.getLogger(__name__)

class RedditDataCollector:
    def __init__(self, data_dir="data/raw/reddit_data"):
        """Initialize the Reddit data collector"""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.reddit = self._init_reddit()
        
    def _init_reddit(self):
        """Initialize Reddit client with refresh token"""
        return Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="script:Market Sentiment Analysis:v1.0 (by /u/kiiiiiiiiirb)",
            refresh_token=os.getenv('REFRESH_TOKEN')
        )
    
    def fetch_subreddit_posts(self, subreddit_name, limit=100, sort='hot'):
        """
        Fetch posts from a subreddit
        
        Args:
            subreddit_name (str): Name of subreddit
            limit (int): Number of posts to fetch
            sort (str): Sort method ('hot', 'new', 'top')
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
                posts = subreddit.top(limit=limit)
            
            for post in posts:
                # Get comment sentiment
                post.comments.replace_more(limit=0)
                top_comments = list(post.comments)[:5]  # Get top 5 comments
                
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
                posts_data.append(post_data)
            
            # Save to CSV
            df = pd.DataFrame(posts_data)
            filename = f"{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Successfully collected {len(posts_data)} posts from r/{subreddit_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")
            return None
            
    def fetch_post_comments(self, post_id, limit=None):
        """Fetch comments for a specific post"""
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=None)  # Expand all comments
            
            comments_data = []
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
            
            return pd.DataFrame(comments_data)
            
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {str(e)}")
            return None 