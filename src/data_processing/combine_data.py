import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from src.utils.path_config import PROCESSED_DIR, MERGED_DIR

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DataCombiner:
    """Combines processed stock and Reddit sentiment data into a single dataset."""
    
    def __init__(self):
        """Initialize paths and ensure output directory exists."""
        self.stock_path = PROCESSED_DIR / "stock_data"
        self.reddit_path = PROCESSED_DIR / "reddit_data"
        self.output_path = MERGED_DIR
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define expected sentiment columns
        self.sentiment_cols = [
            'overall_sentiment_mean',
            'overall_sentiment_std',
            'overall_sentiment_count',
            'vader_sentiment_mean',
            'textblob_sentiment_mean',
            'comment_sentiment_mean',
            'engagement_score_mean',
            'score_sum',
            'num_comments_sum'
        ]
    
    def combine(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Combine stock and Reddit data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: Combined dataset or None if merge fails
        """
        try:
            # Load stock data
            stock_df = self._load_stock_data(ticker)
            if stock_df is None:
                return None
            
            # Load Reddit data
            reddit_df = self._load_reddit_data(ticker)
            
            # Merge data
            merged_df = self._merge_data(stock_df, reddit_df)
            
            # Save results
            self._save_merged_data(merged_df, ticker)
            
            # Log summary statistics
            self._log_summary_stats(merged_df, ticker)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to combine data for {ticker}: {str(e)}")
            return None
    
    def _load_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load and validate stock data."""
        stock_file = self.stock_path / f"{ticker}_processed.csv"
        
        if not stock_file.exists():
            logger.error(f"Missing stock data file: {stock_file}")
            return None
            
        try:
            df = pd.read_csv(stock_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            return df
        except Exception as e:
            logger.error(f"Error loading stock data for {ticker}: {str(e)}")
            return None
    
    def _load_reddit_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load and validate Reddit sentiment data."""
        # Try different possible file names
        possible_files = [
            self.reddit_path / f"{ticker}_daily_sentiment.csv",
            self.reddit_path / f"NVIDIA_daily_sentiment.csv"  # Fallback for NVDA
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df['Date'] = pd.to_datetime(df['date'])  # Note: using 'date' column
                    df = df.set_index('Date')
                    return df
                except Exception as e:
                    logger.warning(f"Error loading Reddit data from {file_path}: {str(e)}")
                    continue
        
        logger.warning(f"No valid Reddit data found for {ticker}")
        return None
    
    def _merge_data(self, stock_df: pd.DataFrame, reddit_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge stock and Reddit data with proper handling of missing values."""
        if reddit_df is None:
            # Create empty DataFrame with expected columns
            reddit_df = pd.DataFrame(index=stock_df.index)
            for col in self.sentiment_cols:
                reddit_df[col] = 0
        
        # Merge on Date index
        merged = stock_df.merge(
            reddit_df,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Fill missing sentiment values with 0
        for col in self.sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        return merged
    
    def _save_merged_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save merged data to file."""
        output_file = self.output_path / f"{ticker}_merged.csv"
        df.to_csv(output_file)
        logger.info(f"Saved merged data to {output_file}")
    
    def _log_summary_stats(self, df: pd.DataFrame, ticker: str) -> None:
        """Log summary statistics about the merged data."""
        logger.info(f"\nSummary for {ticker}:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Total trading days: {len(df)}")
        logger.info(f"Days with Reddit activity: {(df['overall_sentiment_count'] > 0).sum()}")
        
        # Log sentiment statistics
        if 'overall_sentiment_mean' in df.columns:
            logger.info(f"Average sentiment: {df['overall_sentiment_mean'].mean():.4f}")
            logger.info(f"Max sentiment: {df['overall_sentiment_mean'].max():.4f}")
            logger.info(f"Min sentiment: {df['overall_sentiment_mean'].min():.4f}")

def main():
    """Main function to demonstrate data combining."""
    combiner = DataCombiner()
    
    # List of tickers to process
    tickers = ['NVDA', 'AMD']  # Add more as needed
    
    for ticker in tickers:
        logger.info(f"\nProcessing {ticker}")
        combiner.combine(ticker)

if __name__ == "__main__":
    main()
