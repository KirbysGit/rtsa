import logging
from datetime import datetime, timedelta
from src.analysis.topic_identifier import TopicIdentifier
from src.data_collection.reddit_collector import RedditDataCollector
from src.data_collection.stock_data_collector import StockDataCollector
from src.data_processing.reddit_data_processor import RedditDataProcessor
from src.utils.path_config import RAW_DIR, PROCESSED_DIR
import pandas as pd

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.reddit_collector = RedditDataCollector()
        self.reddit_processor = RedditDataProcessor()
        self.topic_identifier = TopicIdentifier()
        self.stock_collector = StockDataCollector()
        
    def run_pipeline(self, top_n_tickers=5, days_to_analyze=7, collect_reddit_data=False):
        """
        Run the complete pipeline:
        1. Collect Reddit data (optional)
        2. Process Reddit data for relevance and sentiment
        3. Identify trending topics from processed data
        4. Collect stock data for trending topics
        5. Process and merge the data

        Args:
            top_n_tickers (int): Number of trending tickers to process
            days_to_analyze (int): Number of days of data to collect
            collect_reddit_data (bool): Whether to collect new Reddit data or use existing
        """
        try:
            # Step 1: Collect Reddit data (optional)
            if collect_reddit_data:
                logger.info("Starting Reddit data collection...")
                reddit_data = self.reddit_collector.fetch_all_subreddits(limit=50)
                if reddit_data is None or reddit_data.empty:
                    logger.error("Failed to collect Reddit data")
                    return
            else:
                logger.info("Skipping Reddit data collection - using existing data")
                # Load existing processed data
                processed_file = PROCESSED_DIR / "reddit_data" / "processed_reddit.csv"
                if not processed_file.exists():
                    logger.error("No existing Reddit data found. Set collect_reddit_data=True to collect new data.")
                    return
                reddit_data = pd.read_csv(processed_file)
            
            # Step 2: Process Reddit data
            logger.info("Processing Reddit data for relevance and sentiment...")
            processed_df, daily_metrics = self.reddit_processor.process_reddit_data(reddit_data)
            if processed_df is None or processed_df.empty:
                logger.error("No relevant Reddit posts after processing")
                return
            
            # Save processed data for traceability
            processed_file = PROCESSED_DIR / "reddit_data" / "processed_reddit.csv"
            processed_df.to_csv(processed_file, index=False)
            logger.info(f"Saved processed Reddit data to {processed_file}")
            
            # Step 3: Identify trending topics from processed data
            logger.info("Identifying trending topics from processed data...")
            trending_tickers = self.topic_identifier.get_trending_tickers(processed_df, top_n=top_n_tickers)
            if not trending_tickers:
                logger.error("No trending tickers identified")
                return
            
            logger.info(f"Identified trending tickers: {', '.join(trending_tickers)}")
            
            # Step 4: Collect stock data for trending topics
            logger.info("Collecting stock data for trending topics...")
            for ticker in trending_tickers:
                try:
                    # Collect stock data
                    stock_data = self.stock_collector.fetch_stock_data(
                        ticker=ticker,
                        start_date=(datetime.now() - timedelta(days=days_to_analyze)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    if stock_data is not None and not stock_data.empty:
                        # Save raw stock data
                        stock_file = RAW_DIR / "stock_data" / f"{ticker}_raw.csv"
                        stock_data.to_csv(stock_file, index=False)
                        logger.info(f"Saved stock data for {ticker} to {stock_file}")
                    else:
                        logger.warning(f"No stock data collected for {ticker}")
                        
                except Exception as e:
                    logger.error(f"Error collecting stock data for {ticker}: {str(e)}")
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")

def main():
    """Main function to run the pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Initialize and run pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline(
        top_n_tickers=5,  # Number of trending tickers to process
        days_to_analyze=7,  # Number of days of data to collect
        collect_reddit_data=False  # Set to True to collect new Reddit data
    )

if __name__ == "__main__":
    main() 