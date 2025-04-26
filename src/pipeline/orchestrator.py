#

import logging
from datetime import datetime, timedelta
from src.analysis.topic_identifier import TopicIdentifier, POTENTIALLY_AMBIGUOUS_TICKERS
from src.data_collection.reddit_collector import RedditDataCollector
from src.data_collection.stock_data_collector import StockDataCollector
from src.data_processing.reddit_data_processor import RedditDataProcessor, COMMON_WORDS
from src.utils.path_config import (
    RAW_DIR, 
    PROCESSED_DIR, 
    TICKER_GENERAL_DIR,
    REDDIT_DATA_DIR,
    STOCK_DATA_DIR,
    PROCESSED_REDDIT_DIR
)
import pandas as pd
from colorama import Fore, Style
from src.modeling.feature_builder import FeatureBuilder
from src.modeling.model_trainer import ModelTrainer
from src.modeling.predictor import Predictor

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        print(f"\n{Fore.CYAN}=== Reddit Stock Trend Analysis Pipeline ==={Style.RESET_ALL}")
        
        # Initialize components
        print(f"\n{Fore.YELLOW}Initializing Components...{Style.RESET_ALL}")
        self.reddit_collector = RedditDataCollector()
        self.reddit_processor = RedditDataProcessor()
        self.topic_identifier = TopicIdentifier()
        self.stock_collector = StockDataCollector()
        self.feature_builder = FeatureBuilder()
        self.model_trainer = None  # Initialize later with config
        self.predictor = None  # Initialize later if needed
        print(f"{Fore.GREEN}✓ All components initialized{Style.RESET_ALL}")
        
    def run_pipeline(self, 
                    top_n_tickers=5, 
                    days_to_analyze=7,
                    lookback_window_days=60,
                    reddit_lookback_days=30,
                    reddit_posts_per_subreddit=100,
                    min_rows_required=30,
                    min_class_balance=0.1,
                    verbose=False,
                    summary_path=None,
                    pipeline_config={
                        'collect_reddit_data': False,
                        'process_reddit_data': False,
                        'analyze_tickers': False,
                        'collect_stock_data': True,
                        'save_debug_info': True,
                        'generate_features': True,
                        'train_models': True,
                        'predict': True  # New config option
                    }):
        """Run the pipeline with configurable steps.
        
        Args:
            top_n_tickers: Number of top trending tickers to analyze
            days_to_analyze: Number of days of historical data to analyze
            lookback_window_days: Number of days to look back for feature generation
            reddit_lookback_days: Number of days to look back for Reddit posts
            reddit_posts_per_subreddit: Number of posts to collect per subreddit/sort method
            min_rows_required: Minimum rows required for model training
            min_class_balance: Minimum class balance required for training
            verbose: Whether to print detailed output
            summary_path: Path to save training summary (CSV or JSON)
            pipeline_config: Dictionary of pipeline step configurations
        """
        try:
            # Print The Current Pipeline Config
            print(f"\n{Fore.CYAN}Pipeline Configuration:{Style.RESET_ALL}")
            for step, enabled in pipeline_config.items():
                status = f"{Fore.GREEN}Enabled" if enabled else f"{Fore.YELLOW}Disabled"
                print(f"• {step.replace('_', ' ').title()}: {status}{Style.RESET_ALL}")
            
            if verbose:
                print(f"\n{Fore.CYAN}Data Collection Parameters:{Style.RESET_ALL}")
                print(f"• Reddit Lookback: {reddit_lookback_days} days")
                print(f"• Posts per Subreddit: {reddit_posts_per_subreddit}")
                print(f"• Feature Lookback: {lookback_window_days} days")
                print(f"• Analysis Period: {days_to_analyze} days")
            
            # Initialize Variables
            reddit_data = None
            trending_tickers = None
            processed_df = None
            
            # ------------------------------------------------------------------------------------------
            
            # Step 1: Data Collection.
            if pipeline_config['collect_reddit_data']:
                print(f"\n{Fore.CYAN}Step 1: Reddit Data Collection{Style.RESET_ALL}")
                
                # Initialize collector with lookback period
                self.reddit_collector = RedditDataCollector(max_days_lookback=reddit_lookback_days)
                
                # Collect Reddit Data.
                reddit_data = self.reddit_collector.fetch_all_subreddits(limit=reddit_posts_per_subreddit)
                
                # Check if Reddit Data was Collected Successfully.
                if reddit_data is None or reddit_data.empty:
                    logger.error("Failed to collect Reddit data")
                    return
                
                # Print the Number of Posts Collected.
                print(f"Collected {len(reddit_data)} posts from Reddit.")
                
            else:
                print(f"\n{Fore.CYAN}Step 1: Loading Existing Reddit Data{Style.RESET_ALL}")
                reddit_data = self._load_existing_reddit_data()
                if reddit_data is None:
                    return
            
            # ------------------------------------------------------------------------------------------
            
            # Step 2: Data Processing.
            if pipeline_config['process_reddit_data']:
                print(f"\n{Fore.CYAN}Step 2: Reddit Data Processing{Style.RESET_ALL}")
                
                # Process the Reddit Data.
                processed_df, daily_metrics = self.reddit_processor.process_reddit_data(reddit_data)
                
                # Check if the Processed Data is Valid.
                if processed_df is None or processed_df.empty:
                    logger.error("No relevant Reddit posts after processing")
                    return
                
                # Save the Processed Data.
                if pipeline_config['save_debug_info']:
                    self._save_processed_data(processed_df, daily_metrics)
            else:
                print(f"\n{Fore.CYAN}Step 2: Loading Latest Processed Data{Style.RESET_ALL}")
                processed_df = self._load_latest_processed_data()
                if processed_df is None:
                    logger.error("Failed to load processed data. Try running with process_reddit_data enabled.")
                    return
                
                print(f"Loaded {len(processed_df)} processed posts with {sum(processed_df['is_relevant'])} relevant posts.")
            
            # ------------------------------------------------------------------------------------------
            
            # Step 3: Ticker Analysis.
            if pipeline_config['analyze_tickers']:
                print(f"\n{Fore.CYAN}Step 3: Ticker Analysis{Style.RESET_ALL}")
                
                # Verify we have the required data for analysis
                if not self._verify_processed_data(processed_df):
                    logger.error("Cannot perform ticker analysis: missing required data columns")
                    return
                
                # Get the Trending Tickers and their analysis
                trending_tickers = self.topic_identifier.get_trending_tickers(processed_df, top_n=top_n_tickers)
                if not trending_tickers:
                    logger.error("No trending tickers identified")
                    return
                
                # Get analysis metrics for all tickers
                analysis_metrics = []
                all_tickers = set()
                for _, row in processed_df.iterrows():
                    all_tickers.update(row['tickers'])
                
                for ticker in all_tickers:
                    ticker_data = processed_df[processed_df['tickers'].apply(lambda x: ticker in x)]
                    if not ticker_data.empty:
                        # Calculate engagement using weighted score
                        total_engagement = (
                            ticker_data['score'].sum() * 0.4 + 
                            ticker_data['num_comments'].sum() * 0.6
                        )
                        
                        # Get example contexts (up to 3)
                        example_contexts = "; ".join(
                            ticker_data['cleaned_text']
                            .str[:100]  # Limit context length
                            .head(3)    # Take up to 3 examples
                            .tolist()
                        )
                        
                        metrics = {
                            'ticker': ticker,
                            'mentions': len(ticker_data),
                            'total_engagement': total_engagement,
                            'avg_confidence': ticker_data['ticker_confidence'].mean(),
                            'passed_filters': True,  # If it's in the processed data, it passed filters
                            'is_common_word': ticker in self.reddit_processor.common_word_tickers,  # Use class attribute
                            'is_valid_ticker': True,  # If it's in the processed data, it's valid
                            'is_ambiguous': ticker in POTENTIALLY_AMBIGUOUS_TICKERS,  # Use imported constant
                            'example_contexts': example_contexts,
                            'min_confidence': ticker_data['ticker_confidence'].min(),
                            'max_confidence': ticker_data['ticker_confidence'].max()
                        }
                        analysis_metrics.append(metrics)
                
                # Sort by total engagement
                analysis_metrics = sorted(analysis_metrics, key=lambda x: x['total_engagement'], reverse=True)
                
                # Save analysis to cache
                timestamp = datetime.now().strftime('%Y%m%d')
                cache_file = TICKER_GENERAL_DIR / f"ticker_analysis_daily_{timestamp}.csv"
                TICKER_GENERAL_DIR.mkdir(parents=True, exist_ok=True)
                
                analysis_df = pd.DataFrame(analysis_metrics)
                analysis_df.to_csv(cache_file, index=False)
                
                print(f"{Fore.GREEN}✓ Saved ticker analysis to {cache_file.name}{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}Analysis Summary:{Style.RESET_ALL}")
                print(f"Total Tickers Analyzed: {len(analysis_metrics)}")
                
                # Display top tickers with their metrics
                print(f"\n{Fore.CYAN}Top Trending Tickers:{Style.RESET_ALL}")
                for metrics in analysis_metrics[:10]:  # Show top 10
                    sentiment_score = metrics['avg_confidence']
                    sentiment_icon = "📈" if sentiment_score > 0.6 else "📉" if sentiment_score < 0.4 else "➖"
                    print(f"• {metrics['ticker']:<5} {sentiment_icon} Score: {sentiment_score:.2f} | "
                          f"Mentions: {int(metrics['mentions'])} | Engagement: {int(metrics['total_engagement'])} | "
                          f"Confidence Range: {metrics['min_confidence']:.2f}-{metrics['max_confidence']:.2f}")
                
                # Only use top N tickers for further analysis
                trending_tickers = [m['ticker'] for m in analysis_metrics[:top_n_tickers]]
                print(f"\nIdentified {len(trending_tickers)} trending tickers for further analysis: {', '.join(trending_tickers)}")
                
            else:
                print(f"\n{Fore.CYAN}Step 3: Loading Cached Analysis{Style.RESET_ALL}")
                cached_data = self._load_cached_ticker_analysis()
                if cached_data:
                    trending_tickers = cached_data['tickers'][:top_n_tickers]  # Get top N tickers
                else:
                    logger.error("No cached ticker analysis found")
                    return
            
            # ------------------------------------------------------------------------------------------
            
            # Step 4: Stock Data Collection.
            if pipeline_config['collect_stock_data'] and trending_tickers:
                print(f"\n{Fore.CYAN}Step 4: Stock Data Collection{Style.RESET_ALL}")
                
                # Collect the Stock Data.
                self._collect_stock_data(trending_tickers, days_to_analyze)
                
                # Print the Stock Data Collection Summary.
                print(f"\n{Fore.GREEN}Stock Data Collection Summary:{Style.RESET_ALL}")
                for ticker in trending_tickers:
                    stock_file = STOCK_DATA_DIR / f"{ticker}_raw.csv"
                    status = "✓" if stock_file.exists() else "✗"
                    print(f"{status} {ticker:<5} {'(Data collected)' if stock_file.exists() else '(No data)'}")
            
            # ------------------------------------------------------------------------------------------
            
            # Step 5: Feature Generation
            if pipeline_config['generate_features'] and trending_tickers:
                print(f"\n{Fore.CYAN}Step 5: Feature Generation{Style.RESET_ALL}")
                
                # Generate features for trending tickers
                feature_results = self.feature_builder.generate_features_for_tickers(
                    tickers=trending_tickers,
                    save_output=True,
                    normalize=True
                )
                
                # Print feature generation summary
                print(f"\n{Fore.CYAN}Feature Generation Summary:{Style.RESET_ALL}")
                for ticker, df in feature_results.items():
                    if df is not None:
                        print(f"{Fore.GREEN}✓ {ticker}: {len(df)} rows, {len(df.columns)} features{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}✗ {ticker}: Failed to generate features{Style.RESET_ALL}")
            
            # ------------------------------------------------------------------------------------------
            
            # Step 6: Model Training
            if pipeline_config['train_models'] and trending_tickers:
                print(f"\n{Fore.CYAN}Step 6: Model Training{Style.RESET_ALL}")
                
                # Initialize model trainer with configuration
                self.model_trainer = ModelTrainer(
                    model_type='xgboost',
                    lookback_window_days=lookback_window_days,
                    min_rows_required=min_rows_required,
                    min_class_balance=min_class_balance,
                    verbose=verbose,
                    summary_path=summary_path
                )
                
                # Train models for trending tickers
                for ticker in trending_tickers:
                    self.model_trainer.train_and_evaluate(ticker, save_results=True)
                
                # Print final training summary
                print(self.model_trainer.get_training_summary())
                self.model_trainer.save_training_summary()
            
            # ------------------------------------------------------------------------------------------
            
            # Step 7: Predictions
            if pipeline_config['predict'] and trending_tickers:
                print(f"\n{Fore.CYAN}Step 7: Running Predictions{Style.RESET_ALL}")
                
                # Initialize predictor
                self.predictor = Predictor(confidence_threshold=0.6)
                
                # Run predictions and get summary
                prediction_summary = self.predictor.predict_all(trending_tickers)
                
                if not prediction_summary.empty:
                    print(f"\n{Fore.GREEN}✓ Generated predictions for {len(prediction_summary)} tickers{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.YELLOW}No predictions could be generated{Style.RESET_ALL}")
            
            # ------------------------------------------------------------------------------------------
            
            # Final Summary
            print(f"\n{Fore.GREEN}Pipeline completed successfully{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------------------------------------------------------------------------------------

    def _load_existing_reddit_data(self):
        """Load Existing Reddit Data Files."""
        try:
            csv_files = list(REDDIT_DATA_DIR.glob("*.csv"))
            
            if not csv_files:
                logger.error("No Reddit data files found")
                return None
            
            dfs = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                    logger.info(f"Loaded data from {file.name}")
                except Exception as e:
                    logger.warning(f"Error reading {file.name}: {str(e)}")
            
            if not dfs:
                logger.error("No valid Reddit data files found")
                return None
            
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
            logger.info(f"Loaded {len(combined_df)} unique posts from {len(csv_files)} files")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading Reddit data: {str(e)}")
            return None
    
    def _save_processed_data(self, processed_df, daily_metrics):
        """Save processed data and metrics."""
        try:
            PROCESSED_REDDIT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            processed_file = PROCESSED_REDDIT_DIR / f"processed_reddit_{timestamp}.csv"
            metrics_file = PROCESSED_REDDIT_DIR / f"daily_metrics_{timestamp}.csv"
            
            processed_df.to_csv(processed_file, index=False)
            if daily_metrics is not None:
                daily_metrics.to_csv(metrics_file, index=False)
            
            logger.info(f"Saved processed data to {processed_file}")
            logger.info(f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
    
    def _verify_processed_data(self, df):
        """Verify that processed data has all required columns for analysis."""
        required_columns = {
            'tickers',
            'ticker_confidence',
            'confidence_class',
            'overall_sentiment',
            'score',
            'num_comments',
            'cleaned_text',
            'is_relevant'
        }
        
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns in processed data: {missing_columns}")
            return False
            
        return True

    def _load_latest_processed_data(self):
        """Load the most recent fully processed Reddit data."""
        try:
            processed_files = list(PROCESSED_REDDIT_DIR.glob("processed_reddit_*.csv"))
            
            if not processed_files:
                logger.error("No processed data files found")
                return None
            
            # Get most recent file
            latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
            
            # Load the data
            df = pd.read_csv(latest_file)
            logger.info(f"Loaded processed data from {latest_file.name}")
            
            # Verify the data has required columns
            if not self._verify_processed_data(df):
                logger.error("Loaded data is missing required columns. Please run with process_reddit_data enabled.")
                return None
                
            # Convert string representation of lists back to actual lists
            if 'tickers' in df.columns:
                df['tickers'] = df['tickers'].apply(eval)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            return None
    
    def _collect_stock_data(self, tickers, days_to_analyze):
        """Collect stock data for the given tickers."""
        STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        for ticker in tickers:
            try:
                stock_data = self.stock_collector.fetch_stock_data(
                    ticker=ticker,
                    start_date=(datetime.now() - timedelta(days=days_to_analyze)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if stock_data is not None and not stock_data.empty:
                    stock_file = STOCK_DATA_DIR / f"{ticker}_raw.csv"
                    stock_data.to_csv(stock_file, index=False)
                    logger.info(f"Saved stock data for {ticker} to {stock_file}")
                else:
                    logger.warning(f"No stock data collected for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error collecting stock data for {ticker}: {str(e)}")

    def _load_cached_ticker_analysis(self):
        """Load the most recent ticker analysis from CSV."""
        try:
            analysis_files = list(TICKER_GENERAL_DIR.glob("ticker_analysis_daily_*.csv"))
            if not analysis_files:
                logger.error("No cached ticker analysis found")
                return None
            
            # Get most recent file
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            # Read CSV file
            df = pd.read_csv(latest_file)
            
            # Sort by total engagement and get top tickers
            df = df.sort_values('total_engagement', ascending=False)
            
            print(f"{Fore.GREEN}✓ Loaded ticker analysis from {latest_file.name}{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Analysis Summary from Cache:{Style.RESET_ALL}")
            print(f"Total Tickers Analyzed: {len(df)}")
            
            # Display top tickers with their metrics
            print(f"\n{Fore.CYAN}Top Trending Tickers:{Style.RESET_ALL}")
            for _, row in df.head(10).iterrows():
                sentiment_score = row['avg_confidence']
                sentiment_icon = "📈" if sentiment_score > 0.6 else "📉" if sentiment_score < 0.4 else "➖"
                print(f"• {row['ticker']:<5} {sentiment_icon} Score: {sentiment_score:.2f} | "
                      f"Mentions: {int(row['mentions'])} | Engagement: {int(row['total_engagement'])}")
            
            return {
                'tickers': df['ticker'].tolist(),
                'data': df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error loading cached ticker analysis: {str(e)}")
            return None

def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Reddit Stock Trend Analysis Pipeline')
    parser.add_argument('--tickers', type=str, help='Path to file containing tickers (one per line)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window for features')
    parser.add_argument('--reddit-days', type=int, default=30, help='Days to look back for Reddit posts')
    parser.add_argument('--reddit-limit', type=int, default=100, help='Posts per subreddit/sort method')
    parser.add_argument('--min-rows', type=int, default=30, help='Minimum rows for training')
    parser.add_argument('--min-balance', type=float, default=0.1, help='Minimum class balance')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--summary', type=str, help='Path to save training summary')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline(
        top_n_tickers=5,
        days_to_analyze=args.days,
        lookback_window_days=args.lookback,
        reddit_lookback_days=args.reddit_days,
        reddit_posts_per_subreddit=args.reddit_limit,
        min_rows_required=args.min_rows,
        min_class_balance=args.min_balance,
        verbose=args.verbose,
        summary_path=args.summary,
        pipeline_config={
            'collect_reddit_data': False,  # Hardcoded to always collect new data
            'process_reddit_data': True,
            'analyze_tickers': True,
            'collect_stock_data': True,
            'save_debug_info': True,
            'generate_features': True,
            'train_models': True,
            'predict': True  # Hardcoded to always run predictions
        }
    )

if __name__ == "__main__":
    main() 