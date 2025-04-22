import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.stock_data_processor import DataProcessor
from src.strategy.baseline_strategy import BaselineStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(symbol: str = 'NVDA') -> pd.DataFrame:
    """Load processed data for backtesting"""
    try:
        file_path = Path('data/processed') / f'{symbol}_processed.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"No processed data found at {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Calculate returns if not present
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        # Process sentiment data
        logger.info("Processing sentiment data...")
        
        # Check if we have real sentiment data
        if 'avg_sentiment' not in df.columns or df['avg_sentiment'].std() == 0:
            logger.warning("No valid sentiment data found. Generating synthetic sentiment for testing.")
            # Generate synthetic sentiment based on price movements and volume
            df['avg_sentiment'] = (
                0.3 * df['Returns'].rolling(5).mean() + 
                0.2 * df['Volume'].pct_change() +
                0.1 * np.random.normal(0, 1, len(df))
            )
        
        # 1. Normalize sentiment
        df['avg_sentiment_normalized'] = (
            (df['avg_sentiment'] - df['avg_sentiment'].mean()) / 
            (df['avg_sentiment'].std() if df['avg_sentiment'].std() != 0 else 1)
        )
        
        # 2. Calculate weighted sentiment
        df['weighted_sentiment'] = df['avg_sentiment_normalized'] * (df['Volume'] / df['Volume'].mean())
        
        # 3. Calculate sentiment momentum
        df['sentiment_momentum'] = df['avg_sentiment_normalized'].diff()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Loaded and processed {len(df)} rows of data for {symbol}")
        logger.info(f"Sentiment stats:")
        logger.info(f"weighted_sentiment range: [{df['weighted_sentiment'].min():.2f}, {df['weighted_sentiment'].max():.2f}]")
        logger.info(f"sentiment_momentum range: [{df['sentiment_momentum'].min():.2f}, {df['sentiment_momentum'].max():.2f}]")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def plot_results(results: Dict, symbol: str):
    """Plot backtest results"""
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['returns'].index, results['returns']['capital'])
    plt.title(f'{symbol} Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.savefig(results_dir / f'{symbol}_equity_curve.png')
    plt.close()
    
    # Plot trade distribution
    plt.figure(figsize=(12, 6))
    trade_returns = [t['pnl'] for t in results['trades']]
    sns.histplot(trade_returns, bins=20)
    plt.title(f'{symbol} Trade Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Count')
    plt.savefig(results_dir / f'{symbol}_trade_distribution.png')
    plt.close()

def main():
    try:
        # Load data
        symbol = 'NVDA'
        df = load_data(symbol)
        
        # Initialize strategy
        strategy = BaselineStrategy(
            lookback_period=5,  # Based on correlation analysis
            position_size=0.1,   # Conservative position sizing
            stop_loss=0.02,      # 2% stop loss
            take_profit=0.05,    # 5% take profit
            transaction_cost=0.001  # 0.1% transaction cost
        )
        
        # Run backtest
        results = strategy.backtest(df)
        
        # Plot results
        plot_results(results, symbol)
        
        # Save results
        results_file = Path('results') / f'{symbol}_baseline_results.json'
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            json_results = {
                'metrics': results['metrics'],
                'trades': [
                    {
                        **trade,
                        'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
                        'exit_date': trade['exit_date'].strftime('%Y-%m-%d')
                    }
                    for trade in results['trades']
                ]
            }
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 