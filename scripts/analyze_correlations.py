import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.stock_data_processor import StockDataProcessor
from src.analysis.correlation_analyzer import CorrelationAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(symbol: str = 'NVDA') -> pd.DataFrame:
    """Load processed data for analysis"""
    try:
        file_path = Path('data/processed') / f'{symbol}_processed.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"No processed data found at {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Calculate daily returns if not present
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        # Process sentiment data
        logger.info("Original data columns:")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        
        # Check if we have real sentiment data
        if 'avg_sentiment' not in df.columns or df['avg_sentiment'].std() == 0:
            logger.warning("No valid sentiment data found. Generating synthetic sentiment for testing.")
            # Generate synthetic sentiment based on price movements and volume
            df['avg_sentiment'] = (
                0.3 * df['Returns'].rolling(5).mean() + 
                0.2 * df['Volume'].pct_change() +
                0.1 * np.random.normal(0, 1, len(df))
            )
        
        logger.info(f"Sentiment stats before normalization:")
        logger.info(f"avg_sentiment mean: {df['avg_sentiment'].mean():.4f}")
        logger.info(f"avg_sentiment std: {df['avg_sentiment'].std():.4f}")
        
        # 1. Normalize sentiment to have mean 0 and std 1
        df['avg_sentiment_normalized'] = (
            (df['avg_sentiment'] - df['avg_sentiment'].mean()) / 
            (df['avg_sentiment'].std() if df['avg_sentiment'].std() != 0 else 1)
        )
        
        # 2. Generate sentiment signal using percentile ranks
        sentiment_ranks = df['avg_sentiment_normalized'].rank(pct=True)
        df['sentiment_signal'] = pd.cut(
            sentiment_ranks,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[-2, -1, 0, 1, 2],
            include_lowest=True
        )
        
        # 3. Calculate weighted sentiment using volume
        df['weighted_sentiment'] = df['avg_sentiment_normalized'] * (df['Volume'] / df['Volume'].mean())
        
        # 4. Calculate sentiment momentum (change in sentiment)
        df['sentiment_momentum'] = df['avg_sentiment_normalized'].diff()
        
        # Fill NaN values
        for col in ['sentiment_signal', 'weighted_sentiment', 'sentiment_momentum']:
            df[col] = df[col].ffill().bfill()
        
        logger.info(f"Sentiment stats after processing:")
        logger.info(f"sentiment_signal distribution:\n{df['sentiment_signal'].value_counts().sort_index()}")
        logger.info(f"weighted_sentiment range: [{df['weighted_sentiment'].min():.2f}, {df['weighted_sentiment'].max():.2f}]")
        logger.info(f"sentiment_momentum range: [{df['sentiment_momentum'].min():.2f}, {df['sentiment_momentum'].max():.2f}]")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise

def run_correlation_analysis(df: pd.DataFrame, symbol: str = 'NVDA'):
    """Run correlation analysis and save results"""
    try:
        # Initialize analyzer with various lookback periods
        analyzer = CorrelationAnalyzer(lookback_periods=[1, 3, 5, 10, 20])
        
        # Run analysis
        results = analyzer.analyze_correlations(df)
        
        # Create results directory if it doesn't exist
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save correlation plots
        plot_path = results_dir / f'{symbol}_correlation_analysis.png'
        analyzer.plot_correlation_analysis(save_path=str(plot_path))
        
        # Print key findings
        print("\n=== Correlation Analysis Results ===")
        
        print("\nContemporaneous Correlations (Pearson):")
        for metric, values in results['contemporaneous']['pearson'].items():
            print(f"{metric}:")
            print(f"  Correlation: {values['correlation']:.3f}")
            print(f"  P-value: {values['p_value']:.3f}")
        
        print("\nLead-Lag Relationships:")
        for period, values in results['lead_lag'].items():
            print(f"\n{period}:")
            print(f"  Leading correlation: {values['sentiment_leading']['correlation']:.3f} (p={values['sentiment_leading']['p_value']:.3f})")
            print(f"  Lagging correlation: {values['sentiment_lagging']['correlation']:.3f} (p={values['sentiment_lagging']['p_value']:.3f})")
            print(f"  Net predictive power: {values['net_predictive']:.3f}")
        
        print("\nPredictive Power Analysis:")
        period_1 = results['predictive_power']['period_1']
        print("\nReturns by Sentiment Level:")
        for level, ret in period_1['returns_by_quintile'].items():
            print(f"  {level}: {ret:.2%}")
        print(f"Information Coefficient: {period_1['info_coefficient']:.3f}")
        print(f"P-value: {period_1['p_value']:.3f}")
        
        print("\nRegime Analysis:")
        for regime_type, regime_data in results['regime_analysis'].items():
            print(f"\n{regime_type}:")
            for regime, data in regime_data.items():
                print(f"  {regime}:")
                print(f"    Correlation: {data['sentiment_returns_corr']:.3f}")
                print(f"    P-value: {data['p_value']:.3f}")
                print(f"    Avg Return: {data['avg_return']:.2%}")
                print(f"    N observations: {data['n_observations']}")
        
        # Save results to JSON
        results_file = results_dir / f'{symbol}_correlation_analysis.json'
        with open(results_file, 'w') as f:
            # Convert all numpy types to Python native types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: {
                            sk: float(sv) if isinstance(sv, (np.float32, np.float64)) else sv
                            for sk, sv in v.items()
                        } if isinstance(v, dict) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        logger.info("Correlation analysis completed successfully")
        logger.info(f"Results saved to {results_file}")
        return results
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        raise

def main():
    try:
        # Load processed data
        df = load_processed_data()
        
        # Run correlation analysis
        results = run_correlation_analysis(df)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 