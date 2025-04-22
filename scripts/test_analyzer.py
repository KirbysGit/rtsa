import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_analysis():
    """Test the data analyzer with merged data"""
    from src.analysis.data_analyzer import DataAnalyzer
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    # Load merged data
    ticker = 'NVDA'
    merged_data_path = f"data/processed/{ticker}_merged_data.csv"
    
    if os.path.exists(merged_data_path):
        merged_data = pd.read_csv(merged_data_path, index_col='Date', parse_dates=True)
        
        # Run correlation analysis
        logger.info("Running correlation analysis...")
        corr_results = analyzer.analyze_correlations(merged_data)
        
        if corr_results:
            logger.info("\nCorrelation Results:")
            logger.info("\nBase Correlations:")
            logger.info(corr_results['correlations']['Returns']['avg_sentiment'])
            
            logger.info("\nLagged Correlations (Sentiment → Returns):")
            for lag, values in corr_results['lagged_correlations'].items():
                logger.info(f"Lag {lag} days: {values['correlation']:.3f} (p={values['p_value']:.3f})")
        
        # Create visualizations
        logger.info("\nCreating visualizations...")
        analyzer.create_visualizations(merged_data, ticker)
        logger.info(f"Visualizations saved to {analyzer.output_dir}/")
        
    else:
        logger.error(f"No merged data found at {merged_data_path}")

if __name__ == "__main__":
    test_analysis() 