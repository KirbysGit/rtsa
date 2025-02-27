import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stock_collection():
    """Test collecting stock data for major tech companies"""
    from src.data_collection.finance_data import StockDataCollector
    
    # Initialize collector
    collector = StockDataCollector()
    
    # Test stocks (focusing on tech companies mentioned frequently on Reddit)
    stocks = ['NVDA', 'AAPL', 'MSFT', 'AMD', 'GOOGL', 'META']
    
    # Time periods to test
    periods = ['1d', '5d', '1mo']
    
    for stock in stocks:
        logger.info(f"\nTesting data collection for {stock}")
        
        for period in periods:
            try:
                data = collector.fetch_stock_data(stock, period=period)
                
                if data is not None:
                    logger.info(f"Successfully collected {period} data:")
                    logger.info(f"Shape: {data.shape}")
                    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
                    logger.info(f"Columns: {', '.join([str(col) for col in data.columns])}")
                    
                    try:
                        logger.info(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
                        logger.info(f"Trading volume: {data['Volume'].iloc[-1]:,}")
                        
                        if len(data) > 1:
                            price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                            logger.info(f"Price change over period: {price_change:.2f}%")
                            
                            # Calculate volatility
                            returns = data['Close'].pct_change().dropna()
                            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                            logger.info(f"Period volatility: {volatility:.2%}")
                        else:
                            logger.info("Single day data - skipping volatility calculation")
                    except Exception as e:
                        logger.error(f"Error processing data: {str(e)}")
                        logger.debug(f"DataFrame columns: {data.columns}")
                        logger.debug(f"DataFrame head:\n{data.head()}")
                    
                else:
                    logger.error(f"Failed to collect {period} data for {stock}")
                    
            except Exception as e:
                logger.error(f"Error processing {stock} for {period}: {str(e)}")

if __name__ == "__main__":
    test_stock_collection() 