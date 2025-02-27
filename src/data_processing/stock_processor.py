import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StockDataProcessor:
    def __init__(self):
        """Initialize the stock data processor"""
        pass
        
    def process_stock_data(self, df):
        """
        Process raw stock data:
        - Calculate technical indicators
        - Handle missing values
        - Add derived features
        """
        try:
            processed_df = df.copy()
            
            # Add technical indicators
            processed_df = self._add_technical_indicators(processed_df)
            
            # Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # Add derived features
            processed_df = self._add_derived_features(processed_df)
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing stock data: {str(e)}")
            return None
            
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        # Add moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Add daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Add volatility (20-day rolling)
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Add RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
        
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Use newer pandas methods
        df = df.ffill().bfill()
        return df
        
    def _add_derived_features(self, df):
        """Add derived features"""
        # Trading volume changes
        df['Volume_Change'] = df['Volume'].pct_change().fillna(0)  # Fill first value with 0
        
        # Price momentum
        # Initialize Price_Momentum with 0s for first 5 days
        df['Price_Momentum'] = df['Close'].diff(5).fillna(0)
        
        return df 