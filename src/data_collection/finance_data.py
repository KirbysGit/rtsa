# src/data_collection/finance_data.py

# Imports.
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Set Up Logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stock Data Collector Class.
class StockDataCollector:
    # Initialize Stock Data Collector.
    def __init__(self, data_dir="data/raw/stock_data"):
        """Initialize the stock data collector."""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    # Fetch Stock Data.
    def fetch_stock_data(self, ticker, period="1y"):
        """Fetch stock data from Yahoo Finance."""
        try:
            # Download Stock Data
            stock_data = yf.download(ticker, period=period)
            
            # Ensure proper column names and format
            # Convert MultiIndex columns to single level if needed
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(-1)
            
            # Set standard column names
            stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Reset index to make date a column
            stock_data = stock_data.reset_index()
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.set_index('Date')
            
            # Save Raw Data
            filename = f"{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            stock_data.to_csv(filepath, index=True)
            
            logger.info(f"Successfully downloaded {ticker} data to {filepath}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
        
    # Fetch Multiple Stocks.
    def get_multiple_stocks(self, tickers, period="1y"):
        """
        Fetch data for multiple stock tickers.
        
        Args:
            tickers (list): List of stock ticker symbols
            period (str): Time period to download
        """
        # Initialize Dictionary to Store Data.
        all_data = {}
        
        # Fetch Data for Each Ticker.
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, period=period)
            if data is not None:
                all_data[ticker] = data
                
        # Return All Data.
        return all_data 