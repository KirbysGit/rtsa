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
    def fetch_stock_data(self, ticker, start_date=None, end_date=None, period="1y"):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            period (str): Time period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        try:
            # Download Stock Data.
            if start_date and end_date:
                stock_data = yf.download(ticker, start=start_date, end=end_date)
            else:
                stock_data = yf.download(ticker, period=period)
            
            # Save Raw Data.
            filename = f"{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            stock_data.to_csv(filepath)
            
            # Log Success.
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