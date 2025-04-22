# src / data_processing / stock_data_processor.py

# Description : This file contains the StockDataProcessor class, which is used to process the stock data.

# Imports.
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

# Setup Logging.
logger = logging.getLogger(__name__)

# Stock Data Processor Class.
class StockDataProcessor:
    """Processes Stock Data with Technical Indicators and Optional Sentiment Data."""
    
    # -----------------------------------------------------------------------------------------------
    
    def __init__(self, data_path: str = 'data'):
        """Initialize the Stock Data Processor."""
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / 'raw'
        self.processed_path = self.data_path / 'processed'
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Define Standard File Names.
        self.STOCK_DATA_SUFFIX = '_stock_data.csv'
        self.SENTIMENT_DATA_SUFFIX = '_sentiment_data.csv'
        self.PROCESSED_DATA_SUFFIX = '_processed.csv'
        self.METADATA_SUFFIX = '_metadata.json'

    # -----------------------------------------------------------------------------------------------
    
    def process_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Process and merge data for a given symbol."""
        try:
            # Load Raw Data.
            stock_data = self._load_stock_data(symbol)
            sentiment_data = self._load_sentiment_data(symbol)
            
            if stock_data is None:
                logger.error(f"No stock data available for {symbol}")
                return None
            
            # Process Stock Data.
            stock_data = self._process_stock_data(stock_data)
            
            # Merge with Sentiment Data if Available.
            if sentiment_data is not None:
                merged_data = self._merge_data(stock_data, sentiment_data)
            else:
                logger.warning(f"No sentiment data found for {symbol}, proceeding with stock data only")
                merged_data = stock_data
            
            # Calculate Technical Indicators.
            processed_data = self._calculate_technical_indicators(merged_data)
            
            # Validate Processed Data.
            if self._validate_processed_data(processed_data):
                # Save Processed Data.
                self._save_processed_data(processed_data, symbol)
                return processed_data
            else:
                logger.error(f"Data Validation Failed for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error Processing Data for {symbol}: {str(e)}")
            return None
    
    # -----------------------------------------------------------------------------------------------
    
    def _load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load Stock Data with Validation."""
        file_path = self.raw_path / f"{symbol}{self.STOCK_DATA_SUFFIX}"
        
        if not file_path.exists():
            logger.error(f"Stock data file not found: {file_path}")
            return None
        
        try:
            # Read the CSV file.
            df = pd.read_csv(file_path)
            
            # Handle Datetime Conversion in Steps.
            try:
                # Step 1: Convert to datetime w/o timezone.
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
                # Step 2: Convert to local time and remove timezone.
                df['Date'] = df['Date'].dt.tz_convert('America/New_York').dt.tz_localize(None)
            except Exception as e:
                logger.warning(f"Complex datetime conversion failed: {str(e)}")
                try:
                    # Fallback: Strip timezone information and convert.
                    df['Date'] = df['Date'].apply(lambda x: x.split('-0')[0] if isinstance(x, str) else x)
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.error(f"Fallback datetime conversion failed: {str(e)}")
                    raise
            
            # Sort by date.
            df = df.sort_values('Date')
            
            # Verify conversion.
            if not pd.api.types.is_datetime64_dtype(df['Date']):
                raise ValueError("Date column is not in datetime format after conversion")
            
            # Log Success.
            logger.info(f"Successfully loaded stock data with {len(df)} rows")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Return DataFrame.
            return df
            
        except Exception as e:
            logger.error(f"Error Loading Stock Data: {str(e)}")
            return None
    
    # -----------------------------------------------------------------------------------------------
    
    def _load_sentiment_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load Sentiment Data with Validation."""
        file_path = self.raw_path / f"{symbol}{self.SENTIMENT_DATA_SUFFIX}"
        
        if not file_path.exists():
            logger.warning(f"Sentiment data file not found: {file_path}")
            try:
                # Get Date Range from Stock Data.
                stock_data = pd.read_csv(self.raw_path / f"{symbol}{self.STOCK_DATA_SUFFIX}")
                start_date = pd.to_datetime(stock_data['Date']).min()
                end_date = pd.to_datetime(stock_data['Date']).max()
                
                # Create Dummy Sentiment Data.
                dates = pd.date_range(start=start_date, end=end_date, freq='B')
                sentiment_data = pd.DataFrame({
                    'Date': dates,
                    'avg_sentiment': 0.0,
                    'total_posts': 0,
                    'total_comments': 0,
                    'avg_engagement': 0
                })

                # Set Date to Local Timezone.
                sentiment_data['Date'] = sentiment_data['Date'].dt.tz_localize(None)

                # Save Dummy Sentiment Data.
                sentiment_data.to_csv(file_path, index=False)
                logger.info(f"Created Placeholder Sentiment Data: {file_path}")
                return sentiment_data
                
            except Exception as e:
                logger.error(f"Error Creating Sentiment Data: {str(e)}")
                return None
        
        try:
            # Read the CSV file.
            df = pd.read_csv(file_path)

            # Set Date to Local Timezone.
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            # Return DataFrame.
            return df
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
            return None
    
    # -----------------------------------------------------------------------------------------------
    
    def _process_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Stock Data with Enhanced Error Handling."""
        try:
            df = df.copy()
            
            # Verify Date Column.
            if not pd.api.types.is_datetime64_dtype(df['Date']):
                logger.warning("Date column not in datetime format, attempting conversion")
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Ensure Data is Sorted by Date.
            df = df.sort_values('Date')
            
            # Calculate Returns and Other Metrics.
            df['Returns'] = df['Close'].pct_change()
            df['High-Low'] = df['High'] - df['Low']
            df['High-PrevClose'] = df['High'] - df['Close'].shift(1)
            df['Low-PrevClose'] = df['Low'] - df['Close'].shift(1)
            
            # Calculate Trading Gaps (in days).
            df['trading_gap'] = df['Date'].diff().dt.total_seconds() / (24 * 60 * 60)
            
            # Forward Fill Any Missing Values.
            for col in ['Open', 'High', 'Low', 'Close']:
                if df[col].isnull().any():
                    logger.warning(f"Filling Missing {col} Values")
                    df[col] = df[col].ffill().bfill()
            
            # Fill Missing Volume with 0.
            df['Volume'] = df['Volume'].fillna(0)
            
            # Return DataFrame.
            return df
            
        except Exception as e:
            logger.error(f"Error processing stock data: {str(e)}")
            raise
    
    # -----------------------------------------------------------------------------------------------
    
    def _merge_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Merge stock and sentiment data with proper alignment."""
        try:
            # Create copies to avoid modifying original data.
            stock_df = stock_data.copy()
            sentiment_df = sentiment_data.copy()
            
            # Ensure Dates are Timezone-Naive.
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
            
            # Merge on Date.
            merged = pd.merge(stock_df, sentiment_df, on='Date', how='left')
            
            # Handle Missing Sentiment Values.
            sentiment_cols = ['avg_sentiment', 'total_posts', 'total_comments', 'avg_engagement']
            for col in sentiment_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(method='ffill')
            
            # Return DataFrame.
            return merged
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            raise
    
    # -----------------------------------------------------------------------------------------------
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with shorter lookback periods"""
        df = df.copy()
        
        # Moving Averages with Shorter Periods.
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # RSI with Shorter Period.
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()  # Reduced from 14
        loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()  # Reduced from 14
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD with Shorter Periods.
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands.
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (std * 2)
        df['BB_lower'] = df['BB_middle'] - (std * 2)
        
        # Return DataFrame.
        return df
    
    # -----------------------------------------------------------------------------------------------
    
    def _validate_processed_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data quality."""
        if df is None or df.empty:
            logger.error("Empty DataFrame")
            return False
        
        # Check Minimum Data Points.
        if len(df) < 60:
            logger.error(f"Insufficient Data Points: {len(df)} < 60")
            return False
        
        # Check for Required Columns.
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing Required Columns: {missing_columns}")
            return False
        
        # Check for Excessive Missing Values.
        missing_pct = df[required_columns].isnull().mean() * 100
        if (missing_pct > 5).any():
            logger.error(f"Excessive Missing Values:\n{missing_pct[missing_pct > 5]}")
            return False
        
        return True
    
    # -----------------------------------------------------------------------------------------------
    
    def _save_processed_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save processed data with metadata."""
        # Save processed data with standardized name.
        output_file = self.processed_path / f"{symbol}{self.PROCESSED_DATA_SUFFIX}"
        df.to_csv(output_file, index=False)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'rows': len(df),
            'columns': df.columns.tolist(),
            'processing_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_version': '1.0'
        }
        
        metadata_file = self.processed_path / f"{symbol}{self.METADATA_SUFFIX}"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Log Success.
        logger.info(f"Processed data saved to {output_file}")
        logger.info(f"Metadata saved to {metadata_file}")

# -----------------------------------------------------------------------------------------------

def main():
    # Set Up Logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Initialize Processor.
    processor = StockDataProcessor()
    
    # Process Data for Symbols.
    symbols = ['NVDA'] 
    
    for symbol in symbols:
        logger.info(f"Processing data for {symbol}")
        processed_data = processor.process_data(symbol)
        
        if processed_data is not None:
            logger.info(f"Successfully processed {len(processed_data)} rows for {symbol}")
        else:
            logger.error(f"Failed to process data for {symbol}")

if __name__ == "__main__":
    main() 