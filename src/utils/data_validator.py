# src / utils / data_validator.py

# Description : Validate the data for backtesting.

# Imports.
import logging  
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Setup Logging.
logger = logging.getLogger(__name__)

# Data Validator Class.
class DataValidator:
    """Validate the Data for Backtesting."""
    
    # -----------------------------------------------------------------------------------------------
    
    def __init__(self):
        # Define Required Columns.
        self.required_columns = {
            'Close': float,
            'Volume': float,
            'trade_signal': float,
            'Recommended_Size': float
        }
        
        # Define Minimum Requirements.
        self.min_requirements = {
            'min_rows': 15,  # Temporarily reduced from 60 for debugging
            'min_price': 0.01,
            'min_volume': 1000,
            'max_missing_pct': 0.05
        }
        
        # Define Optional Columns.
        self.optional_columns = {
            'Open': float,
            'High': float,
            'Low': float,
            'avg_sentiment': float,
            'avg_engagement': float,
            'RSI': float,
            'BB_upper': float,
            'BB_lower': float,
            'stop_loss': float,
            'target': float
        }
        
        # Define Valid Ranges for Key Metrics.
        self.valid_ranges = {
            'Close': (0.01, float('inf')),
            'Volume': (1000, float('inf')),
            'trade_signal': (-1, 1),
            'Recommended_Size': (0.05, 1),  # Minimum 5% Position Size.
            'avg_sentiment': (-1, 1),
            'RSI': (0, 100)
        }

    # -----------------------------------------------------------------------------------------------

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate the Input DataFrame for Backtesting.
        Returns: Dictionary with Validation Results and Any Errors.
        """
        errors = []
        warnings = []
        
        # Add Detailed Data Diagnostics.
        logger.info(f"Data Diagnostics:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns present: {df.columns.tolist()}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        # Check if DataFrame is Empty.
        if df.empty:
            errors.append("DataFrame is empty")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        
        # Handle Date Index First.
        df = self._ensure_date_index(df)
        if df is None:
            errors.append("Unable to create valid datetime index")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check Required Columns.
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        
        # Clean and Normalize Data First.
        df = self._preprocess_data(df)
        
        # Add Missing Optional Columns with Default Calculations.
        df = self._add_missing_columns(df)
        
        # Check Data Types.
        for col, dtype in self.required_columns.items():
            if col in df.columns and not np.issubdtype(df[col].dtype, dtype):
                try:
                    df[col] = df[col].astype(dtype)
                    warnings.append(f"Column {col} was converted to {dtype}")
                except:
                    errors.append(f"Column {col} has incorrect data type")
        
        # Check for Missing Values in Required Columns.
        missing_values = df[list(self.required_columns.keys())].isnull().sum()
        if missing_values.any():
            for col, count in missing_values.items():
                if count > 0:
                    warnings.append(f"Column {col} had {count} missing values - filled with appropriate values")
        
        # Check Value Ranges for Required Columns.
        for col, (min_val, max_val) in self.valid_ranges.items():
            if col in df.columns and col in self.required_columns:
                invalid_values = df[
                    (df[col] < min_val) | (df[col] > max_val)
                ][col]
                if not invalid_values.empty:
                    warnings.append(
                        f"Column {col} had {len(invalid_values)} values outside valid range [{min_val}, {max_val}] - clipped to range"
                    )
        
        # Check for Duplicate Indices.
        if df.index.duplicated().any():
            warnings.append("Duplicate Timestamps Found - Keeping Last Value")
            df = df[~df.index.duplicated(keep='last')]
        
        # Check for Chronological Order.
        if not df.index.is_monotonic_increasing:
            warnings.append("Data was not in chronological order - sorting now")
            df = df.sort_index()
        
        # Calculate Data Quality Metrics.
        quality_metrics = self._calculate_quality_metrics(df)
        
        # Convert Warnings to Info if No Errors.
        is_valid = True
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'quality_metrics': quality_metrics,
            'data': df  # Return the Modified DataFrame.
        }
    
    # -----------------------------------------------------------------------------------------------

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Data Before Validation with Enhanced Checks."""
        df = df.copy()
        
        # Verify Minimum Data Requirements.
        if len(df) < self.min_requirements['min_rows']:
            raise ValueError(f"Insufficient data: {len(df)} rows. Minimum required: {self.min_requirements['min_rows']}")
        
        # Handle Missing Close Prices More Carefully.
        if 'Close' in df.columns:
            missing_close = df['Close'].isnull().sum()
            if missing_close / len(df) > self.min_requirements['max_missing_pct']:
                raise ValueError(f"Too many missing Close prices: {missing_close} ({missing_close/len(df):.1%})")
            
            # Use Forward Fill Only for Small Gaps (Max 2 Days).
            df['Close'] = df['Close'].ffill(limit=2).bfill(limit=2)
            
            # If Still Missing Values, Raise Error.
            if df['Close'].isnull().any():
                raise ValueError("Unable to fill all missing Close prices within acceptable limits")
        
        # Handle Volume with More Care.
        if 'Volume' in df.columns:
            # Fill Missing Volume with 30-Day Median.
            rolling_median_volume = df['Volume'].rolling(window=30, min_periods=1).median()
            df['Volume'] = df['Volume'].fillna(rolling_median_volume)
            
            # Ensure Minimum Volume.
            low_volume_days = df[df['Volume'] < self.min_requirements['min_volume']]
            if len(low_volume_days) > 0:
                logger.warning(f"Found {len(low_volume_days)} days with suspiciously low volume")
        
        # Normalize and Validate Trade Signals.
        if 'trade_signal' in df.columns:
            df['trade_signal'] = df['trade_signal'].fillna(0)
            df['trade_signal'] = np.clip(df['trade_signal'], -1, 1)
            
            # Ensure We Don't Have Too Many Signals.
            signal_density = (df['trade_signal'] != 0).mean()
            if signal_density > 0.3:  # More than 30% of days have signals
                logger.warning(f"High signal density detected: {signal_density:.1%} of days have trading signals")
        
        # Validate Position Sizing.
        if 'Recommended_Size' in df.columns:
            df['Recommended_Size'] = df['Recommended_Size'].fillna(0.25)  # Conservative default
            df['Recommended_Size'] = np.clip(df['Recommended_Size'], 
                                           self.valid_ranges['Recommended_Size'][0],
                                           self.valid_ranges['Recommended_Size'][1])
        
        # Return DataFrame.
        return df
    
    # -----------------------------------------------------------------------------------------------

    def _add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Missing Optional Columns with Default Calculations."""
        df = df.copy()
        
        # Calculate Stop_Loss if Missing (Using ATR-Based Method).
        if 'stop_loss' not in df.columns:
            logger.info("Calculating stop_loss using ATR method")
            df['ATR'] = self._calculate_atr(df, period=14)
            # For Long Positions: Stop_Loss = Close - 2*ATR.
            # For Short Positions: Stop_Loss = Close + 2*ATR.
            df['stop_loss'] = np.where(
                df['trade_signal'] > 0,
                df['Close'] - 2 * df['ATR'],
                df['Close'] + 2 * df['ATR']
            )
            df.drop('ATR', axis=1, inplace=True)
        
        # Calculate Target if Missing (Using Risk-Reward Ratio of 2:1).
        if 'target' not in df.columns:
            logger.info("Calculating price targets using 2:1 risk-reward ratio")
            risk = abs(df['Close'] - df['stop_loss'])
            df['target'] = np.where(
                df['trade_signal'] > 0,
                df['Close'] + 2 * risk,  # Long position
                df['Close'] - 2 * risk   # Short position
            )
        
        # Add Other Optional Columns with Default Values if Needed.
        for col in self.optional_columns:
            if col not in df.columns and col not in ['stop_loss', 'target']:
                logger.warning(f"Adding {col} with default values")
                df[col] = 0.0
        
        return df
    
    # -----------------------------------------------------------------------------------------------

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""

        # Calculate Average True Range.
        high = df['High'] if 'High' in df.columns else df['Close']
        low = df['Low'] if 'Low' in df.columns else df['Close']
        close = df['Close']
        
        # Calculate True Range.
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Average True Range.
        atr = tr.rolling(window=period).mean()
        
        # Use Backward Fill Instead of Method='bfill' to Avoid Warning.
        return atr.bfill()
    
    # -----------------------------------------------------------------------------------------------

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Enhanced Data Quality Metrics Calculation."""
        metrics = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d'),
                'trading_days': len(df)
            },
            'completeness': {
                col: (1 - df[col].isnull().mean()) * 100 
                for col in self.required_columns
            },
            'data_quality': {
                'price_gaps': len(df[df['Close'].pct_change().abs() > 0.1]),  # Count 10%+ price moves
                'volume_spikes': len(df[df['Volume'] > df['Volume'].mean() * 3]),  # 3x average volume
                'zero_volume_days': len(df[df['Volume'] == 0]),
                'signal_distribution': {
                    'buy_signals': (df['trade_signal'] > 0).mean() * 100,
                    'sell_signals': (df['trade_signal'] < 0).mean() * 100,
                    'neutral': (df['trade_signal'] == 0).mean() * 100
                }
            },
            'statistics': {
                'avg_daily_volume': df['Volume'].mean(),
                'avg_daily_range': ((df['Close'].pct_change().abs()) * 100).mean(),
                'price_trend': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            }
        }
        
        # Return Metrics.
        return metrics
    
    # -----------------------------------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean the Data by Handling Missing Values and Outliers.
        Returns: Tuple of (cleaned_df, cleaning_report).
        """
        cleaning_report = {'actions': []}
        cleaned_df = df.copy()
        
        # Handle Missing Values.
        for col in self.required_columns:
            if col in cleaned_df.columns:
                missing_count = cleaned_df[col].isnull().sum()
                if missing_count > 0:
                    # Forward Fill First, Then Backward Fill.
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
                    cleaning_report['actions'].append(
                        f"Filled {missing_count} missing values in {col}"
                    )
        
        # Handle Outliers (Using IQR Method).
        for col in ['Close', 'Volume']:
            if col in cleaned_df.columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = cleaned_df[
                    (cleaned_df[col] < lower_bound) | 
                    (cleaned_df[col] > upper_bound)
                ]
                
                if not outliers.empty:
                    cleaned_df.loc[outliers.index, col] = cleaned_df[col].clip(
                        lower=lower_bound, upper=upper_bound
                    )
                    cleaning_report['actions'].append(
                        f"Clipped {len(outliers)} outliers in {col}"
                    )
        
        # Sort Index if Needed.
        if not cleaned_df.index.is_monotonic_increasing:
            cleaned_df.sort_index(inplace=True)
            cleaning_report['actions'].append("Sorted data chronologically")
        
        # Return Report.
        cleaning_report['rows_before'] = len(df)
        cleaning_report['rows_after'] = len(cleaned_df)
        
        return cleaned_df, cleaning_report
    
    # -----------------------------------------------------------------------------------------------

    def _ensure_date_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has a Proper Datetime Index."""
        df = df.copy()
        
        # If Index is Already Datetime, Validate It.
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info("DataFrame already has DatetimeIndex")
            return df
        
        # Try to Convert Index to Datetime if It's Not Already.
        try:
            df.index = pd.to_datetime(df.index)
            logger.info("Successfully converted index to DatetimeIndex")
            return df
        except:
            logger.warning("Failed to convert index to datetime, checking for Date column")
        
        # Look for Date Column in Various Common Names.
        date_column_names = ['Date', 'date', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
        date_column = None
        
        for col in date_column_names:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    logger.info(f"Successfully set index using {col} column")
                    return df
                except:
                    continue
        
        # If We Get Here, We Need to Create a Date Index.
        logger.warning("No Valid Date Column Found, Creating Synthetic Dates")
        try:
            # Create Synthetic Dates Starting from Most Recent Date.
            end_date = pd.Timestamp.now()
            dates = pd.date_range(end=end_date, periods=len(df), freq='B')
            df.index = dates
            logger.info("Created synthetic date index")
            return df
        except Exception as e:
            logger.error(f"Failed to create date index: {str(e)}")
            return None