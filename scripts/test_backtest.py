import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.backtester import Backtester
from src.utils.data_validator import DataValidator

# Set up logging with custom formatter
class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and sections"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_blue = "\x1b[1;34m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_blue + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Set up logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler('results/backtest.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

class BacktestRunner:
    """Manages backtest execution and reporting"""
    
    def __init__(self, symbol: str = 'NVDA', 
                 data_path: str = 'data/processed',
                 strategy_params: Optional[Dict] = None):
        self.symbol = symbol
        self.data_path = Path(data_path)
        self.validator = DataValidator()
        self.results = None
        self.strategy_params = strategy_params or {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ma_period': 20,
            'risk_per_trade': 0.02,  # 2% risk per trade
            'profit_factor': 2.0,    # 2:1 reward-risk ratio
            'max_position_size': 0.25  # 25% maximum position
        }
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate data for backtesting"""
        file_path = self.data_path / f"{self.symbol}_processed.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        
        # Handle missing values quietly
        for col in ['MA20', 'RSI', 'ATR', 'MACD', 'Signal_Line']:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].ffill().bfill()
        
        # Generate trading signals
        df = _calculate_trading_signals(df)
        
        # Validate the data
        validation_result = self.validator.validate_data(df)
        
        if not validation_result['is_valid']:
            logger.error(f"Data validation failed: {validation_result['errors']}")
            raise ValueError("Data validation failed")
        
        return df
    
    def plot_results(self) -> None:
        """Generate performance visualization plots"""
        if not hasattr(self, 'df') or self.df is None:
            logger.error("No data available for plotting")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Equity Curve
        ax1 = plt.subplot(2, 2, 1)
        self.df['Cumulative_Returns'].plot(ax=ax1, color='blue')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns')
        
        # 2. Drawdown
        ax2 = plt.subplot(2, 2, 2)
        drawdown = (self.df['Cumulative_Returns'] - 
                   self.df['Cumulative_Returns'].expanding().max()) * 100
        drawdown.plot(ax=ax2, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        
        # 3. Daily Returns Distribution
        ax3 = plt.subplot(2, 2, 3)
        sns.histplot(self.df['Strategy_Returns'].dropna() * 100, 
                    ax=ax3, bins=50, kde=True)
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Returns %')
        
        # 4. Position Sizes Over Time
        ax4 = plt.subplot(2, 2, 4)
        self.df['Position'].plot(ax=ax4, color='green')
        ax4.set_title('Position Sizes')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Position Size')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('results') / f"{self.symbol}_analysis.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Analysis plots saved to {plot_path}")
    
    def run_backtest(self) -> Dict:
        """Run the backtest with enhanced strategy"""
        try:
            # Load and validate data
            self.df = self.load_and_validate_data()
            
            # Run backtest with proper date handling
            self.results = run_backtest(self.df, self.symbol, self.strategy_params)
            
            # Generate plots
            self.plot_results()
            
            # Print detailed summary
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise
            
    def calculate_advanced_metrics(self) -> Dict:
        """Calculate additional performance metrics"""
        if self.df is None or len(self.df) == 0:
            return {}
            
        daily_returns = self.df['Strategy_Returns'].dropna()
        
        # Sortino Ratio (downside risk only)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (daily_returns.mean() * 252) / downside_std
        
        # Maximum Consecutive Losses
        wins = daily_returns > 0
        max_consecutive_losses = max(
            sum(1 for _ in group) 
            for key, group in itertools.groupby(wins) 
            if not key
        )
        
        # Average Win/Loss
        avg_win = daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0
        avg_loss = daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0
        
        return {
            'sortino_ratio': round(sortino_ratio, 2),
            'max_consecutive_losses': max_consecutive_losses,
            'avg_win': round(avg_win * 100, 2),
            'avg_loss': round(avg_loss * 100, 2),
            'profit_factor': round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2)
        }
    
    def _print_summary(self) -> None:
        """Print clean backtest summary"""
        if not self.results:
            logger.error("No results available")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 BACKTEST ANALYSIS: {self.symbol}")
        print("=" * 60)
        
        # Data Summary
        print("\n📅 Data Summary:")
        print(f"Period: {self.results['start_date']} to {self.results['end_date']}")
        trading_days = pd.Timestamp(self.results['end_date']) - pd.Timestamp(self.results['start_date'])
        print(f"Trading Days: {trading_days.days}")
        
        # Performance Metrics
        print("\n📈 Performance Metrics:")
        print(f"Total Return: {self.results['total_return']:>8.2f}%")
        print(f"Annualized:   {self.results['annualized_return']:>8.2f}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:>8.2f}")
        
        # Risk Metrics
        print("\n⚠️ Risk Metrics:")
        print(f"Maximum Drawdown: {self.results['max_drawdown']:>8.2f}%")
        print(f"Volatility:       {self.results['risk_metrics']['volatility']:>8.2f}%")
        
        # Trading Statistics
        print("\n🔄 Trading Statistics:")
        if self.results['trade_metrics']['total_trades'] > 0:
            total_trades = self.results['trade_metrics']['total_trades']
            winning_trades = self.results['trade_metrics']['winning_trades']
            losing_trades = self.results['trade_metrics']['losing_trades']
            win_rate = self.results['trade_metrics']['win_rate']
            
            print(f"Total Trades:    {total_trades:>4}")
            print(f"Winning Trades:  {winning_trades:>4} ({win_rate:>6.1f}%)")
            print(f"Losing Trades:   {losing_trades:>4} ({100-win_rate:>6.1f}%)")
        else:
            print("No trades executed during the period")
        
        print("=" * 60 + "\n")

def run_backtest(df: pd.DataFrame, symbol: str, strategy_params: Dict) -> Dict:
    """Run backtest with proper date handling and trading logic"""
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        start_date = df.index[0]
        end_date = df.index[-1]
        
        logger.info(f"Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Initialize results
        results = {
            'symbol': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'risk_metrics': {'volatility': 0.0},
            'trade_metrics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0
            }
        }
        
        # Generate trading signals if not present
        if 'trade_signal' not in df.columns:
            df = _calculate_trading_signals(df)
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Apply trading logic with position sizing
        df['Position'] = df['trade_signal'] * df['Recommended_Size']
        df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)
        
        # Calculate cumulative returns
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod()
        
        # Calculate performance metrics
        total_return = (df['Cumulative_Returns'].iloc[-1] - 1) * 100
        trading_days = len(df)
        annualized_return = ((1 + total_return/100) ** (252/trading_days) - 1) * 100
        
        # Calculate risk metrics
        daily_returns = df['Strategy_Returns'].dropna()
        if len(daily_returns) > 0:
            volatility = daily_returns.std() * np.sqrt(252) * 100
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = daily_returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # Calculate drawdown
            cumulative = df['Cumulative_Returns']
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max * 100
            max_drawdown = abs(drawdowns.min())
            
            # Count trades
            position_changes = df['Position'].diff()
            total_trades = (position_changes != 0).sum()
            winning_trades = len(daily_returns[daily_returns > 0])
            losing_trades = len(daily_returns[daily_returns < 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # Update results
            results.update({
                'total_return': round(total_return, 2),
                'annualized_return': round(annualized_return, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'risk_metrics': {
                    'volatility': round(volatility, 2)
                },
                'trade_metrics': {
                    'total_trades': int(total_trades),
                    'winning_trades': int(winning_trades),
                    'losing_trades': int(losing_trades),
                    'win_rate': round(win_rate, 2)
                }
            })
        
        # Save detailed results
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{symbol}_backtest_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Detailed results saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in backtest execution: {str(e)}")
        raise

def _calculate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trading signals with improved logic"""
    df = df.copy()
    
    # Calculate basic indicators if missing
    if 'MA20' not in df.columns:
        df['MA20'] = df['Close'].rolling(window=20).mean()
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    df['trade_signal'] = 0
    
    # Trend following with RSI filter
    df.loc[(df['Close'] > df['MA20']) & (df['RSI'] < 70), 'trade_signal'] = 1
    df.loc[(df['Close'] < df['MA20']) & (df['RSI'] > 30), 'trade_signal'] = -1
    
    # Position sizing based on RSI strength
    df['Recommended_Size'] = 0.1  # Base size
    rsi_strength = abs(df['RSI'] - 50) / 50
    df['Recommended_Size'] = df['Recommended_Size'] * (1 + rsi_strength)
    
    # Apply volatility adjustment if available
    if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
        bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        high_volatility = bb_width > bb_width.mean() + bb_width.std()
        df.loc[high_volatility, 'Recommended_Size'] *= 0.5
    
    # Clean up and limits
    df['Recommended_Size'] = df['Recommended_Size'].clip(0.05, 0.25)
    df['trade_signal'] = df['trade_signal'].fillna(0)
    df['Recommended_Size'] = df['Recommended_Size'].fillna(0.1)
    
    return df

def main():
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    runner = BacktestRunner()
    
    try:
        # Run backtest (summary is printed inside run_backtest)
        results = runner.run_backtest()
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    main() 