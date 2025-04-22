# src / strategy / baseline_strategy.py

# Description : This file contains the BaselineStrategy class, which is used to implement the baseline trading strategy.

# Imports.

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Setup Logging.
logger = logging.getLogger(__name__)

# Baseline Strategy Class.
class BaselineStrategy:
    """Baseline Trading Strategy using Sentiment Signals."""
    
    # -----------------------------------------------------------------------------------------------
    
    def __init__(
        self,
        lookback_period: int = 5,
        position_size: float = 1.0,
        stop_loss: float = 0.02,
        take_profit: float = 0.05,
        transaction_cost: float = 0.001
    ):
        """
        Initialize Strategy Parameters.
        
        Parameters :
        -----------

        lookback_period : int
            Period for Signal Generation (default: 5 based on correlation analysis)
        position_size : float
            Base Position Size as Fraction of Capital (0.0 to 1.0)
        stop_loss : float
            Stop Loss Level as Fraction of Position Value
        take_profit : float
            Take Profit Level as Fraction of Position Value
        transaction_cost : float
            Transaction Cost as Fraction of Trade Value

        """
        self.lookback_period = lookback_period
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.transaction_cost = transaction_cost
        
        # Trading State.
        self.position = 0
        self.entry_price = 0
        self.capital = 1.0      
        self.trades = []
        
    # -----------------------------------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Trading Signals from Sentiment Data."""
        signals = pd.DataFrame(index=df.index)
        
        # 1. Use Weighted Sentiment as Primary Signal (Strongest Correlation).
        signals['raw_signal'] = -1 * df['weighted_sentiment']  # Negative Correlation.
        
        # 2. Generate Regime Indicators.
        signals['volatility'] = df['Returns'].rolling(20).std()
        signals['trend'] = df['Returns'].rolling(20).mean()
        
        # 3. Generate Trading Signals.
        signals['signal'] = np.where(
            signals['raw_signal'] > signals['raw_signal'].rolling(self.lookback_period).mean(),
            1,  # Buy Signal
            np.where(
                signals['raw_signal'] < signals['raw_signal'].rolling(self.lookback_period).mean(),
                -1,  # Sell Signal
                0  # No Signal
            )
        )
        
        # 4. Position Sizing based on Signal Strength.
        signal_strength = (signals['raw_signal'] - signals['raw_signal'].rolling(self.lookback_period).mean()).abs()
        signals['position_size'] = (signal_strength / signal_strength.rolling(self.lookback_period).max()).fillna(0)
        signals['position_size'] = signals['position_size'].clip(0, 1) * self.position_size
        
        return signals
    
    def calculate_returns(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """Calculate Strategy Returns and Metrics."""
        returns = pd.DataFrame(index=df.index)
        
        # Initialize State Variables.
        position = 0
        entry_price = 0
        trades = []
        capital = self.capital
        
        # Calculate Returns and Metrics.
        for i in range(len(df)):
            date = df.index[i]
            current_price = df['Close'].iloc[i]
            signal = signals['signal'].iloc[i]
            size = signals['position_size'].iloc[i]
            
            # Check Stop Loss / Take Profit.
            if position != 0:
                pnl = (current_price - entry_price) / entry_price * position
                if abs(pnl) >= self.stop_loss or abs(pnl) >= self.take_profit:
                    # Close Position.
                    capital *= (1 + pnl - self.transaction_cost)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': 'sl/tp'
                    })
                    position = 0
            
            # Process New Signals.
            if signal != 0 and position != signal:
                if position != 0:
                    # Close Existing Position.
                    pnl = (current_price - entry_price) / entry_price * position
                    capital *= (1 + pnl - self.transaction_cost)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': 'signal'
                    })
                
                # Open New Position.
                position = signal * size
                entry_price = current_price
                entry_date = date
            
            # Record Daily State.
            returns.loc[date, 'position'] = position
            returns.loc[date, 'capital'] = capital
        
        # Calculate Metrics.
        daily_returns = returns['capital'].pct_change().fillna(0)
        
        metrics = {
            'total_return': (capital / self.capital - 1),
            'annual_return': daily_returns.mean() * 252,
            'annual_volatility': daily_returns.std() * np.sqrt(252),
            'sharpe_ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0,
            'max_drawdown': (returns['capital'] / returns['capital'].cummax() - 1).min(),
            'win_rate': sum(t['pnl'] > 0 for t in trades) / len(trades) if trades else 0,
            'avg_trade': np.mean([t['pnl'] for t in trades]) if trades else 0,
            'n_trades': len(trades)
        }
        
        # Return Results.
        return {
            'returns': returns,
            'trades': trades,
            'metrics': metrics
        }
    
    # -----------------------------------------------------------------------------------------------
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """Run Backtest on Historical Data."""
        try:
            # Generate Signals.
            signals = self.generate_signals(df)
            
            # Calculate Returns and Metrics.
            results = self.calculate_returns(df, signals)
            
            # Log Results.
            logger.info("Backtest completed successfully")
            logger.info(f"Total Return: {results['metrics']['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
            logger.info(f"Win Rate: {results['metrics']['win_rate']:.2%}")
            logger.info(f"Number of Trades: {results['metrics']['n_trades']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise 