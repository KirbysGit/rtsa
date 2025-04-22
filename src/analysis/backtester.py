# src / analysis / backtester.py

# Description : This file contains the Backtester class, which is used to backtest the trading signals.

# Imports.
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Setup Logging.
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------

@dataclass
class TradeResult:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    return_pct: float
    trade_duration: int
    trade_type: str  # 'long' or 'short'
    exit_reason: str  # 'signal', 'stop_loss', 'target'

# -----------------------------------------------------------------------------------------------

class Backtester:
    """Backtests Trading Signals and Calculates Performance Metrics."""
    
    # -----------------------------------------------------------------------------------------------
    
    def __init__(self, initial_capital: float = 100000, 
                 transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.02,
                 max_position_pct: float = 0.25,  # Maximum 25% in single position
                 stop_loss_pct: float = 0.02):    # Default 2% account risk per trade
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.trades: List[TradeResult] = []
        self.equity_curve = None
        self.daily_returns = None
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.consecutive_losses = 0
        
    # -----------------------------------------------------------------------------------------------
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Enhanced backtest with better risk management."""
        if len(df) < 60:  # Minimum 60 days required
            raise ValueError("Insufficient data for backtesting")
        
        # Initialize Tracking Variables.
        current_position = 0
        equity = pd.Series(index=df.index, dtype=float)
        equity.iloc[0] = self.initial_capital
        self.trades = []
        
        # Risk Management Variables.
        max_loss_today = self.initial_capital * self.stop_loss_pct
        trailing_stop = None
        
        # Backtest Loop.
        for i in range(1, len(df)):
            current_date = df.index[i]
            prev_date = df.index[i-1]
            
            # Update Equity and Check for Risk Limits.
            if current_position != 0:
                pnl = position_size * (df['Close'].iloc[i] - df['Close'].iloc[i-1])
                equity.iloc[i] = equity.iloc[i-1] + pnl
                
                # Update Drawdown Tracking.
                peak_equity = equity.iloc[:i+1].max()
                current_drawdown = (peak_equity - equity.iloc[i]) / peak_equity
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                # Check for Risk Limits.
                if pnl < -max_loss_today:
                    logger.warning(f"Daily loss limit hit on {current_date}")
                    self._close_position(df, i, equity, 'risk_limit', current_position)
                    current_position = 0
                    continue
                
                # Update Trailing Stop if Enabled.
                if trailing_stop is not None:
                    if current_position == 1 and df['Close'].iloc[i] < trailing_stop:
                        self._close_position(df, i, equity, 'trailing_stop', current_position)
                        current_position = 0
                        continue
                    elif current_position == -1 and df['Close'].iloc[i] > trailing_stop:
                        self._close_position(df, i, equity, 'trailing_stop', current_position)
                        current_position = 0
                        continue
            else:
                equity.iloc[i] = equity.iloc[i-1]
            
            # Position Sizing with Kelly Criterion.
            if current_position == 0 and i < len(df) - 1:
                signal = df['trade_signal'].iloc[i]
                if signal != 0:
                    # Calculate Kelly Position Size.
                    win_rate = self._calculate_win_rate()
                    avg_win_loss_ratio = self._calculate_win_loss_ratio()
                    kelly_size = (win_rate - ((1 - win_rate) / avg_win_loss_ratio)) if avg_win_loss_ratio > 0 else 0.5
                    
                    # Apply Position Size Limits.
                    position_size = min(
                        equity.iloc[i] * min(kelly_size, self.max_position_pct),
                        equity.iloc[i] * df['Recommended_Size'].iloc[i]
                    )
                    
                    # Enter Position with Proper Risk Management.
                    entry_price = df['Close'].iloc[i]
                    stop_price = df['stop_loss'].iloc[i]
                    risk_amount = abs(entry_price - stop_price) / entry_price
                    
                    if risk_amount > self.stop_loss_pct:
                        position_size *= (self.stop_loss_pct / risk_amount)  # Scale Position to Limit Risk
                    
                    current_position = np.sign(signal)
                    trailing_stop = stop_price
                    
                    # Apply Transaction Costs.
                    equity.iloc[i] -= abs(position_size * entry_price * self.transaction_cost)
        
        # Update Equity Curve and Daily Returns.
        self.equity_curve = equity
        self.daily_returns = self.equity_curve.pct_change().fillna(0)
        
        # Calculate Performance Metrics.
        return self._calculate_performance_metrics()
    
    # -----------------------------------------------------------------------------------------------
    
    def _close_position(self, df: pd.DataFrame, i: int, equity: pd.Series, 
                       reason: str, current_position: int) -> None:
        """Helper Method to Close Positions with Proper Logging."""
        exit_price = df['Close'].iloc[i]
        pnl = position_size * (exit_price - entry_price) * current_position
        
        # Create Trade Result.
        trade = TradeResult(
            entry_date=entry_date,
            exit_date=df.index[i],
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            pnl=pnl,
            return_pct=(pnl / (position_size * entry_price)) * 100,
            trade_duration=(df.index[i] - entry_date).days,
            trade_type='long' if current_position == 1 else 'short',
            exit_reason=reason
        )
        self.trades.append(trade)
        
        # Update Consecutive Losses Tracking.
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Apply Transaction Costs.
        equity.iloc[i] -= abs(position_size * exit_price * self.transaction_cost)

    # -----------------------------------------------------------------------------------------------
    
    def _calculate_win_rate(self) -> float:
        """Calculate Win Rate from Historical Trades."""
        if not self.trades:
            return 0.5  # Default to 50% if No Trade History.
        return len([t for t in self.trades if t.pnl > 0]) / len(self.trades)

    def _calculate_win_loss_ratio(self) -> float:
        """Calculate Win/Loss Ratio from Historical Trades."""
        if not self.trades:
            return 1.0  # Default to 1.0 if No Trade History.
        
        # Calculate Winning and Losing Trades.
        winning_trades = [t.pnl for t in self.trades if t.pnl > 0]
        losing_trades = [abs(t.pnl) for t in self.trades if t.pnl < 0]
        
        # Calculate Average Winning and Losing Trades.
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 1
        
        return avg_win / avg_loss if avg_loss > 0 else 1.0
    
    # -----------------------------------------------------------------------------------------------
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate Comprehensive Performance Metrics."""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            raise ValueError("No equity curve available")
        
        # Calculate Trading Days and Annualization Factor.
        total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if total_days == 0:
            logger.warning("Insufficient data for annualization")
            annualization_factor = 252  # Default to 252 trading days
        else:
            annualization_factor = min(252, 365 / total_days * len(self.equity_curve))
        
        # Calculate Returns Safely. 
        try:
            total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1)
            
            # Ensure Return is Within Realistic Bounds.
            if total_return < -1:  # Cannot lose more than 100%
                logger.warning(f"Unrealistic return detected: {total_return:.2%}, capping at -100%")
                total_return = -0.99
            
            # Calculate Annualized Return Safely.
            if total_return >= -1:  # Only calculate if return is valid
                annualized_return = ((1 + total_return) ** (annualization_factor / len(self.equity_curve)) - 1)
            else:
                annualized_return = total_return  # Use Total Return if Calculation is Impossible.
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            total_return = 0
            annualized_return = 0
        
        # Calculate Drawdown with Safety Checks.
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        max_drawdown = min(abs(drawdown.min()), 1.0)  # Cap at 100%
        
        # Calculate Daily Returns and Volatility Safely.
        self.daily_returns = self.equity_curve.pct_change().fillna(0)
        self.daily_returns = self.daily_returns.clip(-0.5, 0.5)  # Remove Extreme Outliers.
        
        # Calculate Risk Metrics Safely.
        excess_returns = self.daily_returns - (self.risk_free_rate / 252)
        volatility = min(self.daily_returns.std() * np.sqrt(annualization_factor), 2.0)  # Cap at 200%
        
        # Calculate Sharpe and Sortino Ratios.
        if volatility > 0:
            sharpe_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / volatility
            downside_returns = self.daily_returns[self.daily_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else volatility
            sortino_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / downside_std
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calculate Trade Metrics Safely.
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades) * 100
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate Performance Metrics.
        metrics = {
            'start_date': self.equity_curve.index[0].strftime('%Y-%m-%d'),
            'end_date': self.equity_curve.index[-1].strftime('%Y-%m-%d'),
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'risk_metrics': {
                'volatility': volatility * 100,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': abs(annualized_return / max_drawdown) if max_drawdown > 0 else float('inf'),
                'var_95': np.percentile(self.daily_returns, 5) * 100,
                'var_99': np.percentile(self.daily_returns, 1) * 100
            },
            'trade_metrics': {
                'total_trades': len(self.trades),
                'avg_trade_duration': np.mean([t.trade_duration for t in self.trades]) if self.trades else 0,
                'avg_profit_per_trade': np.mean([t.pnl for t in self.trades]) if self.trades else 0,
                'largest_winner': max([t.pnl for t in self.trades], default=0),
                'largest_loser': min([t.pnl for t in self.trades], default=0),
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
        }
        
        return metrics
    
    # -----------------------------------------------------------------------------------------------
    
    def plot_equity_curve(self) -> None:
        """Plot Equity Curve with Drawdown and Trade Markers."""
        if self.equity_curve is None:
            logger.warning("No equity curve available. Run backtest first.")
            return
        
        try:
            # Use Default Style Instead of Seaborn.
            plt.style.use('default')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot Equity Curve with Better Formatting.
            ax1.plot(self.equity_curve.index, self.equity_curve, label='Portfolio Value', color='#1f77b4', linewidth=2)
            
            # Add Trade Markers with Better Visibility.
            for trade in self.trades:
                if trade.trade_type == 'long':
                    entry_color = '#2ecc71'  # Green
                    exit_color = '#e74c3c'   # Red
                else:
                    entry_color = '#e74c3c'   # Red
                    exit_color = '#2ecc71'    # Green
                
                ax1.scatter(trade.entry_date, self.equity_curve.loc[trade.entry_date], 
                          marker='^', color=entry_color, s=100, zorder=5)
                ax1.scatter(trade.exit_date, self.equity_curve.loc[trade.exit_date], 
                          marker='v', color=exit_color, s=100, zorder=5)
            
            # Improve Equity Curve Appearance.
            ax1.set_title('Portfolio Performance', fontsize=12, pad=10)
            ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Format y-axis with Dollar Signs and Commas.
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Plot Drawdown with Better Formatting.
            rolling_max = self.equity_curve.expanding().max()
            drawdown = ((self.equity_curve - rolling_max) / rolling_max) * 100
            ax2.fill_between(drawdown.index, drawdown, 0, color='#ff7675', alpha=0.5)
            ax2.set_title('Drawdown', fontsize=12, pad=10)
            ax2.set_ylabel('Drawdown (%)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Format y-axis with Percentage Signs.
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
            
            plt.tight_layout()
            plt.savefig('results/backtest_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            plt.close('all')  # Clean up any open figures