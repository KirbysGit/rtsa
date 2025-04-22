import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    def __init__(self, output_dir="results"):
        """Initialize risk analyzer with risk thresholds"""
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Adjusted risk thresholds for current market conditions
        self.volatility_thresholds = {
            'low': 0.25,    # 25% annualized volatility
            'medium': 0.40,  # 40% annualized volatility
            'high': 0.55    # 55% annualized volatility
        }
        
        self.volume_thresholds = {
            'low': 0.5,     # 50% of 20-day average
            'medium': 1.0,  # At 20-day average
            'high': 2.0     # 200% of 20-day average
        }
        
        self.sentiment_thresholds = {
            'very_bearish': -2.0,
            'bearish': -0.5,
            'neutral': 0.5,
            'bullish': 2.0,
            'very_bullish': 3.0
        }
        
        # Position sizing parameters
        self.max_position_size = 1.0
        self.min_position_size = 0.1
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """Main analysis method combining all metrics and signals"""
        # Calculate basic risk metrics
        risk_metrics, df = self.calculate_risk_metrics(df)
        
        # Detect market regime
        df = self.detect_market_regime(df)
        
        # Calculate position sizing
        df = self.recommend_position_size(df)
        
        # Generate trading signals
        signals = self.generate_trading_signals(df)
        
        # Combine all signals and metrics
        df = pd.concat([df, signals], axis=1)
        
        # Generate market summary
        market_summary = self.generate_market_summary(df, risk_metrics)
        
        return market_summary, df
    
    def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect current market regime based on multiple factors"""
        df['Market_Regime'] = pd.Series('normal', index=df.index)
        
        # Define regime conditions
        high_vol_regime = df['Historical_Volatility'] > self.volatility_thresholds['high']
        high_sent_regime = df['Risk_Adjusted_Sentiment'] > self.sentiment_thresholds['bullish']
        crisis_regime = (df['Historical_Volatility'] > self.volatility_thresholds['high']) & \
                       (df['Risk_Adjusted_Sentiment'] < self.sentiment_thresholds['very_bearish'])
        recovery_regime = (df['Historical_Volatility'] > self.volatility_thresholds['medium']) & \
                         (df['Risk_Adjusted_Sentiment'] > self.sentiment_thresholds['bullish'])
        
        # Assign regimes
        df.loc[high_vol_regime, 'Market_Regime'] = 'high_volatility'
        df.loc[high_sent_regime, 'Market_Regime'] = 'high_sentiment'
        df.loc[crisis_regime, 'Market_Regime'] = 'crisis'
        df.loc[recovery_regime, 'Market_Regime'] = 'recovery'
        
        return df
    
    def recommend_position_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate recommended position size based on risk factors"""
        # Base position size
        base_size = self.max_position_size
        
        # Volatility adjustment (reduce size as volatility increases)
        vol_factor = 1 - (df['Historical_Volatility'] / (2 * self.volatility_thresholds['high']))
        
        # Sentiment adjustment (increase size with positive sentiment)
        sent_factor = (1 + df['Risk_Adjusted_Sentiment']) / 2
        
        # Volume adjustment (reduce size in low volume conditions)
        vol_adj = df['Volume_Ratio'].clip(0.5, 1.5) / 1.5
        
        # Combine factors
        df['Recommended_Size'] = (base_size * vol_factor * sent_factor * vol_adj)\
            .clip(self.min_position_size, self.max_position_size)
        
        return df
    
    def generate_market_summary(self, df: pd.DataFrame, risk_metrics: Dict) -> Dict:
        """Generate a clean, user-friendly market summary"""
        latest_data = df.iloc[-1]
        
        summary = {
            'market_condition': {
                'regime': latest_data['Market_Regime'],
                'risk_level': latest_data['Volatility_Risk_Level'],
                'sentiment': self._classify_sentiment(latest_data['Risk_Adjusted_Sentiment']),
                'volume_condition': latest_data['Volume_Risk_Level']
            },
            'risk_metrics': {
                'volatility': f"{risk_metrics['Historical_Volatility']:.1%}",
                'var_95': f"{risk_metrics['VaR_95']:.1%}",
                'expected_shortfall': f"{risk_metrics['ES_95']:.1%}",
                'combined_risk': f"{latest_data['Combined_Risk_Score']:.2f}"
            },
            'trading_signals': {
                'primary_signal': self._interpret_trading_signal(latest_data['trade_signal']),
                'position_size': f"{latest_data['Recommended_Size']:.1%}",
                'confidence': self._calculate_signal_confidence(df)
            },
            'key_levels': {
                'stop_loss': self._calculate_stop_loss(df),
                'target': self._calculate_price_target(df)
            }
        }
        
        return summary
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment into readable categories"""
        if sentiment_score <= self.sentiment_thresholds['very_bearish']:
            return 'Very Bearish'
        elif sentiment_score <= self.sentiment_thresholds['bearish']:
            return 'Bearish'
        elif sentiment_score <= self.sentiment_thresholds['neutral']:
            return 'Neutral'
        elif sentiment_score <= self.sentiment_thresholds['bullish']:
            return 'Bullish'
        else:
            return 'Very Bullish'
    
    def _interpret_trading_signal(self, signal: int) -> str:
        """Convert numerical signal to readable recommendation"""
        signal_map = {
            2: "Strong Buy - Aggressive Entry",
            1: "Buy - Gradual Entry",
            0: "Hold - Monitor Position",
            -1: "Sell - Reduce Position",
            -2: "Strong Sell - Exit Position"
        }
        return signal_map.get(signal, "No Clear Signal")
    
    def _calculate_signal_confidence(self, df: pd.DataFrame) -> str:
        """Calculate confidence level in the trading signal"""
        latest = df.iloc[-1]
        
        # Count confirming factors
        confirming_factors = 0
        total_factors = 4
        
        # Sentiment confirmation
        if latest['Risk_Adjusted_Sentiment'] * latest['trade_signal'] > 0:
            confirming_factors += 1
        
        # Volume confirmation
        if latest['Volume_Ratio'] > self.volume_thresholds['medium']:
            confirming_factors += 1
        
        # Trend confirmation
        if 'Trend_Strength' in latest and latest['Trend_Strength'] * latest['trade_signal'] > 0:
            confirming_factors += 1
        
        # Risk level appropriate for signal
        if (latest['trade_signal'] > 0 and latest['Volatility_Risk_Level'] in ['low', 'medium']) or \
           (latest['trade_signal'] < 0 and latest['Volatility_Risk_Level'] in ['high', 'very_high']):
            confirming_factors += 1
        
        confidence = (confirming_factors / total_factors) * 100
        
        if confidence >= 75:
            return "High"
        elif confidence >= 50:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_stop_loss(self, df: pd.DataFrame) -> float:
        """Calculate suggested stop loss level"""
        latest_price = df['Close'].iloc[-1]
        volatility = df['Historical_Volatility'].iloc[-1]
        var_95 = df['VaR_95'].iloc[-1]
        
        # Wider stops in high volatility conditions
        stop_multiple = 2 if volatility > self.volatility_thresholds['high'] else 1.5
        
        # Use both VaR and volatility for stop loss calculation
        stop_loss_pct = max(abs(var_95), volatility / np.sqrt(252)) * stop_multiple
        return latest_price * (1 - stop_loss_pct)
    
    def _calculate_price_target(self, df: pd.DataFrame) -> float:
        """Calculate suggested price target"""
        latest_price = df['Close'].iloc[-1]
        volatility = df['Historical_Volatility'].iloc[-1]
        var_95 = df['VaR_95'].iloc[-1]
        
        # Higher targets in positive sentiment conditions
        target_multiple = 2.5 if df['Risk_Adjusted_Sentiment'].iloc[-1] > 0 else 2
        
        # Use both VaR and volatility for target calculation
        target_pct = max(abs(var_95), volatility / np.sqrt(252)) * target_multiple
        return latest_price * (1 + target_pct)
    
    def calculate_risk_metrics(self, df):
        """Calculate various risk metrics and add them to the DataFrame"""
        risk_metrics = {}
        df = df.copy()
        
        # Calculate returns if not present
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        returns = df['Returns'].dropna()
        
        # Value at Risk (VaR) and Expected Shortfall (ES)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        es_95 = returns[returns <= var_95].mean()
        
        # Store in both risk_metrics and DataFrame
        risk_metrics['VaR_95'] = var_95
        risk_metrics['VaR_99'] = var_99
        risk_metrics['ES_95'] = es_95
        
        # Add to DataFrame as constant columns
        df['VaR_95'] = var_95
        df['VaR_99'] = var_99
        df['ES_95'] = es_95
        
        # Volatility Risk
        df['Historical_Volatility'] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)
        risk_metrics['Historical_Volatility'] = df['Historical_Volatility'].mean()
        
        # Volume-based Risk Metrics
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Trend'] = df['Volume'].rolling(window=5, min_periods=1).mean() / \
                            df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Sentiment and Engagement Risk
        df['Sentiment_Volatility'] = df['avg_sentiment'].rolling(window=20, min_periods=1).std()
        df['Risk_Adjusted_Sentiment'] = df['avg_sentiment'] / (df['Sentiment_Volatility'] + 0.001)
        
        if 'avg_engagement' in df.columns:
            df['Engagement_Risk'] = df['avg_engagement'].rolling(window=20, min_periods=1).std()
            df['Volume_Weighted_Sentiment'] = df['avg_sentiment'] * (df['Volume_Ratio'] ** 0.5)
        else:
            df['Engagement_Risk'] = 0
            df['Volume_Weighted_Sentiment'] = df['avg_sentiment']
        
        # Fill NaN values using forward fill then backward fill
        columns_to_fill = ['Historical_Volatility', 'Sentiment_Volatility', 
                          'Engagement_Risk', 'Volume_Ratio', 'Volume_Trend']
        for col in columns_to_fill:
            df[col] = df[col].ffill().bfill()
        
        # Risk Level Classifications
        df['Volatility_Risk_Level'] = self._classify_risk_level(
            df['Historical_Volatility'], self.volatility_thresholds)
        df['Volume_Risk_Level'] = self._classify_risk_level(
            df['Volume_Ratio'], self.volume_thresholds)
        
        # Combined Risk Score with more factors
        combined_risk = self._calculate_combined_risk(df)
        df['Combined_Risk_Score'] = combined_risk
        risk_metrics['Combined_Risk_Score'] = combined_risk
        
        # Store risk metrics
        for col in ['Historical_Volatility', 'Sentiment_Volatility', 'Engagement_Risk',
                    'Volume_Ratio', 'Volume_Trend']:
            risk_metrics[col] = df[col].mean()
        
        return risk_metrics, df
    
    def _classify_risk_level(self, series, thresholds):
        """Classify risk levels as low/medium/high"""
        conditions = [
            series <= thresholds['low'],
            (series > thresholds['low']) & (series <= thresholds['medium']),
            (series > thresholds['medium']) & (series <= thresholds['high']),
            series > thresholds['high']
        ]
        choices = ['low', 'medium', 'high', 'very_high']
        return pd.Series(np.select(conditions, choices, default='medium'), index=series.index)
    
    def _calculate_combined_risk(self, df):
        """Calculate combined risk score based on multiple factors"""
        try:
            # Normalize factors to 0-1 scale
            price_vol = df['Historical_Volatility'] / self.volatility_thresholds['high']
            sent_vol = df['Sentiment_Volatility'] / df['Sentiment_Volatility'].max()
            vol_risk = df['Volume_Ratio'].clip(0, 2) / 2
            
            # Risk-adjusted sentiment factor
            sent_impact = (1 + df['Risk_Adjusted_Sentiment']) / 2
            
            # Combined score with dynamic weights
            combined = (
                0.35 * price_vol +
                0.25 * sent_vol +
                0.20 * vol_risk +
                0.20 * (1 - sent_impact)  # Inverse of positive sentiment impact
            ).clip(0, 1)  # Ensure 0-1 range
            
            return combined
        except Exception as e:
            logger.error(f"Error calculating combined risk: {str(e)}")
            return pd.Series(0.5, index=df.index)  # Default to medium risk
    
    def generate_trading_signals(self, df):
        """Generate trading signals with more granular risk assessment"""
        signals = pd.DataFrame(index=df.index)
        
        # Risk levels
        signals['volatility_risk'] = df['Volatility_Risk_Level']
        signals['volume_risk'] = df['Volume_Risk_Level']
        
        # Sentiment signals
        signals['sentiment_signal'] = np.where(
            df['Risk_Adjusted_Sentiment'] > 0.5, 1,
            np.where(df['Risk_Adjusted_Sentiment'] < -0.5, -1, 0)
        )
        
        # Volume-weighted sentiment
        signals['volume_weighted_signal'] = np.where(
            (df['Volume_Weighted_Sentiment'] > 0) & (df['Volume_Ratio'] > 1), 1,
            np.where((df['Volume_Weighted_Sentiment'] < 0) & (df['Volume_Ratio'] > 1), -1, 0)
        )
        
        # Combined trading signal
        conditions = [
            # Strong buy: positive sentiment, low risk, high volume
            (df['Risk_Adjusted_Sentiment'] > 0.5) & 
            (df['Volatility_Risk_Level'].isin(['low', 'medium'])) &
            (df['Volume_Risk_Level'].isin(['medium', 'high'])),
            
            # Moderate buy: positive sentiment, medium risk
            (df['Risk_Adjusted_Sentiment'] > 0) &
            (df['Volatility_Risk_Level'] != 'very_high'),
            
            # Strong sell: negative sentiment, high risk
            (df['Risk_Adjusted_Sentiment'] < -0.5) |
            (df['Volatility_Risk_Level'] == 'very_high'),
            
            # Moderate sell: negative sentiment or high risk
            (df['Risk_Adjusted_Sentiment'] < 0) |
            (df['Volatility_Risk_Level'] == 'high')
        ]
        choices = [2, 1, -2, -1]  # 2: strong buy, 1: buy, -1: sell, -2: strong sell
        signals['trade_signal'] = np.select(conditions, choices, default=0)
        
        return signals
    
    def plot_risk_metrics(self, df, ticker):
        """Enhanced plot with risk metrics and signals"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            
            # Price and Volatility
            ax1.plot(df.index, df['Close'], 'b-', label='Price')
            ax1.set_title(f'{ticker} Risk Analysis Dashboard')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left')
            
            ax1_2 = ax1.twinx()
            ax1_2.plot(df.index, df['Historical_Volatility'], 'r--', label='Volatility')
            ax1_2.set_ylabel('Volatility')
            ax1_2.legend(loc='upper right')
            
            # Sentiment and Volume
            ax2.plot(df.index, df['Risk_Adjusted_Sentiment'], 'g-', label='Risk-Adj Sentiment')
            ax2.plot(df.index, df['Volume_Weighted_Sentiment'], 'y--', label='Vol-Weight Sentiment')
            ax2.set_ylabel('Sentiment Scores')
            ax2.legend(loc='upper left')
            
            # Risk Levels and Signals
            ax3.plot(df.index, df['Combined_Risk_Score'], 'r-', label='Combined Risk')
            ax3.plot(df.index, df['Volume_Ratio'], 'b--', label='Volume Ratio')
            ax3.set_ylabel('Risk Levels')
            ax3.legend(loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{ticker}_risk_dashboard.png")
            plt.close()
            
            logger.info(f"Risk dashboard saved to {self.output_dir}/{ticker}_risk_dashboard.png")
            
        except Exception as e:
            logger.error(f"Error plotting risk metrics: {str(e)}")
    
    def detect_trend_reversal(self, df):
        """Detect potential trend reversals with enhanced signals"""
        signals = pd.DataFrame(index=df.index)
        
        # Price-based reversal signals
        if 'BB_lower' in df.columns and 'RSI' in df.columns:
            signals['Price_Reversal'] = (
                (df['Close'] < df['BB_lower']) & 
                (df['RSI'] < 30)
            ).astype(int)
        else:
            # Calculate basic oversold condition if BB and RSI not available
            signals['Price_Reversal'] = (
                (df['Returns'].rolling(window=14).mean() < -0.02) &
                (df['Returns'] > 0)
            ).astype(int)
        
        # Sentiment-based reversal signals
        signals['Sentiment_Reversal'] = (
            (df['Risk_Adjusted_Sentiment'] > 0) & 
            (df['Returns'] < 0)
        ).astype(int)
        
        # Volume-based confirmation
        signals['Volume_Confirmation'] = (
            df['Volume_Ratio'] > self.volume_thresholds['medium']
        ).astype(int)
        
        # Trend strength
        signals['Trend_Strength'] = pd.Series(0, index=df.index)
        
        # Bullish reversal conditions
        bullish_conditions = (
            (signals['Price_Reversal'] == 1) &
            (signals['Sentiment_Reversal'] == 1) &
            (df['Volume_Ratio'] > self.volume_thresholds['medium'])
        )
        
        # Bearish reversal conditions
        bearish_conditions = (
            (df['Returns'].rolling(window=5).mean() > 0.02) &
            (df['Risk_Adjusted_Sentiment'] < -0.5) &
            (df['Volume_Ratio'] > self.volume_thresholds['medium'])
        )
        
        # Assign trend strength
        signals.loc[bullish_conditions, 'Trend_Strength'] = 1
        signals.loc[bearish_conditions, 'Trend_Strength'] = -1
        
        return signals 