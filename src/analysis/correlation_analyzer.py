# src / analysis / correlation_analyzer.py

# Description : This file contains the CorrelationAnalyzer class, which is used to analyze the correlations between sentiment and price movements.

# Imports.
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Setup Logging.
logger = logging.getLogger(__name__)

# Correlation Analyzer Class.
class CorrelationAnalyzer:
    """Analyzes Correlations Between Sentiment and Price Movements."""
    
    # -----------------------------------------------------------------------------------------------
    
    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [1, 3, 5, 10, 20]
        self.results = {}
    
    # -----------------------------------------------------------------------------------------------
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between sentiment metrics and price movements
        
        Parameters:
        -----------
        df : pd.DataFrame
            Must contain columns: ['Close', 'Returns', 'avg_sentiment', 'sentiment_signal', 
                                 'weighted_sentiment', 'sentiment_momentum']
        
        Returns:
        --------
        Dict containing correlation analysis results
        """
        try:
            # Define Required Columns.
            required_cols = ['Close', 'Returns', 'avg_sentiment', 'sentiment_signal', 
                           'weighted_sentiment', 'sentiment_momentum']
            
            # Check if all required columns are present.
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Analyze Correlations.
            results = {
                'contemporaneous': self._analyze_contemporaneous_correlations(df),
                'lead_lag': self._analyze_lead_lag_relationships(df),
                'predictive_power': self._analyze_predictive_power(df),
                'regime_analysis': self._analyze_market_regimes(df)
            }
            
            # Store Results.
            self.results = results
            
            # Return Results.
            return results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise
    
    # -----------------------------------------------------------------------------------------------
    
    def _analyze_contemporaneous_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze Same-Period Correlations."""
        correlations = {}
        
        # Define Sentiment and Price Columns.
        sentiment_cols = ['avg_sentiment', 'sentiment_signal', 'weighted_sentiment', 'sentiment_momentum']
        price_cols = ['Returns', 'Close']
        
        # Calculate Correlations with p-values.
        for s_col in sentiment_cols:
            for p_col in price_cols:
                corr, p_value = stats.pearsonr(df[s_col].fillna(0), df[p_col].fillna(0))
                correlations[f"{s_col}_vs_{p_col}"] = {
                    'correlation': corr,
                    'p_value': p_value
                }
        
        # Add Spearman Rank Correlations.
        spearman_results = {}
        for s_col in sentiment_cols:
            for p_col in price_cols:
                corr, p_value = stats.spearmanr(df[s_col].fillna(0), df[p_col].fillna(0))
                spearman_results[f"{s_col}_vs_{p_col}"] = {
                    'correlation': corr,
                    'p_value': p_value
                }

        # Return Results.
        return {
            'pearson': correlations,
            'spearman': spearman_results
        }
    
    # -----------------------------------------------------------------------------------------------
    
    def _analyze_lead_lag_relationships(self, df: pd.DataFrame) -> Dict:
        """Analyze if sentiment leads or lags price movements."""
        results = {}
        
        for period in self.lookback_periods:
            future_returns = df['Returns'].shift(-period)
            past_returns = df['Returns'].shift(period)
            
            # Calculate Correlations with p-values.
            lead_corr, lead_p = stats.pearsonr(df['sentiment_signal'].fillna(0), future_returns.fillna(0))
            lag_corr, lag_p = stats.pearsonr(df['sentiment_signal'].fillna(0), past_returns.fillna(0))
            
            results[f'period_{period}'] = {
                'sentiment_leading': {'correlation': lead_corr, 'p_value': lead_p},
                'sentiment_lagging': {'correlation': lag_corr, 'p_value': lag_p},
                'net_predictive': lead_corr - lag_corr
            }
        
        # Return Results.
        return results
    
    # -----------------------------------------------------------------------------------------------
    
    def _analyze_predictive_power(self, df: pd.DataFrame) -> Dict:
        """Analyze the Predictive Power of Sentiment Signals."""
        results = {}
        
        # Analyze Predictive Power.
        for period in self.lookback_periods:
            future_returns = df['Returns'].shift(-period)
            
            # Convert sentiment_signal to numeric if it's categorical.
            sentiment_signal = df['sentiment_signal'].astype(float)
            
            # Group by Existing Sentiment Signal Levels (Already Discretized).
            sentiment_groups = pd.Categorical(sentiment_signal).categories
            returns_by_group = {}
            std_by_group = {}
            
            for level in sorted(sentiment_groups):
                mask = sentiment_signal == level
                if mask.any():
                    returns_by_group[f'Level_{int(level)}'] = future_returns[mask].mean()
                    std_by_group[f'Level_{int(level)}'] = future_returns[mask].std()
            
            # Calculate Information Coefficient (Rank Correlation).
            ic, p_value = stats.spearmanr(sentiment_signal, future_returns, nan_policy='omit')
            
            results[f'period_{period}'] = {
                'returns_by_quintile': returns_by_group,
                'std_by_quintile': std_by_group,
                'info_coefficient': ic,
                'p_value': p_value
            }
        
        # Return Results.
        return results
    
    # -----------------------------------------------------------------------------------------------
    
    def _analyze_market_regimes(self, df: pd.DataFrame) -> Dict:
        """Analyze how correlations vary under different market conditions."""
        
        # Calculate Volatility and Trend Indicators.
        returns_std = df['Returns'].rolling(20).std()
        returns_trend = df['Returns'].rolling(20).mean()
        
        # Define Regimes Using Fixed Thresholds Instead of Quantiles.
        vol_thresholds = [-np.inf, returns_std.quantile(0.33), returns_std.quantile(0.66), np.inf]
        trend_thresholds = [-np.inf, returns_trend.quantile(0.33), returns_trend.quantile(0.66), np.inf]
        
        # Create Regime Labels.
        vol_regimes = pd.cut(returns_std, bins=vol_thresholds, labels=['Low', 'Medium', 'High'])
        trend_regimes = pd.cut(returns_trend, bins=trend_thresholds, labels=['Down', 'Flat', 'Up'])
        
        # Analyze Regime Correlations.
        results = {}
        for regime_type, regimes in [('volatility_regime', vol_regimes), ('trend_regime', trend_regimes)]:
            
            regime_correlations = {}

            # Analyze Each Regime.
            for regime in regimes.unique():

                # Skip if Regime is Missing.
                if pd.isna(regime):
                    continue
                
                # Get Data for Current Regime.
                regime_data = df[regimes == regime]

                # Skip if Regime is Empty.
                if len(regime_data) > 0:
                    # Calculate Correlation.    
                    corr, p_value = stats.pearsonr(
                        regime_data['sentiment_signal'].astype(float).fillna(0),
                        regime_data['Returns'].fillna(0)
                    )

                    # Store Results.
                    regime_correlations[str(regime)] = {
                        'sentiment_returns_corr': corr,
                        'p_value': p_value,
                        'avg_sentiment': float(regime_data['avg_sentiment'].mean()),
                        'avg_return': float(regime_data['Returns'].mean()),
                        'n_observations': int(len(regime_data))
                    }
            
            # Store Results.
            results[regime_type] = regime_correlations
        
        # Return Results.
        return results
    
    # -----------------------------------------------------------------------------------------------
    
    def plot_correlation_analysis(self, save_path: Optional[str] = None) -> None:
        """Generate Visualization of Correlation Analysis Results."""
        if not self.results:
            logger.warning("No results available. Run analyze_correlations first.")
            return
            
        # Create Figure.
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Contemporaneous Correlations Heatmap.
        ax1 = plt.subplot(2, 2, 1)

        # Create Correlation Data.
        corr_data = pd.DataFrame({
            k: v['correlation'] 
            for k, v in self.results['contemporaneous']['pearson'].items()
        }, index=[0]).T

        # Plot Heatmap.
        sns.heatmap(corr_data, annot=True, cmap='RdYlBu', center=0, ax=ax1)
        ax1.set_title('Sentiment-Price Correlations')
        
        # 2. Lead-Lag Relationships.
        ax2 = plt.subplot(2, 2, 2)

        # Create Lead-Lag Data.
        lead_lag_data = pd.DataFrame({
            period: data['net_predictive']
            for period, data in self.results['lead_lag'].items()
        }, index=[0]).T

        # Plot Bar Chart.
        lead_lag_data.plot(kind='bar', ax=ax2)
        ax2.set_title('Net Predictive Power by Period')
        ax2.set_xlabel('Lookback Period')
        ax2.set_ylabel('Net Correlation')
        
        # 3. Returns by sentiment quintile.
        ax3 = plt.subplot(2, 2, 3)

        # Create Quintile Data.
        quintile_data = pd.DataFrame(
            self.results['predictive_power']['period_1']['returns_by_quintile'],
            index=[0]
        ).T

        # Plot Bar Chart.
        quintile_data.plot(kind='bar', ax=ax3)
        ax3.set_title('Returns by Sentiment Quintile')
        ax3.set_xlabel('Sentiment Level')
        ax3.set_ylabel('Average Return')
        
        # 4. Regime Analysis.
        ax4 = plt.subplot(2, 2, 4)

        # Create Regime Data.
        regime_data = pd.DataFrame({
            regime: data['sentiment_returns_corr']
            for regime, data in self.results['regime_analysis']['trend_regime'].items()
        }, index=[0]).T

        # Plot Bar Chart.
        regime_data.plot(kind='bar', ax=ax4)
        ax4.set_title('Correlation by Market Regime')
        ax4.set_xlabel('Market Regime')
        ax4.set_ylabel('Correlation')
        
        plt.tight_layout()
                
        # Save Figure.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation analysis plots saved to {save_path}")
        
        # Close Figure.
        plt.close() 