# src / analysis / data_analyzer.py

# Description : This file contains the DataAnalyzer class, which is used to analyze the data.

# Imports.
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# Setup Logging.
logger = logging.getLogger(__name__)

# Data Analyzer Class.
class DataAnalyzer:
    """Analyzes Data for Insights and Visualizations."""
    
    # -----------------------------------------------------------------------------------------------
    
    def __init__(self, output_dir="results"):
        """Initialize the Data Analyzer."""
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    # -----------------------------------------------------------------------------------------------
    
    def analyze_correlations(self, merged_data):
        """Analyze Correlations Between Sentiment and Price Movements."""
        try:
            # Calculate Correlations.
            correlation_cols = ['Close', 'Returns', 'avg_sentiment', 
                              'total_posts', 'total_comments', 'avg_engagement']
            correlations = merged_data[correlation_cols].corr()
            
            # Create Correlation Heatmap.
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Key Metrics')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/correlation_heatmap.png")
            plt.close()
            
            # Calculate Lagged Correlations (Sentiment as Leading Indicator).
            lags = [1, 2, 3, 5]  # Look at different day lags.
            lag_corrs = {}
            
            for lag in lags:
                # Calculate Lagged Sentiment.
                lagged_sentiment = merged_data['avg_sentiment'].shift(lag)

                # Calculate Correlation.
                corr = stats.pearsonr(
                    merged_data['Returns'].dropna(),
                    lagged_sentiment.dropna()
                )

                # Store Results.
                lag_corrs[lag] = {'correlation': corr[0], 'p_value': corr[1]}
            
            # Return Results.
            return {
                'correlations': correlations,
                'lagged_correlations': lag_corrs
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return None
    
    # -----------------------------------------------------------------------------------------------
    
    def create_visualizations(self, merged_data, ticker):
        """Create Visualizations of Price and Sentiment Data."""
        try:
            # 1. Price vs Sentiment Plot.
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Price Plot.
            ax1.plot(merged_data.index, merged_data['Close'], 'b-', label='Price')
            ax1.set_title(f'{ticker} Price and Sentiment Over Time')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left')
            
            # Sentiment Plot.
            ax2.plot(merged_data.index, merged_data['avg_sentiment'], 'g-', label='Sentiment')
            ax2.set_ylabel('Sentiment Score')
            ax2.legend(loc='upper left')
            
            # Save Figure.
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{ticker}_price_sentiment.png")
            plt.close()
            
            # 2. Comment Volume vs Price Changes.
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(merged_data.index, merged_data['total_comments'], alpha=0.3, label='Comments')
            ax2 = ax.twinx()
            ax2.plot(merged_data.index, merged_data['Returns'], 'r-', label='Returns')
            
            # Set Title and Labels.
            ax.set_title(f'{ticker} Comment Volume vs Returns')
            ax.set_ylabel('Number of Comments')
            ax2.set_ylabel('Returns (%)')
            
            # Set Legend.
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Save Figure.
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{ticker}_volume_returns.png")
            plt.close()
            
            # 3. Engagement Patterns.
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(merged_data.index, merged_data['avg_engagement'], 'b-', label='Engagement')
            ax.fill_between(merged_data.index, merged_data['avg_engagement'], alpha=0.3)
            
            # Set Title and Labels.
            ax.set_title(f'{ticker} Reddit Engagement Over Time')
            ax.set_ylabel('Engagement Score')
            ax.legend()
            
            # Save Figure.
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{ticker}_engagement.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}") 