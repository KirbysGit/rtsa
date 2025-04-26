"""
Figure Generator Module for IEEE Publication-Ready Visualizations

This module provides functions for generating various visualizations from Reddit data processing.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
from src.utils.path_config import FIGURES_DIR

# Configure logging
logger = logging.getLogger(__name__)

class FigureGenerator:
    """Generates publication-ready figures following IEEE guidelines."""
    
    def __init__(self):
        """Initialize the FigureGenerator with IEEE style settings."""
        self.figures_dir = FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set IEEE publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            # Font settings
            'font.family': 'Times New Roman',
            'font.size': 10,
            'font.weight': 'normal',
            
            # Axes settings
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.grid.which': 'major',
            'axes.grid.axis': 'y',
            
            # Grid settings
            'grid.color': '#E5E5E5',
            'grid.linestyle': ':',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.5,
            
            # Tick settings
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.major.pad': 5,
            'ytick.major.pad': 5,
            
            # Legend settings
            'legend.fontsize': 9,
            'legend.title_fontsize': 10,
            'legend.frameon': True,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            
            # Figure settings
            'figure.figsize': [6.5, 4.5],
            'figure.dpi': 300,
            'figure.constrained_layout.use': True
        })
        
        # Define color palettes
        self.color_palette = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4']
        self.sequential_palette = 'Blues'
        self.diverging_palette = 'RdYlBu'
        
    def _style_figure(self, fig, ax, title: str, xlabel: str, ylabel: str):
        """Apply IEEE-style formatting to a figure."""
        # Set title and labels
        ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, labelpad=10, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, labelpad=10, fontsize=11, fontweight='bold')
        
        # Style grid
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, color='#E5E5E5')
        ax.xaxis.grid(False)
        
        # Remove top and right spines
        sns.despine(ax=ax)
        
        # Adjust legend if it exists
        if ax.get_legend():
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True,
                edgecolor='black',
                fancybox=False,
                fontsize=9
            )
        
    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save a figure with IEEE publication standards."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.figures_dir / f"{name}_{timestamp}.pdf"  # Save as PDF for publication
        
        # Save with publication-quality settings
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            format='pdf',
            transparent=False
        )
        
        # Also save a PNG version for quick viewing
        png_filename = filename.with_suffix('.png')
        fig.savefig(
            png_filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            format='png'
        )
        
        plt.close(fig)
        logger.info(f"Saved figures to {filename} and {png_filename}")
    
    def generate_subreddit_ticker_heatmap(self, df: pd.DataFrame) -> None:
        """Generate heatmap of ticker mentions across subreddits."""
        try:
            mention_data = []
            for _, row in df.iterrows():
                subreddit = row['subreddit']
                tickers = row['tickers']
                for ticker in tickers:
                    mention_data.append({'subreddit': subreddit, 'ticker': ticker})
            
            mention_df = pd.DataFrame(mention_data)
            pivot_table = pd.crosstab(mention_df['subreddit'], mention_df['ticker'])
            
            # Keep top 15 tickers by total mentions
            top_tickers = pivot_table.sum().nlargest(15).index
            pivot_table = pivot_table[top_tickers]
            
            # Create figure with IEEE-style dimensions
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap with enhanced styling
            sns.heatmap(
                pivot_table,
                cmap='YlOrRd',
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Number of Mentions'},
                ax=ax
            )
            
            self._style_figure(
                fig, ax,
                title='Distribution of Ticker Mentions Across Subreddits',
                xlabel='Stock Ticker Symbol',
                ylabel='Subreddit Community'
            )
            
            self._save_figure(fig, 'subreddit_ticker_heatmap')
            
        except Exception as e:
            logger.error(f"Error generating subreddit ticker heatmap: {str(e)}")
    
    def generate_confidence_sentiment_scatter(self, df: pd.DataFrame) -> None:
        """Generate scatter plot of ticker confidence vs sentiment."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot with enhanced styling
            sns.scatterplot(
                data=df,
                x='ticker_confidence',
                y='overall_sentiment',
                hue='confidence_class',
                alpha=0.6,
                s=100,  # Increased marker size
                ax=ax
            )
            
            self._style_figure(
                fig, ax,
                title='Relationship Between Ticker Confidence and Sentiment',
                xlabel='Ticker Confidence Score',
                ylabel='Overall Sentiment Score'
            )
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            self._save_figure(fig, 'confidence_sentiment_scatter')
            
        except Exception as e:
            logger.error(f"Error generating confidence sentiment scatter: {str(e)}")
    
    def generate_false_positive_breakdown(self, df: pd.DataFrame, rejection_reasons: Dict[str, int]) -> None:
        """Generate stacked bar chart of ticker rejection reasons."""
        try:
            reasons_df = pd.DataFrame.from_dict(
                rejection_reasons,
                orient='index',
                columns=['count']
            ).reset_index()
            reasons_df.columns = ['reason', 'count']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bar plot with enhanced styling
            sns.barplot(
                data=reasons_df,
                x='reason',
                y='count',
                palette='deep',
                ax=ax
            )
            
            self._style_figure(
                fig, ax,
                title='Analysis of Rejected Ticker Mentions by Reason',
                xlabel='Rejection Reason',
                ylabel='Number of Occurrences'
            )
            
            # Rotate x-labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            self._save_figure(fig, 'false_positive_breakdown')
            
        except Exception as e:
            logger.error(f"Error generating false positive breakdown: {str(e)}")
    
    def generate_confidence_distribution(self, df: pd.DataFrame) -> None:
        """Generate stacked bar chart of confidence classes per ticker with IEEE publication styling."""
        try:
            confidence_data = []
            for _, row in df.iterrows():
                for ticker in row['tickers']:
                    confidence_data.append({
                        'ticker': ticker,
                        'confidence_class': row['confidence_class']
                    })
            
            conf_df = pd.DataFrame(confidence_data)
            top_tickers = conf_df['ticker'].value_counts().nlargest(10).index
            conf_df = conf_df[conf_df['ticker'].isin(top_tickers)]
            
            # Create figure with IEEE dimensions and adjusted height
            fig, ax = plt.subplots(figsize=(8, 5))  # Increased width and height
            
            # Create stacked bar chart with monochromatic color scheme
            conf_pivot = pd.crosstab(conf_df['ticker'], conf_df['confidence_class'])
            
            # Define monochromatic gray color scheme for confidence levels
            confidence_colors = {
                'HIGH': '#556b6b',     # Dark slate gray
                'MEDIUM': '#a9a9a9',   # Medium gray
                'LOW': '#dcdcdc'       # Light gray
            }
            
            # Sort confidence classes from highest to lowest
            sorted_confidence = ['HIGH', 'MEDIUM', 'LOW']
            available_confidence = [c for c in sorted_confidence if c in conf_pivot.columns]
            conf_pivot = conf_pivot[available_confidence]
            
            # Create the stacked bar plot
            conf_pivot.plot(
                kind='bar',
                stacked=True,
                ax=ax,
                color=[confidence_colors[level] for level in available_confidence],
                width=0.8
            )
            
            # Clean up the plot style
            sns.despine(ax=ax)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            
            # Configure grid
            ax.grid(False)  # Remove all grid first
            ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
            
            # Style the figure with IEEE formatting
            ax.set_title('Confidence level distribution by ticker', 
                        pad=20, fontsize=12, fontweight='normal', 
                        font='Times New Roman')
            
            # Move x-label down for better spacing
            ax.set_xlabel('Stock ticker symbol', 
                         labelpad=15, fontsize=11, fontweight='normal',
                         font='Times New Roman')
            
            ax.set_ylabel('Number of mentions', 
                         labelpad=10, fontsize=11, fontweight='normal',
                         font='Times New Roman')
            
            # Rotate x-labels for better readability with more space
            plt.setp(ax.get_xticklabels(), 
                    rotation=45, ha='right', 
                    fontsize=9, fontweight='normal',
                    font='Times New Roman')
            
            plt.setp(ax.get_yticklabels(),
                    fontsize=9, fontweight='normal',
                    font='Times New Roman')
            
            # Add value labels on the bars with consistent styling
            for c in ax.containers:
                ax.bar_label(c, label_type='center',
                           fmt='%d',
                           padding=3,
                           fontsize=8,
                           fontweight='normal',
                           color='#333333')  # Dark gray for all labels
            
            # Place legend below the chart with more space
            legend = ax.legend(
                title='Confidence level',
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),  # Moved down
                ncol=3,
                frameon=False,
                fontsize=9,
                columnspacing=1.5  # Added spacing between legend columns
            )
            
            # Style legend title and text
            legend.get_title().set_fontsize(9)
            legend.get_title().set_fontweight('normal')
            legend.get_title().set_font('Times New Roman')
            
            for text in legend.get_texts():
                text.set_font('Times New Roman')
                text.set_fontsize(9)
                text.set_fontweight('normal')
            
            # Adjust layout to accommodate legend and labels
            plt.subplots_adjust(bottom=0.25)  # Increased bottom margin
            plt.tight_layout()
            
            # Additional adjustment after tight_layout to ensure legend fits
            plt.subplots_adjust(bottom=0.25)
            
            self._save_figure(fig, 'confidence_distribution')
            
        except Exception as e:
            logger.error(f"Error generating confidence distribution: {str(e)}")
            raise
    
    def generate_engagement_sentiment_chart(self, df: pd.DataFrame) -> None:
        """Generate bar chart of top tickers by engagement, colored by sentiment."""
        try:
            engagement_data = []
            for _, row in df.iterrows():
                for ticker in row['tickers']:
                    engagement_data.append({
                        'ticker': ticker,
                        'engagement': row['engagement_score'],
                        'sentiment': row['overall_sentiment']
                    })
            
            eng_df = pd.DataFrame(engagement_data)
            ticker_stats = eng_df.groupby('ticker').agg({
                'engagement': 'sum',
                'sentiment': 'mean'
            }).reset_index()
            
            top_tickers = ticker_stats.nlargest(10, 'engagement')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bar chart with enhanced styling
            bars = ax.bar(
                top_tickers['ticker'],
                top_tickers['engagement'],
                color=plt.cm.RdYlGn(
                    (top_tickers['sentiment'] + 1) / 2
                )
            )
            
            self._style_figure(
                fig, ax,
                title='Top 10 Tickers by Reddit Engagement',
                xlabel='Stock Ticker Symbol',
                ylabel='Total Engagement Score'
            )
            
            # Add colorbar with enhanced styling
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label(
                'Average Sentiment',
                rotation=270,
                labelpad=15,
                fontweight='bold',
                fontfamily='Times New Roman'
            )
            
            self._save_figure(fig, 'engagement_sentiment_chart')
            
        except Exception as e:
            logger.error(f"Error generating engagement sentiment chart: {str(e)}")
    
    def generate_all_figures(self, df: pd.DataFrame, rejection_reasons: Optional[Dict[str, int]] = None) -> None:
        """Generate all available figures from the data.
        
        Args:
            df: DataFrame with processed Reddit data
            rejection_reasons: Optional dictionary of rejection reasons and counts
        """
        logger.info("Generating all figures...")
        
        self.generate_subreddit_ticker_heatmap(df)
        self.generate_confidence_sentiment_scatter(df)
        if rejection_reasons:
            self.generate_false_positive_breakdown(df, rejection_reasons)
        self.generate_confidence_distribution(df)
        self.generate_engagement_sentiment_chart(df)
        
        logger.info("Completed generating all figures")
    
    def generate_technical_analysis_plot(self, stock_data: pd.DataFrame, 
                                      sentiment_data: Optional[pd.DataFrame] = None) -> None:
        """Generate a publication-ready technical analysis plot."""
        try:
            # Create figure with specific size for IEEE column width
            fig = plt.figure(figsize=(6.5, 8))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)
            
            # Price subplot
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            
            ticker = stock_data['Ticker'].iloc[0]
            
            # Plot price data with enhanced styling
            ax1.plot(stock_data['Date'], stock_data['Close'], 
                    label='Close price', color=self.color_palette[0], linewidth=1.5)
            ax1.plot(stock_data['Date'], stock_data['Price_MA'], 
                    label='5-day MA', color=self.color_palette[1], linestyle='--', alpha=0.7)
            
            # Add price range with subtle coloring
            ax1.fill_between(stock_data['Date'], stock_data['Low'], stock_data['High'],
                           alpha=0.1, color=self.color_palette[0], label='Daily range')
            
            # Volume subplot with neutral colors
            ax2.bar(stock_data['Date'], stock_data['Volume'],
                   color=self.color_palette[0], alpha=0.3, label='Volume')
            ax2.plot(stock_data['Date'], stock_data['Volume_MA'],
                    color=self.color_palette[0], alpha=0.8, label='Volume MA')
            
            # RSI/Sentiment subplot
            if sentiment_data is not None and not sentiment_data.empty:
                merged_data = pd.merge(stock_data, sentiment_data[['Date', 'overall_sentiment']], 
                                     on='Date', how='left')
                
                # Plot RSI line
                ax3.plot(merged_data['Date'], merged_data['RSI'],
                        color=self.color_palette[2], alpha=0.7, label='RSI')
                
                # Add sentiment scatter with custom colormap
                scatter = ax3.scatter(merged_data['Date'],
                                    merged_data['overall_sentiment'] * 100,
                                    c=merged_data['overall_sentiment'],
                                    cmap=self.diverging_palette,
                                    label='Sentiment',
                                    alpha=0.6,
                                    s=30)
                
                # Add colorbar with IEEE styling
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Sentiment score', fontsize=9, fontweight='bold')
                cbar.ax.tick_params(labelsize=8)
            else:
                ax3.plot(stock_data['Date'], stock_data['RSI'],
                        color=self.color_palette[2], label='RSI')
                ax3.axhline(y=30, color='#E5E5E5', linestyle='--', alpha=0.5)
                ax3.axhline(y=70, color='#E5E5E5', linestyle='--', alpha=0.5)
            
            # Style each subplot
            for ax in [ax1, ax2, ax3]:
                self._style_figure(fig, ax, '', '', '')
                ax.legend(loc='upper left', frameon=True, fontsize=8)
            
            # Set specific labels
            ax1.set_title(f"Technical analysis for {ticker}",
                         pad=20, fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
            ax3.set_ylabel('RSI/Sentiment', fontsize=11, fontweight='bold')
            
            # Format x-axis
            ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
            plt.setp(ax3.xaxis.get_majorticklabels(), 
                    rotation=45, ha='right', fontsize=9)
            
            # Save the figure
            self._save_figure(fig, f"{ticker}_technical_analysis")
            
        except Exception as e:
            logger.error(f"Error generating technical analysis plot: {str(e)}")
            raise

    def generate_multi_ticker_analysis(self, stock_data_dict: Dict[str, pd.DataFrame],
                                     sentiment_data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """Generate technical analysis plots for multiple tickers."""
        for ticker, stock_data in stock_data_dict.items():
            sentiment_data = sentiment_data_dict.get(ticker) if sentiment_data_dict else None
            try:
                self.generate_technical_analysis_plot(stock_data, sentiment_data)
                logger.info(f"Generated plot for {ticker}")
            except Exception as e:
                logger.error(f"Failed to generate plot for {ticker}: {str(e)}")
                continue

    def generate_model_performance_chart(self, performance_data: Dict[str, Dict], metric: str = 'roc_auc') -> None:
        """Generate a bar chart of model performance across tickers."""
        try:
            # Prepare data for plotting
            plot_data = []
            for ticker, result in performance_data.items():
                if result['status'] == 'success':
                    plot_data.append({
                        'ticker': ticker,
                        'metric_value': result['metrics'][metric],
                        'samples': result['n_samples']
                    })
            
            if not plot_data:
                logger.warning("No successful models found for performance chart")
                return
            
            # Convert to DataFrame and sort by metric value
            df = pd.DataFrame(plot_data)
            df = df.nlargest(10, 'metric_value')
            
            # Create figure with IEEE dimensions and manual layout
            fig = plt.figure(figsize=(10, 5), constrained_layout=False)
            ax = fig.add_axes([0.1, 0.15, 0.75, 0.75])
            
            # Create normalized colormap
            norm = plt.Normalize(df['samples'].min(), df['samples'].max())
            colors = plt.cm.Blues(norm(df['samples']))
            
            # Create bars with normalized colors and black edges
            bars = ax.bar(df['ticker'], df['metric_value'], 
                         color=colors,
                         edgecolor='black',     # Add black edges
                         linewidth=1)           # Set edge width
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom',
                       fontsize=9)
            
            # Style the figure
            metric_name = {
                'accuracy': 'Accuracy',
                'f1': 'F1 Score',
                'roc_auc': 'ROC-AUC'
            }.get(metric, metric)
            
            self._style_figure(
                fig, ax,
                title=f'Top 10 Tickers by Model {metric_name}',
                xlabel='Stock Ticker Symbol',
                ylabel=metric_name
            )
            
            # Create colorbar with proper normalization
            sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.75])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Number of Samples', rotation=270, labelpad=15)
            
            # Rotate x-axis labels
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            
            # Save the figure
            self._save_figure(fig, f'model_performance_{metric}')
            
        except Exception as e:
            logger.error(f"Error generating model performance chart: {str(e)}")
            raise
