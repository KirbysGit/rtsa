import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentSignalGenerator:
    """Generates trading signals based on sentiment analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'sentiment_thresholds': {
                'very_bullish': 0.6,
                'bullish': 0.2,
                'neutral': -0.2,
                'bearish': -0.6
            },
            'volume_threshold': 0.75,  # 75th percentile for high volume
            'momentum_period': 5,      # 5-period sentiment momentum
            'sentiment_ma': 20,        # 20-period sentiment moving average
            'min_posts': 10           # Minimum posts for valid sentiment
        }
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on sentiment analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Must contain columns: ['avg_sentiment', 'total_posts', 'total_comments', 'avg_engagement']
            
        Returns:
        --------
        pd.DataFrame
            Original dataframe with additional columns:
            - sentiment_signal: [-1, 0, 1] trading signal
            - sentiment_momentum: Rate of change in sentiment
            - weighted_sentiment: Volume-adjusted sentiment
            - sentiment_confidence: Signal confidence score (0-1)
            - sentiment_ma: Moving average of sentiment
        """
        try:
            required_cols = ['avg_sentiment', 'total_posts', 'total_comments', 'avg_engagement']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            df = df.copy()
            
            # 1. Basic sentiment signals
            df['sentiment_signal'] = self._calculate_sentiment_signal(df)
            
            # 2. Sentiment momentum (rate of change)
            df['sentiment_momentum'] = self._calculate_sentiment_momentum(df)
            
            # 3. Volume-weighted sentiment
            df['weighted_sentiment'] = self._calculate_weighted_sentiment(df)
            
            # 4. Sentiment moving average
            df['sentiment_ma'] = df['avg_sentiment'].rolling(
                window=self.config['sentiment_ma']
            ).mean()
            
            # 5. Confidence score
            df['sentiment_confidence'] = self._calculate_confidence_score(df)
            
            # 6. Adjust signals based on confidence
            df['sentiment_signal'] = df['sentiment_signal'] * df['sentiment_confidence']
            
            logger.info(f"Generated sentiment signals with {df['sentiment_signal'].abs().mean():.2%} average signal strength")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating sentiment signals: {str(e)}")
            raise
    
    def _calculate_sentiment_signal(self, df: pd.DataFrame) -> pd.Series:
        """Convert raw sentiment into trading signals (-1, 0, 1)"""
        signals = pd.Series(0, index=df.index)
        
        # Only generate signals when we have sufficient posts
        valid_sentiment = df['total_posts'] >= self.config['min_posts']
        
        # Generate signals based on thresholds
        thresholds = self.config['sentiment_thresholds']
        signals.loc[valid_sentiment & (df['avg_sentiment'] > thresholds['bullish'])] = 1
        signals.loc[valid_sentiment & (df['avg_sentiment'] < thresholds['bearish'])] = -1
        
        # Strengthen signals for extreme sentiment
        signals.loc[valid_sentiment & (df['avg_sentiment'] > thresholds['very_bullish'])] = 1.5
        signals.loc[valid_sentiment & (df['avg_sentiment'] < -thresholds['very_bullish'])] = -1.5
        
        return signals
    
    def _calculate_sentiment_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the rate of change in sentiment"""
        momentum = df['avg_sentiment'].diff(self.config['momentum_period'])
        
        # Normalize momentum to [-1, 1] range
        momentum = momentum / df['avg_sentiment'].rolling(self.config['momentum_period']).std()
        return momentum.fillna(0).clip(-1, 1)
    
    def _calculate_weighted_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """Weight sentiment by post volume and engagement"""
        # Calculate volume factor
        volume_ma = df['total_posts'].rolling(20).mean()
        volume_factor = (df['total_posts'] / volume_ma).clip(0.5, 2)
        
        # Calculate engagement factor
        engagement_factor = (df['avg_engagement'] / df['avg_engagement'].rolling(20).mean()).clip(0.5, 2)
        
        # Combine factors
        weight = (volume_factor * 0.7 + engagement_factor * 0.3)
        return df['avg_sentiment'] * weight
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score (0-1) for sentiment signals"""
        confidence = pd.Series(0.0, index=df.index)
        
        # 1. Sentiment strength
        sentiment_strength = abs(df['avg_sentiment'])
        
        # 2. Volume significance
        volume_percentile = df['total_posts'].rank(pct=True)
        
        # 3. Engagement quality
        engagement_factor = df['avg_engagement'].rank(pct=True)
        
        # 4. Momentum alignment
        momentum_alignment = (
            np.sign(df['avg_sentiment']) == 
            np.sign(df['avg_sentiment'].diff(self.config['momentum_period']))
        ).astype(float)
        
        # Combine factors with weights
        confidence = (
            sentiment_strength * 0.3 +
            volume_percentile * 0.3 +
            engagement_factor * 0.2 +
            momentum_alignment * 0.2
        )
        
        return confidence.clip(0, 1)
    
    def get_signal_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate metrics about the generated signals"""
        if not all(col in df.columns for col in ['sentiment_signal', 'sentiment_confidence']):
            raise ValueError("Run generate_signals first")
            
        metrics = {
            'signal_count': len(df[df['sentiment_signal'] != 0]),
            'signal_density': len(df[df['sentiment_signal'] != 0]) / len(df),
            'avg_confidence': df['sentiment_confidence'].mean(),
            'strong_signals': len(df[abs(df['sentiment_signal']) > 1]),
            'bullish_ratio': len(df[df['sentiment_signal'] > 0]) / len(df[df['sentiment_signal'] != 0])
            if len(df[df['sentiment_signal'] != 0]) > 0 else 0
        }
        
        return metrics 