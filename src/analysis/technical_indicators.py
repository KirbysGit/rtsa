import pandas as pd
import numpy as np
from scipy.stats import linregress

class TechnicalAnalyzer:
    @staticmethod
    def add_volatility_indicators(df):
        """Add volatility-based indicators"""
        # True Range and ATR
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
        
        # Volatility Ratio
        df['Volatility_Ratio'] = df['ATR'] / df['Close']
        
        return df
    
    @staticmethod
    def add_trend_indicators(df):
        """Add trend detection indicators"""
        # ADX (Average Directional Index)
        df['Plus_DM'] = df['High'].diff()
        df['Minus_DM'] = df['Low'].diff()
        df['Plus_DM'] = df['Plus_DM'].where(
            (df['Plus_DM'] > 0) & (df['Plus_DM'] > df['Minus_DM']), 0
        )
        df['Minus_DM'] = abs(df['Minus_DM'].where(
            (df['Minus_DM'] > 0) & (df['Minus_DM'] > df['Plus_DM']), 0
        ))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df):
        """Add momentum-based indicators"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        df['14-high'] = df['High'].rolling(14).max()
        df['14-low'] = df['Low'].rolling(14).min()
        df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
        df['%D'] = df['%K'].rolling(3).mean()
        
        return df
    
    @staticmethod
    def add_sentiment_indicators(df):
        """Add sentiment-based indicators"""
        # Sentiment Momentum
        df['Sentiment_Momentum'] = df['avg_sentiment'].diff()
        
        # Sentiment Volume Indicator
        df['Sentiment_Volume'] = df['avg_sentiment'] * df['total_posts']
        
        # Engagement Momentum
        df['Engagement_Momentum'] = df['avg_engagement'].diff()
        
        return df 