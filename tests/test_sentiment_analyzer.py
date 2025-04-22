import unittest
import pandas as pd
import numpy as np
from src.analysis.sentiment_analyzer import SentimentSignalGenerator

class TestSentimentSignalGenerator(unittest.TestCase):
    def setUp(self):
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'avg_sentiment': np.random.normal(0, 0.5, 100),
            'total_posts': np.random.randint(5, 100, 100),
            'total_comments': np.random.randint(10, 200, 100),
            'avg_engagement': np.random.uniform(0.1, 1.0, 100)
        }, index=dates)
        
        self.generator = SentimentSignalGenerator()
    
    def test_signal_generation(self):
        # Test basic signal generation
        result = self.generator.generate_signals(self.test_data)
        
        # Check required columns exist
        required_cols = [
            'sentiment_signal', 'sentiment_momentum', 'weighted_sentiment',
            'sentiment_confidence', 'sentiment_ma'
        ]
        for col in required_cols:
            self.assertIn(col, result.columns)
        
        # Check signal bounds
        self.assertTrue(all(result['sentiment_signal'].between(-1.5, 1.5)))
        self.assertTrue(all(result['sentiment_confidence'].between(0, 1)))
    
    def test_confidence_calculation(self):
        # Test with strong sentiment
        self.test_data.loc[:, 'avg_sentiment'] = 0.8
        self.test_data.loc[:, 'total_posts'] = 100
        self.test_data.loc[:, 'avg_engagement'] = 1.0
        
        result = self.generator.generate_signals(self.test_data)
        
        # High sentiment should lead to high confidence
        self.assertTrue(result['sentiment_confidence'].mean() > 0.6)
    
    def test_signal_metrics(self):
        result = self.generator.generate_signals(self.test_data)
        metrics = self.generator.get_signal_metrics(result)
        
        required_metrics = [
            'signal_count', 'signal_density', 'avg_confidence',
            'strong_signals', 'bullish_ratio'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

if __name__ == '__main__':
    unittest.main() 