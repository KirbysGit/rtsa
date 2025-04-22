import unittest
import pandas as pd
import numpy as np
from src.analysis.correlation_analyzer import CorrelationAnalyzer

class TestCorrelationAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate correlated data for testing
        sentiment = np.random.normal(0, 1, 100)
        returns = 0.3 * sentiment + np.random.normal(0, 1, 100)  # Correlated with sentiment
        close = 100 * (1 + returns).cumprod()
        
        self.test_data = pd.DataFrame({
            'Close': close,
            'Returns': returns,
            'avg_sentiment': sentiment,
            'sentiment_signal': np.sign(sentiment),
            'weighted_sentiment': sentiment * np.random.uniform(0.5, 1.5, 100),
            'sentiment_momentum': pd.Series(sentiment).diff().values
        }, index=dates)
        
        self.analyzer = CorrelationAnalyzer()
    
    def test_correlation_analysis(self):
        # Test basic correlation analysis
        results = self.analyzer.analyze_correlations(self.test_data)
        
        # Check all expected sections are present
        required_sections = ['contemporaneous', 'lead_lag', 'predictive_power', 'regime_analysis']
        for section in required_sections:
            self.assertIn(section, results)
        
        # Check contemporaneous correlations
        self.assertIn('pearson', results['contemporaneous'])
        self.assertIn('spearman', results['contemporaneous'])
        
        # Check lead-lag relationships
        for period in self.analyzer.lookback_periods:
            self.assertIn(f'period_{period}', results['lead_lag'])
    
    def test_market_regimes(self):
        results = self.analyzer.analyze_correlations(self.test_data)
        
        # Check regime analysis
        self.assertIn('volatility_regime', results['regime_analysis'])
        self.assertIn('trend_regime', results['regime_analysis'])
        
        # Check regime categories
        regimes = results['regime_analysis']['volatility_regime'].keys()
        self.assertEqual(len(regimes), 3)  # Should have 3 regime categories
    
    def test_predictive_power(self):
        results = self.analyzer.analyze_correlations(self.test_data)
        
        # Check predictive power analysis
        period_1_results = results['predictive_power']['period_1']
        self.assertIn('returns_by_quintile', period_1_results)
        self.assertIn('monotonicity', period_1_results)
        
        # Check quintile counts
        quintiles = period_1_results['returns_by_quintile']
        self.assertEqual(len(quintiles), 5)  # Should have 5 quintiles

if __name__ == '__main__':
    unittest.main() 