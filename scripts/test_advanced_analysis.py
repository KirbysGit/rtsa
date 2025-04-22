# src / scripts / test_advanced_analysis.py

# Imports.
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv

# Add Project Root to Python Path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Logging.
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Format Market Summary.
def format_market_summary(summary: Dict, ticker: str) -> None:
    """Format and Print Market Summary in a Clean, Readable Format."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Print Market Summary.
    print("\n" + "="*80)
    print(f"MARKET ANALYSIS SUMMARY - {ticker}")
    print(f"Generated at: {current_time}")
    print("="*80)
    
    # Market Condition Section.
    print("\n📊 MARKET CONDITIONS:")
    print(f"   • Market Regime: {summary['market_condition']['regime'].replace('_', ' ').title()}")
    print(f"   • Risk Level: {summary['market_condition']['risk_level'].replace('_', ' ').title()}")
    print(f"   • Sentiment: {summary['market_condition']['sentiment']}")
    print(f"   • Volume: {summary['market_condition']['volume_condition'].replace('_', ' ').title()}")
    
    # Risk Metrics Section.
    print("\n📈 RISK METRICS:")
    print(f"   • Volatility: {summary['risk_metrics']['volatility']}")
    print(f"   • Value at Risk (95%): {summary['risk_metrics']['var_95']}")
    print(f"   • Expected Shortfall: {summary['risk_metrics']['expected_shortfall']}")
    print(f"   • Combined Risk Score: {summary['risk_metrics']['combined_risk']}")
    
    # Trading Recommendation Section.
    print("\n🎯 TRADING RECOMMENDATION:")
    print(f"   • Signal: {summary['trading_signals']['primary_signal']}")
    print(f"   • Recommended Position Size: {summary['trading_signals']['position_size']}")
    print(f"   • Signal Confidence: {summary['trading_signals']['confidence']}")
    
    # Key Levels Section.
    print("\n🔑 KEY LEVELS:")
    print(f"   • Suggested Stop Loss: ${summary['key_levels']['stop_loss']:.2f}")
    print(f"   • Price Target: ${summary['key_levels']['target']:.2f}")
    
    print("\n" + "="*80)

def test_advanced_analysis():
    """Test Advanced Technical and Risk Analysis."""
    from src.analysis.technical_indicators import TechnicalAnalyzer
    from src.analysis.risk_analyzer import RiskAnalyzer
    
    # Load Merged Data.
    ticker = 'NVDA'
    merged_data_path = f"data/processed/{ticker}_merged_data.csv"
    
    if os.path.exists(merged_data_path):
        # Load and Process Data.
        data = pd.read_csv(merged_data_path, index_col='Date', parse_dates=True)
        
        # Add Technical Indicators.
        tech_analyzer = TechnicalAnalyzer()
        data = tech_analyzer.add_volatility_indicators(data)
        data = tech_analyzer.add_trend_indicators(data)
        data = tech_analyzer.add_momentum_indicators(data)
        data = tech_analyzer.add_sentiment_indicators(data)
        
        # Perform Advanced Analysis.
        risk_analyzer = RiskAnalyzer()
        market_summary, enhanced_data = risk_analyzer.analyze_market_conditions(data)
        
        # Display Formatted Summary.
        format_market_summary(market_summary, ticker)
        
        # Save Enhanced Data.
        output_path = f"data/processed/{ticker}_enhanced_data.csv"
        enhanced_data.to_csv(output_path)
        logger.info(f"\nDetailed Analysis Saved to: {output_path}")
        logger.info(f"Risk Dashboard Saved to: {risk_analyzer.output_dir}/{ticker}_risk_dashboard.png\n")
        
    else:
        logger.error(f"No merged data found at {merged_data_path}")

if __name__ == "__main__":
    load_dotenv()
    test_advanced_analysis() 