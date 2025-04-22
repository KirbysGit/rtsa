# RTSA /

## / data

- / processed  
  Holds Post Result of Processing Raw Data.

- / raw  
  Raw Data Gathered From Data Collection.


## / results

Stores All of Resulting Graphs & Data.


## / scripts

- analyze_correlations.py –  
  Runs correlation study between sentiment and prices.

- run_baseline_strategy.py –  
  Tests the baseline sentiment strategy.

- test_advanced_analysis.py –  
  Runs deeper tests on performance and analytics.

- test_analyzer.py –  
  Tests general data analysis flow.

- test_backtest.py –  
  Runs full backtesting pipeline.

- test_merger.py –  
  Tests data merging between Reddit and stock data.

- test_processors.py –  
  Tests data preprocessing functions.

- test_reddit_data.py –  
  Tests Reddit scraping.

- test_stock_data.py –  
  Tests historical stock data pulling.


## / src

### / analysis

- backtester.py – "Testing Lab."

  Simulates how our strategy would’ve done in the past.

  Tracks:
  - Trades (entry/exit)
  - Account value over time
  - Profit/Loss
  - Risk mgmt (stop-loss)

  Plots:
  - Portfolio value
  - Entry/exit points
  - Drawdowns

- correlation_analyzer.py – "Research Department."

  Studies link between sentiment and prices.

  Tells us:
  - If sentiment predicts price
  - Best lookback periods
  - When it works best
  - Signal reliability

  Generates:
  - Heatmaps
  - Lead-lag analysis
  - Returns by sentiment
  - Performance by regime

- data_analyzer.py – "Data Exploration Tool."

  Helps us understand what we’re working with.

  Looks at:
  - Prices
  - Sentiment
  - Reddit activity
  - Engagement
  - Correlations

  Outputs:
  - Correlation heatmap
  - Price & sentiment plots
  - Comment volume vs returns
  - Engagement over time

- risk_analyzer.py – "Risk & Signal HQ"

  Adds risk metrics, market regimes, and advanced signals.

- sentiment_analyzer.py –  
  Builds sentiment signals using Reddit data.

- technical_indicators.py –  
  Adds RSI, MACD, Bollinger Bands, and more.


### / data_collection

- reddit_collector.py –  
  Grabs Reddit posts from subs.

- stock_data_collector.py –  
  Gets historical stock prices from Yahoo.


### / data_processing

- data_merger.py –  
  Merges processed Reddit & stock data.

- reddit_processor.py –  
  Cleans and processes Reddit data.

- stock_data_processor.py –  
  Cleans and formats stock data.


### / prediction_model

(Placeholder for future ML models)


### / sentiment_analysis

(Placeholder for advanced sentiment modeling)


### / strategy

- baseline_strategy.py –  
  Implements simple sentiment-based trading logic.


### / utils

- data_validator.py –  
  Validates data for backtesting. *(Currently unused)*

- reddit_config.py –  
  Loads Reddit API credentials.


### / tests

- test_correlation_analyzer.py –  
  Unit tests for correlation analysis.

- test_sentiment_analyzer.py –  
  Tests sentiment signal generation.


## .env

Environment file for Reddit credentials and config.
