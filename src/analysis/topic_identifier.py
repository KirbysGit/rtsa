import re
import logging
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tqdm import tqdm
from colorama import Fore, Style, init
from src.utils.path_config import RAW_DIR, DEBUG_DIR
import requests
import numpy as np
from pathlib import Path

# Initialize colorama
init()

logger = logging.getLogger(__name__)

# Constants for filtering
POTENTIALLY_AMBIGUOUS_TICKERS = {
    'SEE', 'OPEN', 'REAL', 'DAY', 'TOP', 'KEY', 'TRUE', 'SAFE', 'GAIN', 'LOT', 'TURN',
    'MORE', 'LIKE', 'JUST', 'CASH', 'PROP', 'STEP', 'ELSE', 'JUNE', 'NEXT', 'GOOD',
    'BEST', 'WELL', 'FAST', 'FREE', 'LIVE', 'PLAY', 'STAY', 'MOVE', 'MIND', 'LIFE',
    'PEAK', 'FUND', 'HUGE', 'NICE', 'EASY', 'BEAT', 'HOPE', 'CARE', 'MAIN', 'RIDE'
}

class TopicIdentifier:
    def __init__(self, data_dir=None):
        """Initialize the Topic Identifier."""
        print(f"\n{Fore.CYAN}Initializing Topic Identifier...{Style.RESET_ALL}")
        self.data_dir = data_dir or DEBUG_DIR
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})')
        
        # Common words to filter out
        self.common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'WAS', 'YOU', 'HAS', 'HAD', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
            'FROM', 'THIS', 'THAT', 'WITH', 'WHICH', 'WHEN', 'WHERE', 'WHAT', 'WHY', 'HOW', 'WHO'
        }
        
        # Common finance terms that might be mistaken for tickers
        self.finance_terms = {
            'STOCK', 'MARKET', 'BUY', 'SELL', 'HOLD', 'SHORT', 'LONG', 'CALL', 'PUT', 'OPTION',
            'PRICE', 'SHARE', 'TRADE', 'TRADING', 'INVEST', 'INVESTING', 'MONEY', 'CASH', 'GAIN',
            'LOSS', 'PROFIT', 'LOSS', 'RISK', 'SAFE', 'BEAR', 'BULL', 'PUMP', 'DUMP', 'MOON'
        }
        
        # Load valid tickers from NASDAQ and NYSE
        self.valid_tickers = self._load_valid_tickers()
        
        # Track ambiguous ticker stats
        self.ambiguous_stats = defaultdict(lambda: {
            'total_mentions': 0,
            'high_confidence_mentions': 0,
            'contexts': [],
            'subreddits': set()
        })
        
        print(f"{Fore.GREEN}✓ Topic Identifier initialized successfully{Style.RESET_ALL}\n")
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text with enhanced filtering."""
        if pd.isna(text):
            return []
        
        # Find $TICKER format
        tickers = self.ticker_pattern.findall(text.upper())
        
        # Find standalone tickers (3-5 capital letters)
        standalone = re.findall(r'\b[A-Z]{3,5}\b', text.upper())
        tickers.extend(standalone)
        
        # Filter out common words and finance terms
        tickers = [
            t for t in tickers 
            if t not in self.common_words 
            and t not in self.finance_terms
            and t in self.valid_tickers
        ]
        
        return list(set(tickers))
    
    def _load_valid_tickers(self) -> set:
        """Load valid stock tickers from NASDAQ and NYSE."""
        print(f"{Fore.CYAN}Loading valid tickers from exchanges...{Style.RESET_ALL}")
        try:
            # Try to load from local cache first
            cache_file = self.data_dir / "valid_tickers.csv"
            if cache_file.exists():
                print(f"{Fore.YELLOW}Loading tickers from cache at {cache_file}...{Style.RESET_ALL}")
                df = pd.read_csv(cache_file)
                tickers = set(df['Symbol'].str.upper().tolist())
                print(f"{Fore.GREEN}✓ Loaded {len(tickers)} tickers from cache{Style.RESET_ALL}")
                return tickers

            # If no cache, try to download from NASDAQ
            print(f"{Fore.YELLOW}Downloading NASDAQ tickers...{Style.RESET_ALL}")
            nasdaq_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(nasdaq_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                tickers = set()
                
                # Extract tickers from response
                for row in data.get('data', {}).get('table', {}).get('rows', []):
                    symbol = row.get('symbol', '').upper()
                    if len(symbol) <= 5 and symbol.isalpha():
                        tickers.add(symbol)
                
                # Save to cache in debug directory
                self.data_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({'Symbol': list(tickers)}).to_csv(cache_file, index=False)
                print(f"{Fore.GREEN}✓ Downloaded and cached {len(tickers)} tickers to {cache_file}{Style.RESET_ALL}")
                return tickers
            
            raise Exception("Failed to download ticker list")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error loading valid tickers: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Using fallback ticker list...{Style.RESET_ALL}")
            return {
                'NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'INTC', 'IBM',
                'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD',
                'NFLX', 'DIS', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'PEP',
                'KO', 'PG', 'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'CVS', 'WBA', 'LLY',
                'XOM', 'CVX', 'COP', 'BP', 'SHEL', 'PBR', 'BABA', 'JD', 'PDD', 'BIDU',
                'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM', 'TM', 'HMC', 'TSM',
                'ASML', 'QCOM', 'AVGO', 'TXN', 'MU', 'LRCX', 'AMAT', 'KLAC', 'ADI', 'MRVL'
            }
    
    def calculate_ticker_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate scores for each ticker based on mentions, engagement, and confidence class."""
        if df.empty:
            return {}
        
        print(f"\n{Fore.CYAN}Analyzing posts for ticker mentions...{Style.RESET_ALL}")
        
        # Filter for relevant posts with ticker mentions
        relevant_df = df[df['is_relevant'] & df['tickers'].notna()]
        if relevant_df.empty:
            print(f"{Fore.YELLOW}No relevant ticker mentions found{Style.RESET_ALL}")
            return {}
        
        ticker_scores = {}
        ticker_mentions = Counter()
        ticker_engagement = Counter()
        debug_info = defaultdict(lambda: {
            'mentions': 0, 
            'engagement': 0, 
            'confidence': [],
            'contexts': [],
            'confidence_class_counts': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        })
        
        for _, row in tqdm(relevant_df.iterrows(), total=len(relevant_df), desc=f"{Fore.YELLOW}Processing posts{Style.RESET_ALL}"):
            for ticker in row['tickers']:
                # Skip common words early
                if ticker in self.common_words:
                    continue
                    
                # Only consider tickers in our valid list
                if ticker not in self.valid_tickers:
                    continue
                
                # Enhanced filtering for ambiguous tickers
                if ticker in POTENTIALLY_AMBIGUOUS_TICKERS:
                    confidence = row.get('ticker_confidence', 0)
                    # Require higher confidence and entity match for ambiguous tickers
                    if confidence < 0.9:
                        continue
                    
                    # Track stats for ambiguous tickers
                    self.ambiguous_stats[ticker]['total_mentions'] += 1
                    if confidence >= 0.9:
                        self.ambiguous_stats[ticker]['high_confidence_mentions'] += 1
                    self.ambiguous_stats[ticker]['contexts'].append(row.get('cleaned_text', '')[:100])
                    self.ambiguous_stats[ticker]['subreddits'].add(row.get('subreddit', ''))
                
                # Base score from mentions
                ticker_mentions[ticker] += 1
                
                # Calculate weighted engagement based on confidence class
                confidence_class = row.get('confidence_class', 'LOW')
                class_multiplier = {'HIGH': 1.0, 'MEDIUM': 0.7, 'LOW': 0.4}.get(confidence_class, 0.4)
                
                engagement = (
                    row.get('score', 0) + 
                    row.get('num_comments', 0)
                ) * row.get('ticker_confidence', 0.3) * class_multiplier
                
                ticker_engagement[ticker] += engagement
                
                # Store debug info
                debug_info[ticker]['mentions'] += 1
                debug_info[ticker]['engagement'] += engagement
                debug_info[ticker]['confidence'].append(row.get('ticker_confidence', 0))
                debug_info[ticker]['contexts'].append(row.get('cleaned_text', '')[:100])
                debug_info[ticker]['confidence_class_counts'][confidence_class] += 1
                
                # Add to scores
                ticker_scores[ticker] = ticker_scores.get(ticker, 0) + engagement
        
        # Apply stricter post-processing filters
        min_mentions = max(3, len(relevant_df) * 0.01)  # At least 3 mentions or 1% of posts
        min_engagement = len(relevant_df) * 0.02  # At least 2% of total possible engagement
        
        filtered_scores = {
            k: v 
            for k, v in ticker_scores.items() 
            if (ticker_mentions[k] >= min_mentions and ticker_engagement[k] >= min_engagement) and
               (k not in POTENTIALLY_AMBIGUOUS_TICKERS or 
                (ticker_mentions[k] >= min_mentions * 2 and  # Double the requirements for ambiguous tickers
                 debug_info[k]['confidence_class_counts']['HIGH'] > 0))  # Require at least one HIGH confidence mention
        }
        
        # Save enhanced debug information
        self._save_debug_info(debug_info, filtered_scores, ticker_mentions, ticker_engagement)
        
        # Save ambiguous ticker analysis
        self._save_ambiguous_analysis()
        
        # Normalize scores
        if filtered_scores:
            max_score = max(filtered_scores.values())
            normalized_scores = {
                k: v/max_score 
                for k, v in filtered_scores.items()
            }
            
            print(f"{Fore.GREEN}✓ Identified {len(normalized_scores)} valid tickers{Style.RESET_ALL}")
            
            # Print top tickers with enhanced stats
            print(f"\n{Fore.CYAN}Top Tickers Analysis:{Style.RESET_ALL}")
            for ticker, score in sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
                info = debug_info[ticker]
                print(f"{Fore.GREEN}{ticker}:{Style.RESET_ALL}")
                print(f"  Score: {score:.3f}")
                print(f"  Mentions: {info['mentions']}")
                print(f"  Confidence Classes:")
                print(f"    HIGH: {info['confidence_class_counts']['HIGH']}")
                print(f"    MEDIUM: {info['confidence_class_counts']['MEDIUM']}")
                print(f"    LOW: {info['confidence_class_counts']['LOW']}")
                print(f"  Avg Confidence: {np.mean(info['confidence']):.3f}")
                print(f"  Total Engagement: {info['engagement']:.0f}")
                if ticker in POTENTIALLY_AMBIGUOUS_TICKERS:
                    print(f"  {Fore.YELLOW}⚠️ Ambiguous ticker - verified with entity matches{Style.RESET_ALL}")
            
            return normalized_scores
        
        print(f"{Fore.YELLOW}No tickers passed post-processing filters{Style.RESET_ALL}")
        return {}
    
    def _save_debug_info(self, debug_info, filtered_scores, ticker_mentions, ticker_engagement):
        """Save enhanced debug information about ticker processing."""
        debug_rows = []
        for ticker, info in debug_info.items():
            debug_rows.append({
                'ticker': ticker,
                'mentions': info['mentions'],
                'total_engagement': info['engagement'],
                'avg_confidence': np.mean(info['confidence']),
                'passed_filters': ticker in filtered_scores,
                'is_common_word': ticker in self.common_words,
                'is_valid_ticker': ticker in self.valid_tickers,
                'is_ambiguous': ticker in POTENTIALLY_AMBIGUOUS_TICKERS,
                'example_contexts': '; '.join(info['contexts'][:3]),  # Save up to 3 example contexts
                'min_confidence': min(info['confidence']),
                'max_confidence': max(info['confidence'])
            })
        
        if debug_rows:
            debug_df = pd.DataFrame(debug_rows)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_file = self.data_dir / f"ticker_analysis_{timestamp}.csv"
            debug_df.to_csv(debug_file, index=False)
            print(f"{Fore.GREEN}✓ Saved ticker analysis to {debug_file}{Style.RESET_ALL}")

    def _save_ambiguous_analysis(self):
        """Save detailed analysis of ambiguous ticker mentions."""
        if not self.ambiguous_stats:
            return
            
        rows = []
        for ticker, stats in self.ambiguous_stats.items():
            rows.append({
                'ticker': ticker,
                'total_mentions': stats['total_mentions'],
                'high_confidence_mentions': stats['high_confidence_mentions'],
                'confidence_ratio': stats['high_confidence_mentions'] / stats['total_mentions'] if stats['total_mentions'] > 0 else 0,
                'example_contexts': '; '.join(stats['contexts'][:3]),
                'subreddits': '; '.join(stats['subreddits']),
            })
        
        if rows:
            df = pd.DataFrame(rows)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.data_dir / f"ambiguous_tickers_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            print(f"{Fore.GREEN}✓ Saved ambiguous ticker analysis to {file_path}{Style.RESET_ALL}")
    
    def identify_trending_topics(self, df: pd.DataFrame, min_mentions: int = 5) -> Dict[str, float]:
        """Identify trending tickers from processed Reddit posts."""
        try:
            if df.empty:
                return {}
            
            # Calculate ticker scores
            ticker_scores = self.calculate_ticker_scores(df)
            
            # Sort by score
            trending = dict(sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True))
            
            print(f"\n{Fore.CYAN}Trending Analysis Complete:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✓ Identified {len(trending)} trending tickers{Style.RESET_ALL}")
            return trending
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error identifying trending topics: {str(e)}{Style.RESET_ALL}")
            return {}
    
    def get_trending_tickers(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        """Get the top N trending tickers from processed data."""
        trending = self.identify_trending_topics(df)
        return list(trending.keys())[:top_n]

def main():
    """Test the TopicIdentifier."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print(f"\n{Fore.CYAN}=== Reddit Trending Ticker Analysis ==={Style.RESET_ALL}")
    
    # Initialize identifier
    identifier = TopicIdentifier()
    
    # Load and process test data
    test_file = RAW_DIR / "reddit_data" / "processed_reddit.csv"
    if test_file.exists():
        df = pd.read_csv(test_file)
        trending = identifier.identify_trending_topics(df)
        
        if trending:
            print(f"\n{Fore.CYAN}Top Trending Tickers:{Style.RESET_ALL}")
            for ticker, score in list(trending.items())[:10]:
                print(f"{Fore.GREEN}{ticker}: {score:.2f}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}No trending tickers found{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No processed data found for testing{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 