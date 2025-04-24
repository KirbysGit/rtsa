# src / data_processing / reddit_processor.py

# Description : This file contains the RedditDataProcessor class, which is used to process the Reddit data.

# Imports.
import re
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.utils.path_config import RAW_DIR, PROCESSED_DIR, DEBUG_DIR
from src.analysis.topic_identifier import POTENTIALLY_AMBIGUOUS_TICKERS  # Added import
from typing import Tuple, List, Set
from colorama import Fore, Style, init
from tqdm import tqdm
import time
from src.analysis.entity_linker import EntityLinker
import json
from contextlib import nullcontext

# Initialize colorama
init()

# Setup Logging.
logger = logging.getLogger(__name__)

# Constants
TICKERS = ['NVDA', 'NVIDIA', 'AMD', 'INTC', 'TSMC']  # Add more as needed
MAX_TEXT_LENGTH = 500  # Maximum words in text
SUMMARY_LENGTH = 100  # Words in summary

# Subreddit to ticker mapping (all lowercase keys)
SUBREDDIT_TICKERS = {
    'nvidia': 'NVDA',
    'amd': 'AMD',
    'intel': 'INTC',
    'tsmc': 'TSMC',
    'wallstreetbets': None,  # General trading subreddit
    'stocks': None,          # General trading subreddit
    'investing': None,       # General trading subreddit
    'stockmarket': None      # General trading subreddit
}

# Strong financial context words (high confidence)
STRONG_FINANCE_WORDS = {
    'stock', 'shares', 'ticker', 'earnings', 'revenue', 'dividend', 'market cap',
    'trading', 'investor', 'bullish', 'bearish', '$', 'calls', 'puts', 'options',
    'portfolio', 'shareholders', 'eps', 'pe ratio', 'market share', 'guidance',
    'analyst', 'upgrade', 'downgrade', 'price target', 'short interest', 'float',
    'institutional', 'hedge fund', 'etf', 'ipo', 'spac', 'merger', 'acquisition'
}

# Weak financial context words (lower confidence)
WEAK_FINANCE_WORDS = {
    'buy', 'sell', 'price', 'trade', 'invest', 'market', 'portfolio', 'position',
    'profit', 'loss', 'analysis', 'company', 'corporation', 'inc', 'ltd', 'tech',
    'up', 'down', 'gain', 'drop', 'rise', 'fall', 'quarter', 'growth', 'decline',
    'performance', 'trend', 'sector', 'industry', 'competition', 'partnership',
    'deal', 'contract', 'launch', 'product', 'service', 'expansion', 'strategy'
}

# Common words that might be mistaken for tickers
COMMON_WORDS = {
    'THE', 'AND', 'FOR', 'ARE', 'WAS', 'YOU', 'HAS', 'HAD', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
    'FROM', 'THIS', 'THAT', 'WITH', 'WHICH', 'WHEN', 'WHERE', 'WHAT', 'WHY', 'HOW', 'WHO',
    'CAN', 'MAN', 'POST', 'LIVE', 'HAS', 'HAD', 'WAS', 'WERE', 'BEEN', 'BEING', 'HAVE', 'HAS',
    'WILL', 'WOULD', 'SHALL', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'COULD', 'SHOULD', 'WOULD',
    'NOT', 'BUT', 'LIKE', 'MORE', 'JUST', 'NOW', 'OUT', 'ALL', 'THEY', 'SAID', 'TIME', 'ABOUT',
    'SOME', 'INTO', 'ALSO', 'THAN', 'THEN', 'WHEN', 'WHERE', 'WHY', 'HOW', 'WHAT', 'WHICH',
    'THERE', 'HERE', 'THOSE', 'THESE', 'THEIR', 'THEM', 'THIS', 'THAT', 'THOSE', 'THESE',
    'NOT', 'BUT', 'LIKE', 'MORE', 'JUST', 'SOME', 'TIME', 'GOOD', 'SAY', 'WAY', 'MOVE',
    'BACK', 'LOOK', 'THINK', 'KNOW', 'MAKE', 'TAKE', 'COME', 'WELL', 'EVEN', 'WANT',
    'NEED', 'MUCH', 'MANY', 'SUCH', 'MOST', 'PART', 'OVER', 'YEAR', 'HELP', 'WORK',
    'LIFE', 'TELL', 'CASE', 'DAYS', 'FIND', 'NEXT', 'LAST', 'WEEK', 'GIVE', 'NAME',
    'BEST', 'IDEA', 'TALK', 'SURE', 'KIND', 'HEAD', 'HAND', 'FACT', 'TYPE', 'LINE'
}

class RedditDataProcessor:
    def __init__(self):
        """Initialize the Reddit Data Processor."""
        # Set up paths using path_config
        self.raw_path = RAW_DIR / "reddit_data"
        self.processed_path = PROCESSED_DIR / "reddit_data"
        self.debug_path = DEBUG_DIR
        
        # Create necessary directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.debug_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentiment analyzers
        self.textblob = TextBlob
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize common word tickers
        self.common_word_tickers = COMMON_WORDS
        
        # Initialize FinBERT with optimized GPU settings
        print(f"{Fore.YELLOW}Initializing FinBERT model...{Style.RESET_ALL}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set CUDA optimization flags if GPU is available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"{Fore.GREEN}✓ CUDA optimizations enabled{Style.RESET_ALL}")
        
        # Load model and tokenizer with optimized settings
        self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.finbert_model = (AutoModelForSequenceClassification
            .from_pretrained('ProsusAI/finbert')
            .to(self.device)
            .eval()  # Set to eval mode immediately
        )
        
        # Enable half-precision if on GPU
        if self.device.type == 'cuda':
            self.finbert_model = self.finbert_model.half()  # Convert to FP16
            print(f"{Fore.GREEN}✓ Using FP16 precision for faster inference{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}✓ FinBERT initialized on {self.device}{Style.RESET_ALL}")
        
        # Cache for FinBERT scores with size limit
        self.finbert_cache = {}
        self.max_cache_size = 10000  # Limit cache size
        
        # Compile regex patterns
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})')
        self.standalone_ticker_pattern = re.compile(r'\b[A-Z]{3,5}\b')
        self.sentence_end_pattern = re.compile(r'[.!?]\s+')
        
        # Initialize EntityLinker
        print(f"{Fore.YELLOW}Initializing Entity Linker...{Style.RESET_ALL}")
        self.entity_linker = EntityLinker()
        
        # Track confidence classes
        self.confidence_classes = {
            'HIGH': [],    # FinBERT + Entity match
            'MEDIUM': [],  # FinBERT only
            'LOW': []      # Weak context
        }
        
        # Add well-known ETFs and their categories
        self.etf_categories = {
            'MARKET_INDEX': {
                'SPY',  # S&P 500
                'QQQ',  # Nasdaq-100
                'IWM',  # Russell 2000
                'DIA',  # Dow Jones
                'VOO',  # Vanguard S&P 500
                'VTI'   # Vanguard Total Market
            },
            'SECTOR': {
                'XLF',  # Financial
                'XLE',  # Energy
                'XLV',  # Healthcare
                'XLK',  # Technology
                'XLI',  # Industrial
                'XLP',  # Consumer Staples
                'XLY',  # Consumer Discretionary
                'XLB',  # Materials
                'XLU',  # Utilities
                'XLRE', # Real Estate
                'XLC'   # Communication Services
            },
            'COMMODITY': {
                'GLD',  # Gold
                'SLV',  # Silver
                'USO',  # Oil
                'UNG'   # Natural Gas
            },
            'BOND': {
                'TLT',  # 20+ Year Treasury
                'IEF',  # 7-10 Year Treasury
                'HYG',  # High Yield Corporate
                'LQD',  # Investment Grade Corporate
                'AGG',  # Aggregate Bond
                'BND'   # Vanguard Total Bond
            },
            'INTERNATIONAL': {
                'EFA',  # Developed Markets
                'EEM',  # Emerging Markets
                'VEA',  # Vanguard Developed Markets
                'VWO',  # Vanguard Emerging Markets
                'VGK'   # Vanguard European
            }
        }
        
        # Flatten ETF list for quick lookup
        self.valid_etfs = {etf for category in self.etf_categories.values() for etf in category}
        
        # Well-known stock tickers that should have lower validation requirements
        self.well_known_tickers = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC',
            'INTC', 'AMD', 'CSCO', 'ORCL', 'IBM', 'PLTR', 'COIN', 'GME', 'AMC', 'BB',
            'F', 'GM', 'GE', 'BA', 'RTX', 'LMT', 'NOC',
            'PFE', 'JNJ', 'MRK', 'CVS', 'UNH',
            'KO', 'PEP', 'MCD', 'WMT', 'TGT',
            'DIS', 'NFLX', 'CMCSA', 'T', 'VZ'
        }
        
        # Add negative context patterns that invalidate ticker matches
        self.negative_context_patterns = {
            'COIN': [
                'meme coin', 'shit coin', 'shitcoin', 'alt coin', 'altcoin', 'stable coin', 'stablecoin',
                'dog coin', 'dogcoin', 'moon coin', 'mooncoin', 'pump coin', 'dump coin', 'new coin',
                'this coin', 'the coin', 'that coin', 'any coin', 'my coin', 'your coin', 'their coin',
                'crypto coin', 'cryptocurrency', 'token'
            ],
            'GOLD': ['gold standard', 'gold medal', 'gold mine', 'gold rush', 'gold price'],
            'GOOD': ['good morning', 'good night', 'good day', 'good luck', 'good job'],
            'CASH': ['cash app', 'cash out', 'cash flow', 'cash back', 'cash money'],
            'MOON': ['to the moon', 'moon shot', 'moon boy', 'moon mission'],
            'PUMP': ['pump and dump', 'pump scheme', 'pump group'],
            'HOLD': ['hold on', 'hold up', 'hold tight', 'hold steady'],
            'GAS': ['gas price', 'gas fee', 'gas station', 'gas tank']
        }
        
        # Add ambiguous financial tickers that need extra validation
        self.ambiguous_financial_tickers = {
            'COIN': {
                'required_context': ['coinbase', 'nasdaq:coin', 'nyse:coin'],
                'company_terms': ['coinbase', 'armstrong', 'crypto exchange', 'cryptocurrency exchange'],
                'min_confidence': 0.9
            },
            'GOLD': {
                'required_context': ['gld etf', 'gold etf', 'gold shares'],
                'company_terms': ['spdr', 'state street', 'gold trust'],
                'min_confidence': 0.85
            },
            'CASH': {
                'required_context': ['money market', 'cash management'],
                'company_terms': ['money market fund', 'cash equivalent'],
                'min_confidence': 0.9
            }
        }
    
    def _get_context_window(self, text: str, target: str, window_size: int = 20) -> str:
        """Get a window of words around a target word."""
        words = text.split()
        try:
            idx = words.index(target.lower())
            start = max(0, idx - window_size)
            end = min(len(words), idx + window_size + 1)
            return ' '.join(words[start:end])
        except ValueError:
            return ""
    
    def _get_finbert_scores_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Process multiple texts with FinBERT in optimized batches."""
        scores = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache first
            batch_scores = []
            uncached_texts = []
            uncached_indices = []
            
            for idx, text in enumerate(batch_texts):
                if text in self.finbert_cache:
                    batch_scores.append(self.finbert_cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
            
            if uncached_texts:
                try:
                    # Tokenize all texts in batch
                    with torch.cuda.amp.autocast() if self.device.type == 'cuda' else nullcontext():
                        inputs = self.finbert_tokenizer(
                            uncached_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(self.device)
                        
                        # Process batch with memory optimization
                        with torch.no_grad():
                            outputs = self.finbert_model(**inputs)
                            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                            financial_scores = (probabilities[:, 0] + probabilities[:, 2]).cpu().numpy()
                        
                        # Update cache and scores
                        for idx, (text, score) in enumerate(zip(uncached_texts, financial_scores)):
                            # Manage cache size
                            if len(self.finbert_cache) >= self.max_cache_size:
                                # Remove a random item if cache is full
                                self.finbert_cache.pop(next(iter(self.finbert_cache)))
                            
                            self.finbert_cache[text] = float(score)
                            batch_scores.insert(uncached_indices[idx], float(score))
                            
                        # Clear CUDA cache periodically
                        if self.device.type == 'cuda' and i % (batch_size * 10) == 0:
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    logger.error(f"Error in FinBERT batch processing: {str(e)}")
                    batch_scores.extend([0.0] * len(uncached_texts))
            
            scores.extend(batch_scores)
        
        return scores

    def _get_finbert_score(self, text: str, ticker: str) -> float:
        """Get FinBERT-based financial context score with optimized processing."""
        try:
            # Get context window around ticker
            context = self._get_context_window(text.lower(), ticker.lower(), window_size=20)
            if not context:
                return 0.0
            
            # Process in batch of 1 using the optimized batch function
            scores = self._get_finbert_scores_batch([context])
            return scores[0] if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in FinBERT scoring: {str(e)}")
            return 0.0

    def _has_financial_context(self, text: str, ticker: str) -> Tuple[bool, float]:
        """Check if text has financial context using FinBERT and entity matching."""
        if pd.isna(text) or pd.isna(ticker):
            return False, 0.0
            
        text = text.lower()
        ticker = ticker.upper()
        
        # Get context window
        context = self._get_context_window(text, ticker.lower(), window_size=20)
        if not context:
            return False, 0.0
        
        # Track feature attribution
        attribution = {
            'finbert_score': 0.0,
            'entity_match': False,
            'keyword_match': False,
            'dollar_symbol': False,
            'well_known': ticker in self.well_known_tickers,
            'etf': ticker in self.valid_etfs
        }
        
        # Get traditional keyword-based scores
        strong_terms = sum(1 for term in STRONG_FINANCE_WORDS if term in context)
        weak_terms = sum(1 for term in WEAK_FINANCE_WORDS if term in context)
        keyword_confidence = min(1.0, (strong_terms * 0.4 + weak_terms * 0.15))
        attribution['keyword_match'] = strong_terms > 0 or weak_terms > 0
        
        # Get FinBERT score
        finbert_score = self._get_finbert_score(text, ticker)
        attribution['finbert_score'] = finbert_score
        
        # Get entity validation
        has_entity, entity_boost = self.entity_linker.validate_context(text, ticker)
        attribution['entity_match'] = has_entity
        
        # Check for dollar symbol
        attribution['dollar_symbol'] = f"${ticker}" in text
        
        # Calculate final confidence
        confidence = (
            finbert_score * 0.4 + 
            keyword_confidence * 0.3 + 
            entity_boost * 0.3
        )
        
        if attribution['dollar_symbol']:
            confidence = min(1.0, confidence + 0.3)
        
        # Determine if context is financial
        has_context = (
            finbert_score > 0.3 or
            (strong_terms >= 1 and weak_terms >= 1) or
            has_entity or
            attribution['dollar_symbol']
        )
        
        # Determine confidence class and log attribution
        if has_context:
            attribution_info = {
                'ticker': ticker,
                'context': context,
                'finbert_score': finbert_score,
                'entity_matches': [ticker],
                'entity_confidence': entity_boost,
                'final_confidence': confidence,
                'attribution': attribution
            }
            
            if ((has_entity and finbert_score > 0.4) or
                (attribution['dollar_symbol'] and finbert_score > 0.3) or
                (ticker in self.well_known_tickers and finbert_score > 0.3)):
                self.confidence_classes['HIGH'].append(attribution_info)
            elif finbert_score > 0.3 or entity_boost > 0.2:
                self.confidence_classes['MEDIUM'].append(attribution_info)
            else:
                self.confidence_classes['LOW'].append(attribution_info)
        
        return has_context, min(1.0, confidence)
    
    def _has_negative_context(self, text: str, ticker: str) -> bool:
        """Check if the ticker appears in a negative context that invalidates it."""
        if ticker not in self.negative_context_patterns:
            return False
            
        text = text.lower()
        return any(pattern in text for pattern in self.negative_context_patterns[ticker])

    def _validate_ambiguous_financial_ticker(self, text: str, ticker: str) -> Tuple[bool, float]:
        """Extra validation for tickers that are common financial terms."""
        if ticker not in self.ambiguous_financial_tickers:
            return True, 1.0
            
        text = text.lower()
        validation = self.ambiguous_financial_tickers[ticker]
        
        # Check for required context
        has_required = any(context in text for context in validation['required_context'])
        
        # Check for company-specific terms
        has_company_terms = any(term in text for term in validation['company_terms'])
        
        # Calculate confidence based on context
        confidence = 0.0
        if has_required:
            confidence += 0.6
        if has_company_terms:
            confidence += 0.4
        if f"${ticker.lower()}" in text:
            confidence = min(1.0, confidence + 0.2)
            
        # Must meet minimum confidence threshold
        return confidence >= validation['min_confidence'], confidence

    def _extract_and_validate_tickers(self, text: str, subreddit: str = None) -> Set[str]:
        """Extract and validate tickers from text with enhanced context validation."""
        if pd.isna(text):
            return set()
            
        text = text.lower()
        tickers = set()
        
        try:
            # Find $TICKER format (highest confidence)
            dollar_tickers = set(self.ticker_pattern.findall(text.upper()))
            
            # Process dollar tickers with validation
            for ticker in dollar_tickers:
                # Skip if in negative context
                if self._has_negative_context(text, ticker):
                    continue
                    
                # Extra validation for ambiguous financial tickers
                if ticker in self.ambiguous_financial_tickers:
                    is_valid, _ = self._validate_ambiguous_financial_ticker(text, ticker)
                    if not is_valid:
                        continue
                
                if ticker in self.valid_etfs:
                    tickers.add(ticker)  # Auto-accept ETFs with $ symbol
                elif ticker in self.well_known_tickers:
                    tickers.add(ticker)  # Auto-accept well-known tickers with $ symbol
                elif ticker not in COMMON_WORDS:
                    tickers.add(ticker)
            
            # Find standalone tickers
            standalone_tickers = set(self.standalone_ticker_pattern.findall(text.upper()))
            
            # Validate standalone tickers with category-specific rules
            for ticker in standalone_tickers:
                # Skip if in negative context
                if self._has_negative_context(text, ticker):
                    continue
                    
                # Extra validation for ambiguous financial tickers
                if ticker in self.ambiguous_financial_tickers:
                    is_valid, confidence = self._validate_ambiguous_financial_ticker(text, ticker)
                    if not is_valid:
                        continue
                
                if ticker in self.valid_etfs:
                    # For ETFs, require basic financial context
                    try:
                        has_context, _ = self._has_financial_context(text, ticker)
                        if has_context or any(term in text for term in ['etf', 'fund', 'index', 'market', 'trading']):
                            tickers.add(ticker)
                    except Exception as e:
                        logger.debug(f"Error validating ETF {ticker}: {str(e)}")
                        continue
                        
                elif ticker in self.well_known_tickers:
                    # For well-known tickers, require basic context
                    try:
                        has_context, _ = self._has_financial_context(text, ticker)
                        if has_context:
                            tickers.add(ticker)
                    except Exception as e:
                        logger.debug(f"Error validating well-known ticker {ticker}: {str(e)}")
                        continue
                        
                elif ticker in COMMON_WORDS:
                    # For common words, require BOTH $ symbol AND very strong financial context
                    try:
                        if f"${ticker.lower()}" in text:
                            has_context, confidence = self._has_financial_context(text, ticker)
                            if has_context and confidence >= 0.95:
                                tickers.add(ticker)
                    except Exception as e:
                        logger.debug(f"Error validating common word ticker {ticker}: {str(e)}")
                        continue
                        
                elif ticker in POTENTIALLY_AMBIGUOUS_TICKERS:
                    # For ambiguous tickers, require strong context and entity validation
                    try:
                        has_context, confidence = self._has_financial_context(text, ticker)
                        has_entity, entity_boost = self.entity_linker.validate_context(text, ticker)
                        if has_context and has_entity and confidence >= 0.8:
                            tickers.add(ticker)
                    except Exception as e:
                        logger.debug(f"Error validating ambiguous ticker {ticker}: {str(e)}")
                        continue
                        
                else:
                    # Regular ticker validation with entity check
                    try:
                        has_context, confidence = self._has_financial_context(text, ticker)
                        has_entity, entity_boost = self.entity_linker.validate_context(text, ticker)
                        if has_context and (has_entity or confidence >= 0.7):
                            tickers.add(ticker)
                    except Exception as e:
                        logger.debug(f"Error validating ticker {ticker}: {str(e)}")
                        continue
            
            # Add subreddit-based tickers if from specific company subreddit
            if subreddit and not pd.isna(subreddit):
                subreddit = str(subreddit).lower()
                if subreddit in SUBREDDIT_TICKERS and SUBREDDIT_TICKERS[subreddit]:
                    tickers.add(SUBREDDIT_TICKERS[subreddit])
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error in ticker extraction: {str(e)}")
            return set()
    
    def _calculate_ticker_confidence(self, tickers: Set[str], text: str, subreddit: str = None) -> float:
        """Calculate confidence score for ticker mentions with enhanced context validation."""
        if not tickers:
            return 0.0
            
        text = text.lower()
        max_confidence = 0.0
        
        for ticker in tickers:
            # Skip if in negative context
            if self._has_negative_context(text, ticker):
                continue
            
            # Start with category-specific base confidence
            if ticker in self.valid_etfs:
                confidence = 0.7  # Higher base confidence for ETFs
            elif ticker in self.well_known_tickers:
                confidence = 0.6  # Higher base confidence for well-known tickers
            elif ticker in self.ambiguous_financial_tickers:
                # Use specialized validation for ambiguous financial tickers
                _, conf = self._validate_ambiguous_financial_ticker(text, ticker)
                confidence = conf * 0.5  # Start with half of the validation confidence
            elif ticker in COMMON_WORDS:
                confidence = 0.3  # Lower base confidence for common words
            else:
                confidence = 0.5  # Standard base confidence
            
            # Get entity validation
            has_entity, entity_boost = self.entity_linker.validate_context(text, ticker)
            
            # Add entity match confidence
            if has_entity:
                confidence += entity_boost * 0.3
            
            # Check financial context
            has_context, context_confidence = self._has_financial_context(text, ticker)
            if has_context:
                confidence += context_confidence * 0.2
            
            # Add confidence from specific company subreddit
            if (subreddit and not pd.isna(subreddit) and 
                subreddit.lower() in SUBREDDIT_TICKERS and 
                SUBREDDIT_TICKERS[subreddit] == ticker):
                confidence += 0.3
            
            # Add confidence from trading subreddit
            elif (subreddit and not pd.isna(subreddit) and 
                  subreddit.lower() in SUBREDDIT_TICKERS and 
                  SUBREDDIT_TICKERS[subreddit] is None):
                confidence += 0.1
            
            # Add $ symbol boost
            if f"${ticker.lower()}" in text:
                confidence = min(1.0, confidence + 0.2)
            
            # Category-specific adjustments
            if ticker in COMMON_WORDS:
                confidence *= 0.5  # 50% penalty for common words
            elif ticker in POTENTIALLY_AMBIGUOUS_TICKERS:
                confidence *= 0.7  # 30% penalty for ambiguous tickers
            elif ticker in self.valid_etfs:
                confidence = min(1.0, confidence * 1.2)  # 20% boost for ETFs
            elif ticker in self.well_known_tickers:
                confidence = min(1.0, confidence * 1.1)  # 10% boost for well-known tickers
            
            max_confidence = max(max_confidence, min(confidence, 1.0))
        
        return max_confidence
        
    def process_reddit_data(self, df):
        """Process raw Reddit data with enhanced features and debug logging."""
        try:
            print(f"\n{Fore.CYAN}Processing Reddit Data Pipeline{Style.RESET_ALL}")
            total_steps = 6
            
            # Step 1: Initial Processing
            print(f"\n{Fore.YELLOW}[1/{total_steps}] Initial Data Processing{Style.RESET_ALL}")
            logger.info(f"Initial post count: {len(df)}")
            processed_df = df.copy()
            
            if 'subreddit' not in processed_df.columns:
                logger.warning("No subreddit column found in data")
                processed_df['subreddit'] = None
            
            # Convert subreddit names to lowercase
            processed_df['subreddit'] = processed_df['subreddit'].str.lower() if processed_df['subreddit'].notna().any() else processed_df['subreddit']
            
            # Step 2: Text Cleaning
            print(f"\n{Fore.YELLOW}[2/{total_steps}] Cleaning Text Data{Style.RESET_ALL}")
            with tqdm(total=2, desc="Cleaning text") as pbar:
                processed_df['cleaned_title'] = processed_df['title'].apply(self._clean_text)
                pbar.update(1)
                processed_df['cleaned_text'] = processed_df['text'].apply(self._clean_text)
                pbar.update(1)
            
            # Step 3: Ticker Extraction
            print(f"\n{Fore.YELLOW}[3/{total_steps}] Extracting Tickers{Style.RESET_ALL}")
            
            # Process in smaller batches to show progress
            batch_size = 10
            num_batches = (len(processed_df) + batch_size - 1) // batch_size
            
            with tqdm(total=len(processed_df), desc="Analyzing posts") as pbar:
                potential_tickers_list = []
                tickers_list = []
                confidence_classes_list = []  # New list to store confidence classes
                
                for i in range(0, len(processed_df), batch_size):
                    batch = processed_df.iloc[i:i+batch_size]
                    
                    # Reset confidence classes for this batch
                    self.confidence_classes = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
                    
                    # Extract and validate tickers
                    tickers = batch.apply(
                        lambda row: self._extract_and_validate_tickers(
                            row['cleaned_text'], 
                            row.get('subreddit')
                        ), 
                        axis=1
                    )
                    tickers_list.extend(tickers)
                    
                    # Store confidence classes for each post
                    batch_classes = []
                    for ticker_set in tickers:
                        post_classes = set()
                        for ticker in ticker_set:
                            # Check which confidence class contains this ticker
                            if any(entry['ticker'] == ticker for entry in self.confidence_classes['HIGH']):
                                post_classes.add('HIGH')
                            elif any(entry['ticker'] == ticker for entry in self.confidence_classes['MEDIUM']):
                                post_classes.add('MEDIUM')
                            else:
                                post_classes.add('LOW')
                        # Use highest confidence class for the post
                        if 'HIGH' in post_classes:
                            batch_classes.append('HIGH')
                        elif 'MEDIUM' in post_classes:
                            batch_classes.append('MEDIUM')
                        else:
                            batch_classes.append('LOW')
                    
                    confidence_classes_list.extend(batch_classes)
                    pbar.update(len(batch))
                
                processed_df['tickers'] = tickers_list
                processed_df['confidence_class'] = confidence_classes_list  # Add confidence classes to DataFrame
            
            # Step 4: Confidence Scoring
            print(f"\n{Fore.YELLOW}[4/{total_steps}] Calculating Confidence Scores{Style.RESET_ALL}")
            with tqdm(total=len(processed_df), desc="Scoring tickers") as pbar:
                processed_df['ticker_confidence'] = processed_df.apply(
                    lambda row: self._calculate_ticker_confidence(
                        row['tickers'],
                        row['cleaned_text'],
                        row.get('subreddit')
                    ),
                    axis=1
                )
                pbar.update(len(processed_df))
            
            # Step 5: Sentiment Analysis
            print(f"\n{Fore.YELLOW}[5/{total_steps}] Performing Sentiment Analysis{Style.RESET_ALL}")
            with tqdm(total=len(processed_df), desc="Analyzing sentiment") as pbar:
                processed_df = self._calculate_sentiment_scores(processed_df)
                if 'top_comments' in processed_df.columns:
                    processed_df = self._process_comments(processed_df)
                processed_df = self._add_derived_features(processed_df)
                pbar.update(len(processed_df))
            
            # Step 6: Final Processing
            print(f"\n{Fore.YELLOW}[6/{total_steps}] Finalizing Results{Style.RESET_ALL}")
            processed_df['date'] = pd.to_datetime(processed_df['created_utc']).dt.date
            daily_metrics = self._aggregate_daily_metrics(processed_df)
            
            # Save entity matching debug info
            self.entity_linker.save_debug_info()
            
            # Save confidence class debug info with enhanced details
            debug_file = self.debug_path / f"confidence_classes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(debug_file, 'w') as f:
                json.dump({
                    'high_confidence': [
                        {
                            'ticker': entry['ticker'],
                            'context': entry['context'],
                            'finbert_score': entry['finbert_score'],
                            'entity_matches': entry['entity_matches'],
                            'entity_confidence': entry['entity_confidence']
                        }
                        for entry in self.confidence_classes['HIGH']
                    ],
                    'medium_confidence': [
                        {
                            'ticker': entry['ticker'],
                            'context': entry['context'],
                            'finbert_score': entry['finbert_score'],
                            'entity_matches': entry['entity_matches'],
                            'entity_confidence': entry['entity_confidence']
                        }
                        for entry in self.confidence_classes['MEDIUM']
                    ],
                    'low_confidence': [
                        {
                            'ticker': entry['ticker'],
                            'context': entry['context'],
                            'finbert_score': entry['finbert_score'],
                            'entity_matches': entry['entity_matches'],
                            'entity_confidence': entry['entity_confidence']
                        }
                        for entry in self.confidence_classes['LOW']
                    ]
                }, f, indent=2)
            
            print(f"{Fore.GREEN}✓ Saved enhanced confidence class debug info to {debug_file}{Style.RESET_ALL}")
            
            # Print Summary
            print(f"\n{Fore.GREEN}✓ Processing Complete{Style.RESET_ALL}")
            print(f"  • Processed {len(processed_df)} posts")
            print(f"  • Generated {len(daily_metrics)} daily records")
            print(f"  • Found {processed_df['tickers'].apply(len).sum()} ticker mentions")
            
            return processed_df, daily_metrics
            
        except Exception as e:
            logger.error(f"Error Processing Reddit Data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    # -----------------------------------------------------------------------------------------------
    
    def _clean_text(self, text):
        """Clean and Preprocess Text."""
        if pd.isna(text):
            return ""
            
        # Convert to Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove Special Characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    # -----------------------------------------------------------------------------------------------
    
    def _summarize_text(self, text):
        """Summarize text if it's too long."""
        try:
            words = text.split()
            if len(words) > SUMMARY_LENGTH:
                return ' '.join(words[:SUMMARY_LENGTH]) + '...'
            return text
        except:
            return text
    
    def _calculate_sentiment_scores(self, df):
        """Calculate multiple sentiment scores using different methods."""
        tqdm.pandas(desc="Calculating sentiment")
        
        # TextBlob sentiment
        df['textblob_sentiment'] = df['cleaned_text'].progress_apply(
            lambda x: self.textblob(x).sentiment.polarity if x else 0
        )
        
        # VADER sentiment
        df['vader_sentiment'] = df['cleaned_text'].progress_apply(
            lambda x: self.vader.polarity_scores(x)['compound'] if x else 0
        )
        
        # Combined sentiment score (weighted average)
        df['overall_sentiment'] = (
            df['textblob_sentiment'] * 0.4 + 
            df['vader_sentiment'] * 0.6
        )
        
        return df
    
    # -----------------------------------------------------------------------------------------------
    
    def _process_comments(self, df):
        """Process and analyze top comments."""
        def analyze_comments(comments):
            if not isinstance(comments, list):
                return 0, 0
            
            sentiments = []
            for comment in comments:
                if comment:
                    # Use VADER for comment sentiment (better for short text)
                    sentiments.append(self.vader.polarity_scores(comment)['compound'])
            
            if sentiments:
                return np.mean(sentiments), len(sentiments)
            return 0, 0
        
        # Apply comment analysis
        df[['comment_sentiment', 'comment_count']] = pd.DataFrame(
            df['top_comments'].apply(analyze_comments).tolist(),
            index=df.index
        )
        
        return df
    
    # -----------------------------------------------------------------------------------------------
        
    def _add_derived_features(self, df):
        """Add derived features from Reddit Data."""
        # Engagement Score (weighted)
        df['engagement_score'] = (
            df['score'] * 0.4 + 
            df['num_comments'] * 0.3 +
            df['comment_count'] * 0.3
        )
        
        # Normalize Engagement Score
        df['engagement_score'] = (df['engagement_score'] - df['engagement_score'].mean()) / df['engagement_score'].std()
        
        return df 
    
    # -----------------------------------------------------------------------------------------------
    
    def _aggregate_daily_metrics(self, df):
        """Aggregate metrics by date for pipeline integration."""
        # Ensure required columns exist
        if 'is_relevant' not in df.columns:
            df['is_relevant'] = df['ticker_confidence'].apply(lambda x: 1 if x >= 0.7 else 0)
        
        # Group by date and calculate metrics
        metrics = {
            'overall_sentiment': ['mean', 'std', 'count'],
            'vader_sentiment': 'mean',
            'textblob_sentiment': 'mean',
            'comment_sentiment': 'mean',
            'engagement_score': 'mean',
            'score': 'sum',
            'num_comments': 'sum',
            'is_relevant': 'sum'  # Count of relevant posts per day
        }
        
        # Filter metrics based on available columns
        available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
        
        # Group by date and calculate metrics
        daily = df.groupby('date').agg(available_metrics).round(4)
        
        # Flatten column names
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        
        # Reset index and rename date column
        daily.reset_index(inplace=True)
        daily.rename(columns={'date': 'Date'}, inplace=True)
        
        # Convert Date to datetime
        daily['Date'] = pd.to_datetime(daily['Date'])
        
        return daily

    def _get_rejection_reason(self, row) -> str:
        """Determine why a potential ticker was rejected."""
        reasons = []
        
        # Check if any potential tickers were found
        potential = row['potential_tickers']
        actual = row['tickers']
        
        if not potential:
            return "No potential tickers found"
        
        # Check which tickers were rejected and why
        for ticker in potential:
            if ticker not in actual:
                if ticker in COMMON_WORDS:
                    reasons.append(f"{ticker}: Common word")
                else:
                    has_context, conf = self._has_financial_context(row['cleaned_text'], ticker)
                    if not has_context:
                        reasons.append(f"{ticker}: No financial context")
                    elif conf < 0.3:
                        reasons.append(f"{ticker}: Low confidence ({conf:.2f})")
        
        return "; ".join(reasons) if reasons else "Unknown"

def main():
    """Main function to process Reddit data files."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Initialize processor
    processor = RedditDataProcessor()

    # Process each CSV file in the raw reddit data directory
    for file in os.listdir(processor.raw_path):
        if file.endswith(".csv"):
            try:
                filepath = processor.raw_path / file
                logger.info(f"Processing file: {filepath}")
                
                # Read the CSV file
                df = pd.read_csv(filepath)
                
                # Process the data
                processed_df, daily_metrics = processor.process_reddit_data(df)
                
                if processed_df is not None and daily_metrics is not None:
                    # Map common company names to tickers
                    name_to_ticker = {
                        'NVIDIA': 'NVDA',
                        'AMD': 'AMD',
                        'INTEL': 'INTC',
                        'TSMC': 'TSM'
                    }
                    
                    # Extract name from filename and convert to ticker
                    input_name = file.split("_")[0].upper()
                    output_name = name_to_ticker.get(input_name, input_name)
                    
                    # Save detailed processed data
                    detailed_file = processor.processed_path / f"{output_name}_detailed_sentiment.csv"
                    processed_df.to_csv(detailed_file, index=False)
                    
                    # Save daily aggregated data
                    daily_file = processor.processed_path / f"{output_name}_daily_sentiment.csv"
                    daily_metrics.to_csv(daily_file, index=False)
                    
                    logger.info(f"Saved detailed sentiment data to {detailed_file}")
                    logger.info(f"Saved daily sentiment data to {daily_file}")
                else:
                    logger.error(f"Failed to process data from {file}")
                    
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main() 