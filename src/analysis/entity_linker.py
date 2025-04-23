# src.analysis.entity_linker.py  

# Imports.
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Set, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from colorama import Fore, Style
import requests
from bs4 import BeautifulSoup

# Local Imports.
from src.utils.path_config import DEBUG_DIR

# Initialize logger.
logger = logging.getLogger(__name__)

class EntityLinker:
    def __init__(self, cache_duration_hours: int = 24):
        """Initialize the EntityLinker with caching."""
        # Initialize paths and basic attributes
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_file = DEBUG_DIR / "entity_cache.json"
        self.match_history = []
        
        # Initialize common word tickers to filter out
        self.common_word_tickers = {
            'A', 'I', 'AM', 'BE', 'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE',
            'ALL', 'AND', 'ANY', 'ARE', 'BAD', 'BIG', 'BUT', 'CAN', 'DAY', 'DID', 'FOR', 'GET', 'GOT', 'HAD', 'HAS', 'HER', 'HIM', 'HIS',
            'HOW', 'LAW', 'LET', 'MAN', 'NEW', 'NOT', 'NOW', 'OLD', 'ONE', 'OUR', 'OUT', 'PAY', 'PUT', 'SAY', 'SEE', 'SHE', 'TWO', 'USE',
            'WAY', 'WHO', 'WHY', 'YES', 'YOU', 'ABLE', 'ALSO', 'AWAY', 'BACK', 'BEST', 'CALM', 'CASE', 'COST', 'DARE', 'DEAL', 'DEAR',
            'EACH', 'ELSE', 'EVEN', 'EVER', 'FACT', 'FAIR', 'FAST', 'FIND', 'FREE', 'FULL', 'GAVE', 'GIVE', 'GONE', 'GOOD', 'GREW',
            'GROW', 'HAVE', 'HERE', 'HIGH', 'INTO', 'JUST', 'KEEP', 'KIND', 'KNEW', 'KNOW', 'LAST', 'LATE', 'LESS', 'LIFE', 'LIKE',
            'LINE', 'LIVE', 'LONG', 'LOOK', 'LOVE', 'MADE', 'MAKE', 'MANY', 'MIND', 'MORE', 'MOST', 'MOVE', 'MUST', 'NEAR', 'NEED',
            'NEXT', 'NICE', 'ONCE', 'ONLY', 'OPEN', 'OVER', 'PART', 'PAST', 'PLAN', 'PLAY', 'REAL', 'SAID', 'SAME', 'SAVE', 'SEEM',
            'SELF', 'SEND', 'SENT', 'SHOW', 'SIDE', 'SOME', 'SOON', 'STAY', 'SURE', 'TAKE', 'TALK', 'TELL', 'THAN', 'THAT', 'THEM',
            'THEN', 'THEY', 'THIS', 'TIME', 'TRUE', 'TURN', 'VERY', 'WANT', 'WELL', 'WENT', 'WERE', 'WHAT', 'WHEN', 'WILL', 'WITH',
            'WORK', 'YEAR', 'YOUR'
        }
        
        # Initialize ETF tickers that should always be considered valid
        self.valid_etfs = {
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'GLD', 'SLV', 'VGK', 'EEM', 'TLT', 'LQD', 'HYG',
            'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ'
        }
        
        # Initialize fallback entities first
        self._fallback_entities = self._get_fallback_entities()
        
        # Load or update cache last
        self.entity_map = self._load_or_update_cache()
    
    def _load_or_update_cache(self) -> Dict:
        """Load entity cache or update if expired."""
        try:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        cache = json.load(f)
                    
                    last_update = datetime.fromisoformat(cache['last_update'])
                    if datetime.now() - last_update < self.cache_duration:
                        print(f"{Fore.GREEN}✓ Loaded entity cache{Style.RESET_ALL}")
                        
                        # Convert lists back to sets for aliases
                        for ticker, data in cache['entities'].items():
                            if 'aliases' in data:
                                data['aliases'] = set(data['aliases'])
                        
                        return cache['entities']
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error reading cache file: {str(e)}")
                    # Don't return here, continue to update cache
            
            print(f"{Fore.YELLOW}Updating entity cache...{Style.RESET_ALL}")
            entities = self._fetch_current_entities()
            
            # Prepare cache data with list conversion for JSON serialization
            cache_data = {
                'last_update': datetime.now().isoformat(),
                'entities': {
                    ticker: {
                        **entity_data,
                        'aliases': list(entity_data['aliases']) if isinstance(entity_data.get('aliases'), set) else []
                    }
                    for ticker, entity_data in entities.items()
                }
            }
            
            # Save to cache with error handling
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                print(f"{Fore.GREEN}✓ Updated entity cache with {len(entities)} stocks{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error loading/updating entity cache: {str(e)}")
            return self._fallback_entities.copy()
    
    def _fetch_current_entities(self) -> Dict:
        """Fetch current hot stocks and their related entities."""
        entities = {}
        
        try:
            # Get top movers from Yahoo Finance
            movers = self._get_market_movers()
            
            # Get company info for each ticker
            for ticker in movers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if not info or 'longName' not in info:
                        continue
                    
                    # Extract and clean company name variations
                    company_name = info.get('longName', '').lower()
                    short_name = info.get('shortName', '').lower()
                    
                    # Create base aliases set
                    aliases = {
                        company_name,
                        short_name,
                        ticker.lower(),
                        company_name.split()[0]  # First word of company name
                    }
                    
                    # Add common variations
                    if 'inc.' in company_name:
                        aliases.add(company_name.replace('inc.', '').strip())
                    if 'corp.' in company_name:
                        aliases.add(company_name.replace('corp.', '').strip())
                    if 'corporation' in company_name:
                        aliases.add(company_name.replace('corporation', '').strip())
                    
                    # Remove empty strings and clean aliases
                    aliases = {alias.strip() for alias in aliases if alias and len(alias.strip()) > 1}
                    
                    entities[ticker] = {
                        'company_name': company_name,
                        'short_name': short_name,
                        'industry': info.get('industry', ''),
                        'sector': info.get('sector', ''),
                        'officers': [
                            officer.get('name', '').lower()
                            for officer in info.get('officers', [])
                            if officer.get('name')
                        ],
                        'products': self._extract_products(info.get('longBusinessSummary', '')),
                        'aliases': aliases,
                        'confidence_class': 'HIGH'  # Default for verified stocks
                    }
                except Exception as e:
                    logger.debug(f"Error fetching info for {ticker}: {str(e)}")
                    continue
            
            return entities
            
        except Exception as e:
            logger.error(f"Error fetching entities: {str(e)}")
            return self._fallback_entities
    
    def _get_market_movers(self) -> List[str]:
        """Get current market movers from Yahoo Finance."""
        movers = set()
        
        try:
            # Most active stocks
            url = "https://finance.yahoo.com/most-active"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract tickers from the table
            for row in soup.select('tr'):
                ticker_cell = row.select_one('td:first-child')
                if ticker_cell and ticker_cell.text.strip().isalpha():
                    ticker = ticker_cell.text.strip().upper()
                    if len(ticker) <= 5:  # Only consider standard length tickers
                        movers.add(ticker)
            
            if not movers:  # If no tickers found, use fallback
                return list(self._fallback_entities.keys())
            
            return list(movers)[:50]  # Limit to top 50
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {str(e)}")
            return list(self._fallback_entities.keys())
    
    def _extract_products(self, text: str) -> List[str]:
        """Extract potential product names from business summary."""
        if not text:
            return []
        
        # Split on common separators and filter
        words = text.lower().replace(',', ' ').replace(';', ' ').split()
        products = []
        
        # Look for product-like terms
        product_indicators = ['product', 'platform', 'service', 'technology', 'solution']
        
        current_product = []
        for word in words:
            if word in product_indicators:
                if current_product:
                    products.append(' '.join(current_product))
                current_product = []
            elif len(word) > 3 and word.isalpha():
                current_product.append(word)
            else:
                if current_product:
                    products.append(' '.join(current_product))
                current_product = []
        
        if current_product:
            products.append(' '.join(current_product))
        
        return list(set(products))
    
    def _get_fallback_entities(self) -> Dict:
        """Provide fallback entity data for key tech stocks."""
        return {
            'NVDA': {
                'company_name': 'nvidia corporation',
                'short_name': 'nvidia',
                'aliases': {'nvidia', 'jensen huang', 'cuda', 'geforce', 'rtx', 'gpu'},
                'products': ['cuda', 'geforce', 'rtx', 'gpu', 'tensor', 'drive'],
                'confidence_class': 'HIGH'
            },
            'AAPL': {
                'company_name': 'apple inc.',
                'short_name': 'apple',
                'aliases': {'apple', 'tim cook', 'iphone', 'macbook', 'ios'},
                'products': ['iphone', 'macbook', 'ipad', 'airpods', 'mac'],
                'confidence_class': 'HIGH'
            },
            'MSFT': {
                'company_name': 'microsoft corporation',
                'short_name': 'microsoft',
                'aliases': {'microsoft', 'satya nadella', 'windows', 'azure', 'xbox'},
                'products': ['windows', 'azure', 'office', 'xbox', 'teams'],
                'confidence_class': 'HIGH'
            },
            'GOOGL': {
                'company_name': 'alphabet inc.',
                'short_name': 'google',
                'aliases': {'google', 'alphabet', 'sundar pichai', 'android', 'chrome'},
                'products': ['search', 'android', 'chrome', 'gmail', 'maps'],
                'confidence_class': 'HIGH'
            },
            'META': {
                'company_name': 'meta platforms inc.',
                'short_name': 'meta',
                'aliases': {'meta', 'facebook', 'mark zuckerberg', 'instagram', 'whatsapp'},
                'products': ['facebook', 'instagram', 'whatsapp', 'oculus', 'messenger'],
                'confidence_class': 'HIGH'
            }
        }
    
    def validate_context(self, text: str, ticker: str, context_window: int = 50) -> Tuple[bool, float, List[str]]:
        """Validate if ticker appears in meaningful context with known entities."""
        if not text or not ticker:
            return False, 0.0, []
        
        text = text.lower()
        ticker = ticker.upper()
        
        # Always accept valid ETFs
        if ticker in self.valid_etfs:
            return True, 1.0, [f"Valid ETF: {ticker}"]
        
        # Reject common word tickers unless they have very strong context
        if ticker in self.common_word_tickers:
            # Get context window
            context = self._get_context_window(text, ticker.lower(), context_window)
            if not context:
                return False, 0.0, []
            
            # Check for strong financial indicators
            dollar_symbol = f"${ticker.lower()}" in text
            stock_mention = any(term in context.lower() for term in ['stock', 'share', 'ticker', 'etf', 'trading'])
            
            # Only allow if it has explicit stock context
            if not (dollar_symbol and stock_mention):
                return False, 0.0, [f"Rejected common word ticker: {ticker}"]
        
        # Get entity info
        entity_info = self.entity_map.get(ticker)
        if not entity_info:
            entity_info = self._fallback_entities.get(ticker)
        
        if not entity_info:
            return False, 0.0, []
        
        # Get context window
        context = self._get_context_window(text, ticker.lower(), context_window)
        if not context:
            return False, 0.0, []
        
        # Track matches found
        matches = []
        match_strength = 0.0
        
        # Check for company name and aliases
        for alias in entity_info['aliases']:
            if alias in context:
                matches.append(f"Company reference: {alias}")
                match_strength += 0.4  # Strong match for company name
        
        # Check for officer mentions
        for officer in entity_info.get('officers', []):
            if officer in context:
                matches.append(f"Officer mention: {officer}")
                match_strength += 0.3  # Good match for officer names
        
        # Check for product mentions
        for product in entity_info.get('products', []):
            if product in context:
                matches.append(f"Product mention: {product}")
                match_strength += 0.2  # Moderate match for products
        
        # Check for industry/sector terms
        industry = entity_info.get('industry', '').lower()
        sector = entity_info.get('sector', '').lower()
        if industry and industry in context:
            matches.append(f"Industry mention: {industry}")
            match_strength += 0.1
        if sector and sector in context:
            matches.append(f"Sector mention: {sector}")
            match_strength += 0.1
        
        # Cap the match strength
        match_strength = min(1.0, match_strength)
        
        # Track match for debugging
        self.match_history.append({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'context': context,
            'matches': matches,
            'match_strength': match_strength
        })
        
        return bool(matches), match_strength, matches
    
    def _get_context_window(self, text: str, target: str, window_size: int) -> str:
        """Get text window around target word."""
        words = text.split()
        try:
            idx = words.index(target.lower())
            start = max(0, idx - window_size)
            end = min(len(words), idx + window_size + 1)
            return ' '.join(words[start:end])
        except ValueError:
            return ""
    
    def save_debug_info(self):
        """Save entity matching debug information."""
        if not self.match_history:
            return
        
        # Convert sets to lists for JSON serialization
        serializable_history = []
        for match in self.match_history:
            serializable_match = match.copy()
            if isinstance(match.get('matches'), set):
                serializable_match['matches'] = list(match['matches'])
            serializable_history.append(serializable_match)
        
        debug_file = DEBUG_DIR / f"entity_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(debug_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"{Fore.GREEN}✓ Saved entity matching debug info to {debug_file}{Style.RESET_ALL}")
    
    def get_confidence_class(self, ticker: str) -> str:
        """Get the confidence class for a ticker."""
        if ticker in self.entity_map:
            return self.entity_map[ticker].get('confidence_class', 'LOW')
        elif ticker in self._fallback_entities:
            return self._fallback_entities[ticker].get('confidence_class', 'LOW')
        return 'LOW' 