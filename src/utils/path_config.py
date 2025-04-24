from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Assumes this file is in src/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Data subdirectories
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = DATA_DIR / "merged"
DEBUG_DIR = DATA_DIR / "debug"
REFERENCES_DIR = DATA_DIR / "references"

# Raw data subdirectories
TICKER_ANALYSIS_DIR = RAW_DIR / "ticker_analysis"
REDDIT_DATA_DIR = RAW_DIR / "reddit_data"
STOCK_DATA_DIR = RAW_DIR / "stock_data"

# Processed data subdirectories
PROCESSED_REDDIT_DIR = PROCESSED_DIR / "reddit_data"
PROCESSED_STOCK_DIR = PROCESSED_DIR / "stock_data"

# Reports directory in src
REPORTS_DIR = SRC_DIR / "reports"

# Results directory for final outputs
RESULTS_DIR = PROJECT_ROOT / "results"

# Reference files paths
VALID_TICKERS_FILE = REFERENCES_DIR / "valid_tickers.csv"
ENTITY_CACHE_FILE = REFERENCES_DIR / "entity_cache.json"

# Directory structure with descriptions
DIRECTORY_STRUCTURE = {
    'data': {
        'raw': {
            'ticker_analysis': 'Stores daily ticker analysis results',
            'reddit_data': 'Raw Reddit post data',
            'stock_data': 'Raw stock price data'
        },
        'processed': {
            'reddit_data': 'Processed Reddit sentiment data',
            'stock_data': 'Processed stock analysis'
        },
        'merged': 'Combined datasets',
        'debug': 'Debug logs and analysis',
        'references': 'Essential reference files for pipeline operation'
    },
    'src': {
        'reports': 'Generated analysis reports'
    },
    'results': 'Final analysis outputs'
}

# List of all directories to ensure they exist
_REQUIRED_DIRECTORIES = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MERGED_DIR,
    DEBUG_DIR,
    RESULTS_DIR,
    TICKER_ANALYSIS_DIR,
    REDDIT_DATA_DIR,
    STOCK_DATA_DIR,
    PROCESSED_REDDIT_DIR,
    PROCESSED_STOCK_DIR,
    REPORTS_DIR,
    REFERENCES_DIR
]

def _ensure_directories_exist():
    """Silently create required directories if they don't exist."""
    for directory in _REQUIRED_DIRECTORIES:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")

# Create directories silently when module is imported
_ensure_directories_exist()
