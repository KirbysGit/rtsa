from pathlib import Path

# Assumes this file is in src/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = DATA_DIR / "merged"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MERGED_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
