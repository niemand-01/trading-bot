"""
Normalize all CSV files in data/ib/ to match GOOGL format:
Date, Symbol, Open, High, Low, Close, Adj Close, Volume

This script:
- Standardizes column names (capitalization)
- Adds 'Adj Close' column (duplicates 'Close' if not present)
- Ensures 'Date' column name
- Saves normalized files back to the same location
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def normalize_csv_file(csv_path):
    """Normalize a single CSV file to match GOOGL format"""
    logging.info(f"Processing: {csv_path.name}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get original columns
    original_cols = df.columns.tolist()
    logging.debug(f"  Original columns: {original_cols}")
    
    # Normalize column names (case-insensitive mapping)
    col_mapping = {}
    
    # Date column
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['date', 'datetime']:
            col_mapping[col] = 'Date'
            break
    
    # Symbol column
    for col in df.columns:
        if col.lower() == 'symbol':
            col_mapping[col] = 'Symbol'
            break
    
    # OHLC columns
    ohlc_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adj close': 'Adj Close',
        'volume': 'Volume'
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ohlc_mapping:
            if col not in col_mapping:  # Don't overwrite if already mapped
                col_mapping[col] = ohlc_mapping[col_lower]
    
    # Apply column renaming
    df = df.rename(columns=col_mapping)
    
    # Ensure Date column exists
    if 'Date' not in df.columns:
        raise ValueError(f"No date column found in {csv_path.name}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure Symbol column exists (extract from filename if not present)
    if 'Symbol' not in df.columns:
        symbol = csv_path.stem.split('_')[0]  # Extract symbol from filename (e.g., TSLA_ohlcv_1d -> TSLA
        df['Symbol'] = symbol
        logging.info(f"  Added Symbol column: {symbol}")
    
    # Ensure Close column exists
    if 'Close' not in df.columns:
        raise ValueError(f"No Close column found in {csv_path.name}")
    
    # Add Adj Close if not present (duplicate Close)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        logging.info(f"  Added Adj Close column (duplicated from Close)")
    
    # Ensure all required columns exist
    required_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing_cols}")
    
    # Select and reorder columns
    df = df[required_cols].copy()
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Save normalized file (backup original first)
    backup_path = csv_path.with_suffix('.csv.backup')
    if not backup_path.exists():
        csv_path.rename(backup_path)
        logging.info(f"  Created backup: {backup_path.name}")
    
    # Save normalized file
    df.to_csv(csv_path, index=False)
    logging.info(f"  Normalized and saved: {csv_path.name}")
    logging.info(f"  Rows: {len(df)}, Date range: {df['Date'].min()} to {df['Date'].max()}")


def main():
    """Normalize all CSV files in data/ib/"""
    data_dir = Path(__file__).parent.parent / "data" / "ib"
    
    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        return
    
    csv_files = list(data_dir.glob("*_ohlcv_1d.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {data_dir}")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to normalize")
    logging.info("=" * 80)
    
    for csv_file in csv_files:
        try:
            normalize_csv_file(csv_file)
            logging.info("")
        except Exception as e:
            logging.error(f"Failed to normalize {csv_file.name}: {e}")
            logging.info("")
    
    logging.info("=" * 80)
    logging.info("Normalization complete!")


if __name__ == "__main__":
    main()

