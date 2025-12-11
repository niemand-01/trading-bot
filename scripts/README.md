# Training and Evaluation Framework

The `train_eval_framework.py` script orchestrates the complete training and evaluation pipeline for multiple stock symbols.

## Features

- **Automated Data Splitting**: Automatically splits data into:
  - Training: Data prior to 2023
  - Validation: Data from 2024
  - Testing: Data from 2025

- **Format Handling**: Handles different CSV formats (with/without Adj Close column, different date column names)

- **Organized Outputs**: 
  - Models saved to `models/<symbol>/`
  - Evaluation results saved to `evaluation_output/<symbol>/`

- **Batch Processing**: Process multiple symbols in a single run

## Usage

### Basic Usage (Default: AAPL and GOOGL)

```bash
python scripts/train_eval_framework.py
```

### Process Specific Symbols

```bash
python scripts/train_eval_framework.py --symbols AAPL GOOGL MSFT
```

### Custom Training Parameters

```bash
python scripts/train_eval_framework.py --symbols GOOGL --episode-count 100 --window-size 20 --batch-size 64
```

### With Debug Logging

```bash
python scripts/train_eval_framework.py --symbols AAPL GOOGL --debug
```

## Command Line Options

- `--symbols SYMBOL1 SYMBOL2 ...`: List of symbols to process (default: AAPL GOOGL)
- `--window-size SIZE`: Window size for feature vector (default: 10)
- `--batch-size SIZE`: Batch size for training (default: 128)
- `--episode-count COUNT`: Number of training episodes (default: 50)
- `--strategy STRATEGY`: Q-learning strategy - `dqn`, `t-dqn`, or `double-dqn` (default: t-dqn)
- `--debug`: Enable debug logging

## Data Requirements

Each symbol requires a CSV file in `data/ib/` with the naming pattern: `<SYMBOL>_ohlcv_1d.csv`

The CSV should contain:
- A date column (named `Date` or `Datetime`)
- A close price column (preferably `Adj Close`, but `close` or `Close` will work)

## Output Structure

After running the framework:

```
models/
  ├── AAPL/
  │   ├── (model files)
  │   └── AAPL_<episode>/ (checkpoints)
  └── GOOGL/
      ├── (model files)
      └── GOOGL_<episode>/ (checkpoints)

evaluation_output/
  ├── AAPL/
  │   └── YYYYMMDD_HHMMSS/
  │       ├── nav_chart.html
  │       ├── trades_chart.html
  │       ├── trades_report.html
  │       └── quantstats_report.html
  └── GOOGL/
      └── YYYYMMDD_HHMMSS/
          └── (same structure)
```

## Notes

- The script creates temporary CSV files for training and validation, which are automatically cleaned up
- Test evaluation uses the original data file with date filtering (no temporary files needed)
- If a symbol doesn't have sufficient data for a period, the script will log a warning and skip that symbol
- Models are organized by symbol after training completes

