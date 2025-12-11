# Evaluation Visualization Guide

## Overview

The `eval.py` script now includes comprehensive visualization capabilities inspired by `custom_analyzer.py`. You can generate interactive HTML reports showing:

1. **NAV Chart** - Portfolio Net Asset Value over time with drawdown analysis
2. **Trades Candle Chart** - OHLCV candlestick chart with buy/sell markers and trade lines
3. **Trades HTML Report** - Detailed table of all completed trades with performance metrics

## Usage

### Basic Evaluation (No Visualization)
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50
```

### Evaluation with Visualization Export
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --export-viz
```

### Evaluate All Models with Visualization
```bash
python eval.py data/GOOG_2019.csv --export-viz
```

## Output Files

When `--export-viz` is used, the following files are created in `evaluation_output/<model_name>/`:

1. **`nav_chart.html`** - Interactive Plotly chart showing:
   - Portfolio NAV over time
   - Drawdown percentage
   - Initial/Final values, total return, max drawdown

2. **`trades_chart.html`** - Interactive Plotly candlestick chart showing:
   - OHLCV candlestick chart
   - Buy markers (green triangles up)
   - Sell markers (red triangles down)
   - Trade lines connecting entry to exit (green for wins, red for losses)
   - Volume chart below
   - Summary statistics in title

3. **`trades_report.html`** - HTML table report showing:
   - Performance summary (total trades, win rate, average return, etc.)
   - Detailed trade table with:
     - Trade ID
     - Entry/Exit dates and prices
     - Gross P&L
     - Return percentage
     - Duration in days
     - Win/Loss indicator

## Features

### NAV Chart Features
- **Portfolio Value Tracking**: Shows how your portfolio value changes over time
- **Drawdown Analysis**: Visualizes maximum drawdown periods
- **Performance Metrics**: Displays initial capital, final value, total return, and max drawdown
- **Interactive**: Zoom, pan, and hover for detailed information

### Trades Candle Chart Features
- **OHLCV Candlesticks**: Full price action visualization
- **Trade Markers**: 
  - Green triangles (▲) for BUY signals
  - Red triangles (▼) for SELL signals
- **Trade Lines**: Dashed lines connecting entry to exit
  - Green for winning trades
  - Red for losing trades
- **Volume Chart**: Trading volume below price chart
- **Summary Stats**: Total trades, win rate, total P&L in chart title

### Trades HTML Report Features
- **Performance Summary Cards**: 
  - Total trades
  - Win rate (color-coded: green if ≥50%, red if <50%)
  - Average return
  - Total P&L
  - Best/worst trade
  - Average duration
- **Detailed Trade Table**: Sortable, hoverable table with all trade details
- **Color Coding**: 
  - Green for winning trades
  - Red for losing trades
- **Responsive Design**: Works on desktop and mobile devices

## Example Output Structure

```
evaluation_output/
├── model_debug_50/
│   ├── nav_chart.html
│   ├── trades_chart.html
│   └── trades_report.html
├── model_debug_40/
│   ├── nav_chart.html
│   ├── trades_chart.html
│   └── trades_report.html
└── ...
```

## Requirements

### Required Packages
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `plotly` - Interactive visualizations (install with: `pip install plotly`)

### Optional Packages
- If Plotly is not available, visualization functions will log a warning and skip export

## Technical Details

### Data Alignment
- The script automatically aligns trading history with OHLCV data
- History entries (BUY/SELL/HOLD) are matched with corresponding dates from the CSV
- NAV calculation tracks cash and inventory value over time

### Trade Matching
- Uses FIFO (First In, First Out) inventory management
- Matches BUY actions with subsequent SELL actions
- Calculates duration, return percentage, and P&L for each complete trade

### Portfolio NAV Calculation
- Starts with initial capital (first day's price by default)
- Tracks cash balance and inventory
- Calculates NAV as: `cash + (inventory_value)`
- Updates after each trading action

## Customization

### Change Initial Capital
Modify the `initial_capital` parameter in the visualization functions:
```python
initial_capital = 10000  # Start with $10,000
export_nav_chart(ohlcv_data, history, model_name, nav_path, initial_capital)
```

### Change Output Directory
Modify the `output_dir` path in `main()`:
```python
output_dir = Path("custom_output") / model_name
```

### Customize Chart Colors
Modify color values in the visualization functions:
- Buy markers: `color="#00ff00"` (green)
- Sell markers: `color="#ff0000"` (red)
- Winning trades: `color="#2e7d32"` (dark green)
- Losing trades: `color="#c62828"` (dark red)

## Troubleshooting

### Plotly Not Available
**Error**: `Plotly not available. Install with: pip install plotly`

**Solution**: 
```bash
pip install plotly
```

### Data Alignment Issues
If you see errors about mismatched data lengths:
- Ensure the CSV file has Date, Open, High, Low, Close, Adj Close, Volume columns
- Check that the evaluation completed successfully (history was generated)
- Verify the CSV file matches the data used for evaluation

### Empty Trades
If no trades appear in the charts:
- The model may not have made any trades (all HOLD actions)
- Check the evaluation output to see if trades were executed
- Verify the model is working correctly

## Comparison with custom_analyzer.py

This implementation is adapted from `custom_analyzer.py` but simplified for the DQN trading bot:

| Feature | custom_analyzer.py | eval.py |
|---------|-------------------|---------|
| **Trading Type** | Long/Short with leverage | Long only (BUY/SELL) |
| **Position Management** | Complex portfolio with multiple symbols | Simple FIFO inventory |
| **Leverage** | Supports leveraged trades | No leverage |
| **Commissions** | Includes commission tracking | No commissions (can be added) |
| **Visualization** | Multiple chart types | NAV, Candles, Trades table |
| **Complexity** | Full backtesting framework | Simplified for DQN bot |

## Future Enhancements

Potential improvements:
- Add commission tracking
- Support for multiple stocks
- Add more performance metrics (Sharpe ratio, Sortino ratio)
- Export to PDF
- Add comparison charts (model vs buy-and-hold)
- Add trade distribution charts

## Examples

### Example 1: Evaluate Single Model
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --export-viz
```

Output: `evaluation_output/model_debug_50/` with 3 HTML files

### Example 2: Compare All Models
```bash
python eval.py data/GOOG_2019.csv --export-viz
```

Output: Separate directories for each model in `evaluation_output/`

### Example 3: Debug Mode with Visualization
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --debug --export-viz
```

Output: Detailed logs + visualization files

