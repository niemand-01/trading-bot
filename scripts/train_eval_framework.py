"""
Training and Evaluation Framework for DQN Trading Bot

This script orchestrates the training and evaluation process for multiple symbols:
- Trains models using data prior to 2023
- Validates using 2024 data
- Tests using 2025 data
- Organizes models and evaluation results by symbol

Features:
- Supports both long and short positions (short selling enabled)
- Position-aware actions: BUY/SELL adapt based on current position
- Symmetric reward function for both long and short trades
- Enhanced state representation includes position information

Note: The agent's state size is automatically set to (window_size + 1) to include
position state information. Existing models trained without short selling support
will need to be retrained.

Usage:
    python scripts/train_eval_framework.py [--symbols SYMBOL1 SYMBOL2 ...] [--window-size SIZE] 
        [--batch-size SIZE] [--episode-count COUNT] [--strategy STRATEGY] [--debug]

Options:
    --symbols SYMBOL1 SYMBOL2 ...  List of symbols to process (default: AAPL GOOGL)
    --window-size SIZE             Window size for feature vector [default: 10]
                                    Note: Actual state size is (window_size + 1) to include position
    --batch-size SIZE              Batch size for training [default: 128]
    --episode-count COUNT          Number of training episodes [default: 50]
    --strategy STRATEGY            Q-learning strategy (dqn, t-dqn, double-dqn) [default: t-dqn]
    --debug                        Enable debug logging
"""

import os
import sys
import logging
import pandas as pd
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import coloredlogs

# Add parent directory to path to import trading_bot modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import training and evaluation functions directly
from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    show_eval_result,
    switch_k_backend_device
)

# Import evaluation functions from eval.py
from eval import (
    get_ohlcv_data,
    export_nav_chart,
    export_trades_candle_chart,
    export_trades_html,
    export_quantstats_report
)


def setup_logging(debug=False):
    """Setup logging configuration"""
    level = "DEBUG" if debug else "INFO"
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Also configure TensorFlow/Keras backend
    switch_k_backend_device(use_gpu=True)


def normalize_csv_format(df, symbol):
    """Normalize CSV format to ensure it has required columns
    
    Now that all CSV files are normalized, this function is simplified.
    Returns normalized DataFrame with: Date, Adj Close columns
    """
    df = df.copy()
    
    # All files should now have 'Date' and 'Adj Close' columns after normalization
    if 'Date' not in df.columns:
        raise ValueError(f"No 'Date' column found in {symbol} data. Columns: {df.columns.tolist()}")
    
    if 'Adj Close' not in df.columns:
        raise ValueError(f"No 'Adj Close' column found in {symbol} data. Columns: {df.columns.tolist()}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Select only required columns
    df = df[['Date', 'Adj Close']].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def split_data_by_date(df, symbol):
    """Split data into train (< 2023), validation (2024), and test (2025) periods
    
    Returns:
        train_df: Data before 2023
        val_df: Data from 2024
        test_df: Data from 2025
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Split by year
    train_df = df[df['Date'] < pd.Timestamp('2023-01-01')].copy()
    val_df = df[(df['Date'] >= pd.Timestamp('2024-01-01')) & 
                (df['Date'] < pd.Timestamp('2025-01-01'))].copy()
    test_df = df[df['Date'] >= pd.Timestamp('2025-01-01')].copy()
    
    # Log data availability
    logging.info(f"{symbol} data split:")
    logging.info(f"  Train (< 2023): {len(train_df)} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
    logging.info(f"  Validation (2024): {len(val_df)} rows ({val_df['Date'].min() if len(val_df) > 0 else 'N/A'} to {val_df['Date'].max() if len(val_df) > 0 else 'N/A'})")
    logging.info(f"  Test (2025): {len(test_df)} rows ({test_df['Date'].min() if len(test_df) > 0 else 'N/A'} to {test_df['Date'].max() if len(test_df) > 0 else 'N/A'})")
    
    return train_df, val_df, test_df


# Removed create_temp_csv function - no longer needed with direct function calls


def train_model_direct(symbol, train_data, val_data, window_size, batch_size, episode_count, 
                       strategy, model_name, debug=False):
    """Train a model directly using imported functions
    
    The model will be trained with short selling support:
    - Position-aware actions (BUY/SELL adapt based on current position)
    - Symmetric rewards for both long and short positions
    - State includes position information (state_size = window_size + 1)
    
    Args:
        symbol: Symbol name
        train_data: List of training prices
        val_data: List of validation prices
        window_size: Window size (actual state size will be window_size + 1)
        batch_size: Batch size
        episode_count: Number of episodes
        strategy: Training strategy
        model_name: Model name
        debug: Enable debug logging
    
    Returns:
        success: bool indicating if training succeeded
    """
    logging.info(f"Training {symbol} model with short selling support...")
    
    try:
        agent = Agent(window_size, strategy=strategy, pretrained=False, model_name=model_name)
        
        initial_offset = val_data[1] - val_data[0] if len(val_data) > 1 else 0
        
        # Track best validation result to save best model
        best_val_result = float('-inf')
        best_episode = 0
        
        for episode in range(1, episode_count + 1):
            train_result = train_model(agent, episode, train_data, ep_count=episode_count,
                                      batch_size=batch_size, window_size=window_size)
            val_result, _ = evaluate_model(agent, val_data, window_size, debug)
            show_train_result(train_result, val_result, initial_offset)
            
            # Save model if this is the best validation result so far
            if val_result > best_val_result:
                best_val_result = val_result
                best_episode = episode
                # Save best model to model folder (overwrites previous best)
                agent.save_best()
                # Also save checkpoint with episode number for reference
                agent.save(episode)
                logging.info(f"New best validation result: {format_position(val_result)} at episode {episode}.")
                logging.info(f"Best model saved to: models/{model_name}/ (overwrites previous best)")
                logging.info(f"Checkpoint saved to: models/{model_name}_{episode}/")
        
        # Log final best model info
        if best_episode > 0:
            logging.info(f"Training completed. Best model: episode {best_episode} with validation result: {format_position(best_val_result)}")
            logging.info(f"Best model location: models/{model_name}/")
            logging.info(f"Best checkpoint location: models/{model_name}_{best_episode}/")
        
        logging.info(f"Training completed successfully for {symbol}")
        return True
        
    except Exception as e:
        logging.error(f"Training failed for {symbol}: {e}", exc_info=debug)
        return False


def evaluate_model_direct(symbol, original_data_file, window_size, model_name, debug=False, 
                          start_date=None, end_date=None):
    """Evaluate a model directly using imported functions
    
    Evaluation includes:
    - NAV calculation with both long and short positions
    - Trade tracking distinguishing LONG vs SHORT trades
    - Visualizations showing short positions (orange squares, purple diamonds)
    
    Args:
        symbol: Symbol name
        original_data_file: Path to original CSV file with full OHLCV data
        window_size: Window size (must match training window_size)
        model_name: Model name
        debug: Enable debug logging
        start_date: Start date for evaluation (YYYY-MM-DD)
        end_date: End date for evaluation (YYYY-MM-DD)
    
    Returns:
        success: bool indicating if evaluation succeeded
    """
    logging.info(f"Evaluating {symbol} model on test data (with short selling support)...")
    
    try:
        # Get full OHLCV data first (with date filtering if specified)
        ohlcv_data = get_ohlcv_data(original_data_file, start_date, end_date)
        
        # Extract Adj Close as list for evaluation (matching get_stock_data format)
        data = ohlcv_data["Adj Close"].tolist()
        
        if len(data) < 2:
            logging.error(
                f"Insufficient data after date filtering. Found {len(data)} rows."
            )
            logging.error(f"Date range: {start_date} to {end_date}")
            return False
        
        initial_offset = data[1] - data[0]
        
        # Log date range being used
        if start_date or end_date:
            actual_start = ohlcv_data.index[0].strftime("%Y-%m-%d")
            actual_end = ohlcv_data.index[-1].strftime("%Y-%m-%d")
            logging.info(
                f"Evaluating on date range: {actual_start} to {actual_end} ({len(data)} days)"
            )
        else:
            actual_start = ohlcv_data.index[0].strftime("%Y-%m-%d")
            actual_end = ohlcv_data.index[-1].strftime("%Y-%m-%d")
            logging.info(
                f"Evaluating on full dataset: {actual_start} to {actual_end} ({len(data)} days)"
            )
        
        # Load agent and evaluate
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        result = evaluate_model(agent, data, window_size, debug)
        
        # Handle both tuple and single value returns
        profit = result[0] if isinstance(result, tuple) else result
        history = result[1] if isinstance(result, tuple) and len(result) > 1 else []
        show_eval_result(model_name, profit, initial_offset)
        
        # Export visualizations
        if ohlcv_data is not None and history:
            # Create output directory with timestamp: evaluation_output/<model_name>/<timestamp>/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("evaluation_output") / model_name / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a fixed initial capital for consistent NAV calculation
            # This prevents unrealistic returns when using first stock price as initial capital
            # Using $100,000 as standard initial capital for portfolio evaluation
            initial_capital = 100000.0
            
            # Trim ohlcv_data to match history length
            if len(history) < len(ohlcv_data):
                ohlcv_data_trimmed = ohlcv_data.iloc[: len(history)].copy()
            else:
                ohlcv_data_trimmed = ohlcv_data.copy()
            
            # Export NAV chart
            nav_path = output_dir / "nav_chart.html"
            export_nav_chart(
                ohlcv_data_trimmed, history, model_name, str(nav_path), initial_capital
            )
            
            # Export trades candle chart
            trades_chart_path = output_dir / "trades_chart.html"
            export_trades_candle_chart(
                ohlcv_data_trimmed, history, model_name, str(trades_chart_path)
            )
            
            # Export trades HTML table
            trades_html_path = output_dir / "trades_report.html"
            export_trades_html(
                ohlcv_data_trimmed,
                history,
                model_name,
                str(trades_html_path),
                initial_capital,
            )
            
            # Export QuantStats report
            quantstats_path = output_dir / "quantstats_report.html"
            export_quantstats_report(
                ohlcv_data_trimmed,
                history,
                model_name,
                str(quantstats_path),
                initial_capital,
            )
            
            logging.info(f"Visualizations exported to: {output_dir}")
        
        logging.info(f"Evaluation completed successfully for {symbol}")
        return True
        
    except Exception as e:
        logging.error(f"Evaluation failed for {symbol}: {e}", exc_info=debug)
        return False


def organize_outputs(symbol, model_name):
    """Organize model and evaluation outputs by symbol
    
    Moves:
    - models/{model_name}/ -> models/{symbol}/
    - models/{model_name}_* -> models/{symbol}/{model_name}_*
    - evaluation_output/{model_name}/* -> evaluation_output/{symbol}/
    """
    base_dir = Path(__file__).parent.parent
    
    # Organize model
    model_source = base_dir / "models" / model_name
    model_target = base_dir / "models" / symbol
    
    # Skip if source and target are the same (model already organized)
    if model_source == model_target:
        logging.debug(f"Model already organized: {model_name} == {symbol}, skipping")
    elif model_source.exists():
        model_target.mkdir(parents=True, exist_ok=True)
        
        if model_source.is_dir():
            # If target already has content, merge by moving items
            # Otherwise, rename the entire directory
            if not any(model_target.iterdir()):
                # Target is empty, can rename the whole directory
                try:
                    model_source.rename(model_target)
                    logging.info(f"Renamed model directory from {model_source} to {model_target}")
                except OSError:
                    # Fallback: move contents
                    for item in model_source.iterdir():
                        shutil.move(str(item), str(model_target / item.name))
                    try:
                        model_source.rmdir()
                    except OSError:
                        pass
                    logging.info(f"Moved model contents from {model_source} to {model_target}")
            else:
                # Target has content, move items
                for item in model_source.iterdir():
                    target_item = model_target / item.name
                    if target_item.exists():
                        # If item exists, merge directories or overwrite
                        if item.is_dir() and target_item.is_dir():
                            # Merge directory contents
                            for subitem in item.iterdir():
                                shutil.move(str(subitem), str(target_item / subitem.name))
                            item.rmdir()
                        else:
                            # Overwrite file
                            shutil.move(str(item), str(target_item))
                    else:
                        shutil.move(str(item), str(target_item))
                try:
                    model_source.rmdir()
                except OSError:
                    pass
                logging.info(f"Moved model contents from {model_source} to {model_target}")
        else:
            # Move file
            shutil.move(str(model_source), str(model_target / model_name))
            logging.info(f"Moved model file from {model_source} to {model_target}")
    
    # Also move checkpoint directories (model_name_*)
    # Only move if model_name != symbol (otherwise they're already in the right place)
    if model_name != symbol:
        for checkpoint_dir in base_dir.glob(f"models/{model_name}_*"):
            if checkpoint_dir.is_dir():
                # Move checkpoint to symbol directory
                checkpoint_target = model_target / checkpoint_dir.name
                if checkpoint_target.exists():
                    # Merge if exists
                    for item in checkpoint_dir.iterdir():
                        shutil.move(str(item), str(checkpoint_target / item.name))
                    checkpoint_dir.rmdir()
                else:
                    shutil.move(str(checkpoint_dir), str(checkpoint_target))
                logging.info(f"Moved checkpoint {checkpoint_dir.name} to {checkpoint_target}")
    
    # Organize evaluation outputs
    # Only move if model_name != symbol (otherwise they're already in the right place)
    if model_name != symbol:
        eval_source = base_dir / "evaluation_output" / model_name
        eval_target = base_dir / "evaluation_output" / symbol
        eval_target.mkdir(parents=True, exist_ok=True)
        
        if eval_source.exists():
            # Move all timestamped directories
            for item in eval_source.iterdir():
                if item.is_dir():
                    shutil.move(str(item), str(eval_target / item.name))
            # Remove empty source directory
            try:
                eval_source.rmdir()
            except OSError:
                pass
            logging.info(f"Moved evaluation outputs from {eval_source} to {eval_target}")
    else:
        logging.debug(f"Evaluation outputs already organized: {model_name} == {symbol}")


def process_symbol(symbol, data_dir, window_size, batch_size, episode_count, 
                   strategy, debug=False):
    """Process a single symbol: train, validate, and test
    
    Returns:
        success: bool indicating if processing succeeded
    """
    logging.info(f"=" * 80)
    logging.info(f"Processing symbol: {symbol}")
    logging.info(f"=" * 80)
    
    # Find data file
    data_file = data_dir / f"{symbol}_ohlcv_1d.csv"
    if not data_file.exists():
        logging.error(f"Data file not found: {data_file}")
        return False
    
    try:
        # Load and normalize data
        df = pd.read_csv(data_file)
        df = normalize_csv_format(df, symbol)
        
        # Split data by date
        train_df, val_df, test_df = split_data_by_date(df, symbol)
        
        # Check data availability
        if len(train_df) < window_size + 1:
            logging.error(f"{symbol}: Insufficient training data ({len(train_df)} rows, need at least {window_size + 1})")
            return False
        
        if len(val_df) < window_size + 1:
            logging.warning(f"{symbol}: Insufficient validation data ({len(val_df)} rows, need at least {window_size + 1})")
            logging.warning(f"{symbol}: Proceeding with available validation data")
        
        if len(test_df) < window_size + 1:
            logging.warning(f"{symbol}: Insufficient test data ({len(test_df)} rows, need at least {window_size + 1})")
            logging.warning(f"{symbol}: Proceeding with available test data")
        
        # Extract data as lists for training (matching get_stock_data format)
        train_data = train_df['Adj Close'].tolist()
        val_data = val_df['Adj Close'].tolist()
        
        # Model name (will be organized by symbol later)
        model_name = symbol
        
        try:
            # Train model directly
            train_success = train_model_direct(
                symbol, train_data, val_data, window_size, batch_size, 
                episode_count, strategy, model_name, debug
            )
            
            if not train_success:
                logging.error(f"{symbol}: Training failed")
                return False
            
            # Evaluate model on test data using original file with date filtering
            # Get date range for test data
            test_start = test_df['Date'].min().strftime('%Y-%m-%d') if len(test_df) > 0 else None
            test_end = test_df['Date'].max().strftime('%Y-%m-%d') if len(test_df) > 0 else None
            
            eval_success = evaluate_model_direct(
                symbol, data_file, window_size, model_name, debug,
                start_date=test_start, end_date=test_end
            )
            
            if not eval_success:
                logging.warning(f"{symbol}: Evaluation failed, but model was trained")
            
            # Organize outputs by symbol
            organize_outputs(symbol, model_name)
            
            logging.info(f"{symbol}: Processing completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"{symbol}: Error during processing: {e}", exc_info=debug)
            return False
    
    except Exception as e:
        logging.error(f"{symbol}: Error during processing: {e}", exc_info=debug)
        return False


def main(symbols, window_size, batch_size, episode_count, strategy, debug=False):
    """Main function to process all symbols
    
    Note: Models trained with this framework support short selling:
    - Actions are position-aware (BUY/SELL adapt to current position)
    - State size is (window_size + 1) to include position information
    - Rewards are symmetric for both long and short positions
    """
    setup_logging(debug)
    
    # Log short selling support
    logging.info("=" * 80)
    logging.info("DQN Trading Bot - Training & Evaluation Framework")
    logging.info("Short Selling Support: ENABLED")
    logging.info(f"State Size: {window_size} (price features) + 1 (position state) = {window_size + 1}")
    logging.info("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data" / "ib"
    
    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        return
    
    # Process each symbol
    results = {}
    for symbol in symbols:
        success = process_symbol(
            symbol, data_dir, window_size, batch_size, episode_count, 
            strategy, debug
        )
        results[symbol] = success
    
    # Summary
    logging.info("=" * 80)
    logging.info("Processing Summary")
    logging.info("=" * 80)
    for symbol, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"{symbol}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    logging.info(f"\nTotal: {successful}/{total} symbols processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training and Evaluation Framework for DQN Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default symbols (AAPL, GOOGL)
  python scripts/train_eval_framework.py
  
  # Process specific symbols
  python scripts/train_eval_framework.py --symbols AAPL GOOGL MSFT
  
  # Custom training parameters
  python scripts/train_eval_framework.py --symbols AAPL --episode-count 100 --window-size 20
        """
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL"],
        help="List of symbols to process (default: AAPL GOOGL)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Window size for feature vector (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)"
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=50,
        help="Number of training episodes (default: 50)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="t-dqn",
        choices=["dqn", "t-dqn", "double-dqn"],
        help="Q-learning strategy (default: t-dqn)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    main(
        args.symbols,
        args.window_size,
        args.batch_size,
        args.episode_count,
        args.strategy,
        args.debug
    )

