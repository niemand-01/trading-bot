"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug] [--export-viz] [--start-date=<start-date>] [--end-date=<end-date>]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
  --export-viz                  Export visualization HTML files (NAV chart, trades chart, trades table).
  --start-date=<start-date>     Start date for evaluation (format: YYYY-MM-DD). If not specified, uses first date in CSV.
  --end-date=<end-date>         End date for evaluation (format: YYYY-MM-DD). If not specified, uses last date in CSV.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import coloredlogs

from docopt import docopt

# Plotly imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

# QuantStats imports
try:
    import quantstats as qs

    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    logging.warning("QuantStats not available. Install with: pip install quantstats")

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device,
)


def get_ohlcv_data(stock_file, start_date=None, end_date=None):
    """Reads full OHLCV data from CSV file

    Args:
        stock_file: Path to CSV file
        start_date: Start date for filtering (YYYY-MM-DD format or None)
        end_date: End date for filtering (YYYY-MM-DD format or None)

    Returns:
        Filtered DataFrame with Date as index
    """
    df = pd.read_csv(stock_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Filter by date range if provided
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]

    return df


def calculate_portfolio_nav(data, history, initial_capital=None):
    """Calculate portfolio NAV over time from trading history with short selling support

    This matches the profit calculation in evaluate_model:
    - BUY: Open long if flat, close short if short
    - SELL: Close long if long, open short if flat
    - SHORT_SELL: Open short position
    - COVER_SHORT: Close short position
    - NAV = initial_capital + cumulative_realized_profit + unrealized_profit

    Note: For realistic returns calculation, use a fixed initial capital (e.g., $100,000)
    instead of the first stock price to avoid inflated percentage returns.
    """
    if initial_capital is None:
        # Use fixed initial capital for consistent evaluation
        # This prevents unrealistic returns when using first stock price
        initial_capital = 100000.0

    nav_history = [initial_capital]
    cumulative_realized_profit = 0.0
    long_inventory = []  # Store long buy prices (FIFO)
    short_inventory = []  # Store short sell prices (FIFO)

    # Get price data as list for compatibility
    prices = data["Adj Close"].values

    # Process history - it should align with data indices
    for i, (price, action) in enumerate(history):
        if i >= len(prices):
            break

        current_price = prices[i] if i < len(prices) else prices[-1]

        if action == "BUY":
            # Open long position
            long_inventory.append(price)
        elif action == "SELL" and len(long_inventory) > 0:
            # Close long position
            bought_price = long_inventory.pop(0)
            profit = price - bought_price
            cumulative_realized_profit += profit
        elif action == "SHORT_SELL":
            # Open short position
            short_inventory.append(price)
        elif action == "COVER_SHORT" and len(short_inventory) > 0:
            # Close short position
            sold_price = short_inventory.pop(0)
            profit = sold_price - price  # Profit when price goes down
            cumulative_realized_profit += profit

        # Calculate current NAV: initial_capital + realized_profit + unrealized_profit
        # Unrealized profit for longs = sum of (current_price - buy_price)
        # Unrealized profit for shorts = sum of (sell_price - current_price)
        unrealized_profit_long = (
            sum(current_price - buy_price for buy_price in long_inventory)
            if long_inventory
            else 0.0
        )
        unrealized_profit_short = (
            sum(sell_price - current_price for sell_price in short_inventory)
            if short_inventory
            else 0.0
        )
        unrealized_profit = unrealized_profit_long + unrealized_profit_short

        # NAV = initial + realized profit + unrealized profit
        current_nav = initial_capital + cumulative_realized_profit + unrealized_profit
        nav_history.append(current_nav)

    return nav_history


def get_complete_trades(history, data):
    """Extract complete trades (entry/exit pairs) from history with short selling support"""
    trades = []
    long_inventory = []  # Track long positions
    short_inventory = []  # Track short positions
    trade_id = 1

    prices = data["Adj Close"].values
    dates = data.index

    # Process history and match with data indices
    for i, (price, action) in enumerate(history):
        if i >= len(dates):
            break

        if action == "BUY":
            # Open long position
            long_inventory.append(
                {
                    "entry_price": price,
                    "entry_date": dates[i],
                    "entry_index": i,
                    "type": "LONG",
                }
            )
        elif action == "SELL":
            if len(long_inventory) > 0:
                # Close long position
                entry = long_inventory.pop(0)
                exit_price = price
                exit_date = dates[i]

                gross_pnl = exit_price - entry["entry_price"]
                return_pct = (
                    (gross_pnl / entry["entry_price"]) * 100
                    if entry["entry_price"] > 0
                    else 0
                )

                # Calculate duration
                if isinstance(exit_date, pd.Timestamp) and isinstance(
                    entry["entry_date"], pd.Timestamp
                ):
                    duration_days = (exit_date - entry["entry_date"]).days
                else:
                    duration_days = (
                        exit_date - entry["entry_date"]
                        if hasattr(exit_date - entry["entry_date"], "days")
                        else 0
                    )

                trades.append(
                    {
                        "trade_id": trade_id,
                        "entry_date": entry["entry_date"],
                        "exit_date": exit_date,
                        "entry_price": entry["entry_price"],
                        "exit_price": exit_price,
                        "quantity": 1.0,
                        "gross_pnl": gross_pnl,
                        "return_pct": return_pct,
                        "duration_days": duration_days,
                        "is_winning": gross_pnl > 0,
                        "trade_type": "LONG",
                    }
                )
                trade_id += 1
            else:
                # Open short position
                short_inventory.append(
                    {
                        "entry_price": price,
                        "entry_date": dates[i],
                        "entry_index": i,
                        "type": "SHORT",
                    }
                )
        elif action == "SHORT_SELL":
            # Open short position
            short_inventory.append(
                {
                    "entry_price": price,
                    "entry_date": dates[i],
                    "entry_index": i,
                    "type": "SHORT",
                }
            )
        elif action == "COVER_SHORT":
            if len(short_inventory) > 0:
                # Close short position
                entry = short_inventory.pop(0)
                exit_price = price
                exit_date = dates[i]

                gross_pnl = (
                    entry["entry_price"] - exit_price
                )  # Profit when price goes down
                return_pct = (
                    (gross_pnl / entry["entry_price"]) * 100
                    if entry["entry_price"] > 0
                    else 0
                )

                # Calculate duration
                if isinstance(exit_date, pd.Timestamp) and isinstance(
                    entry["entry_date"], pd.Timestamp
                ):
                    duration_days = (exit_date - entry["entry_date"]).days
                else:
                    duration_days = (
                        exit_date - entry["entry_date"]
                        if hasattr(exit_date - entry["entry_date"], "days")
                        else 0
                    )

                trades.append(
                    {
                        "trade_id": trade_id,
                        "entry_date": entry["entry_date"],
                        "exit_date": exit_date,
                        "entry_price": entry["entry_price"],
                        "exit_price": exit_price,
                        "quantity": 1.0,
                        "gross_pnl": gross_pnl,
                        "return_pct": return_pct,
                        "duration_days": duration_days,
                        "is_winning": gross_pnl > 0,
                        "trade_type": "SHORT",
                    }
                )
                trade_id += 1

    return pd.DataFrame(trades)


def export_nav_chart(
    data, history, model_name, output_path="nav_chart.html", initial_capital=None
):
    """Export portfolio NAV chart with drawdown"""
    if not PLOTLY_AVAILABLE:
        logging.error("Plotly not available. Install with: pip install plotly")
        return ""

    nav_history = calculate_portfolio_nav(data, history, initial_capital)
    dates = data.index.tolist()

    # NAV history structure:
    # - nav_history[0] = initial capital (before any trading)
    # - nav_history[1] = NAV after first action (should align with dates[0])
    # - nav_history[i] = NAV after action i-1 (should align with dates[i-1])
    # - nav_history[-1] = NAV after last action
    #
    # nav_history has len(history) + 1 entries (initial + one per action)
    # dates has len(data) entries
    #
    # We want to plot NAV for each date, so:
    # - Skip nav_history[0] (initial state before trading)
    # - Use nav_history[1:] which has len(history) entries
    # - Align with dates, padding the last NAV value if needed

    # Start from nav_history[1] (skip initial state)
    nav_to_plot = nav_history[1:] if len(nav_history) > 1 else nav_history

    # Align with dates
    if len(nav_to_plot) < len(dates):
        # Pad with last NAV value for remaining dates
        nav_to_plot = list(nav_to_plot) + [nav_to_plot[-1]] * (
            len(dates) - len(nav_to_plot)
        )
    elif len(nav_to_plot) > len(dates):
        # Trim to match dates length
        nav_to_plot = nav_to_plot[: len(dates)]

    nav_history = nav_to_plot

    # Calculate drawdown
    peak = nav_history[0]
    drawdowns = []
    max_drawdown = 0

    for value in nav_history:
        if value > peak:
            peak = value
        drawdown = ((value / peak) - 1) * 100 if peak > 0 else 0
        drawdowns.append(drawdown)
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Portfolio Net Asset Value (NAV)", "Drawdown"),
        row_heights=[0.7, 0.3],
    )

    # Add NAV chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=nav_history,
            mode="lines",
            name="Portfolio NAV",
            line=dict(color="#2E86AB", width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>NAV:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add drawdown chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            mode="lines",
            name="Drawdown",
            fill="tonexty",
            line=dict(color="#A23B72", width=1),
            fillcolor="rgba(162, 59, 114, 0.3)",
            hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    initial_value = nav_history[0]
    final_value = nav_history[-1]
    total_return = ((final_value / initial_value) - 1) * 100 if initial_value > 0 else 0

    # Update layout
    fig.update_layout(
        title={
            "text": f"Portfolio NAV Analysis - {model_name}<br><sub>Initial: ${initial_value:,.2f} | Final: ${final_value:,.2f} | Return: {total_return:.2f}% | Max DD: {max_drawdown:.2f}%</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        height=800,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        dragmode="pan",
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Save the chart with scroll zoom enabled
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    # Enable scroll zoom in config
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [],
    }
    fig.write_html(output_path, config=config)

    logging.info(f"NAV chart exported to: {output_path}")
    return output_path


def export_trades_candle_chart(
    data, history, model_name, output_path="trades_chart.html"
):
    """Export OHLCV candle chart with buy/sell markers"""
    if not PLOTLY_AVAILABLE:
        logging.error("Plotly not available. Install with: pip install plotly")
        return ""

    # Get complete trades
    trades_df = get_complete_trades(history, data)

    # Extract buy/sell points from history (including short positions)
    buy_points = []
    sell_points = []
    short_sell_points = []
    cover_short_points = []

    prices = data["Adj Close"].values
    dates = data.index

    for i, (price, action) in enumerate(history):
        if i >= len(dates):
            break
        if action == "BUY":
            buy_points.append({"date": dates[i], "price": price})
        elif action == "SELL":
            # Check if this is closing a long or opening a short
            # We'll determine this from the trades_df
            sell_points.append({"date": dates[i], "price": price})
        elif action == "SHORT_SELL":
            short_sell_points.append({"date": dates[i], "price": price})
        elif action == "COVER_SHORT":
            cover_short_points.append({"date": dates[i], "price": price})

    # Create subplots: Price with trades, Volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price with Trading Signals", "Volume"),
        row_heights=[0.7, 0.3],
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # Add buy markers
    if buy_points:
        buy_dates = [p["date"] for p in buy_points]
        buy_prices = [p["price"] for p in buy_points]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#00ff00",
                    line=dict(width=2, color="darkgreen"),
                ),
                text=["BUY"] * len(buy_points),
                textposition="top center",
                textfont=dict(color="darkgreen", size=10, family="Arial Black"),
                name="Buy",
                hovertemplate="<b>Buy</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add sell markers (closing long positions)
    if sell_points:
        sell_dates = [p["date"] for p in sell_points]
        sell_prices = [p["price"] for p in sell_points]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers+text",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ff0000",
                    line=dict(width=2, color="darkred"),
                ),
                text=["SELL"] * len(sell_points),
                textposition="bottom center",
                textfont=dict(color="darkred", size=10, family="Arial Black"),
                name="Sell (Close Long)",
                hovertemplate="<b>Sell</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add short sell markers
    if short_sell_points:
        short_dates = [p["date"] for p in short_sell_points]
        short_prices = [p["price"] for p in short_sell_points]
        fig.add_trace(
            go.Scatter(
                x=short_dates,
                y=short_prices,
                mode="markers+text",
                marker=dict(
                    symbol="square",
                    size=12,
                    color="#ff8800",
                    line=dict(width=2, color="darkorange"),
                ),
                text=["SHORT"] * len(short_sell_points),
                textposition="top center",
                textfont=dict(color="darkorange", size=10, family="Arial Black"),
                name="Short Sell",
                hovertemplate="<b>Short Sell</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add cover short markers
    if cover_short_points:
        cover_dates = [p["date"] for p in cover_short_points]
        cover_prices = [p["price"] for p in cover_short_points]
        fig.add_trace(
            go.Scatter(
                x=cover_dates,
                y=cover_prices,
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="#8800ff",
                    line=dict(width=2, color="purple"),
                ),
                text=["COVER"] * len(cover_short_points),
                textposition="bottom center",
                textfont=dict(color="purple", size=10, family="Arial Black"),
                name="Cover Short",
                hovertemplate="<b>Cover Short</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add trade lines (entry to exit) with different colors for long vs short
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            trade_type = trade.get("trade_type", "LONG")
            if trade_type == "SHORT":
                # Short trades: green when winning (price goes down), red when losing
                line_color = "#2e7d32" if trade["is_winning"] else "#c62828"
            else:
                # Long trades: green when winning (price goes up), red when losing
                line_color = "#2e7d32" if trade["is_winning"] else "#c62828"

            fig.add_trace(
                go.Scatter(
                    x=[trade["entry_date"], trade["exit_date"]],
                    y=[trade["entry_price"], trade["exit_price"]],
                    mode="lines",
                    line=dict(
                        color=line_color,
                        width=2,
                        dash="dash" if trade_type == "SHORT" else "solid",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color="rgba(158,202,225,0.6)",
            marker_line_color="rgba(8,48,107,1.0)",
            marker_line_width=1,
            hovertemplate="<b>Volume:</b> %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Calculate summary stats
    total_trades = len(trades_df)
    winning_trades = (
        len(trades_df[trades_df["is_winning"]]) if len(trades_df) > 0 else 0
    )
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df["gross_pnl"].sum() if len(trades_df) > 0 else 0

    # Update layout
    fig.update_layout(
        title={
            "text": f"Trading Signals - {model_name}<br><sub>Total Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:.2f}</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        height=900,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Save the chart with scroll zoom enabled
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    # Enable scroll zoom in config
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [],
    }
    fig.write_html(output_path, config=config)

    logging.info(f"Trades candle chart exported to: {output_path}")
    return output_path


def export_trades_html(
    data, history, model_name, output_path="trades_report.html", initial_capital=None
):
    """Export trades to HTML table"""
    trades_df = get_complete_trades(history, data)

    if len(trades_df) == 0:
        logging.warning("No complete trades to export")
        return ""

    # Calculate summary statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["is_winning"]])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_return = trades_df["return_pct"].mean()
    total_gross_pnl = trades_df["gross_pnl"].sum()
    best_trade = trades_df["return_pct"].max()
    worst_trade = trades_df["return_pct"].min()
    avg_duration = trades_df["duration_days"].mean()

    # Format dates
    trades_display = trades_df.copy()
    trades_display["entry_date"] = pd.to_datetime(
        trades_display["entry_date"]
    ).dt.strftime("%Y-%m-%d")
    trades_display["exit_date"] = pd.to_datetime(
        trades_display["exit_date"]
    ).dt.strftime("%Y-%m-%d")

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} - Trade Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            color: #34495e;
            margin-top: 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .summary-item h3 {{
            margin: 0 0 5px 0;
            color: #2c3e50;
            font-size: 14px;
        }}
        .summary-item .value {{
            font-size: 20px;
            font-weight: bold;
            color: #34495e;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #34495e; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: bold;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .winning {{ color: #27ae60; }}
        .losing {{ color: #e74c3c; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{model_name} - Trade Report</h1>
        
        <div class="summary">
            <h2>Performance Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Total Trades</h3>
                    <div class="value neutral">{total_trades}</div>
                </div>
                <div class="summary-item">
                    <h3>Win Rate</h3>
                    <div class="value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
                </div>
                <div class="summary-item">
                    <h3>Average Return</h3>
                    <div class="value {'positive' if avg_return >= 0 else 'negative'}">{avg_return:.2f}%</div>
                </div>
                <div class="summary-item">
                    <h3>Total P&L</h3>
                    <div class="value {'positive' if total_gross_pnl >= 0 else 'negative'}">${total_gross_pnl:.2f}</div>
                </div>
                <div class="summary-item">
                    <h3>Best Trade</h3>
                    <div class="value positive">{best_trade:.2f}%</div>
                </div>
                <div class="summary-item">
                    <h3>Worst Trade</h3>
                    <div class="value negative">{worst_trade:.2f}%</div>
                </div>
                <div class="summary-item">
                    <h3>Average Duration</h3>
                    <div class="value neutral">{avg_duration:.1f} days</div>
                </div>
            </div>
        </div>
        
        <h2>Individual Trades</h2>
        <table>
            <thead>
                <tr>
                    <th>Trade ID</th>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Type</th>
                    <th>Gross P&L</th>
                    <th>Return %</th>
                    <th>Duration (Days)</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add trade rows
    for _, trade in trades_display.iterrows():
        result_class = "winning" if trade["is_winning"] else "losing"
        trade_type = trade.get("trade_type", "LONG")
        html_content += f"""
                <tr>
                    <td>{int(trade['trade_id'])}</td>
                    <td>{trade['entry_date']}</td>
                    <td>{trade['exit_date']}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td><strong>{trade_type}</strong></td>
                    <td class="{result_class}">${trade['gross_pnl']:.2f}</td>
                    <td class="{result_class}">{trade['return_pct']:.2f}%</td>
                    <td>{trade['duration_days']:.1f}</td>
                    <td class="{result_class}">{'WIN' if trade['is_winning'] else 'LOSS'}</td>
                </tr>
"""

    html_content += (
        """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by DQN Trading Bot Evaluation</p>
            <p>Report generated on """
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + """</p>
        </div>
    </div>
</body>
</html>
"""
    )

    # Write to file
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logging.info(f"Trades HTML exported to: {output_path}")
    return output_path


def export_quantstats_report(
    data,
    history,
    model_name,
    output_path="quantstats_report.html",
    initial_capital=None,
):
    """Export QuantStats HTML report with benchmark using underlying stock price

    Args:
        data: DataFrame with OHLCV data (Date as index)
        history: List of (price, action) tuples from trading history
        model_name: Name of the model
        output_path: Path where to save the HTML file
        initial_capital: Initial capital (defaults to first price if None)

    Returns:
        Path to the saved HTML file or empty string if failed
    """
    if not QUANTSTATS_AVAILABLE:
        logging.error("QuantStats not available. Install with: pip install quantstats")
        return ""

    try:
        # Calculate portfolio NAV history
        nav_history = calculate_portfolio_nav(data, history, initial_capital)
        dates = data.index.tolist()

        # Align NAV with dates (same logic as export_nav_chart)
        nav_to_plot = nav_history[1:] if len(nav_history) > 1 else nav_history
        if len(nav_to_plot) < len(dates):
            nav_to_plot = list(nav_to_plot) + [nav_to_plot[-1]] * (
                len(dates) - len(nav_to_plot)
            )
        elif len(nav_to_plot) > len(dates):
            nav_to_plot = nav_to_plot[: len(dates)]

        # Create portfolio returns series
        nav_series = pd.Series(nav_to_plot, index=dates, name="Strategy")
        portfolio_returns = nav_series.pct_change().dropna()

        # Create benchmark returns from underlying stock price (buy and hold)
        stock_prices = data["Adj Close"]
        benchmark_returns = stock_prices.pct_change().dropna()

        # Align dates between portfolio and benchmark
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]

        # Convert to timezone-naive if needed (QuantStats requirement)
        if portfolio_returns.index.tz is not None:
            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
        if benchmark_returns.index.tz is not None:
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Generate QuantStats HTML report
        qs.reports.html(
            portfolio_returns,
            benchmark=benchmark_returns,
            output=output_path,
            title=f"{model_name} - QuantStats Report",
        )

        logging.info(f"QuantStats report exported to: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Failed to export QuantStats report: {e}")
        import traceback

        logging.debug(traceback.format_exc())
        return ""


def main(
    eval_stock,
    window_size,
    model_name,
    debug,
    export_viz=False,
    start_date=None,
    end_date=None,
):
    """Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args:
        eval_stock: Path to CSV file with stock data
        window_size: Size of the n-day window
        model_name: Name of the pretrained model
        debug: Whether to use verbose logs
        export_viz: Whether to export visualization HTML files
        start_date: Start date for evaluation (YYYY-MM-DD format or None)
        end_date: End date for evaluation (YYYY-MM-DD format or None)
    """
    # Get full OHLCV data first (with date filtering if specified)
    ohlcv_data = get_ohlcv_data(eval_stock, start_date, end_date)

    # Extract Adj Close as list for evaluation (matching get_stock_data format)
    data = ohlcv_data["Adj Close"].tolist()

    if len(data) < 2:
        logging.error(
            f"Insufficient data after date filtering. Found {len(data)} rows."
        )
        logging.error(f"Date range: {start_date} to {end_date}")
        return

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

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        result = evaluate_model(agent, data, window_size, debug)
        # Handle both tuple and single value returns
        profit = result[0] if isinstance(result, tuple) else result
        history = result[1] if isinstance(result, tuple) and len(result) > 1 else []
        show_eval_result(model_name, profit, initial_offset)

        # Export visualizations if requested
        if export_viz and ohlcv_data is not None and history:
            # Create output directory with timestamp: evaluation_output/<model_name>/<timestamp>/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("evaluation_output") / model_name / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            initial_capital = ohlcv_data["Adj Close"].iloc[0]

            # Trim ohlcv_data to match history length (history has data_length entries)
            # history length = data_length, ohlcv_data has data_length + 1 rows
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

    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            model_path = os.path.join("models", model)
            # Check if it's a directory (SavedModel format) or file (old format)
            if os.path.isdir(model_path) or os.path.isfile(model_path):
                try:
                    agent = Agent(window_size, pretrained=True, model_name=model)
                    result = evaluate_model(agent, data, window_size, debug)
                    # Handle both tuple and single value returns
                    profit = result[0] if isinstance(result, tuple) else result
                    history = (
                        result[1]
                        if isinstance(result, tuple) and len(result) > 1
                        else []
                    )
                    show_eval_result(model, profit, initial_offset)

                    # Export visualizations if requested
                    if export_viz and ohlcv_data is not None and history:
                        # Create output directory with timestamp: evaluation_output/<model_name>/<timestamp>/
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = Path("evaluation_output") / model / timestamp
                        output_dir.mkdir(parents=True, exist_ok=True)

                        initial_capital = ohlcv_data["Adj Close"].iloc[0]

                        # Trim ohlcv_data to match history length
                        if len(history) < len(ohlcv_data):
                            ohlcv_data_trimmed = ohlcv_data.iloc[: len(history)].copy()
                        else:
                            ohlcv_data_trimmed = ohlcv_data.copy()

                        export_nav_chart(
                            ohlcv_data_trimmed,
                            history,
                            model,
                            str(output_dir / "nav_chart.html"),
                            initial_capital,
                        )
                        export_trades_candle_chart(
                            ohlcv_data_trimmed,
                            history,
                            model,
                            str(output_dir / "trades_chart.html"),
                        )
                        export_trades_html(
                            ohlcv_data_trimmed,
                            history,
                            model,
                            str(output_dir / "trades_report.html"),
                            initial_capital,
                        )
                        export_quantstats_report(
                            ohlcv_data_trimmed,
                            history,
                            model,
                            str(output_dir / "quantstats_report.html"),
                            initial_capital,
                        )

                    del agent
                except Exception as e:
                    logging.warning(f"Failed to evaluate {model}: {e}")
                    continue


if __name__ == "__main__":
    args = docopt(__doc__)

    eval_stock = args["<eval-stock>"]
    window_size = int(args["--window-size"])
    model_name = args["--model-name"]
    debug = args["--debug"]
    export_viz = args.get("--export-viz", False)
    start_date = args.get("--start-date")
    end_date = args.get("--end-date")

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device(use_gpu=True)  # Use GPU by default

    try:
        main(
            eval_stock, window_size, model_name, debug, export_viz, start_date, end_date
        )
    except KeyboardInterrupt:
        print("Aborted")
