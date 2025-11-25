"""
Real Analysis Module for Algo Performance
Provides utility functions for analyzing trading performance metrics.
"""

import pandas as pd
import numpy as np
import streamlit as st


def parse_duration_to_seconds(duration_str):
    """
    Convert duration string (H:MM:SS) to total seconds.

    Args:
        duration_str: String in format "H:MM:SS" or "HH:MM:SS"

    Returns:
        Total seconds as integer
    """
    try:
        parts = duration_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except:
        return 0


@st.cache_data
def load_algo_performance():
    """
    Load and preprocess algo performance data.

    Columns used:
        - Entry_Time, Exit_Time: timestamps from CSV
        - Entry_Price, Exit_Price: prices from CSV
        - Return_Pct: CALCULATED from (Exit_Price - Entry_Price) / Entry_Price * 100
          (adjusted for Long/Short direction)

    Returns:
        DataFrame with parsed timestamps and duration in seconds
    """
    df = pd.read_csv('data/algo_performance_report.csv')

    # Parse timestamps (ISO8601 format)
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='ISO8601')
    df['Exit_Time'] = pd.to_datetime(df['Exit_Time'], format='ISO8601')

    # Convert duration to seconds
    df['Duration_Seconds'] = df['Duration'].apply(parse_duration_to_seconds)
    df['Duration_Minutes'] = df['Duration_Seconds'] / 60
    df['Duration_Hours'] = df['Duration_Seconds'] / 3600

    # Convert Outlier to boolean
    df['Outlier'] = df['Outlier'].astype(str).str.upper() == 'TRUE'

    # Ensure numeric columns
    df['Raw_PnL'] = pd.to_numeric(df['Raw_PnL'], errors='coerce')
    df['Entry_Price'] = pd.to_numeric(df['Entry_Price'], errors='coerce')
    df['Exit_Price'] = pd.to_numeric(df['Exit_Price'], errors='coerce')

    # Calculate Return_Pct from entry/exit prices (not from CSV column)
    df['Return_Pct'] = np.where(
        df['Side'] == 'Long',
        (df['Exit_Price'] - df['Entry_Price']) / df['Entry_Price'] * 100,
        (df['Entry_Price'] - df['Exit_Price']) / df['Entry_Price'] * 100
    )

    return df


@st.cache_data
def load_algo_performance_fixed():
    """
    Load and preprocess algo performance data with fixed exits.

    Columns used:
        - Entry_Time: from CSV
        - Exit_Time: from fixed_exit_time
        - Entry_Price: from CSV
        - Exit_Price: from fixed_exit_price
        - Return_Pct: CALCULATED from (fixed_exit_price - Entry_Price) / Entry_Price * 100
          (adjusted for Long/Short direction)

    Returns:
        DataFrame with fixed exit data mapped to standard column names
    """
    df = pd.read_csv('data/algo_performance_fixed_exits.csv')

    # Parse timestamps (ISO8601 format)
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='ISO8601')

    # Map fixed columns to standard column names
    df['Exit_Time'] = pd.to_datetime(df['fixed_exit_time'], format='ISO8601')
    df['Exit_Price'] = df['fixed_exit_price']
    df['Raw_PnL'] = df['fixed_pnl']

    # Calculate Return_Pct from entry/exit prices (not from CSV column)
    df['Return_Pct'] = np.where(
        df['Side'] == 'Long',
        (df['Exit_Price'] - df['Entry_Price']) / df['Entry_Price'] * 100,
        (df['Entry_Price'] - df['Exit_Price']) / df['Entry_Price'] * 100
    )

    # Calculate duration from fixed exit times (not original Duration column)
    df['Duration_Seconds'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds()
    df['Duration_Minutes'] = df['Duration_Seconds'] / 60
    df['Duration_Hours'] = df['Duration_Seconds'] / 3600

    # Convert Outlier to boolean
    df['Outlier'] = df['Outlier'].astype(str).str.upper() == 'TRUE'

    return df


@st.cache_data
def load_algo_performance_rd2():
    """
    Load and preprocess algo performance RD2 data with fixed exits.

    Columns used:
        - Entry_Time: from CSV
        - Exit_Time: from fixed_exit_time
        - Entry_Price: from CSV
        - Exit_Price: from fixed_exit_price
        - Return_Pct: CALCULATED from (fixed_exit_price - Entry_Price) / Entry_Price * 100
          (adjusted for Long/Short direction)

    Returns:
        DataFrame with RD2 fixed exit data mapped to standard column names
    """
    df = pd.read_csv('data/algo_performance_rd2_fixed_exits.csv')

    # Parse entry timestamp
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], utc=True)

    # Map fixed columns to standard column names
    df['Exit_Time'] = pd.to_datetime(df['fixed_exit_time'], format='mixed', utc=True)
    df['Exit_Price'] = df['fixed_exit_price']
    df['Raw_PnL'] = df['fixed_pnl']

    # Calculate Return_Pct from entry/exit prices (not from CSV column)
    df['Return_Pct'] = np.where(
        df['Side'] == 'Long',
        (df['Exit_Price'] - df['Entry_Price']) / df['Entry_Price'] * 100,
        (df['Entry_Price'] - df['Exit_Price']) / df['Entry_Price'] * 100
    )

    # Add missing Coin column
    df['Coin'] = 'SOL'

    # Calculate duration from fixed exit times
    df['Duration_Seconds'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds()
    df['Duration_Minutes'] = df['Duration_Seconds'] / 60
    df['Duration_Hours'] = df['Duration_Seconds'] / 3600

    # Convert Outlier to boolean
    df['Outlier'] = df['Outlier'].astype(str).str.upper() == 'TRUE'

    return df


def calculate_pnl_statistics(df):
    """
    Calculate comprehensive PnL statistics.

    Args:
        df: DataFrame with trading data

    Returns:
        Dictionary of PnL metrics
    """
    stats = {
        'total_trades': len(df),
        'total_pnl': df['Raw_PnL'].sum(),
        'sum_returns': df['Return_Pct'].sum() if 'Return_Pct' in df.columns else 0,
        'avg_pnl': df['Raw_PnL'].mean(),
        'median_pnl': df['Raw_PnL'].median(),
        'std_pnl': df['Raw_PnL'].std(),
        'max_win': df['Raw_PnL'].max(),
        'max_loss': df['Raw_PnL'].min(),
        'wins': len(df[df['Raw_PnL'] > 0]),
        'losses': len(df[df['Raw_PnL'] <= 0]),
        'win_rate': (len(df[df['Raw_PnL'] > 0]) / len(df) * 100) if len(df) > 0 else 0,
        'avg_win': df[df['Raw_PnL'] > 0]['Raw_PnL'].mean() if len(df[df['Raw_PnL'] > 0]) > 0 else 0,
        'avg_loss': df[df['Raw_PnL'] <= 0]['Raw_PnL'].mean() if len(df[df['Raw_PnL'] <= 0]) > 0 else 0,
        'profit_factor': abs(df[df['Raw_PnL'] > 0]['Raw_PnL'].sum() / df[df['Raw_PnL'] <= 0]['Raw_PnL'].sum()) if df[df['Raw_PnL'] <= 0]['Raw_PnL'].sum() != 0 else float('inf')
    }
    return stats


def calculate_side_statistics(df):
    """
    Calculate statistics by trade side (Long/Short).

    Args:
        df: DataFrame with trading data

    Returns:
        Dictionary with Long and Short statistics
    """
    sides = {}
    for side in ['Long', 'Short']:
        side_df = df[df['Side'] == side]
        if len(side_df) > 0:
            sides[side] = calculate_pnl_statistics(side_df)
        else:
            sides[side] = None
    return sides


def calculate_hold_time_statistics(df):
    """
    Calculate hold time statistics.

    Args:
        df: DataFrame with trading data

    Returns:
        Dictionary of hold time metrics
    """
    stats = {
        'avg_seconds': df['Duration_Seconds'].mean(),
        'median_seconds': df['Duration_Seconds'].median(),
        'avg_minutes': df['Duration_Minutes'].mean(),
        'median_minutes': df['Duration_Minutes'].median(),
        'avg_hours': df['Duration_Hours'].mean(),
        'median_hours': df['Duration_Hours'].median(),
        'min_seconds': df['Duration_Seconds'].min(),
        'max_seconds': df['Duration_Seconds'].max(),
        'std_seconds': df['Duration_Seconds'].std()
    }
    return stats


def calculate_stop_loss_frequency(df):
    """
    Calculate stop loss frequency (negative returns).

    Args:
        df: DataFrame with trading data

    Returns:
        Dictionary with stop loss metrics
    """
    stop_losses = len(df[df['Return_Pct'] < 0])
    total = len(df)

    stats = {
        'stop_loss_count': stop_losses,
        'total_trades': total,
        'stop_loss_pct': (stop_losses / total * 100) if total > 0 else 0,
        'avg_stop_loss_pct': df[df['Return_Pct'] < 0]['Return_Pct'].mean() if stop_losses > 0 else 0,
        'avg_stop_loss_pnl': df[df['Return_Pct'] < 0]['Raw_PnL'].mean() if stop_losses > 0 else 0
    }
    return stats


def get_outliers_table(df):
    """
    Get outlier trades formatted for display.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with outlier trades, sorted by PnL
    """
    outliers = df[df['Outlier'] == True].copy()

    if len(outliers) == 0:
        return pd.DataFrame()

    # Select and rename columns for display
    display_df = outliers[[
        'Entry_Time',
        'Coin',
        'Side',
        'Duration',
        'Duration_Hours',
        'Raw_PnL',
        'Return_Pct',
        'Entry_Price',
        'Exit_Price'
    ]].copy()

    # Sort by absolute PnL (largest impact first)
    display_df['Abs_PnL'] = display_df['Raw_PnL'].abs()
    display_df = display_df.sort_values('Abs_PnL', ascending=False)
    display_df = display_df.drop('Abs_PnL', axis=1)

    # Rename columns for better display
    display_df.columns = [
        'Entry Time',
        'Coin',
        'Side',
        'Hold Time',
        'Hold Hours',
        'PnL ($)',
        'Return %',
        'Entry Price',
        'Exit Price'
    ]

    return display_df


def format_duration(seconds):
    """
    Format seconds into H:MM:SS string.

    Args:
        seconds: Total seconds

    Returns:
        Formatted string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def create_summary_comparison(all_stats, filtered_stats):
    """
    Create a comparison table of all trades vs filtered trades.

    Args:
        all_stats: Statistics for all trades
        filtered_stats: Statistics for filtered trades (e.g., excluding outliers)

    Returns:
        DataFrame for display
    """
    comparison = pd.DataFrame({
        'Metric': [
            'Total Trades',
            'Total PnL',
            'Sum of Returns',
            'Average PnL',
            'Median PnL',
            'Win Rate %',
            'Average Win',
            'Average Loss',
            'Profit Factor'
        ],
        'All Trades': [
            f"{all_stats['total_trades']:,}",
            f"${all_stats['total_pnl']:,.2f}",
            f"{all_stats['sum_returns']:.2f}%",
            f"${all_stats['avg_pnl']:,.2f}",
            f"${all_stats['median_pnl']:,.2f}",
            f"{all_stats['win_rate']:.2f}%",
            f"${all_stats['avg_win']:,.2f}",
            f"${all_stats['avg_loss']:,.2f}",
            f"{all_stats['profit_factor']:.2f}" if all_stats['profit_factor'] != float('inf') else '∞'
        ],
        'Filtered': [
            f"{filtered_stats['total_trades']:,}",
            f"${filtered_stats['total_pnl']:,.2f}",
            f"{filtered_stats['sum_returns']:.2f}%",
            f"${filtered_stats['avg_pnl']:,.2f}",
            f"${filtered_stats['median_pnl']:,.2f}",
            f"{filtered_stats['win_rate']:.2f}%",
            f"${filtered_stats['avg_win']:,.2f}",
            f"${filtered_stats['avg_loss']:,.2f}",
            f"{filtered_stats['profit_factor']:.2f}" if filtered_stats['profit_factor'] != float('inf') else '∞'
        ]
    })

    return comparison


@st.cache_data
def load_backtest_data(offset: str = "5s"):
    """
    Load and preprocess backtest data.

    Args:
        offset: "0s" for no offset or "5s" for 5-second offset

    Returns:
        DataFrame with parsed timestamps, calculated PnL, and duration
    """
    df = pd.read_csv('data/backtest_on_filtered_news_offset5s_sl0.33pct.csv')

    # Select columns based on offset
    if offset == "5s":
        entry_price_col = 'entry_price_post_sentiment_5'
        exit_price_col = 'exit_price_post_5'
        exit_time_col = 'exit_timestamp_post_5'
        exit_reason_col = 'exit_reason_post_5'
    else:  # 0s offset
        entry_price_col = 'entry_price_post_sentiment'
        exit_price_col = 'exit_price_post'
        exit_time_col = 'exit_timestamp_post'
        exit_reason_col = 'exit_reason_post'

    # Parse timestamps (both as UTC for consistency)
    df['Entry_Time'] = pd.to_datetime(df['timestamp_post_sentiment'], format='mixed', utc=True)
    df['Exit_Time'] = pd.to_datetime(df[exit_time_col], format='mixed', utc=True)

    # Map prices
    df['Entry_Price'] = df[entry_price_col]
    df['Exit_Price'] = df[exit_price_col]

    # Map side: LONG/SHORT -> Long/Short
    df['Side'] = df['action'].map({'LONG': 'Long', 'SHORT': 'Short'})

    # Add Coin
    df['Coin'] = 'SOL'

    # Calculate return percentage
    df['Return_Pct'] = np.where(
        df['Side'] == 'Long',
        (df['Exit_Price'] - df['Entry_Price']) / df['Entry_Price'] * 100,
        (df['Entry_Price'] - df['Exit_Price']) / df['Entry_Price'] * 100
    )

    # Calculate duration from entry to exit
    df['Duration_Seconds'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds()
    df['Duration_Minutes'] = df['Duration_Seconds'] / 60
    df['Duration_Hours'] = df['Duration_Seconds'] / 3600
    df['Duration'] = df['Duration_Seconds'].apply(format_duration)

    # Keep exit reason for stop loss analysis
    df['Exit_Reason'] = df[exit_reason_col]

    return df


def calculate_stop_loss_frequency_backtest(df):
    """
    Calculate stop loss frequency using exit_reason label.

    Args:
        df: DataFrame with backtest trading data

    Returns:
        Dictionary with stop loss metrics
    """
    stop_losses = len(df[df['Exit_Reason'] == 'stop_loss'])
    total = len(df)
    stop_loss_df = df[df['Exit_Reason'] == 'stop_loss']

    return {
        'stop_loss_count': stop_losses,
        'total_trades': total,
        'stop_loss_pct': (stop_losses / total * 100) if total > 0 else 0,
        'avg_stop_loss_return': stop_loss_df['Return_Pct'].mean() if stop_losses > 0 else 0
    }


def calculate_return_statistics(df):
    """
    Calculate return-based statistics for backtest data.

    Args:
        df: DataFrame with trading data containing Return_Pct

    Returns:
        Dictionary of return metrics
    """
    wins = df[df['Return_Pct'] > 0]
    losses = df[df['Return_Pct'] <= 0]

    return {
        'total_trades': len(df),
        'sum_returns': df['Return_Pct'].sum(),
        'avg_return': df['Return_Pct'].mean(),
        'median_return': df['Return_Pct'].median(),
        'std_return': df['Return_Pct'].std(),
        'max_win': df['Return_Pct'].max(),
        'max_loss': df['Return_Pct'].min(),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': (len(wins) / len(df) * 100) if len(df) > 0 else 0,
        'avg_win': wins['Return_Pct'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['Return_Pct'].mean() if len(losses) > 0 else 0,
        'profit_factor': abs(wins['Return_Pct'].sum() / losses['Return_Pct'].sum()) if losses['Return_Pct'].sum() != 0 else float('inf')
    }


def calculate_return_statistics_by_side(df):
    """
    Calculate return statistics by trade side (Long/Short).

    Args:
        df: DataFrame with trading data

    Returns:
        Dictionary with Long and Short statistics
    """
    sides = {}
    for side in ['Long', 'Short']:
        side_df = df[df['Side'] == side]
        if len(side_df) > 0:
            sides[side] = calculate_return_statistics(side_df)
        else:
            sides[side] = None
    return sides


# =============================================================================
# RECONCILIATION FUNCTIONS - Live vs Backtest Comparison
# =============================================================================

@st.cache_data
def load_reconciliation_data():
    """
    Load and prepare both live and backtest datasets for reconciliation.

    Returns:
        Tuple of (live_df, backtest_df) with normalized columns for matching
    """
    # Load live data with fixed exits
    live_df = pd.read_csv('data/algo_performance_fixed_exits.csv')

    # Parse live timestamps
    live_df['Entry_Time'] = pd.to_datetime(live_df['Entry_Time'], format='ISO8601')
    live_df['Exit_Time'] = pd.to_datetime(live_df['fixed_exit_time'], format='ISO8601')
    live_df['Exit_Price'] = live_df['fixed_exit_price']
    # Map 'original' to 'hold_time' for comparison (original means held until timeout)
    live_df['Exit_Reason'] = live_df['fixed_exit_reason'].replace({'original': 'hold_time'})
    live_df['Return_Pct'] = live_df['fixed_pnl_pct']

    # Create match key (rounded to minute)
    live_df['match_key'] = live_df['Entry_Time'].dt.floor('min')

    # Load backtest data (0s offset columns - no suffix)
    backtest_df = pd.read_csv('data/backtest_on_filtered_news_offset5s_sl0.33pct.csv')

    # Parse backtest timestamps
    backtest_df['Entry_Time'] = pd.to_datetime(
        backtest_df['timestamp_post_sentiment'], format='mixed', utc=True
    )
    backtest_df['Entry_Price'] = backtest_df['entry_price_post_sentiment']
    backtest_df['Exit_Price'] = backtest_df['exit_price_post']
    backtest_df['Exit_Time'] = pd.to_datetime(
        backtest_df['exit_timestamp_post'], format='mixed', utc=True
    )
    backtest_df['Exit_Reason'] = backtest_df['exit_reason_post']

    # Normalize side format: LONG/SHORT -> Long/Short
    backtest_df['Side'] = backtest_df['action'].map({'LONG': 'Long', 'SHORT': 'Short'})

    # Calculate backtest return percentage
    backtest_df['Return_Pct'] = np.where(
        backtest_df['Side'] == 'Long',
        (backtest_df['Exit_Price'] - backtest_df['Entry_Price']) / backtest_df['Entry_Price'] * 100,
        (backtest_df['Entry_Price'] - backtest_df['Exit_Price']) / backtest_df['Entry_Price'] * 100
    )

    # Create match key (rounded to minute, remove timezone for matching)
    backtest_df['match_key'] = backtest_df['Entry_Time'].dt.tz_localize(None).dt.floor('min')

    return live_df, backtest_df


def aggregate_live_trades(live_df):
    """
    Aggregate multiple live fills into one record per news event.
    Uses size-weighted average for prices.

    Args:
        live_df: DataFrame with live trade data

    Returns:
        DataFrame with aggregated trades (one per match_key + Side)
    """
    # Group by match_key and Side
    agg_df = live_df.groupby(['match_key', 'Side']).agg({
        'Entry_Time': 'first',  # Use first entry time
        'Entry_Price': lambda x: np.average(x, weights=live_df.loc[x.index, 'Size_Matched']),
        'Exit_Price': lambda x: np.average(x, weights=live_df.loc[x.index, 'Size_Matched']),
        'Exit_Time': 'last',  # Use last exit time
        'Exit_Reason': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common
        'Return_Pct': lambda x: np.average(x, weights=live_df.loc[x.index, 'Size_Matched']),
        'Size_Matched': 'sum',  # Total size
        'fixed_pnl': 'sum',  # Total PnL
    }).reset_index()

    agg_df['num_fills'] = live_df.groupby(['match_key', 'Side']).size().values

    return agg_df


def match_trades(live_agg, backtest_df):
    """
    Match live trades to backtest trades by timestamp + direction.

    Args:
        live_agg: Aggregated live trades DataFrame
        backtest_df: Backtest trades DataFrame

    Returns:
        Tuple of (matched_df, live_only_df, backtest_only_df)
    """
    # Perform outer merge
    merged = pd.merge(
        live_agg,
        backtest_df[['match_key', 'Side', 'Entry_Time', 'Entry_Price', 'Exit_Price',
                     'Exit_Time', 'Exit_Reason', 'Return_Pct', 'post_text', 'source']],
        on=['match_key', 'Side'],
        how='outer',
        suffixes=('_live', '_bt')
    )

    # Identify matched, live-only, and backtest-only
    matched_df = merged[merged['Entry_Time_live'].notna() & merged['Entry_Time_bt'].notna()].copy()
    live_only_df = merged[merged['Entry_Time_live'].notna() & merged['Entry_Time_bt'].isna()].copy()
    backtest_only_df = merged[merged['Entry_Time_live'].isna() & merged['Entry_Time_bt'].notna()].copy()

    return matched_df, live_only_df, backtest_only_df


def calculate_deviations(matched_df):
    """
    Calculate deviation metrics for matched trades.

    Args:
        matched_df: DataFrame with matched live and backtest trades

    Returns:
        DataFrame with deviation columns added
    """
    df = matched_df.copy()

    # Entry slippage (in basis points)
    df['entry_slippage_pct'] = (
        (df['Entry_Price_live'] - df['Entry_Price_bt']) / df['Entry_Price_bt'] * 100
    )
    df['entry_slippage_bps'] = df['entry_slippage_pct'] * 100

    # Exit slippage (in basis points)
    df['exit_slippage_pct'] = (
        (df['Exit_Price_live'] - df['Exit_Price_bt']) / df['Exit_Price_bt'] * 100
    )
    df['exit_slippage_bps'] = df['exit_slippage_pct'] * 100

    # Return deviation
    df['return_deviation'] = df['Return_Pct_live'] - df['Return_Pct_bt']

    # Exit reason mismatch
    df['exit_reason_mismatch'] = df['Exit_Reason_live'] != df['Exit_Reason_bt']

    # Fill latency (seconds between backtest timestamp and live entry)
    df['fill_latency_seconds'] = (
        df['Entry_Time_live'] - df['Entry_Time_bt'].dt.tz_localize(None)
    ).dt.total_seconds()

    # Absolute return deviation for sorting
    df['abs_return_deviation'] = df['return_deviation'].abs()

    return df


def get_reconciliation_summary(matched_df, live_only_df, backtest_only_df):
    """
    Calculate summary statistics for reconciliation.

    Args:
        matched_df: DataFrame with matched trades
        live_only_df: DataFrame with live-only trades
        backtest_only_df: DataFrame with backtest-only trades

    Returns:
        Dictionary with summary metrics
    """
    total_live = len(matched_df) + len(live_only_df)
    total_backtest = len(matched_df) + len(backtest_only_df)

    summary = {
        'total_matched': len(matched_df),
        'total_live_only': len(live_only_df),
        'total_backtest_only': len(backtest_only_df),
        'match_rate_live': (len(matched_df) / total_live * 100) if total_live > 0 else 0,
        'match_rate_backtest': (len(matched_df) / total_backtest * 100) if total_backtest > 0 else 0,
    }

    if len(matched_df) > 0:
        summary.update({
            'avg_entry_slippage_bps': matched_df['entry_slippage_bps'].mean(),
            'avg_exit_slippage_bps': matched_df['exit_slippage_bps'].mean(),
            'avg_return_deviation': matched_df['return_deviation'].mean(),
            'median_return_deviation': matched_df['return_deviation'].median(),
            'exit_reason_mismatch_count': matched_df['exit_reason_mismatch'].sum(),
            'exit_reason_mismatch_pct': (matched_df['exit_reason_mismatch'].sum() / len(matched_df) * 100),
            'avg_fill_latency_seconds': matched_df['fill_latency_seconds'].mean(),
        })
    else:
        summary.update({
            'avg_entry_slippage_bps': 0,
            'avg_exit_slippage_bps': 0,
            'avg_return_deviation': 0,
            'median_return_deviation': 0,
            'exit_reason_mismatch_count': 0,
            'exit_reason_mismatch_pct': 0,
            'avg_fill_latency_seconds': 0,
        })

    return summary


def get_exit_reason_crosstab(matched_df):
    """
    Create crosstab of backtest exit reason vs live exit reason.

    Args:
        matched_df: DataFrame with matched trades

    Returns:
        Crosstab DataFrame
    """
    if len(matched_df) == 0:
        return pd.DataFrame()

    crosstab = pd.crosstab(
        matched_df['Exit_Reason_bt'],
        matched_df['Exit_Reason_live'],
        margins=True,
        margins_name='Total'
    )

    return crosstab


def get_significant_deviations(matched_df, return_threshold=0.1, entry_slip_threshold=0.05):
    """
    Get trades with significant deviations.

    Args:
        matched_df: DataFrame with deviation metrics
        return_threshold: Minimum absolute return deviation (%)
        entry_slip_threshold: Minimum absolute entry slippage (%)

    Returns:
        DataFrame with significant deviations, sorted by impact
    """
    if len(matched_df) == 0:
        return pd.DataFrame()

    # Filter for significant deviations
    significant = matched_df[
        (matched_df['abs_return_deviation'] > return_threshold) |
        (matched_df['entry_slippage_pct'].abs() > entry_slip_threshold) |
        (matched_df['exit_reason_mismatch'] == True)
    ].copy()

    # Sort by absolute return deviation
    significant = significant.sort_values('abs_return_deviation', ascending=False)

    return significant
