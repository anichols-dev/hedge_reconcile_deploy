"""
Real Analysis Module for Algo Performance
Provides utility functions for analyzing trading performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
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
    df['Return_Pct'] = pd.to_numeric(df['Return_Pct'], errors='coerce')

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
            f"${filtered_stats['avg_pnl']:,.2f}",
            f"${filtered_stats['median_pnl']:,.2f}",
            f"{filtered_stats['win_rate']:.2f}%",
            f"${filtered_stats['avg_win']:,.2f}",
            f"${filtered_stats['avg_loss']:,.2f}",
            f"{filtered_stats['profit_factor']:.2f}" if filtered_stats['profit_factor'] != float('inf') else '∞'
        ]
    })

    return comparison
