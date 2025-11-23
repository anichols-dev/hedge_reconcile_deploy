#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Reconcile Analysis
Analyzes the timing between news releases and actual trade execution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import glob
warnings.filterwarnings('ignore')

# Import real_analysis module
import real_analysis as ra

# Page configuration
st.set_page_config(
    page_title="Reconcile Analysis Dashboard",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_open_trades_data():
    """Load open trades analysis data for reconciliation"""
    # Find the most recent open trades analysis file
    pattern = 'data/computed/clean/open_trades_analysis_*.csv'
    files = glob.glob(pattern)

    if not files:
        return None

    # Get the most recent file
    latest_file = max(files)

    df = pd.read_csv(latest_file)
    df['trade_time_utc'] = pd.to_datetime(df['trade_time_utc'])
    df['news_timestamp'] = pd.to_datetime(df['news_timestamp'])

    return df

@st.cache_data
def load_open_trades_data_v2():
    """Load open trades analysis data (version 2) for reconciliation"""
    # Find the most recent open trades analysis v2 file
    pattern = 'data/computed/clean/open_trades_analysis_v2_*.csv'
    files = glob.glob(pattern)

    if not files:
        return None

    # Get the most recent file
    latest_file = max(files)

    df = pd.read_csv(latest_file)
    df['trade_time_utc'] = pd.to_datetime(df['trade_time_utc'], utc=True)
    df['news_timestamp'] = pd.to_datetime(df['news_timestamp'], utc=True)

    return df

def create_response_time_histogram(df, max_seconds=None, log_scale=False):
    """Create histogram of response times with optional filtering"""

    data = df.copy()
    if max_seconds is not None:
        data = data[data['news_to_trade_seconds'] <= max_seconds]

    fig = px.histogram(
        data,
        x='news_to_trade_seconds',
        color='trade_action',
        nbins=50,
        title='Response Time Distribution (News to Trade Execution)',
        labels={'news_to_trade_seconds': 'Response Time (seconds)', 'count': 'Number of Trades'},
        color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545'},
        log_y=log_scale
    )

    fig.update_layout(
        height=400,
        xaxis_title="Response Time (seconds)",
        yaxis_title="Number of Trades" + (" (log scale)" if log_scale else ""),
        showlegend=True,
        hovermode='x unified'
    )

    return fig

def create_response_time_boxplot(df, max_seconds=None):
    """Create box plot of response times by trade direction"""

    data = df.copy()
    if max_seconds is not None:
        data = data[data['news_to_trade_seconds'] <= max_seconds]

    fig = px.box(
        data,
        x='trade_action',
        y='news_to_trade_seconds',
        color='trade_action',
        title='Response Time by Trade Direction',
        labels={'news_to_trade_seconds': 'Response Time (seconds)', 'trade_action': 'Trade Action'},
        color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545'}
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis_title="Response Time (seconds)",
        xaxis_title="Trade Direction"
    )

    return fig


def main():
    st.title("🔄 Reconcile Analysis Dashboard")
    st.markdown("Analyze news release timing and trading performance")

    # Create tabs for different reconcile analyses
    tab1, tab2 = st.tabs(["📊 Open Analysis", "📈 Real Analysis"])

    with tab1:
        st.header("Open Trades Timing Analysis")

        # Add version toggle
        analysis_version = st.radio(
            "Select Analysis Version",
            ["Version 1 (Original)", "Version 2 (Algo Performance)"],
            horizontal=True,
            help="Version 1: Original trade history analysis | Version 2: Algo performance data analysis"
        )

        # Load appropriate data based on selection
        if analysis_version == "Version 1 (Original)":
            open_trades_df = load_open_trades_data()
        else:
            open_trades_df = load_open_trades_data_v2()

        if open_trades_df is None or open_trades_df.empty:
            st.error(f"No open trades analysis data found for {analysis_version}. Please ensure the file exists in data/computed/clean/")
        else:
            # Sidebar filters
            with st.sidebar:
                st.header("🔄 Reconcile Filters")

                # Outlier filter
                outlier_option = st.selectbox(
                    "Outlier Handling",
                    ["Show All", "Exclude >1 hour", "Exclude >5 minutes", "Exclude >60 seconds"],
                    index=1
                )

                max_seconds = None
                if outlier_option == "Exclude >1 hour":
                    max_seconds = 3600
                elif outlier_option == "Exclude >5 minutes":
                    max_seconds = 300
                elif outlier_option == "Exclude >60 seconds":
                    max_seconds = 60

                # Trade direction filter
                selected_directions = st.multiselect(
                    "Trade Direction",
                    options=['BUY', 'SELL'],
                    default=['BUY', 'SELL']
                )

                # News source filter
                news_sources = open_trades_df['news_source'].unique()
                selected_news_sources = st.multiselect(
                    "News Sources",
                    options=news_sources,
                    default=news_sources
                )

                # Log scale option
                log_scale = st.checkbox("Use Log Scale", value=False)

            # Apply filters
            filtered_trades = open_trades_df[
                (open_trades_df['trade_action'].isin(selected_directions)) &
                (open_trades_df['news_source'].isin(selected_news_sources))
            ].copy()

            if max_seconds is not None:
                initial_count = len(filtered_trades)
                filtered_trades = filtered_trades[filtered_trades['news_to_trade_seconds'] <= max_seconds]
                excluded_count = initial_count - len(filtered_trades)

            # Overview metrics
            st.subheader("📈 Overview Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Trades", f"{len(filtered_trades):,}")

            with col2:
                median_time = filtered_trades['news_to_trade_seconds'].median()
                st.metric("Median Response", f"{median_time:.1f}s")

            with col3:
                fast_trades = len(filtered_trades[filtered_trades['news_to_trade_seconds'] <= 10])
                fast_pct = (fast_trades / len(filtered_trades) * 100) if len(filtered_trades) > 0 else 0
                st.metric("Fast Trades (≤10s)", f"{fast_trades} ({fast_pct:.1f}%)")

            with col4:
                if max_seconds and 'excluded_count' in locals():
                    st.metric("Excluded Outliers", f"{excluded_count}")
                else:
                    delayed = len(open_trades_df[open_trades_df['news_to_trade_seconds'] > 60])
                    st.metric("Delayed (>60s)", f"{delayed}")

            # Timing visualizations
            st.subheader("⏱️ Response Time Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig_hist = create_response_time_histogram(filtered_trades, max_seconds, log_scale)
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Box plot
                fig_box = create_response_time_boxplot(filtered_trades, max_seconds)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)

            # Statistics table
            st.subheader("📊 Response Time Statistics")

            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            stats_data = []

            for action in selected_directions:
                action_df = filtered_trades[filtered_trades['trade_action'] == action]
                if len(action_df) > 0:
                    row = {'Action': action, 'Count': len(action_df)}
                    for p in percentiles:
                        value = action_df['news_to_trade_seconds'].quantile(p/100)
                        if value < 60:
                            row[f'P{p}'] = f"{value:.1f}s"
                        elif value < 3600:
                            row[f'P{p}'] = f"{value/60:.1f}m"
                        else:
                            row[f'P{p}'] = f"{value/3600:.1f}h"
                    stats_data.append(row)

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

            # Outliers table
            if max_seconds:
                outliers = open_trades_df[open_trades_df['news_to_trade_seconds'] > max_seconds]
                if len(outliers) > 0:
                    st.subheader(f"🔍 Excluded Outliers (>{outlier_option.split('>')[1]})")

                    outlier_display = outliers[['trade_time_utc', 'trade_action', 'news_source',
                                               'news_to_trade_minutes', 'trade_pnl', 'news_text']].copy()
                    outlier_display = outlier_display.sort_values('news_to_trade_minutes', ascending=False).head(20)
                    outlier_display['news_text'] = outlier_display['news_text'].str[:100] + '...'

                    st.dataframe(outlier_display, use_container_width=True)

    with tab2:
        st.header("Algo Performance Real Analysis")

        # Load algo performance data
        try:
            algo_df = ra.load_algo_performance()

            # Sidebar filters for Real Analysis
            with st.sidebar:
                st.header("📈 Real Analysis Filters")

                # Exclude outliers checkbox
                exclude_outliers = st.checkbox("Exclude Outliers", value=True)

                # Side filter
                selected_sides = st.multiselect(
                    "Trade Side",
                    options=['Long', 'Short'],
                    default=['Long', 'Short'],
                    key="real_analysis_sides"
                )

                # Date range filter
                min_date = algo_df['Entry_Time'].min().date()
                max_date = algo_df['Entry_Time'].max().date()

                date_range_real = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="real_analysis_date_range"
                )

            # Apply filters
            filtered_algo = algo_df.copy()

            # Filter by side
            filtered_algo = filtered_algo[filtered_algo['Side'].isin(selected_sides)]

            # Filter by date range
            if len(date_range_real) == 2:
                filtered_algo = filtered_algo[
                    (filtered_algo['Entry_Time'].dt.date >= date_range_real[0]) &
                    (filtered_algo['Entry_Time'].dt.date <= date_range_real[1])
                ]

            # Calculate statistics for both all trades and filtered (excluding outliers)
            all_stats = ra.calculate_pnl_statistics(filtered_algo)

            if exclude_outliers:
                non_outlier_df = filtered_algo[filtered_algo['Outlier'] == False]
                display_df = non_outlier_df
            else:
                display_df = filtered_algo

            filtered_stats = ra.calculate_pnl_statistics(display_df)
            hold_stats = ra.calculate_hold_time_statistics(display_df)
            stop_loss_stats = ra.calculate_stop_loss_frequency(display_df)
            side_stats = ra.calculate_side_statistics(display_df)

            # Overview metrics
            st.subheader("📊 Key Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total PnL", f"${filtered_stats['total_pnl']:,.2f}",
                         delta=f"{filtered_stats['total_trades']} trades")

            with col2:
                st.metric("Win Rate", f"{filtered_stats['win_rate']:.1f}%",
                         delta=f"{filtered_stats['wins']}/{filtered_stats['total_trades']}")

            with col3:
                avg_hold_mins = hold_stats['avg_minutes']
                if avg_hold_mins < 60:
                    st.metric("Avg Hold Time", f"{avg_hold_mins:.1f} min")
                else:
                    st.metric("Avg Hold Time", f"{hold_stats['avg_hours']:.1f} hrs")

            with col4:
                st.metric("Stop Loss %", f"{stop_loss_stats['stop_loss_pct']:.1f}%",
                         delta=f"{stop_loss_stats['stop_loss_count']} trades")

            # Cumulative PnL Chart
            st.subheader("💰 Cumulative PnL Over Time")

            # Sort by entry time and calculate cumulative PnL
            pnl_df = display_df.sort_values('Entry_Time').copy()
            pnl_df['Cumulative_PnL'] = pnl_df['Raw_PnL'].cumsum()

            # Create cumulative PnL line chart
            fig_cumulative = go.Figure()

            # Add cumulative PnL line
            fig_cumulative.add_trace(go.Scatter(
                x=pnl_df['Entry_Time'],
                y=pnl_df['Cumulative_PnL'],
                mode='lines',
                name='Cumulative PnL',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='<b>%{x}</b><br>Cumulative PnL: $%{y:,.2f}<extra></extra>'
            ))

            # Add individual trade markers colored by side
            for side, color in [('Long', '#28a745'), ('Short', '#dc3545')]:
                side_df = pnl_df[pnl_df['Side'] == side]
                fig_cumulative.add_trace(go.Scatter(
                    x=side_df['Entry_Time'],
                    y=side_df['Cumulative_PnL'],
                    mode='markers',
                    name=f'{side} Trades',
                    marker=dict(
                        color=color,
                        size=6,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Trade PnL: $%{customdata[0]:,.2f}<br>' +
                                  'Cumulative: $%{y:,.2f}<br>' +
                                  'Return: %{customdata[1]:.2f}%<extra></extra>',
                    text=[f'{side} - {coin}' for coin in side_df['Coin']],
                    customdata=side_df[['Raw_PnL', 'Return_Pct']].values
                ))

            # Add zero line
            fig_cumulative.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )

            fig_cumulative.update_layout(
                title='Cumulative PnL Performance',
                xaxis_title='Entry Time',
                yaxis_title='Cumulative PnL ($)',
                height=500,
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_cumulative, use_container_width=True)

            # Visualizations
            st.subheader("📈 Hold Time Distribution")

            # Hold time histogram
            fig_hold = px.histogram(
                display_df,
                x='Duration_Minutes',
                color='Side',
                nbins=50,
                title='Hold Time Distribution by Side',
                labels={'Duration_Minutes': 'Hold Time (minutes)', 'count': 'Number of Trades'},
                color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'}
            )

            fig_hold.update_layout(
                height=400,
                xaxis_title="Hold Time (minutes)",
                yaxis_title="Number of Trades",
                showlegend=True,
                hovermode='x unified'
            )

            st.plotly_chart(fig_hold, use_container_width=True)

            # PnL Visualizations
            st.subheader("💰 PnL Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # PnL box plot by side
                fig_pnl_box = px.box(
                    display_df,
                    x='Side',
                    y='Raw_PnL',
                    color='Side',
                    title='PnL Distribution by Side',
                    labels={'Raw_PnL': 'PnL ($)', 'Side': 'Trade Side'},
                    color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'}
                )

                fig_pnl_box.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title="PnL ($)"
                )

                st.plotly_chart(fig_pnl_box, use_container_width=True)

            with col2:
                # PnL over time
                fig_pnl_time = px.scatter(
                    display_df,
                    x='Entry_Time',
                    y='Raw_PnL',
                    color='Side',
                    title='PnL Over Time',
                    labels={'Entry_Time': 'Entry Time', 'Raw_PnL': 'PnL ($)'},
                    color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'},
                    hover_data=['Coin', 'Duration', 'Return_Pct']
                )

                fig_pnl_time.update_layout(
                    height=400,
                    yaxis_title="PnL ($)",
                    xaxis_title="Entry Time"
                )

                st.plotly_chart(fig_pnl_time, use_container_width=True)

            # Outliers Section
            st.subheader("⚠️ Outlier Trades")

            outlier_df = ra.get_outliers_table(filtered_algo)

            if not outlier_df.empty:
                st.write(f"**Found {len(outlier_df)} outlier trades** (sorted by absolute PnL impact)")
                st.dataframe(outlier_df, use_container_width=True)
            else:
                st.info("No outlier trades found in the filtered data")

            # Summary Statistics Comparison
            st.subheader("📋 Summary Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Overall Statistics Comparison**")
                comparison_df = ra.create_summary_comparison(all_stats, filtered_stats)
                st.dataframe(comparison_df, use_container_width=True)

            with col2:
                st.write("**Statistics by Side**")
                side_stats_list = []
                for side in selected_sides:
                    if side_stats.get(side):
                        stats = side_stats[side]
                        side_stats_list.append({
                            'Side': side,
                            'Trades': stats['total_trades'],
                            'Win Rate': f"{stats['win_rate']:.1f}%",
                            'Total PnL': f"${stats['total_pnl']:,.2f}",
                            'Avg PnL': f"${stats['avg_pnl']:,.2f}",
                            'Profit Factor': f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else '∞'
                        })

                if side_stats_list:
                    side_stats_df = pd.DataFrame(side_stats_list)
                    st.dataframe(side_stats_df, use_container_width=True)

            # Additional Hold Time Stats
            st.subheader("⏱️ Hold Time Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average", f"{hold_stats['avg_minutes']:.1f} min")
                st.metric("Median", f"{hold_stats['median_minutes']:.1f} min")

            with col2:
                st.metric("Minimum", f"{hold_stats['min_seconds']:.0f} sec")
                st.metric("Maximum", ra.format_duration(hold_stats['max_seconds']))

            with col3:
                st.metric("Std Dev", f"{hold_stats['std_seconds']/60:.1f} min")

            # Stop Loss Details
            st.subheader("🛑 Stop Loss Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Stop Losses", stop_loss_stats['stop_loss_count'])

            with col2:
                st.metric("Avg Stop Loss Return", f"{stop_loss_stats['avg_stop_loss_pct']:.2f}%")

            with col3:
                st.metric("Avg Stop Loss PnL", f"${stop_loss_stats['avg_stop_loss_pnl']:,.2f}")

        except FileNotFoundError:
            st.error("Algo performance data not found. Please ensure 'data/algo_performance_report.csv' exists.")
        except Exception as e:
            st.error(f"Error loading algo performance data: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
