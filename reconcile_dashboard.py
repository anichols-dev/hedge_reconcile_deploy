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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Open Analysis", "📈 Real Analysis", "🧪 Backtest Analysis", "🔍 Reconciliation"])

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

        # Add data version toggle
        data_version = st.radio(
            "Select Data Version",
            ["Original Data", "Fixed Exits Data", "RD2 Data"],
            horizontal=True,
            help="Original: Standard exits | Fixed Exits: Corrected exit prices/times | RD2: Round 2 data"
        )

        # Load algo performance data
        try:
            if data_version == "Original Data":
                algo_df = ra.load_algo_performance()
            elif data_version == "Fixed Exits Data":
                algo_df = ra.load_algo_performance_fixed()
            else:
                algo_df = ra.load_algo_performance_rd2()

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

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total PnL", f"${filtered_stats['total_pnl']:,.2f}",
                         delta=f"{filtered_stats['total_trades']} trades")

            with col2:
                st.metric("Sum of Returns", f"{filtered_stats['sum_returns']:.2f}%")

            with col3:
                st.metric("Win Rate", f"{filtered_stats['win_rate']:.1f}%",
                         delta=f"{filtered_stats['wins']}/{filtered_stats['total_trades']}")

            with col4:
                avg_hold_mins = hold_stats['avg_minutes']
                if avg_hold_mins < 60:
                    st.metric("Avg Hold Time", f"{avg_hold_mins:.1f} min")
                else:
                    st.metric("Avg Hold Time", f"{hold_stats['avg_hours']:.1f} hrs")

            with col5:
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

            # Cumulative Returns Chart
            st.subheader("📈 Cumulative Returns Over Time")

            # Sort by entry time and calculate cumulative returns
            returns_df = display_df.sort_values('Entry_Time').copy()
            returns_df['Cumulative_Return'] = returns_df['Return_Pct'].cumsum()

            # Create cumulative returns line chart
            fig_cumulative_returns = go.Figure()

            # Add cumulative returns line
            fig_cumulative_returns.add_trace(go.Scatter(
                x=returns_df['Entry_Time'],
                y=returns_df['Cumulative_Return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='<b>%{x}</b><br>Cumulative Return: %{y:.2f}%<extra></extra>'
            ))

            # Add individual trade markers colored by side
            for side, color in [('Long', '#28a745'), ('Short', '#dc3545')]:
                side_df = returns_df[returns_df['Side'] == side]
                fig_cumulative_returns.add_trace(go.Scatter(
                    x=side_df['Entry_Time'],
                    y=side_df['Cumulative_Return'],
                    mode='markers',
                    name=f'{side} Trades',
                    marker=dict(
                        color=color,
                        size=6,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Trade Return: %{customdata[0]:.2f}%<br>' +
                                  'Cumulative: %{y:.2f}%<extra></extra>',
                    text=[f'{side} - {coin}' for coin in side_df['Coin']],
                    customdata=side_df[['Return_Pct']].values
                ))

            # Add zero line
            fig_cumulative_returns.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )

            fig_cumulative_returns.update_layout(
                title='Cumulative Return Performance',
                xaxis_title='Entry Time',
                yaxis_title='Cumulative Return (%)',
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

            st.plotly_chart(fig_cumulative_returns, use_container_width=True)

            # Visualizations
            st.subheader("📊 Hold Time Distribution")

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
                            'Sum Returns': f"{stats['sum_returns']:.2f}%",
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

    with tab3:
        st.header("Backtest Analysis")

        # Add offset toggle
        offset_selection = st.radio(
            "Entry Offset",
            ["0s Offset", "5s Offset"],
            horizontal=True,
            help="0s: Entry at sentiment timestamp | 5s: Entry 5 seconds after sentiment"
        )
        offset = "0s" if offset_selection == "0s Offset" else "5s"

        # Load backtest data
        try:
            backtest_df = ra.load_backtest_data(offset=offset)

            # Sidebar filters for Backtest Analysis
            with st.sidebar:
                st.header("🧪 Backtest Filters")

                # Side filter
                selected_sides_bt = st.multiselect(
                    "Trade Side",
                    options=['Long', 'Short'],
                    default=['Long', 'Short'],
                    key="backtest_sides"
                )

                # Date range filter
                min_date_bt = backtest_df['Entry_Time'].min().date()
                max_date_bt = backtest_df['Entry_Time'].max().date()

                date_range_bt = st.date_input(
                    "Date Range",
                    value=(min_date_bt, max_date_bt),
                    min_value=min_date_bt,
                    max_value=max_date_bt,
                    key="backtest_date_range"
                )

            # Apply filters
            filtered_bt = backtest_df.copy()

            # Filter by side
            filtered_bt = filtered_bt[filtered_bt['Side'].isin(selected_sides_bt)]

            # Filter by date range
            if len(date_range_bt) == 2:
                filtered_bt = filtered_bt[
                    (filtered_bt['Entry_Time'].dt.date >= date_range_bt[0]) &
                    (filtered_bt['Entry_Time'].dt.date <= date_range_bt[1])
                ]

            # Calculate statistics (return-based)
            bt_stats = ra.calculate_return_statistics(filtered_bt)
            hold_stats_bt = ra.calculate_hold_time_statistics(filtered_bt)
            stop_loss_stats_bt = ra.calculate_stop_loss_frequency_backtest(filtered_bt)
            side_stats_bt = ra.calculate_return_statistics_by_side(filtered_bt)

            # Overview metrics
            st.subheader("📊 Key Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Sum of Returns", f"{bt_stats['sum_returns']:.2f}%",
                         delta=f"{bt_stats['total_trades']} trades")

            with col2:
                st.metric("Win Rate", f"{bt_stats['win_rate']:.1f}%",
                         delta=f"{bt_stats['wins']}/{bt_stats['total_trades']}")

            with col3:
                avg_hold_mins_bt = hold_stats_bt['avg_minutes']
                if avg_hold_mins_bt < 60:
                    st.metric("Avg Hold Time", f"{avg_hold_mins_bt:.1f} min")
                else:
                    st.metric("Avg Hold Time", f"{hold_stats_bt['avg_hours']:.1f} hrs")

            with col4:
                st.metric("Stop Loss %", f"{stop_loss_stats_bt['stop_loss_pct']:.1f}%",
                         delta=f"{stop_loss_stats_bt['stop_loss_count']} trades")

            # Cumulative Returns Chart
            st.subheader("💰 Cumulative Returns Over Time")

            # Sort by entry time and calculate cumulative returns
            returns_df_bt = filtered_bt.sort_values('Entry_Time').copy()
            returns_df_bt['Cumulative_Return'] = returns_df_bt['Return_Pct'].cumsum()

            # Create cumulative returns line chart
            fig_cumulative_bt = go.Figure()

            # Add cumulative returns line
            fig_cumulative_bt.add_trace(go.Scatter(
                x=returns_df_bt['Entry_Time'],
                y=returns_df_bt['Cumulative_Return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='<b>%{x}</b><br>Cumulative Return: %{y:.2f}%<extra></extra>'
            ))

            # Add individual trade markers colored by side
            for side, color in [('Long', '#28a745'), ('Short', '#dc3545')]:
                side_df_bt = returns_df_bt[returns_df_bt['Side'] == side]
                fig_cumulative_bt.add_trace(go.Scatter(
                    x=side_df_bt['Entry_Time'],
                    y=side_df_bt['Cumulative_Return'],
                    mode='markers',
                    name=f'{side} Trades',
                    marker=dict(
                        color=color,
                        size=6,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Trade Return: %{customdata[0]:.2f}%<br>' +
                                  'Cumulative: %{y:.2f}%<extra></extra>',
                    text=[f'{side}' for _ in side_df_bt['Coin']],
                    customdata=side_df_bt[['Return_Pct']].values
                ))

            # Add zero line
            fig_cumulative_bt.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )

            fig_cumulative_bt.update_layout(
                title='Cumulative Return Performance',
                xaxis_title='Entry Time',
                yaxis_title='Cumulative Return (%)',
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

            st.plotly_chart(fig_cumulative_bt, use_container_width=True)

            # Hold Time Distribution
            st.subheader("📈 Hold Time Distribution")

            fig_hold_bt = px.histogram(
                filtered_bt,
                x='Duration_Minutes',
                color='Side',
                nbins=50,
                title='Hold Time Distribution by Side',
                labels={'Duration_Minutes': 'Hold Time (minutes)', 'count': 'Number of Trades'},
                color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'}
            )

            fig_hold_bt.update_layout(
                height=400,
                xaxis_title="Hold Time (minutes)",
                yaxis_title="Number of Trades",
                showlegend=True,
                hovermode='x unified'
            )

            st.plotly_chart(fig_hold_bt, use_container_width=True)

            # Return Visualizations
            st.subheader("💰 Return Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Return box plot by side
                fig_return_box_bt = px.box(
                    filtered_bt,
                    x='Side',
                    y='Return_Pct',
                    color='Side',
                    title='Return Distribution by Side',
                    labels={'Return_Pct': 'Return (%)', 'Side': 'Trade Side'},
                    color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'}
                )

                fig_return_box_bt.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title="Return (%)"
                )

                st.plotly_chart(fig_return_box_bt, use_container_width=True)

            with col2:
                # Return over time
                fig_return_time_bt = px.scatter(
                    filtered_bt,
                    x='Entry_Time',
                    y='Return_Pct',
                    color='Side',
                    title='Return Over Time',
                    labels={'Entry_Time': 'Entry Time', 'Return_Pct': 'Return (%)'},
                    color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'},
                    hover_data=['Duration', 'Exit_Reason']
                )

                fig_return_time_bt.update_layout(
                    height=400,
                    yaxis_title="Return (%)",
                    xaxis_title="Entry Time"
                )

                st.plotly_chart(fig_return_time_bt, use_container_width=True)

            # Summary Statistics
            st.subheader("📋 Summary Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Overall Statistics**")
                stats_df_bt = pd.DataFrame({
                    'Metric': [
                        'Total Trades',
                        'Sum of Returns',
                        'Average Return',
                        'Median Return',
                        'Win Rate %',
                        'Average Win',
                        'Average Loss',
                        'Profit Factor'
                    ],
                    'Value': [
                        f"{bt_stats['total_trades']:,}",
                        f"{bt_stats['sum_returns']:.2f}%",
                        f"{bt_stats['avg_return']:.2f}%",
                        f"{bt_stats['median_return']:.2f}%",
                        f"{bt_stats['win_rate']:.2f}%",
                        f"{bt_stats['avg_win']:.2f}%",
                        f"{bt_stats['avg_loss']:.2f}%",
                        f"{bt_stats['profit_factor']:.2f}" if bt_stats['profit_factor'] != float('inf') else '∞'
                    ]
                })
                st.dataframe(stats_df_bt, use_container_width=True)

            with col2:
                st.write("**Statistics by Side**")
                side_stats_list_bt = []
                for side in selected_sides_bt:
                    if side_stats_bt.get(side):
                        stats = side_stats_bt[side]
                        side_stats_list_bt.append({
                            'Side': side,
                            'Trades': stats['total_trades'],
                            'Win Rate': f"{stats['win_rate']:.1f}%",
                            'Sum Returns': f"{stats['sum_returns']:.2f}%",
                            'Avg Return': f"{stats['avg_return']:.2f}%",
                            'Profit Factor': f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else '∞'
                        })

                if side_stats_list_bt:
                    side_stats_df_bt = pd.DataFrame(side_stats_list_bt)
                    st.dataframe(side_stats_df_bt, use_container_width=True)

            # Hold Time Stats
            st.subheader("⏱️ Hold Time Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average", f"{hold_stats_bt['avg_minutes']:.1f} min")
                st.metric("Median", f"{hold_stats_bt['median_minutes']:.1f} min")

            with col2:
                st.metric("Minimum", f"{hold_stats_bt['min_seconds']:.0f} sec")
                st.metric("Maximum", ra.format_duration(hold_stats_bt['max_seconds']))

            with col3:
                st.metric("Std Dev", f"{hold_stats_bt['std_seconds']/60:.1f} min")

            # Stop Loss Details
            st.subheader("🛑 Stop Loss Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Stop Losses", stop_loss_stats_bt['stop_loss_count'])

            with col2:
                st.metric("Avg Stop Loss Return", f"{stop_loss_stats_bt['avg_stop_loss_return']:.2f}%")

        except FileNotFoundError:
            st.error("Backtest data not found. Please ensure 'data/backtest_on_filtered_news_offset5s_sl0.33pct.csv' exists.")
        except Exception as e:
            st.error(f"Error loading backtest data: {str(e)}")
            st.exception(e)

    with tab4:
        st.header("Live vs Backtest Reconciliation")
        st.markdown("Compare live trading results against backtest predictions to identify deviations")

        try:
            # Load reconciliation data
            live_df, backtest_df = ra.load_reconciliation_data()

            # Aggregate live trades
            live_agg = ra.aggregate_live_trades(live_df)

            # Match trades
            matched_df, live_only_df, backtest_only_df = ra.match_trades(live_agg, backtest_df)

            # Calculate deviations for matched trades
            if len(matched_df) > 0:
                matched_df = ra.calculate_deviations(matched_df)

            # Get summary statistics
            summary = ra.get_reconciliation_summary(matched_df, live_only_df, backtest_only_df)

            # Sidebar filters for Reconciliation
            with st.sidebar:
                st.header("🔍 Reconciliation Filters")

                # Return deviation threshold
                return_threshold = st.slider(
                    "Return Deviation Threshold (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help="Highlight trades with return deviation above this threshold"
                )

                # Entry slippage threshold
                entry_slip_threshold = st.slider(
                    "Entry Slippage Threshold (%)",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    help="Highlight trades with entry slippage above this threshold"
                )

                # Side filter
                selected_sides_recon = st.multiselect(
                    "Trade Side",
                    options=['Long', 'Short'],
                    default=['Long', 'Short'],
                    key="reconciliation_sides"
                )

            # Apply side filter to matched data
            if len(matched_df) > 0:
                filtered_matched = matched_df[matched_df['Side'].isin(selected_sides_recon)]
            else:
                filtered_matched = matched_df

            # Overview metrics
            st.subheader("📊 Reconciliation Summary")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Matched Trades", f"{summary['total_matched']:,}")

            with col2:
                st.metric("Live Only", f"{summary['total_live_only']:,}",
                         help="Trades in live data with no backtest match")

            with col3:
                st.metric("Backtest Only", f"{summary['total_backtest_only']:,}",
                         help="Backtest signals with no live execution")

            with col4:
                st.metric("Match Rate", f"{summary['match_rate_live']:.1f}%")

            with col5:
                st.metric("Exit Mismatches", f"{summary['exit_reason_mismatch_count']:.0f}",
                         delta=f"{summary['exit_reason_mismatch_pct']:.1f}%")

            # Deviation metrics
            st.subheader("📈 Deviation Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Avg Entry Slippage", f"{summary['avg_entry_slippage_bps']:.1f} bps")

            with col2:
                st.metric("Avg Exit Slippage", f"{summary['avg_exit_slippage_bps']:.1f} bps")

            with col3:
                st.metric("Avg Return Deviation", f"{summary['avg_return_deviation']:.3f}%")

            with col4:
                st.metric("Avg Fill Latency", f"{summary['avg_fill_latency_seconds']:.1f}s")

            # Significant deviations table
            st.subheader("⚠️ Significant Deviations")

            if len(filtered_matched) > 0:
                significant = ra.get_significant_deviations(
                    filtered_matched,
                    return_threshold=return_threshold,
                    entry_slip_threshold=entry_slip_threshold
                )

                if len(significant) > 0:
                    st.write(f"**Found {len(significant)} trades with significant deviations**")

                    # Create display dataframe
                    display_cols = [
                        'match_key', 'Side', 'num_fills',
                        'Entry_Price_bt', 'Entry_Price_live', 'entry_slippage_bps',
                        'Exit_Price_bt', 'Exit_Price_live', 'exit_slippage_bps',
                        'Return_Pct_bt', 'Return_Pct_live', 'return_deviation',
                        'Exit_Reason_bt', 'Exit_Reason_live', 'exit_reason_mismatch'
                    ]
                    display_df = significant[display_cols].copy()
                    display_df.columns = [
                        'Timestamp', 'Side', 'Fills',
                        'BT Entry', 'Live Entry', 'Entry Slip (bps)',
                        'BT Exit', 'Live Exit', 'Exit Slip (bps)',
                        'BT Return %', 'Live Return %', 'Return Δ %',
                        'BT Exit Reason', 'Live Exit Reason', 'Mismatch'
                    ]

                    # Format numeric columns
                    display_df['BT Entry'] = display_df['BT Entry'].apply(lambda x: f"${x:.2f}")
                    display_df['Live Entry'] = display_df['Live Entry'].apply(lambda x: f"${x:.2f}")
                    display_df['BT Exit'] = display_df['BT Exit'].apply(lambda x: f"${x:.2f}")
                    display_df['Live Exit'] = display_df['Live Exit'].apply(lambda x: f"${x:.2f}")
                    display_df['Entry Slip (bps)'] = display_df['Entry Slip (bps)'].apply(lambda x: f"{x:.1f}")
                    display_df['Exit Slip (bps)'] = display_df['Exit Slip (bps)'].apply(lambda x: f"{x:.1f}")
                    display_df['BT Return %'] = display_df['BT Return %'].apply(lambda x: f"{x:.3f}")
                    display_df['Live Return %'] = display_df['Live Return %'].apply(lambda x: f"{x:.3f}")
                    display_df['Return Δ %'] = display_df['Return Δ %'].apply(lambda x: f"{x:.3f}")

                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.success("No significant deviations found with current thresholds")
            else:
                st.info("No matched trades to analyze")

            # Deviation distribution charts
            st.subheader("📊 Deviation Distributions")

            if len(filtered_matched) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    # Entry slippage histogram
                    fig_entry_slip = px.histogram(
                        filtered_matched,
                        x='entry_slippage_bps',
                        nbins=50,
                        title='Entry Slippage Distribution (bps)',
                        labels={'entry_slippage_bps': 'Entry Slippage (bps)', 'count': 'Count'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_entry_slip.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_entry_slip.update_layout(height=350)
                    st.plotly_chart(fig_entry_slip, use_container_width=True)

                with col2:
                    # Return deviation histogram
                    fig_return_dev = px.histogram(
                        filtered_matched,
                        x='return_deviation',
                        nbins=50,
                        title='Return Deviation Distribution (%)',
                        labels={'return_deviation': 'Return Deviation (%)', 'count': 'Count'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_return_dev.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_return_dev.update_layout(height=350)
                    st.plotly_chart(fig_return_dev, use_container_width=True)

                # Scatter plot: Backtest Return vs Live Return
                st.subheader("📈 Backtest vs Live Returns")

                fig_scatter = px.scatter(
                    filtered_matched,
                    x='Return_Pct_bt',
                    y='Return_Pct_live',
                    color='Side',
                    title='Backtest Return vs Live Return',
                    labels={'Return_Pct_bt': 'Backtest Return (%)', 'Return_Pct_live': 'Live Return (%)'},
                    color_discrete_map={'Long': '#28a745', 'Short': '#dc3545'},
                    hover_data=['match_key', 'entry_slippage_bps', 'exit_reason_mismatch']
                )

                # Add y=x reference line
                min_val = min(filtered_matched['Return_Pct_bt'].min(), filtered_matched['Return_Pct_live'].min())
                max_val = max(filtered_matched['Return_Pct_bt'].max(), filtered_matched['Return_Pct_live'].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Match (y=x)',
                    line=dict(color='gray', dash='dash')
                ))

                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Exit reason analysis
            st.subheader("🔄 Exit Reason Analysis")

            if len(filtered_matched) > 0:
                crosstab = ra.get_exit_reason_crosstab(filtered_matched)

                if len(crosstab) > 0:
                    st.write("**Exit Reason Crosstab** (Backtest rows × Live columns)")
                    st.dataframe(crosstab, use_container_width=True)

                    # Highlight mismatches
                    mismatches = filtered_matched[filtered_matched['exit_reason_mismatch'] == True]
                    if len(mismatches) > 0:
                        st.write(f"**Exit Reason Mismatches: {len(mismatches)} trades**")

                        # Group by exit reason pair
                        mismatch_summary = mismatches.groupby(['Exit_Reason_bt', 'Exit_Reason_live']).agg({
                            'match_key': 'count',
                            'return_deviation': 'mean'
                        }).reset_index()
                        mismatch_summary.columns = ['Backtest Exit', 'Live Exit', 'Count', 'Avg Return Deviation %']
                        mismatch_summary['Avg Return Deviation %'] = mismatch_summary['Avg Return Deviation %'].apply(lambda x: f"{x:.3f}")
                        mismatch_summary = mismatch_summary.sort_values('Count', ascending=False)

                        st.dataframe(mismatch_summary, use_container_width=True)

            # Unmatched trades
            st.subheader("❓ Unmatched Trades")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Live Only ({len(live_only_df)} trades)**")
                st.caption("Trades executed live but not in backtest")

                if len(live_only_df) > 0:
                    live_only_display = live_only_df[['match_key', 'Side', 'Entry_Price_live',
                                                       'Return_Pct_live', 'num_fills']].copy()
                    live_only_display.columns = ['Timestamp', 'Side', 'Entry Price', 'Return %', 'Fills']
                    live_only_display['Entry Price'] = live_only_display['Entry Price'].apply(lambda x: f"${x:.2f}")
                    live_only_display['Return %'] = live_only_display['Return %'].apply(lambda x: f"{x:.3f}")
                    st.dataframe(live_only_display.head(20), use_container_width=True)
                else:
                    st.success("All live trades matched to backtest")

            with col2:
                st.write(f"**Backtest Only ({len(backtest_only_df)} trades)**")
                st.caption("Backtest signals not executed live")

                if len(backtest_only_df) > 0:
                    bt_only_display = backtest_only_df[['match_key', 'Side', 'Entry_Price_bt',
                                                         'Return_Pct_bt', 'source']].copy()
                    bt_only_display.columns = ['Timestamp', 'Side', 'Entry Price', 'Return %', 'Source']
                    bt_only_display['Entry Price'] = bt_only_display['Entry Price'].apply(lambda x: f"${x:.2f}")
                    bt_only_display['Return %'] = bt_only_display['Return %'].apply(lambda x: f"{x:.3f}")
                    st.dataframe(bt_only_display.head(20), use_container_width=True)
                else:
                    st.success("All backtest signals were executed live")

            # Summary statistics by side
            st.subheader("📋 Deviation Statistics by Side")

            if len(filtered_matched) > 0:
                side_stats = []
                for side in selected_sides_recon:
                    side_df = filtered_matched[filtered_matched['Side'] == side]
                    if len(side_df) > 0:
                        side_stats.append({
                            'Side': side,
                            'Trades': len(side_df),
                            'Avg Entry Slip (bps)': f"{side_df['entry_slippage_bps'].mean():.1f}",
                            'Avg Exit Slip (bps)': f"{side_df['exit_slippage_bps'].mean():.1f}",
                            'Avg Return Dev %': f"{side_df['return_deviation'].mean():.3f}",
                            'Exit Mismatches': side_df['exit_reason_mismatch'].sum(),
                            'Mismatch %': f"{(side_df['exit_reason_mismatch'].sum() / len(side_df) * 100):.1f}"
                        })

                if side_stats:
                    st.dataframe(pd.DataFrame(side_stats), use_container_width=True)

        except FileNotFoundError as e:
            st.error(f"Data file not found: {str(e)}")
            st.info("Please ensure both 'data/algo_performance_fixed_exits.csv' and 'data/backtest_on_filtered_news_offset5s_sl0.33pct.csv' exist.")
        except Exception as e:
            st.error(f"Error loading reconciliation data: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
