# Reconcile Analysis Dashboard

Lightweight Streamlit dashboard for analyzing trading reconciliation data.

## Features

- **Open Analysis**: Analyze news release timing vs actual trade execution
- **Real Analysis**: Algo trading performance metrics and statistics

## Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `reconcile_dashboard.py`
4. Deploy!

### Local Development

```bash
pip install -r requirements.txt
streamlit run reconcile_dashboard.py
```

## Data Files

Total data size: ~150KB

- `data/algo_performance_report.csv` (25KB)
- `data/computed/clean/open_trades_analysis_v2_*.csv` (68KB)
- `data/computed/clean/open_trades_analysis_*.csv` (48KB)

## Repository Size

This is a lightweight deployment repository with no Git LFS dependencies.
Total size: <200KB
