# Deployment Guide

## 🎯 Quick Deploy to Streamlit Community Cloud

### Step 1: Create GitHub Repository

```bash
# You're already in the hedge_reconcile_deploy directory with git initialized!

# Create a new repo on GitHub (go to github.com/new)
# Name it: hedge-reconcile-dashboard (or whatever you prefer)
# Don't initialize with README (we already have one)

# Then run:
git remote add origin https://github.com/YOUR_USERNAME/hedge-reconcile-dashboard.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/hedge-reconcile-dashboard`
4. Branch: `main`
5. Main file path: `reconcile_dashboard.py`
6. Click "Deploy!"

That's it! Your app will be live in ~2 minutes.

## ✅ Repository Stats

- **Total size**: 256KB
- **Git repo size**: 220KB
- **No Git LFS required**
- **Data files**: 200KB total
  - `algo_performance_report.csv`: 25KB
  - `open_trades_analysis*.csv`: 175KB combined

## 🧪 Test Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run reconcile_dashboard.py

# Open browser to http://localhost:8501
```

## 🔄 Future Updates

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push

# Streamlit will auto-redeploy within seconds!
```

## 📊 Features Available

✅ **Open Analysis Tab**
- Response time analysis (news to trade)
- Histogram and box plots
- Percentile statistics
- Outlier detection
- Supports both v1 and v2 data formats

✅ **Real Analysis Tab**
- Cumulative PnL tracking
- Hold time distribution
- Win rate analysis
- Stop loss statistics
- Side-by-side comparison (Long vs Short)
- Outlier trade identification

## 🚀 Alternative Deployment Options

### Heroku
```bash
# Create Procfile
echo "web: streamlit run reconcile_dashboard.py --server.port=\$PORT" > Procfile
git add Procfile
git commit -m "Add Heroku Procfile"
git push heroku main
```

### Render
1. Connect your GitHub repo
2. Select "Web Service"
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run reconcile_dashboard.py --server.port=$PORT --server.address=0.0.0.0`

### Railway
1. Connect GitHub repo
2. Will auto-detect Streamlit
3. Deploy automatically

All platforms support this lightweight repo with no issues!
