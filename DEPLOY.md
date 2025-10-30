# Render Deployment Guide

## Quick Deploy to Render

This repository is configured for easy deployment to Render with three separate Streamlit applications:

1. **Main Insurance Prediction App** (`app.py`)
2. **Fairness Analysis App** (`fairness_app.py`)
3. **Data & Model Monitoring App** (`monitoring_app.py`) - **NEW!**

### Prerequisites

1. GitHub account with this repository
2. Render account (free tier available)
3. Gemini API key (for AI insights feature)

### Deployment Steps

#### Option 1: Deploy All Apps Separately (Recommended)

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Deploy Main Prediction App:**
   - Click "New +" → "Web Service"
   - Connect your GitHub account and select this repository
   - Configure:
     - **Name**: `insurance-prediction-app`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
     - **Instance Type**: Free tier

3. **Deploy Fairness Analysis App:**
   - Repeat process with:
     - **Name**: `fairness-analysis-app`
     - **Start Command**: `streamlit run fairness_app.py --server.port=$PORT --server.address=0.0.0.0`

4. **Deploy Data Monitoring App:**
   - Repeat process with:
     - **Name**: `data-monitoring-app`
     - **Start Command**: `streamlit run monitoring_app.py --server.port=$PORT --server.address=0.0.0.0`

#### Option 2: Use render.yaml (Blueprint)

1. **Fork/Clone this repository**
2. **Push to your GitHub**
3. **In Render Dashboard:**
   - Click "New +" → "Blueprint"
   - Connect this repository
   - Render will automatically detect `render.yaml` and deploy all three services

### Environment Variables

For the AI Insights feature to work, add this environment variable in Render:

- **Key**: `GEMINI_API_KEY`
- **Value**: Your Google Gemini API key

### Post-Deployment

1. **Test the applications** at the provided Render URLs
2. **Configure API key** in the sidebar of the main app
3. **Monitor logs** in Render dashboard for any issues

### Costs

- **Free Tier**: Both apps can run on Render's free tier
- **Limitations**: Free apps sleep after 15 minutes of inactivity
- **Upgrade**: For production use, consider upgrading to paid tier

### Troubleshooting

If deployment fails:

1. Check build logs in Render dashboard
2. Verify all files are committed to GitHub
3. Ensure `requirements.txt` is up to date
4. Check Python version compatibility (3.12.0 specified)

### Files Added for Deployment

- `render.yaml` - Blueprint configuration for all three services
- `runtime.txt` - Python version specification
- `start_main_app.sh` - Startup script for main app
- `start_fairness_app.sh` - Startup script for fairness app
- `start_monitoring_app.sh` - Startup script for monitoring app
- Updated `requirements.txt` - Complete dependency list including plotly and scipy
- `monitoring_app.py` - Comprehensive data and model monitoring dashboard

### Live URLs

After deployment, you'll receive URLs like:
- Main App: `https://insurance-prediction-app-xxxx.onrender.com`
- Fairness App: `https://fairness-analysis-app-xxxx.onrender.com`
- Monitoring App: `https://data-monitoring-app-xxxx.onrender.com`

Share these URLs to provide access to your deployed applications!

## New Monitoring App Features

### 📊 Data & Model Monitoring Dashboard

The new monitoring app provides comprehensive CI/CD monitoring capabilities:

#### 📋 Data Quality Monitoring
- **Missing Values Detection**: Identify and track missing data patterns
- **Outlier Detection**: IQR-based outlier identification and comparison
- **Data Type Validation**: Ensure data consistency across deployments
- **Quality Score**: Overall data health metric (0-1 scale)

#### 📊 Distribution Analysis  
- **Feature Distribution Comparison**: Visual comparison between baseline and current data
- **Statistical Tests**: Kolmogorov-Smirnov and Mann-Whitney U tests
- **Box Plot Analysis**: Detailed statistical summaries and comparisons

#### 🔍 Drift Detection
- **Population Stability Index (PSI)**: Industry-standard drift metric
  - PSI < 0.1: Stable (✅)
  - PSI 0.1-0.25: Moderate drift (⚠️)
  - PSI > 0.25: High drift (🚨)
- **Feature-level Analysis**: Detailed drift investigation for each feature
- **Categorical and Numerical Support**: Handles both data types

#### 🎯 Model Performance Monitoring
- **Prediction Drift**: PSI analysis on model predictions
- **Performance Metrics**: RMSE, MAE, R² tracking
- **Residual Analysis**: Distribution and pattern analysis
- **Actual vs Predicted Plots**: Visual performance assessment

#### ⚡ Real-time Alerts
- **Configurable Thresholds**: Customize alert sensitivity
- **Multi-level Alerts**: High, Medium, Low severity levels
- **Actionable Recommendations**: Specific guidance for each alert type
- **Integration Ready**: Email, Slack, Webhook support

### Usage Instructions

1. **Upload Current Data**: Use the sidebar to upload your production dataset
2. **Generate Demo Data**: Click "Generate Demo Drift Data" to see the system in action
3. **Monitor Across Tabs**: Navigate through different monitoring aspects
4. **Configure Alerts**: Set custom thresholds in the Real-time Alerts tab
5. **Take Action**: Follow recommended actions for any detected issues