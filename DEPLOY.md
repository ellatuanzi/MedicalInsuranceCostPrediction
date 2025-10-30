# Render Deployment Guide

## Quick Deploy to Render

This repository is configured for easy deployment to Render with two separate Streamlit applications:

1. **Main Insurance Prediction App** (`app.py`)
2. **Fairness Analysis App** (`fairness_app.py`)

### Prerequisites

1. GitHub account with this repository
2. Render account (free tier available)
3. Gemini API key (for AI insights feature)

### Deployment Steps

#### Option 1: Deploy Both Apps Separately (Recommended)

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Deploy Main Prediction App:**
   - Click "New +" → "Web Service"
   - Connect your GitHub account and select this repository
   - Configure:
     - **Name**: `insurance-prediction-app`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
     - **Instance Type**: Free tier

3. **Deploy Fairness Analysis App:**
   - Repeat process with:
     - **Name**: `fairness-analysis-app`
     - **Start Command**: `streamlit run fairness_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

#### Option 2: Use render.yaml (Blueprint)

1. **Fork/Clone this repository**
2. **Push to your GitHub**
3. **In Render Dashboard:**
   - Click "New +" → "Blueprint"
   - Connect this repository
   - Render will automatically detect `render.yaml` and deploy both services

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

- `render.yaml` - Blueprint configuration for both services
- `runtime.txt` - Python version specification
- `start_main_app.sh` - Startup script for main app
- `start_fairness_app.sh` - Startup script for fairness app
- Updated `requirements.txt` - Complete dependency list

### Live URLs

After deployment, you'll receive URLs like:
- Main App: `https://insurance-prediction-app-xxxx.onrender.com`
- Fairness App: `https://fairness-analysis-app-xxxx.onrender.com`

Share these URLs to provide access to your deployed applications!