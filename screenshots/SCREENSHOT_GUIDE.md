# Screenshots Guide for Monitoring Dashboard

## 📸 Required Screenshots for README

To complete the documentation, please take the following screenshots of the monitoring dashboard at http://localhost:8503:

### 1. Data Quality Monitoring (`monitoring_data_quality.png`)
- Navigate to the **Data Quality** tab
- Upload `deployment_data/data_quality_issues.csv`
- Capture the view showing:
  - Quality score comparison (baseline vs current)
  - Missing values heatmap
  - Outlier detection table

### 2. Drift Detection (`monitoring_drift_detection.png`)
- Navigate to the **Drift Detection** tab  
- Upload `deployment_data/high_drift_data.csv`
- Capture the view showing:
  - PSI scores bar chart with red/yellow/green indicators
  - Feature-level drift analysis
  - Alert levels (High/Medium/Low)

### 3. Model Performance Tracking (`monitoring_model_performance.png`)
- Navigate to the **Model Performance** tab
- Upload `deployment_data/performance_degradation_data.csv`
- Capture the view showing:
  - RMSE/MAE metrics comparison
  - Prediction distribution plots
  - Actual vs Predicted scatter plot

### 4. Real-time Alerts (`monitoring_alerts.png`)
- Navigate to the **Real-time Alerts** tab
- Upload `deployment_data/high_drift_data.csv` (to trigger alerts)
- Capture the view showing:
  - Alert summary cards with severity levels
  - Recommended actions
  - Alert configuration sliders

## 📋 Screenshot Instructions

1. **Start the monitoring app**: `streamlit run monitoring_app.py`
2. **Use the demo datasets** from `deployment_data/` folder
3. **Ensure full tab content is visible** in each screenshot
4. **Include sidebar** showing upload status
5. **Save as PNG files** in the `screenshots/` directory

## 🎯 Optimal Screenshot Settings

- **Browser width**: 1400px minimum for full content
- **Capture full tab content**: Scroll to show complete visualizations
- **Clear data upload status**: Show successful file upload in sidebar
- **High quality**: PNG format with good resolution

After taking screenshots, the README will automatically display them in the Screenshots section.