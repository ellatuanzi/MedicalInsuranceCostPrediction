# Deployment Test Datasets

This directory contains synthetic deployment datasets for testing the Data & Model Monitoring Dashboard.

## 📁 Dataset Files

### 1. `stable_deployment_data.csv`
- **Purpose**: Test stable production scenario
- **Expected PSI**: < 0.1 (Green/Stable)
- **Characteristics**: 
  - Minimal drift from baseline
  - 2% noise added to numerical features
  - Normal data quality
- **Use Case**: Validate that monitoring doesn't trigger false positives

### 2. `moderate_drift_data.csv`
- **Purpose**: Test moderate drift detection
- **Expected PSI**: 0.1-0.25 (Yellow/Moderate)
- **Characteristics**:
  - Population aging (+3 years average)
  - Income inflation (+8%)
  - BMI increase (+1.2 average)
  - Increased healthcare utilization (+15%)
  - Urban migration (20% rural→urban)
- **Use Case**: Test early warning systems

### 3. `high_drift_data.csv`
- **Purpose**: Test high drift detection
- **Expected PSI**: > 0.25 (Red/High)
- **Characteristics**:
  - Major demographic shift (younger population, -8 years)
  - Bimodal income distribution (economic inequality)
  - Significant BMI changes (+25%)
  - Healthcare delivery changes (-40% visits)
  - Provider quality shifts
  - Regional migration patterns
- **Use Case**: Test critical alert systems

### 4. `data_quality_issues.csv`
- **Purpose**: Test data quality monitoring
- **Issues Introduced**:
  - Missing values (5-15% in key columns)
  - Outliers (2% extreme values)
  - Duplicate rows (2%)
  - Inconsistent categorical values
- **Use Case**: Test data quality score calculations

### 5. `performance_degradation_data.csv`
- **Purpose**: Test model performance monitoring
- **Characteristics**:
  - Systematic bias in model features
  - Feature distribution shifts that affect predictions
  - Expected RMSE increase > 15%
- **Use Case**: Test model retraining alerts

## 🧪 How to Use

1. **Start the monitoring app**: `streamlit run monitoring_app.py`
2. **Upload any dataset** using the sidebar file uploader
3. **Observe different drift levels** in the Drift Detection tab
4. **Check data quality scores** in the Data Quality tab
5. **Monitor model performance** in the Model Performance tab
6. **Review alerts** in the Real-time Alerts tab

## 📊 Expected Results

| Dataset | Data Quality | PSI Range | Model Impact | Alert Level |
|---------|-------------|-----------|--------------|-------------|
| Stable | High (>0.9) | < 0.1 | Minimal | ✅ None |
| Moderate Drift | Good (>0.8) | 0.1-0.25 | Low | ⚠️ Medium |
| High Drift | Good (>0.8) | > 0.25 | High | 🚨 High |
| Quality Issues | Poor (<0.7) | Variable | Variable | 🚨 High |
| Performance Deg | Good (>0.8) | Variable | Very High | 🚨 High |

## 🔄 Regenerating Data

To create new test datasets:
```bash
python generate_deployment_data.py
```

All datasets are based on the original `medical_insurance.csv` with controlled modifications to test specific monitoring scenarios.
