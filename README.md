# Medical Insurance Claims Prediction

A machine learning project for predicting medical insurance claims with interpretable AI and fairness analysis.

## 📋 Overview

This project uses a **Kaggle medical insurance dataset** to predict **total_claims_paid** for patients based on demographics, health metrics, and claims history. The project includes:

- **LightGBM model** with 12 selected features
- **Interactive Streamlit apps** for predictions, interpretability, fairness analysis, and monitoring
- **SHAP explanations** for model transparency
- **AI-powered insights** via Google's Gemini API
- **Fairness analysis** across demographic groups
- **CI/CD monitoring dashboard** for data drift detection and model performance tracking

## 📊 Dataset Description

**Source:** [Kaggle Medical Insurance Cost Prediction Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction)

**Size:** 100,000 rows × 54 columns

**Target Variable:** `total_claims_paid` (continuous, amount paid in insurance claims)

**Missing Data:** 30,083 missing values across various features (handled during preprocessing)

### Key Features (54 total):

**Demographics:**
- `age`, `sex`, `region`, `urban_rural`, `education`, `marital_status`, `employment_status`
- `household_size`, `dependents`, `income`

**Health Metrics:**
- `bmi`, `systolic_bp`, `diastolic_bp`, `ldl`, `hba1c`
- `smoker`, `alcohol_freq`

**Medical History:**
- `visits_last_year`, `hospitalizations_last_3yrs`, `days_hospitalized_last_3yrs`
- `medication_count`, `chronic_count`
- Binary indicators: `hypertension`, `diabetes`, `asthma`, `copd`, `cardiovascular_disease`, 
  `cancer_history`, `kidney_disease`, `liver_disease`, `arthritis`, `mental_health`

**Insurance Details:**
- `plan_type`, `network_tier`, `deductible`, `copay`, `policy_term_years`
- `policy_changes_last_2yrs`, `provider_quality`, `risk_score`

**Medical Procedures:**
- `proc_imaging_count`, `proc_surgery_count`, `proc_physio_count`, `proc_consult_count`, `proc_lab_count`
- `is_high_risk`, `had_major_procedure`

**Data Types:**
- Integer (31 features)
- Float (13 features)  
- Categorical (10 features)

## 🎯 Model Performance

### Training Configuration
- **Training Set:** 80,000 samples (80%)
- **Test/Validation Set:** 20,000 samples (20%)
- **Feature Selection:** Top 12 features selected via Random Forest importance

### Final Model Metrics (LightGBM)

**Test Set Performance:**
- **RMSE:** $1,905.67
- **MAE:** $1,027.30

**Training Set Performance:**
- **RMSE:** $1,930.50 (at best iteration)

**Model Characteristics:**
- **Best Iteration:** 95 (out of 100 max iterations)
- **Early Stopping:** Applied with patience of 10 rounds
- **Low Overfitting:** Training RMSE ($1,930.50) ≈ Validation RMSE ($1,905.67)

### Baseline Comparison (GLM with Tweedie Distribution)

**GLM Performance:**
- **RMSE:** $7,937.49
- **MAE:** $1,279.41

**Improvement:** LightGBM achieves **76% reduction in RMSE** compared to GLM baseline

### Selected Features (12 Final Features)

The model uses these 12 features ranked by importance:
1. `visits_last_year` - Number of medical visits
2. `chronic_count` - Count of chronic conditions
3. `ldl` - LDL cholesterol level
4. `income` - Annual income
5. `hba1c` - Blood sugar control metric
6. `bmi` - Body Mass Index
7. `provider_quality` - Healthcare provider rating
8. `systolic_bp` - Systolic blood pressure
9. `diastolic_bp` - Diastolic blood pressure
10. `risk_score` - Calculated risk assessment
11. `days_hospitalized_last_3yrs` - Recent hospitalization days
12. `policy_term_years` - Insurance policy duration

## �🖼️ Screenshots

### Main Prediction App

#### Prediction Interface
![Prediction Tab](screenshots/prediction_tab.png)

#### SHAP Analysis
![SHAP Analysis](screenshots/shap_analysis.png)

#### AI Insights
![AI Insights](screenshots/ai_insights.png)

### Fairness Analysis App

#### Group Comparison
![Fairness Group Comparison](screenshots/fairness_group_comparison.png)

#### Fairness Metrics
![Fairness Metrics](screenshots/fairness_metrics.png)

#### Prediction Distribution
![Fairness Distribution](screenshots/fairness_distribution.png)

#### Detailed Analysis
![Fairness Detailed](screenshots/fairness_detailed.png)

### Data & Model Monitoring Dashboard

#### Data Quality and Distribution Monitoring
![Monitoring Data Quality](screenshots/monitoring_data_quality.png)
![Monitoring Data Distribution](screenshots/monitoring_data_distribution.png)

#### Drift Detection (PSI Analysis)
![Monitoring Drift Detection](screenshots/monitoring_drift_detection.png)

#### Model Performance Tracking
![Monitoring Model Performance](screenshots/monitoring_model_performance.png)

#### Real-time Alerts
![Monitoring Alerts](screenshots/monitoring_alerts.png)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Apps

**Main Prediction App:**
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

**Fairness Analysis App:**
```bash
streamlit run fairness_app.py
```
Opens at: http://localhost:8502

**Data & Model Monitoring Dashboard:**
```bash
streamlit run monitoring_app.py
```
Opens at: http://localhost:8503

## 📊 Applications

## 📊 Applications

### 1. Main Prediction App (`app.py`)

**Features:**
- 🔮 **Prediction Tab**: Get claims predictions for individual patients
- 📊 **SHAP Analysis Tab**: Visualize feature contributions with waterfall plots
- 💡 **AI Insights Tab**: Generate natural language explanations (requires Gemini API key)

**Usage:**
1. Select a sample or use random selection
2. View patient demographics and health metrics
3. Get instant predictions with detailed explanations
4. Understand which features drive the prediction

### 2. Fairness Analysis App (`fairness_app.py`)

**Features:**
- ⚖️ **Group Comparison**: Compare predictions across demographic groups
- 📈 **Fairness Metrics**: Statistical parity and disparity analysis
- 📊 **Distribution Analysis**: Visualize prediction distributions by group
- 🔍 **Detailed Reports**: Exportable fairness assessments

**Usage:**
1. Select a protected attribute (sex, age, region, education, etc.)
2. Analyze prediction differences across groups
3. Review statistical parity and bias metrics
4. Export fairness reports as CSV

### 3. Data & Model Monitoring Dashboard (`monitoring_app.py`) 🆕

**Features:**
- 📋 **Data Quality Monitoring**: Track missing values, outliers, and data type consistency
- 📊 **Distribution Analysis**: Compare baseline vs current data with statistical tests
- 🔍 **Drift Detection**: Population Stability Index (PSI) calculation for all features
- 🎯 **Model Performance Tracking**: Monitor RMSE, MAE, prediction drift, and residual analysis
- ⚡ **Real-time Alerts**: Configurable thresholds with actionable recommendations

**Usage:**
1. Upload current/production dataset via sidebar
2. **Data Quality Tab**: Compare data quality scores and identify issues
3. **Distribution Analysis Tab**: Run Kolmogorov-Smirnov and Mann-Whitney U tests
4. **Drift Detection Tab**: Monitor PSI scores and feature-level drift analysis
5. **Model Performance Tab**: Track prediction accuracy and detect model degradation
6. **Real-time Alerts Tab**: Configure alert thresholds and review recommendations

**Drift Detection Levels:**
- 🟢 **Stable** (PSI < 0.1): No action needed
- 🟡 **Moderate Drift** (PSI 0.1-0.25): Monitor closely
- 🔴 **High Drift** (PSI > 0.25): Investigation and potential retraining required

**Demo Datasets Available:**
- `deployment_data/stable_deployment_data.csv` - Normal production scenario
- `deployment_data/moderate_drift_data.csv` - Gradual population changes  
- `deployment_data/high_drift_data.csv` - Significant demographic shifts
- `deployment_data/data_quality_issues.csv` - Missing values and outliers
- `deployment_data/performance_degradation_data.csv` - Model accuracy decline

## 🧠 Model Information

- **Algorithm**: LightGBM Regressor
- **Target Variable**: `total_claims_paid`
- **Features**: 12 selected features
  - `visits_last_year`, `chronic_count`, `ldl`, `income`, `hba1c`, `bmi`
  - `provider_quality`, `systolic_bp`, `diastolic_bp`, `risk_score`
  - `days_hospitalized_last_3yrs`, `policy_term_years`
- **Feature Selection**: Random Forest importance ranking
- **Evaluation**: RMSE, MAE, Gini coefficient, lift charts

## 📁 Project Structure

```
├── app.py                              # Main prediction & interpretation app
├── fairness_app.py                     # Model fairness analysis app
├── monitoring_app.py                   # Data & model monitoring dashboard
├── generate_deployment_data.py         # Script to create test datasets
├── insurance_claim_analysis.ipynb      # Model training & analysis notebook
├── lightgbm_model.pkl                  # Trained model (exported from notebook)
├── medical_insurance.csv               # Dataset
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version for deployment
├── render.yaml                         # Render deployment configuration
├── start_main_app.sh                   # Startup script for main app
├── start_fairness_app.sh               # Startup script for fairness app
├── start_monitoring_app.sh             # Startup script for monitoring app
├── DEPLOY.md                           # Deployment guide
├── README.md                           # Project documentation
├── deployment_data/                    # Test datasets for monitoring
│   ├── README.md                       # Dataset documentation
│   ├── stable_deployment_data.csv      # Normal production data
│   ├── moderate_drift_data.csv         # Moderate drift scenario
│   ├── high_drift_data.csv             # High drift scenario
│   ├── data_quality_issues.csv         # Data quality problems
│   └── performance_degradation_data.csv # Model performance decline
└── screenshots/                        # App screenshots
    ├── prediction_tab.png
    ├── shap_analysis.png
    ├── ai_insights.png
    ├── fairness_group_comparison.png
    ├── fairness_metrics.png
    ├── fairness_distribution.png
    ├── fairness_detailed.png
    ├── monitoring_data_quality.png
    ├── monitoring_drift_detection.png
    ├── monitoring_model_performance.png
    └── monitoring_alerts.png
```

## 🔑 Gemini API Setup (Optional)

For AI-powered insights in the main app:

1. Visit https://ai.google.dev/
2. Create a free API key
3. Enter the key in the app sidebar
4. Generate natural language explanations with highlighted factors

## 📈 Key Features

### Interpretability
- **SHAP Values**: Understand individual predictions
- **Feature Importance**: See which factors matter most
- **Waterfall Plots**: Visualize positive/negative contributions

### Fairness Analysis
- **Statistical Parity**: Measure equal treatment across groups
- **Bias Detection**: Identify systematic prediction differences
- **Group Comparisons**: Analyze performance by demographics
- **Actionable Recommendations**: Get suggestions for fairness improvements

### CI/CD Monitoring
- **Data Drift Detection**: Population Stability Index (PSI) monitoring
- **Data Quality Tracking**: Missing values, outliers, type consistency
- **Model Performance Monitoring**: Accuracy degradation detection
- **Real-time Alerts**: Configurable thresholds with severity levels
- **Statistical Testing**: Kolmogorov-Smirnov and Mann-Whitney U tests

### User Experience
- **Interactive Dashboards**: Easy-to-use Streamlit interfaces
- **Real-time Predictions**: Instant results
- **Export Capabilities**: Download fairness reports
- **Visual Analytics**: Comprehensive charts and plots

## 📝 Notes

- Model and data files must be in the same directory as the apps
- SHAP calculations may take a few seconds
- Gemini API requires internet connection
- Keep API keys secure and never commit them to version control
- Fairness analysis works with or without ground truth labels

## 🛠️ Development

The project workflow:
1. **Data exploration** in Jupyter notebook
2. **Feature engineering** and selection
3. **Model training** with LightGBM
4. **Model export** to pickle file
5. **App development** for deployment
6. **Fairness evaluation** across groups

## 📚 Resources

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Library](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Gemini API](https://ai.google.dev/)

## ⚖️ Fairness & Ethics

This project includes comprehensive fairness analysis tools to ensure responsible AI deployment. Always:
- Review predictions for potential bias
- Validate fairness across protected groups
- Consider the societal impact of automated decisions
- Use fairness metrics alongside performance metrics


## 🔮 Future Work

### Model Improvements
- **Hyperparameter Optimization**: Systematic tuning using Optuna or similar frameworks
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Feature Engineering**: Advanced feature creation and selection techniques

### Fairness Enhancements
- **Intersectional Fairness**: Analyze fairness across multiple protected attributes simultaneously
- **Counterfactual Explanations**: Generate "what-if" scenarios for fairer predictions
- **Continuous Monitoring**: Set up automated fairness drift detection in production

### Application Features
- **Real-time Predictions**: API deployment for production use
- **Advanced Visualizations**: Interactive 3D plots and animated charts
- **Multi-language Support**: Internationalization for global deployment
- **Mobile Optimization**: Responsive design for mobile devices

### Data & Infrastructure
- **Automated Data Pipeline**: Scheduled data refresh and validation
- **Cloud Deployment**: AWS/Azure/GCP integration with auto-scaling
- **Database Integration**: PostgreSQL/MongoDB for data persistence
- **A/B Testing Framework**: Systematic model comparison and rollout

### Research Extensions
- **Causal Inference**: Identify causal relationships in insurance claims
- **Time Series Analysis**: Predict claims trends over time
- **Anomaly Detection**: Identify unusual claim patterns and potential fraud
- **Personalized Recommendations**: Health and policy recommendations for individuals