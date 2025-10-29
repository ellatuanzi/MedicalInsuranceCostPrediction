# Medical Insurance Claims Prediction

A machine learning project for predicting medical insurance claims with interpretable AI and fairness analysis.

## 📋 Overview

This project uses a **Kaggle medical insurance dataset** to predict **total_claims_paid** for patients based on demographics, health metrics, and claims history. The project includes:

- **LightGBM model** with 12 selected features
- **Interactive Streamlit apps** for predictions, interpretability, and fairness analysis
- **SHAP explanations** for model transparency
- **AI-powered insights** via Google's Gemini API
- **Fairness analysis** across demographic groups

## 🖼️ Screenshots

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
Opens at: http://localhost:8505

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
├── insurance_claim_analysis.ipynb      # Model training & analysis notebook
├── lightgbm_model.pkl                  # Trained model (exported from notebook)
├── medical_insurance.csv               # Dataset
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── screenshots/                        # App screenshots
    ├── prediction_tab.png
    ├── shap_analysis.png
    └── ai_insights.png
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
