import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data & Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Data & Model Monitoring Dashboard")
st.markdown("""
**CI/CD Data Monitoring Platform** for tracking data quality, distribution changes, drift detection, and model performance.
Upload your baseline (training) data and current (production) data to monitor system health.
""")

# Utility Functions
@st.cache_data
def load_baseline_data():
    """Load the original training dataset as baseline"""
    try:
        df = pd.read_csv('medical_insurance.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Baseline data 'medical_insurance.csv' not found.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model for predictions"""
    try:
        with open('lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'lightgbm_model.pkl' not found.")
        return None

def calculate_psi(baseline_data, current_data, feature, bins=10):
    """
    Calculate Population Stability Index (PSI) for a feature
    PSI > 0.25: Significant shift
    PSI 0.1-0.25: Moderate shift  
    PSI < 0.1: Stable
    """
    try:
        # Handle categorical variables
        if baseline_data[feature].dtype == 'object' or current_data[feature].dtype == 'object':
            baseline_counts = baseline_data[feature].value_counts(normalize=True)
            current_counts = current_data[feature].value_counts(normalize=True)
            
            # Align indices
            all_categories = set(baseline_counts.index) | set(current_counts.index)
            baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0.001)
            current_aligned = current_counts.reindex(all_categories, fill_value=0.001)
        else:
            # Numerical variables - use binning
            min_val = min(baseline_data[feature].min(), current_data[feature].min())
            max_val = max(baseline_data[feature].max(), current_data[feature].max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            baseline_binned = pd.cut(baseline_data[feature], bins=bin_edges, include_lowest=True)
            current_binned = pd.cut(current_data[feature], bins=bin_edges, include_lowest=True)
            
            baseline_aligned = baseline_binned.value_counts(normalize=True).sort_index()
            current_aligned = current_binned.value_counts(normalize=True).sort_index()
            
            # Add small epsilon to avoid division by zero
            baseline_aligned = baseline_aligned + 0.001
            current_aligned = current_aligned + 0.001
        
        # Calculate PSI
        psi = sum((current_aligned - baseline_aligned) * np.log(current_aligned / baseline_aligned))
        return float(psi)
    except Exception as e:
        st.warning(f"Could not calculate PSI for {feature}: {str(e)}")
        return None

def calculate_data_quality_score(df):
    """Calculate overall data quality score"""
    scores = {}
    
    # Missing values score
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    scores['missing'] = max(0, 1 - missing_ratio * 2)  # Penalize missing values
    
    # Duplicate rows score
    duplicate_ratio = df.duplicated().sum() / len(df)
    scores['duplicates'] = max(0, 1 - duplicate_ratio * 5)  # Heavily penalize duplicates
    
    # Data type consistency (assume all should be numeric except specific columns)
    categorical_cols = ['sex', 'smoker', 'region', 'urban_rural', 'education', 'marital_status', 
                       'employment_status', 'alcohol_freq', 'plan_type', 'network_tier']
    numeric_expected = [col for col in df.columns if col not in categorical_cols and col != 'person_id']
    
    type_consistency = 0
    for col in numeric_expected:
        if col in df.columns:
            try:
                pd.to_numeric(df[col])
                type_consistency += 1
            except:
                pass
    scores['types'] = type_consistency / max(1, len(numeric_expected))
    
    # Overall score
    overall_score = np.mean(list(scores.values()))
    return overall_score, scores

def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    return outliers, len(series)

def preprocess_for_prediction(df):
    """Preprocess data to match model expectations"""
    model_features = ['visits_last_year', 'chronic_count', 'ldl', 'income', 'hba1c', 
                     'bmi', 'provider_quality', 'systolic_bp', 'diastolic_bp', 
                     'risk_score', 'days_hospitalized_last_3yrs', 'policy_term_years']
    
    # Return only the features the model expects
    return df[model_features] if all(col in df.columns for col in model_features) else None

# Sidebar for data upload
st.sidebar.header("📁 Data Upload")

# Load baseline data automatically
baseline_data = load_baseline_data()
model = load_model()

# Upload current/production data
uploaded_file = st.sidebar.file_uploader(
    "Upload Current Dataset (CSV)", 
    type=['csv'],
    help="Upload the current/production dataset to compare against baseline"
)

current_data = None
if uploaded_file is not None:
    current_data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Current data loaded: {current_data.shape[0]} rows, {current_data.shape[1]} columns")

# Generate synthetic drift data for demo
if st.sidebar.button("🎲 Generate Demo Drift Data"):
    if baseline_data is not None:
        # Create synthetic data with intentional drift
        demo_data = baseline_data.copy().sample(n=min(5000, len(baseline_data)), random_state=42)
        
        # Introduce drift in key features
        if 'age' in demo_data.columns:
            demo_data['age'] = demo_data['age'] + np.random.normal(5, 2, len(demo_data))  # Age shift
        if 'bmi' in demo_data.columns:
            demo_data['bmi'] = demo_data['bmi'] * np.random.normal(1.1, 0.1, len(demo_data))  # BMI inflation
        if 'income' in demo_data.columns:
            demo_data['income'] = demo_data['income'] * np.random.normal(1.2, 0.15, len(demo_data))  # Income increase
        
        current_data = demo_data
        st.sidebar.success("🎲 Demo drift data generated!")

# Main dashboard
if baseline_data is None:
    st.error("Please ensure 'medical_insurance.csv' is available in the working directory.")
    st.stop()

# Create tabs for different monitoring aspects
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Quality", 
    "📊 Distribution Analysis", 
    "🔍 Drift Detection", 
    "🎯 Model Performance",
    "⚡ Real-time Alerts"
])

with tab1:
    st.header("📋 Data Quality Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Data Quality")
        baseline_score, baseline_scores = calculate_data_quality_score(baseline_data)
        
        st.metric("Overall Quality Score", f"{baseline_score:.3f}", help="1.0 = Perfect, 0.0 = Poor")
        
        # Detailed breakdown
        st.write("**Quality Breakdown:**")
        st.write(f"- Missing Values Score: {baseline_scores['missing']:.3f}")
        st.write(f"- Duplicate Records Score: {baseline_scores['duplicates']:.3f}")
        st.write(f"- Data Type Consistency: {baseline_scores['types']:.3f}")
        
        # Missing values heatmap
        if baseline_data.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(baseline_data.isnull(), cbar=True, ax=ax)
            ax.set_title("Missing Values Pattern (Baseline)")
            st.pyplot(fig)
        else:
            st.success("✅ No missing values in baseline data")
    
    with col2:
        if current_data is not None:
            st.subheader("Current Data Quality")
            current_score, current_scores = calculate_data_quality_score(current_data)
            
            # Compare scores
            score_diff = current_score - baseline_score
            st.metric(
                "Overall Quality Score", 
                f"{current_score:.3f}", 
                delta=f"{score_diff:.3f}",
                help="Change compared to baseline"
            )
            
            # Detailed breakdown
            st.write("**Quality Breakdown:**")
            st.write(f"- Missing Values Score: {current_scores['missing']:.3f} ({current_scores['missing'] - baseline_scores['missing']:+.3f})")
            st.write(f"- Duplicate Records Score: {current_scores['duplicates']:.3f} ({current_scores['duplicates'] - baseline_scores['duplicates']:+.3f})")
            st.write(f"- Data Type Consistency: {current_scores['types']:.3f} ({current_scores['types'] - baseline_scores['types']:+.3f})")
            
            # Missing values heatmap
            if current_data.isnull().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(current_data.isnull(), cbar=True, ax=ax)
                ax.set_title("Missing Values Pattern (Current)")
                st.pyplot(fig)
            else:
                st.success("✅ No missing values in current data")
        else:
            st.info("Upload current dataset to compare quality metrics")
    
    # Outlier Detection
    if current_data is not None:
        st.subheader("🎯 Outlier Detection")
        
        numeric_columns = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'person_id' in numeric_columns:
            numeric_columns.remove('person_id')
        
        outlier_data = []
        for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
            if col in current_data.columns:
                baseline_outliers, baseline_total = detect_outliers_iqr(baseline_data[col])
                current_outliers, current_total = detect_outliers_iqr(current_data[col])
                
                outlier_data.append({
                    'Feature': col,
                    'Baseline Outliers': f"{baseline_outliers}/{baseline_total} ({baseline_outliers/baseline_total*100:.1f}%)",
                    'Current Outliers': f"{current_outliers}/{current_total} ({current_outliers/current_total*100:.1f}%)",
                    'Change': f"{(current_outliers/current_total - baseline_outliers/baseline_total)*100:+.1f}%"
                })
        
        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)

with tab2:
    st.header("📊 Distribution Analysis")
    
    if current_data is not None:
        # Feature selection for distribution comparison
        numeric_features = [col for col in baseline_data.select_dtypes(include=[np.number]).columns 
                          if col in current_data.columns and col != 'person_id']
        
        selected_features = st.multiselect(
            "Select features to analyze",
            numeric_features,
            default=numeric_features[:4] if len(numeric_features) >= 4 else numeric_features
        )
        
        if selected_features:
            # Create distribution plots
            n_cols = 2
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_features,
                vertical_spacing=0.1
            )
            
            for i, feature in enumerate(selected_features):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                # Add baseline distribution
                fig.add_trace(
                    go.Histogram(
                        x=baseline_data[feature],
                        name=f'Baseline {feature}',
                        opacity=0.7,
                        nbinsx=30,
                        histnorm='probability'
                    ),
                    row=row, col=col
                )
                
                # Add current distribution
                fig.add_trace(
                    go.Histogram(
                        x=current_data[feature],
                        name=f'Current {feature}',
                        opacity=0.7,
                        nbinsx=30,
                        histnorm='probability'
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=300 * n_rows,
                title_text="Distribution Comparison: Baseline vs Current",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            st.subheader("📈 Statistical Tests")
            
            stat_results = []
            for feature in selected_features:
                if feature in baseline_data.columns and feature in current_data.columns:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(baseline_data[feature].dropna(), current_data[feature].dropna())
                    
                    # Mann-Whitney U test
                    mw_stat, mw_p = stats.mannwhitneyu(
                        baseline_data[feature].dropna(), 
                        current_data[feature].dropna(),
                        alternative='two-sided'
                    )
                    
                    stat_results.append({
                        'Feature': feature,
                        'KS Statistic': f"{ks_stat:.4f}",
                        'KS p-value': f"{ks_p:.4f}",
                        'KS Significant': "🚨 Yes" if ks_p < 0.05 else "✅ No",
                        'MW p-value': f"{mw_p:.4f}",
                        'MW Significant': "🚨 Yes" if mw_p < 0.05 else "✅ No"
                    })
            
            if stat_results:
                stat_df = pd.DataFrame(stat_results)
                st.dataframe(stat_df, use_container_width=True)
                st.caption("🚨 Significant = Distribution likely changed (p < 0.05)")
        
    else:
        st.info("Upload current dataset to analyze distribution changes")

with tab3:
    st.header("🔍 Drift Detection")
    
    if current_data is not None:
        # Calculate PSI for all features
        st.subheader("📊 Population Stability Index (PSI)")
        
        # Feature selection
        all_features = [col for col in baseline_data.columns 
                       if col in current_data.columns and col not in ['person_id', 'total_claims_paid']]
        
        psi_results = []
        for feature in all_features:
            psi_score = calculate_psi(baseline_data, current_data, feature)
            if psi_score is not None:
                # Determine alert level
                if psi_score > 0.25:
                    alert_level = "🚨 High"
                    alert_class = "alert-high"
                elif psi_score > 0.1:
                    alert_level = "⚠️ Medium"
                    alert_class = "alert-medium"
                else:
                    alert_level = "✅ Low"
                    alert_class = "alert-low"
                
                psi_results.append({
                    'Feature': feature,
                    'PSI Score': psi_score,
                    'PSI Formatted': f"{psi_score:.4f}",
                    'Alert Level': alert_level,
                    'Alert Class': alert_class
                })
        
        if psi_results:
            # Sort by PSI score descending
            psi_results_sorted = sorted(psi_results, key=lambda x: x['PSI Score'], reverse=True)
            
            # Display results in cards
            st.subheader("🎯 PSI Scores by Feature")
            
            # Show top concerning features
            high_drift_features = [r for r in psi_results_sorted if r['PSI Score'] > 0.1]
            if high_drift_features:
                st.warning(f"⚠️ {len(high_drift_features)} features showing moderate to high drift!")
                
                for result in high_drift_features[:5]:  # Show top 5
                    st.markdown(f"""
                    <div class="metric-card {result['Alert Class']}">
                        <strong>{result['Feature']}</strong><br>
                        PSI Score: {result['PSI Formatted']} - {result['Alert Level']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All features show stable distributions (PSI < 0.1)")
            
            # Create PSI visualization
            psi_df = pd.DataFrame(psi_results_sorted)
            
            fig = px.bar(
                psi_df.head(15), 
                x='PSI Score', 
                y='Feature',
                title="Population Stability Index by Feature (Top 15)",
                color='PSI Score',
                color_continuous_scale=['green', 'yellow', 'red'],
                orientation='h'
            )
            
            # Add reference lines
            fig.add_vline(x=0.1, line_dash="dash", line_color="orange", 
                         annotation_text="Moderate Drift Threshold")
            fig.add_vline(x=0.25, line_dash="dash", line_color="red", 
                         annotation_text="High Drift Threshold")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed PSI Results")
            display_df = psi_df[['Feature', 'PSI Formatted', 'Alert Level']].copy()
            display_df.columns = ['Feature', 'PSI Score', 'Drift Level']
            st.dataframe(display_df, use_container_width=True)
        
        # Feature-level drift analysis
        st.subheader("🔬 Detailed Feature Analysis")
        
        if psi_results:
            selected_feature = st.selectbox(
                "Select feature for detailed analysis",
                [r['Feature'] for r in psi_results_sorted]
            )
            
            if selected_feature:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution comparison
                    if baseline_data[selected_feature].dtype == 'object':
                        # Categorical feature
                        baseline_counts = baseline_data[selected_feature].value_counts(normalize=True)
                        current_counts = current_data[selected_feature].value_counts(normalize=True)
                        
                        comparison_df = pd.DataFrame({
                            'Baseline': baseline_counts,
                            'Current': current_counts
                        }).fillna(0)
                        
                        fig = px.bar(
                            comparison_df.reset_index(), 
                            x='index', 
                            y=['Baseline', 'Current'],
                            title=f"{selected_feature} Distribution Comparison",
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Numerical feature
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=baseline_data[selected_feature],
                            name='Baseline',
                            opacity=0.7,
                            nbinsx=30,
                            histnorm='probability'
                        ))
                        fig.add_trace(go.Histogram(
                            x=current_data[selected_feature],
                            name='Current',
                            opacity=0.7,
                            nbinsx=30,
                            histnorm='probability'
                        ))
                        fig.update_layout(
                            title=f"{selected_feature} Distribution Comparison",
                            barmode='overlay'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Statistics comparison
                    if baseline_data[selected_feature].dtype != 'object':
                        baseline_stats = baseline_data[selected_feature].describe()
                        current_stats = current_data[selected_feature].describe()
                        
                        stats_comparison = pd.DataFrame({
                            'Baseline': baseline_stats,
                            'Current': current_stats,
                            'Change %': ((current_stats - baseline_stats) / baseline_stats * 100).round(2)
                        })
                        
                        st.write("**Statistical Summary:**")
                        st.dataframe(stats_comparison)
                        
                        # Box plot comparison
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=baseline_data[selected_feature],
                            name='Baseline',
                            boxpoints='outliers'
                        ))
                        fig.add_trace(go.Box(
                            y=current_data[selected_feature],
                            name='Current',
                            boxpoints='outliers'
                        ))
                        fig.update_layout(title=f"{selected_feature} Box Plot Comparison")
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload current dataset to perform drift detection analysis")

with tab4:
    st.header("🎯 Model Performance Monitoring")
    
    if current_data is not None and model is not None:
        # Preprocess data for model prediction
        baseline_model_data = preprocess_for_prediction(baseline_data)
        current_model_data = preprocess_for_prediction(current_data)
        
        if baseline_model_data is not None and current_model_data is not None:
            # Make predictions
            baseline_predictions = model.predict(baseline_model_data)
            current_predictions = model.predict(current_model_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Statistics")
                
                # Basic prediction stats
                baseline_pred_stats = pd.Series(baseline_predictions).describe()
                current_pred_stats = pd.Series(current_predictions).describe()
                
                pred_comparison = pd.DataFrame({
                    'Baseline Predictions': baseline_pred_stats,
                    'Current Predictions': current_pred_stats,
                    'Change %': ((current_pred_stats - baseline_pred_stats) / baseline_pred_stats * 100).round(2)
                })
                
                st.dataframe(pred_comparison)
                
                # Prediction drift (PSI on predictions)
                pred_psi = calculate_psi(
                    pd.DataFrame({'pred': baseline_predictions}),
                    pd.DataFrame({'pred': current_predictions}),
                    'pred'
                )
                
                if pred_psi is not None:
                    if pred_psi > 0.25:
                        st.error(f"🚨 High prediction drift detected! PSI: {pred_psi:.4f}")
                    elif pred_psi > 0.1:
                        st.warning(f"⚠️ Moderate prediction drift detected. PSI: {pred_psi:.4f}")
                    else:
                        st.success(f"✅ Stable predictions. PSI: {pred_psi:.4f}")
            
            with col2:
                st.subheader("📈 Prediction Distribution")
                
                # Prediction distribution comparison
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=baseline_predictions,
                    name='Baseline Predictions',
                    opacity=0.7,
                    nbinsx=30,
                    histnorm='probability'
                ))
                fig.add_trace(go.Histogram(
                    x=current_predictions,
                    name='Current Predictions',
                    opacity=0.7,
                    nbinsx=30,
                    histnorm='probability'
                ))
                fig.update_layout(
                    title="Prediction Distribution Comparison",
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model performance if actual values are available
            if 'total_claims_paid' in current_data.columns:
                st.subheader("🎯 Model Performance Metrics")
                
                current_actuals = current_data['total_claims_paid'].values
                
                # Calculate metrics
                current_rmse = np.sqrt(mean_squared_error(current_actuals, current_predictions))
                current_mae = mean_absolute_error(current_actuals, current_predictions)
                current_r2 = r2_score(current_actuals, current_predictions)
                
                # Compare with baseline if available
                if 'total_claims_paid' in baseline_data.columns:
                    baseline_actuals = baseline_data['total_claims_paid'].values[:len(baseline_predictions)]
                    baseline_rmse = np.sqrt(mean_squared_error(baseline_actuals, baseline_predictions))
                    baseline_mae = mean_absolute_error(baseline_actuals, baseline_predictions)
                    baseline_r2 = r2_score(baseline_actuals, baseline_predictions)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        rmse_change = current_rmse - baseline_rmse
                        st.metric("RMSE", f"${current_rmse:.2f}", f"{rmse_change:+.2f}")
                    
                    with col2:
                        mae_change = current_mae - baseline_mae
                        st.metric("MAE", f"${current_mae:.2f}", f"{mae_change:+.2f}")
                    
                    with col3:
                        r2_change = current_r2 - baseline_r2
                        st.metric("R² Score", f"{current_r2:.4f}", f"{r2_change:+.4f}")
                else:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RMSE", f"${current_rmse:.2f}")
                    
                    with col2:
                        st.metric("MAE", f"${current_mae:.2f}")
                    
                    with col3:
                        st.metric("R² Score", f"{current_r2:.4f}")
                
                # Residual analysis
                residuals = current_actuals - current_predictions
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residual distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=30,
                        name='Residuals'
                    ))
                    fig.update_layout(title="Residual Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Actual vs Predicted scatter
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=current_actuals,
                        y=current_predictions,
                        mode='markers',
                        name='Predictions',
                        opacity=0.6
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(current_actuals.min(), current_predictions.min())
                    max_val = max(current_actuals.max(), current_predictions.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    fig.update_layout(
                        title="Actual vs Predicted",
                        xaxis_title="Actual Claims",
                        yaxis_title="Predicted Claims"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Cannot perform model predictions. Required features missing from dataset.")
    else:
        if current_data is None:
            st.info("Upload current dataset to monitor model performance")
        if model is None:
            st.error("Model not available for performance monitoring")

with tab5:
    st.header("⚡ Real-time Alerts")
    
    if current_data is not None:
        st.subheader("🚨 Alert Summary")
        
        alerts = []
        
        # Data quality alerts
        baseline_score, _ = calculate_data_quality_score(baseline_data)
        current_score, _ = calculate_data_quality_score(current_data)
        
        if current_score < baseline_score - 0.1:
            alerts.append({
                'Type': '📋 Data Quality',
                'Severity': 'High',
                'Message': f'Data quality dropped by {(baseline_score - current_score)*100:.1f}%',
                'Action': 'Review data pipeline and validation rules'
            })
        
        # PSI alerts
        if 'psi_results' in locals():
            high_drift_count = len([r for r in psi_results if r['PSI Score'] > 0.25])
            medium_drift_count = len([r for r in psi_results if 0.1 < r['PSI Score'] <= 0.25])
            
            if high_drift_count > 0:
                alerts.append({
                    'Type': '🔍 Data Drift',
                    'Severity': 'High',
                    'Message': f'{high_drift_count} features show high drift (PSI > 0.25)',
                    'Action': 'Investigate data sources and consider model retraining'
                })
            elif medium_drift_count > 3:
                alerts.append({
                    'Type': '🔍 Data Drift',
                    'Severity': 'Medium',
                    'Message': f'{medium_drift_count} features show moderate drift',
                    'Action': 'Monitor closely and prepare for potential model update'
                })
        
        # Model performance alerts
        if model is not None and 'total_claims_paid' in current_data.columns:
            current_model_data = preprocess_for_prediction(current_data)
            if current_model_data is not None:
                current_predictions = model.predict(current_model_data)
                current_actuals = current_data['total_claims_paid'].values
                current_rmse = np.sqrt(mean_squared_error(current_actuals, current_predictions))
                
                # Assuming baseline RMSE of around $1900 (from previous analysis)
                baseline_rmse_expected = 1905.67
                rmse_increase = (current_rmse - baseline_rmse_expected) / baseline_rmse_expected
                
                if rmse_increase > 0.15:  # 15% increase
                    alerts.append({
                        'Type': '🎯 Model Performance',
                        'Severity': 'High',
                        'Message': f'RMSE increased by {rmse_increase*100:.1f}% (${current_rmse:.2f})',
                        'Action': 'Model retraining required'
                    })
                elif rmse_increase > 0.05:  # 5% increase
                    alerts.append({
                        'Type': '🎯 Model Performance',
                        'Severity': 'Medium',
                        'Message': f'RMSE increased by {rmse_increase*100:.1f}%',
                        'Action': 'Monitor performance closely'
                    })
        
        # Display alerts
        if alerts:
            for alert in alerts:
                severity_color = {
                    'High': '🔴',
                    'Medium': '🟡', 
                    'Low': '🟢'
                }
                
                st.markdown(f"""
                <div class="metric-card alert-{alert['Severity'].lower()}">
                    <strong>{severity_color[alert['Severity']]} {alert['Type']} - {alert['Severity']} Severity</strong><br>
                    {alert['Message']}<br>
                    <em>Recommended Action: {alert['Action']}</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No alerts detected. All systems operating normally.")
        
        # Alert configuration
        st.subheader("⚙️ Alert Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Quality Thresholds:**")
            quality_threshold = st.slider("Quality Score Drop Alert", 0.05, 0.3, 0.1, 0.01)
            
            st.write("**Data Drift Thresholds:**")
            psi_medium = st.slider("PSI Medium Drift", 0.05, 0.2, 0.1, 0.01)
            psi_high = st.slider("PSI High Drift", 0.15, 0.5, 0.25, 0.01)
        
        with col2:
            st.write("**Model Performance Thresholds:**")
            rmse_medium = st.slider("RMSE Increase Medium Alert (%)", 5, 20, 5, 1)
            rmse_high = st.slider("RMSE Increase High Alert (%)", 10, 30, 15, 1)
            
            st.write("**Notification Settings:**")
            email_alerts = st.checkbox("Email Alerts", value=True)
            slack_alerts = st.checkbox("Slack Notifications", value=False)
            webhook_alerts = st.checkbox("Webhook Integration", value=False)
    
    else:
        st.info("Upload current dataset to see real-time alerts")

# Footer
st.markdown("---")
st.markdown("""
**Data & Model Monitoring Dashboard** | Built with Streamlit | 
[📚 Documentation](DEPLOY.md) | [🔗 GitHub Repository](https://github.com/ellatuanzi/MedicalInsuranceCostPrediction)
""")