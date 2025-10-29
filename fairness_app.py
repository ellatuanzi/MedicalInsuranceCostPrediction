import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Model Fairness Analysis",
    page_icon="⚖️",
    layout="wide"
)

# Title and description
st.title("⚖️ Model Fairness Analysis")
st.markdown("""
This app analyzes the fairness of the LightGBM insurance claims prediction model across different demographic groups.
Assess potential biases in predictions based on protected attributes like sex, age, region, and more.
""")

@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    try:
        with open('lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'lightgbm_model.pkl' not found.")
        return None

@st.cache_data
def load_data():
    """Load the full dataset"""
    try:
        df = pd.read_csv('medical_insurance.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'medical_insurance.csv' not found.")
        return None

def preprocess_for_prediction(df):
    """Preprocess data to match model expectations"""
    # Model expects these 12 features in order
    model_features = ['visits_last_year', 'chronic_count', 'ldl', 'income', 'hba1c', 
                     'bmi', 'provider_quality', 'systolic_bp', 'diastolic_bp', 
                     'risk_score', 'days_hospitalized_last_3yrs', 'policy_term_years']
    
    return df[model_features]

def calculate_fairness_metrics(df, predictions, group_col, group_values):
    """Calculate fairness metrics for different groups"""
    results = []
    
    for value in group_values:
        mask = df[group_col] == value
        group_data = df[mask].copy()
        group_preds = predictions[mask]
        
        if len(group_data) == 0:
            continue
        
        # Calculate metrics
        if 'total_claims_paid' in df.columns:
            actual = group_data['total_claims_paid'].values
            rmse = np.sqrt(mean_squared_error(actual, group_preds))
            mae = mean_absolute_error(actual, group_preds)
            bias = np.mean(group_preds - actual)
        else:
            rmse = mae = bias = None
        
        results.append({
            'Group': f"{group_col}={value}",
            'Count': len(group_data),
            'Avg Prediction': np.mean(group_preds),
            'Std Prediction': np.std(group_preds),
            'Min Prediction': np.min(group_preds),
            'Max Prediction': np.max(group_preds),
            'RMSE': rmse,
            'MAE': mae,
            'Bias': bias
        })
    
    return pd.DataFrame(results)

def calculate_statistical_parity(df, predictions, group_col, threshold_percentile=75):
    """Calculate statistical parity difference"""
    threshold = np.percentile(predictions, threshold_percentile)
    high_risk_predictions = predictions >= threshold
    
    results = []
    group_values = df[group_col].unique()
    
    for value in group_values:
        mask = df[group_col] == value
        group_high_risk_rate = high_risk_predictions[mask].mean()
        
        results.append({
            'Group': value,
            'High Risk Rate': group_high_risk_rate,
            'Sample Size': mask.sum()
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate parity difference (max - min)
    if len(results_df) > 0:
        parity_diff = results_df['High Risk Rate'].max() - results_df['High Risk Rate'].min()
        results_df['Parity Difference'] = parity_diff
    
    return results_df

# Main app
def main():
    # Load model and data
    model = load_model()
    df = load_data()
    
    if model is None or df is None:
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        # Select protected attribute
        protected_attrs = ['sex', 'age', 'region', 'urban_rural', 'education', 
                          'marital_status', 'employment_status', 'plan_type']
        
        selected_attr = st.selectbox(
            "Select Protected Attribute",
            protected_attrs,
            help="Choose a demographic attribute to analyze for fairness"
        )
        
        # Age binning option
        if selected_attr == 'age':
            use_age_bins = st.checkbox("Use Age Groups", value=True)
            if use_age_bins:
                age_bins = st.slider("Number of Age Groups", 3, 10, 5)
        else:
            use_age_bins = False
        
        st.markdown("---")
        st.markdown("### About Fairness Metrics")
        st.markdown("""
        - **Statistical Parity**: Equal prediction rates across groups
        - **Equal Opportunity**: Equal true positive rates
        - **Bias**: Average prediction error by group
        - **RMSE/MAE**: Prediction accuracy by group
        """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Group Comparison", 
        "⚖️ Fairness Metrics", 
        "📈 Prediction Distribution",
        "🔍 Detailed Analysis"
    ])
    
    # Prepare data for prediction
    X = preprocess_for_prediction(df)
    predictions = model.predict(X)
    
    # Handle age binning if selected
    df_analysis = df.copy()
    if selected_attr == 'age' and use_age_bins:
        df_analysis['age_group'] = pd.cut(df_analysis['age'], bins=age_bins, 
                                          labels=[f"Group {i+1}" for i in range(age_bins)])
        analysis_col = 'age_group'
        group_values = df_analysis['age_group'].cat.categories
    else:
        analysis_col = selected_attr
        group_values = df_analysis[selected_attr].unique()
    
    with tab1:
        st.header(f"Group Comparison by {selected_attr.replace('_', ' ').title()}")
        
        # Calculate fairness metrics
        fairness_df = calculate_fairness_metrics(df_analysis, predictions, analysis_col, group_values)
        
        # Display summary statistics
        st.subheader("Summary Statistics by Group")
        st.dataframe(fairness_df.style.format({
            'Avg Prediction': '${:,.2f}',
            'Std Prediction': '${:,.2f}',
            'Min Prediction': '${:,.2f}',
            'Max Prediction': '${:,.2f}',
            'RMSE': '${:,.2f}',
            'MAE': '${:,.2f}',
            'Bias': '${:,.2f}'
        }), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Prediction by Group")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(fairness_df)), fairness_df['Avg Prediction'])
            ax.set_xticks(range(len(fairness_df)))
            ax.set_xticklabels(fairness_df['Group'], rotation=45, ha='right')
            ax.set_ylabel('Average Prediction ($)')
            ax.set_title(f'Average Predicted Claims by {selected_attr.replace("_", " ").title()}')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, fairness_df['Avg Prediction'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'${val:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            if fairness_df['RMSE'].notna().any():
                st.subheader("Prediction Error by Group")
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(fairness_df))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], fairness_df['RMSE'], width, label='RMSE', alpha=0.8)
                ax.bar([i + width/2 for i in x], fairness_df['MAE'], width, label='MAE', alpha=0.8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(fairness_df['Group'], rotation=45, ha='right')
                ax.set_ylabel('Error ($)')
                ax.set_title('Prediction Error by Group')
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        st.header("Fairness Metrics Analysis")
        
        # Statistical Parity
        st.subheader("Statistical Parity")
        st.markdown("Measures whether different groups have equal rates of high-risk predictions.")
        
        threshold_percentile = st.slider(
            "High-Risk Threshold (Percentile)",
            50, 95, 75,
            help="Predictions above this percentile are considered 'high-risk'"
        )
        
        parity_df = calculate_statistical_parity(df_analysis, predictions, analysis_col, threshold_percentile)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(parity_df.style.format({
                'High Risk Rate': '{:.2%}',
                'Parity Difference': '{:.2%}'
            }), use_container_width=True)
        
        with col2:
            if len(parity_df) > 0:
                parity_diff = parity_df['Parity Difference'].iloc[0]
                
                # Color based on severity
                if parity_diff < 0.05:
                    color = "green"
                    assessment = "Low"
                elif parity_diff < 0.10:
                    color = "orange"
                    assessment = "Moderate"
                else:
                    color = "red"
                    assessment = "High"
                
                st.metric(
                    "Parity Difference",
                    f"{parity_diff:.2%}",
                    help="Difference between highest and lowest group rates"
                )
                st.markdown(f"**Disparity Level:** :{color}[{assessment}]")
        
        # Visualization
        st.subheader("High-Risk Rate by Group")
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(parity_df)), parity_df['High Risk Rate'])
        ax.set_xticks(range(len(parity_df)))
        ax.set_xticklabels(parity_df['Group'], rotation=45, ha='right')
        ax.set_ylabel('High-Risk Rate')
        ax.set_title(f'High-Risk Prediction Rate by {selected_attr.replace("_", " ").title()}')
        ax.axhline(y=parity_df['High Risk Rate'].mean(), color='r', linestyle='--', 
                  label='Average Rate')
        
        # Add value labels
        for bar, val in zip(bars, parity_df['High Risk Rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.1%}', ha='center', va='bottom')
        
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.header("Prediction Distribution Analysis")
        
        # Distribution plots
        st.subheader("Prediction Distribution by Group")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for value in group_values:
            mask = df_analysis[analysis_col] == value
            group_preds = predictions[mask]
            ax.hist(group_preds, bins=50, alpha=0.5, label=str(value))
        
        ax.set_xlabel('Predicted Claims ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Predictions by {selected_attr.replace("_", " ").title()}')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Box plot
        st.subheader("Box Plot Comparison")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_data = []
        plot_labels = []
        
        for value in group_values:
            mask = df_analysis[analysis_col] == value
            plot_data.append(predictions[mask])
            plot_labels.append(str(value))
        
        ax.boxplot(plot_data, labels=plot_labels)
        ax.set_xlabel('Group')
        ax.set_ylabel('Predicted Claims ($)')
        ax.set_title(f'Prediction Distribution by {selected_attr.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab4:
        st.header("Detailed Fairness Analysis")
        
        st.subheader("Key Findings")
        
        # Calculate key statistics
        fairness_df = calculate_fairness_metrics(df_analysis, predictions, analysis_col, group_values)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_range = fairness_df['Avg Prediction'].max() - fairness_df['Avg Prediction'].min()
            st.metric(
                "Prediction Range",
                f"${pred_range:,.2f}",
                help="Difference between highest and lowest average predictions"
            )
        
        with col2:
            if fairness_df['RMSE'].notna().any():
                rmse_range = fairness_df['RMSE'].max() - fairness_df['RMSE'].min()
                st.metric(
                    "RMSE Range",
                    f"${rmse_range:,.2f}",
                    help="Difference in prediction accuracy across groups"
                )
        
        with col3:
            if fairness_df['Bias'].notna().any():
                max_bias = fairness_df['Bias'].abs().max()
                st.metric(
                    "Max Absolute Bias",
                    f"${max_bias:,.2f}",
                    help="Largest prediction bias across all groups"
                )
        
        # Recommendations
        st.subheader("📋 Fairness Assessment")
        
        parity_df = calculate_statistical_parity(df_analysis, predictions, analysis_col, 75)
        parity_diff = parity_df['Parity Difference'].iloc[0] if len(parity_df) > 0 else 0
        
        if parity_diff < 0.05 and pred_range < 1000:
            st.success("✅ **Good**: The model shows low disparity across groups.")
        elif parity_diff < 0.10 or pred_range < 2000:
            st.warning("⚠️ **Moderate**: Some disparity detected. Consider reviewing model inputs and training data.")
        else:
            st.error("❌ **High**: Significant disparity detected. Model may need retraining or additional fairness constraints.")
        
        # Detailed recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = []
        
        if parity_diff > 0.10:
            recommendations.append("- Consider applying fairness constraints during model training")
            recommendations.append("- Review if protected attributes are indirectly encoded in features")
        
        if fairness_df['RMSE'].notna().any() and (fairness_df['RMSE'].max() - fairness_df['RMSE'].min()) > 500:
            recommendations.append("- Model performance varies significantly across groups")
            recommendations.append("- Consider separate models or group-specific calibration")
        
        if pred_range > 2000:
            recommendations.append("- Large prediction differences detected across groups")
            recommendations.append("- Validate that differences are justified by legitimate risk factors")
        
        if len(recommendations) > 0:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.info("No major fairness concerns detected. Continue monitoring with new data.")
        
        # Export results
        st.subheader("📥 Export Analysis")
        
        export_data = fairness_df.to_csv(index=False)
        st.download_button(
            label="Download Fairness Report (CSV)",
            data=export_data,
            file_name=f"fairness_analysis_{selected_attr}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
