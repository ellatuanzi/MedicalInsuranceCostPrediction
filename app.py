import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Page configuration
st.set_page_config(
    page_title="Insurance Claims Prediction & Interpretation",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = None

# Title and description
st.title("🏥 Medical Insurance Claims Prediction & Interpretation")
st.markdown("""
This app uses a LightGBM model to predict insurance claims and provides interpretable insights using SHAP values.
Get AI-powered explanations via Gemini API.
""")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Gemini API Key input
    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key if st.session_state.gemini_api_key else "",
        help="Get your API key from https://ai.google.dev/"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        try:
            genai.configure(api_key=api_key)
            st.success("✅ API Key configured successfully!")
        except Exception as e:
            st.error(f"❌ Error configuring API: {str(e)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    - **Model**: LightGBM
    - **Features**: 11 selected features
    - **Target**: Total claims paid
    """)

@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    try:
        with open('lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'lightgbm_model.pkl' not found. Please ensure the model is exported.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data from CSV"""
    try:
        df = pd.read_csv('medical_insurance.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'medical_insurance.csv' not found.")
        return None

def preprocess_data(df, sample_row):
    """Preprocess the data to match model training"""
    # Exact features expected by the model (in order)
    model_features = ['visits_last_year', 'chronic_count', 'ldl', 'income', 'hba1c', 
                     'bmi', 'provider_quality', 'systolic_bp', 'diastolic_bp', 
                     'risk_score', 'days_hospitalized_last_3yrs', 'policy_term_years']
    
    # Get the specific row
    sample = df.iloc[sample_row:sample_row+1]
    
    # Extract only the features needed by the model
    sample_processed = sample[model_features].copy()
    
    return sample_processed, df

def calculate_shap_values(model, X):
    """Calculate SHAP values for the prediction"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values
    except Exception as e:
        st.error(f"Error calculating SHAP values: {str(e)}")
        return None, None

def get_gemini_explanation(feature_impacts, prediction, api_key, sample_data):
    """Get AI explanation using Gemini API"""
    if not api_key:
        return None, "Please configure your Gemini API key in the sidebar to get AI-powered explanations."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt with feature impacts and sample data
        prompt = f"""
You are an insurance claims analyst explaining a prediction to a client. 

The predicted insurance claim amount is ${prediction:.2f}.

Patient Profile:
- Age: {sample_data.get('age', 'N/A')}
- BMI: {sample_data.get('bmi', 'N/A')}
- Income: ${sample_data.get('income', 'N/A'):,.0f}
- Chronic Conditions: {sample_data.get('chronic_count', 'N/A')}
- Risk Score: {sample_data.get('risk_score', 'N/A')}

Here are the key factors influencing this prediction (SHAP values show impact):

{feature_impacts}

Please provide a comprehensive analysis with:

1. **Executive Summary** (2-3 sentences): Overall assessment of the claim prediction
2. **Key Drivers** (3-5 bullet points): Most important factors and their specific impact on the prediction
3. **Risk Assessment**: Categorize as Low/Medium/High risk and explain why
4. **Actionable Recommendations** (3-5 specific suggestions): What the client can do to potentially reduce future claims
5. **Model Insights**: Brief explanation of how the machine learning model arrived at this prediction

Keep the tone professional but accessible. Use analogies where helpful. Format using markdown with clear headers and bullet points.
"""
        
        response = model.generate_content(prompt)
        
        # Generate highlights separately
        highlights_prompt = f"""
Based on this insurance claim prediction analysis, generate 4-6 key highlights in a very concise format.
Each highlight should be a single sentence (max 15 words) capturing the most important insights.

Prediction: ${prediction:.2f}
Top factors: {feature_impacts}

Format each highlight as:
• [Brief insight]

Focus on actionable insights, risk factors, and surprising findings.
"""
        
        highlights_response = model.generate_content(highlights_prompt)
        
        return highlights_response.text, response.text
    except Exception as e:
        return None, f"Error generating explanation: {str(e)}\n\nPlease check your API key and try again."

# Main app
def main():
    # Load model and data
    model = load_model()
    df = load_sample_data()
    
    if model is None or df is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 SHAP Analysis", "💡 AI Insights"])
    
    with tab1:
        st.header("Select a Sample for Prediction")
        
        # Sample selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sample_idx = st.number_input(
                "Sample Index",
                min_value=0,
                max_value=len(df)-1,
                value=0,
                help="Select a sample from the dataset"
            )
            
            if st.button("🎲 Random Sample", use_container_width=True):
                sample_idx = np.random.randint(0, len(df))
                st.rerun()
        
        with col2:
            # Display sample information
            sample = df.iloc[sample_idx]
            
            st.subheader("Sample Information")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Age", sample['age'])
                st.metric("BMI", f"{sample['bmi']:.1f}")
            
            with col_b:
                st.metric("Sex", sample['sex'])
                st.metric("Smoker", sample['smoker'])
            
            with col_c:
                st.metric("Income", f"${sample['income']:,.0f}")
                st.metric("Plan Type", sample['plan_type'])
            
            with col_d:
                st.metric("Risk Score", f"{sample['risk_score']:.2f}")
                st.metric("Chronic Count", int(sample['chronic_count']))
        
        # Make prediction
        if st.button("🔮 Predict Claims", type="primary", use_container_width=True):
            with st.spinner("Processing prediction..."):
                # Preprocess data
                X_sample, _ = preprocess_data(df, sample_idx)
                
                # Make prediction
                prediction = model.predict(X_sample)[0]
                
                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.X_sample = X_sample
                st.session_state.sample_idx = sample_idx
                
                # Display prediction
                st.success("Prediction completed!")
                st.metric(
                    "Predicted Total Claims",
                    f"${prediction:,.2f}",
                    help="This is the predicted total claims amount for this individual"
                )
                
                # Compare with actual if available
                if 'total_claims_paid' in df.columns:
                    actual = df.iloc[sample_idx]['total_claims_paid']
                    error = prediction - actual
                    error_pct = (error / actual * 100) if actual > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Actual Claims", f"${actual:,.2f}")
                    col2.metric("Prediction Error", f"${error:,.2f}")
                    col3.metric("Error %", f"{error_pct:.1f}%")
    
    with tab2:
        st.header("SHAP Analysis - Feature Importance")
        
        if 'X_sample' not in st.session_state:
            st.info("👈 Please make a prediction first in the 'Make Prediction' tab.")
        else:
            with st.spinner("Calculating SHAP values..."):
                X_sample = st.session_state.X_sample
                
                # Calculate SHAP values
                explainer, shap_values = calculate_shap_values(model, X_sample)
                
                if explainer and shap_values is not None:
                    # Store in session state
                    st.session_state.shap_values = shap_values
                    st.session_state.explainer = explainer
                    
                    # Waterfall plot
                    st.subheader("Feature Contribution (Waterfall Plot)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[0],
                            base_values=explainer.expected_value,
                            data=X_sample.iloc[0].values,
                            feature_names=X_sample.columns.tolist()
                        ),
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    # Feature importance table
                    st.subheader("Feature Importance Breakdown")
                    feature_importance = pd.DataFrame({
                        'Feature': X_sample.columns,
                        'Value': X_sample.iloc[0].values,
                        'SHAP Value': shap_values[0],
                        'Impact': ['Increases' if x > 0 else 'Decreases' for x in shap_values[0]]
                    })
                    feature_importance['Abs SHAP'] = abs(feature_importance['SHAP Value'])
                    feature_importance = feature_importance.sort_values('Abs SHAP', ascending=False)
                    
                    st.dataframe(
                        feature_importance[['Feature', 'Value', 'SHAP Value', 'Impact']].head(10),
                        use_container_width=True,
                        hide_index=True
                    )
    
    with tab3:
        st.header("AI-Powered Insights")
        
        if 'shap_values' not in st.session_state:
            st.info("👈 Please complete the SHAP Analysis first in the 'SHAP Analysis' tab.")
        else:
            if st.button("✨ Generate AI Explanation", type="primary", use_container_width=True):
                with st.spinner("Generating AI insights with Gemini..."):
                    # Prepare feature impacts for Gemini
                    X_sample = st.session_state.X_sample
                    shap_values = st.session_state.shap_values
                    prediction = st.session_state.prediction
                    sample_idx = st.session_state.sample_idx
                    
                    # Get original sample data for context
                    sample_data = df.iloc[sample_idx].to_dict()
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X_sample.columns,
                        'SHAP Value': shap_values[0]
                    })
                    feature_importance['Abs SHAP'] = abs(feature_importance['SHAP Value'])
                    feature_importance = feature_importance.sort_values('Abs SHAP', ascending=False).head(5)
                    
                    feature_impacts = "\n".join([
                        f"- {row['Feature']}: Impact of {row['SHAP Value']:.2f} ({'increases' if row['SHAP Value'] > 0 else 'decreases'} prediction)"
                        for _, row in feature_importance.iterrows()
                    ])
                    
                    # Get Gemini explanation
                    highlights, explanation = get_gemini_explanation(
                        feature_impacts,
                        prediction,
                        st.session_state.gemini_api_key,
                        sample_data
                    )
                    
                    # Store in session state
                    st.session_state.ai_highlights = highlights
                    st.session_state.ai_explanation = explanation
            
            # Display results if available
            if 'ai_explanation' in st.session_state:
                # Display highlights at the top with special styling
                if st.session_state.ai_highlights:
                    st.markdown("### 🎯 Key Highlights")
                    st.info(st.session_state.ai_highlights)
                    st.markdown("---")
                
                # Main explanation
                st.markdown("### 🤖 Detailed Analysis")
                st.markdown(st.session_state.ai_explanation)
                
                # Additional insights
                st.markdown("---")
                st.markdown("### 📈 Key Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Positive Factors**")
                    feature_importance = pd.DataFrame({
                        'Feature': st.session_state.X_sample.columns,
                        'SHAP Value': st.session_state.shap_values[0]
                    })
                    positive_factors = feature_importance[feature_importance['SHAP Value'] > 0].head(3)
                    for _, row in positive_factors.iterrows():
                        st.markdown(f"- {row['Feature']}: +${row['SHAP Value']:.2f}")
                
                with col2:
                    st.markdown("**Top Negative Factors**")
                    negative_factors = feature_importance[feature_importance['SHAP Value'] < 0].head(3)
                    for _, row in negative_factors.iterrows():
                        st.markdown(f"- {row['Feature']}: ${row['SHAP Value']:.2f}")

if __name__ == "__main__":
    main()
