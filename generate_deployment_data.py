#!/usr/bin/env python3
"""
Deployment Data Generator for Medical Insurance Monitoring

This script creates synthetic production/deployment datasets based on the original
medical insurance data to test the monitoring dashboard's drift detection capabilities.

Generates multiple scenarios:
1. Stable data (minimal drift)
2. Moderate drift (PSI 0.1-0.25)  
3. High drift (PSI > 0.25)
4. Data quality issues
5. Model performance degradation scenarios
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def load_baseline_data():
    """Load the original medical insurance dataset"""
    try:
        df = pd.read_csv('medical_insurance.csv')
        print(f"✅ Loaded baseline data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("❌ Error: 'medical_insurance.csv' not found")
        return None

def create_stable_deployment_data(baseline_df, n_samples=5000):
    """
    Create deployment data with minimal drift (PSI < 0.1)
    Simulates normal production data
    """
    print("📊 Generating stable deployment data...")
    
    # Sample from baseline with slight random variation
    stable_data = baseline_df.sample(n=min(n_samples, len(baseline_df)), random_state=42).copy()
    
    # Add minimal noise to numerical features
    numerical_cols = stable_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col not in ['person_id', 'total_claims_paid']:
            noise_factor = 0.02  # 2% noise
            noise = np.random.normal(0, stable_data[col].std() * noise_factor, len(stable_data))
            stable_data[col] = stable_data[col] + noise
            
            # Ensure non-negative values where appropriate
            if col in ['age', 'bmi', 'income', 'visits_last_year', 'chronic_count']:
                stable_data[col] = np.maximum(stable_data[col], 0)
    
    # Round age and other integer columns
    integer_cols = ['age', 'visits_last_year', 'chronic_count', 'household_size', 'dependents']
    for col in integer_cols:
        if col in stable_data.columns:
            stable_data[col] = stable_data[col].round().astype(int)
    
    print(f"✅ Generated stable data: {len(stable_data)} rows")
    return stable_data

def create_moderate_drift_data(baseline_df, n_samples=5000):
    """
    Create deployment data with moderate drift (PSI 0.1-0.25)
    Simulates gradual population changes
    """
    print("⚠️ Generating moderate drift deployment data...")
    
    drift_data = baseline_df.sample(n=min(n_samples, len(baseline_df)), random_state=123).copy()
    
    # Age shift: Population getting slightly older
    if 'age' in drift_data.columns:
        age_shift = np.random.normal(3, 1.5, len(drift_data))  # Average 3 years older
        drift_data['age'] = drift_data['age'] + age_shift
        drift_data['age'] = np.clip(drift_data['age'], 18, 85).round().astype(int)
    
    # Income inflation: 8% increase on average
    if 'income' in drift_data.columns:
        income_multiplier = np.random.normal(1.08, 0.05, len(drift_data))
        drift_data['income'] = drift_data['income'] * income_multiplier
        drift_data['income'] = np.maximum(drift_data['income'], 10000)  # Minimum income
    
    # BMI trend: Slight increase reflecting population health trends
    if 'bmi' in drift_data.columns:
        bmi_shift = np.random.normal(1.2, 0.8, len(drift_data))
        drift_data['bmi'] = drift_data['bmi'] + bmi_shift
        drift_data['bmi'] = np.clip(drift_data['bmi'], 15, 50)
    
    # Healthcare utilization increase
    if 'visits_last_year' in drift_data.columns:
        visit_multiplier = np.random.normal(1.15, 0.1, len(drift_data))
        drift_data['visits_last_year'] = (drift_data['visits_last_year'] * visit_multiplier).round().astype(int)
        drift_data['visits_last_year'] = np.clip(drift_data['visits_last_year'], 0, 50)
    
    # Regional distribution shift (more urban)
    if 'urban_rural' in drift_data.columns:
        # Convert 20% of rural to urban
        rural_mask = drift_data['urban_rural'] == 'Rural'
        rural_indices = drift_data[rural_mask].index
        convert_indices = np.random.choice(rural_indices, size=int(len(rural_indices) * 0.2), replace=False)
        drift_data.loc[convert_indices, 'urban_rural'] = 'Urban'
    
    print(f"✅ Generated moderate drift data: {len(drift_data)} rows")
    return drift_data

def create_high_drift_data(baseline_df, n_samples=5000):
    """
    Create deployment data with high drift (PSI > 0.25)
    Simulates significant population or market changes
    """
    print("🚨 Generating high drift deployment data...")
    
    high_drift_data = baseline_df.sample(n=min(n_samples, len(baseline_df)), random_state=456).copy()
    
    # Dramatic age shift: Much younger population (new market segment)
    if 'age' in high_drift_data.columns:
        # Shift towards younger demographics
        age_shift = np.random.normal(-8, 3, len(high_drift_data))
        high_drift_data['age'] = high_drift_data['age'] + age_shift
        high_drift_data['age'] = np.clip(high_drift_data['age'], 18, 85).round().astype(int)
    
    # Major income changes: Economic boom/recession effects
    if 'income' in high_drift_data.columns:
        # Bimodal distribution: some much richer, some poorer
        income_multiplier = np.where(
            np.random.random(len(high_drift_data)) < 0.3,
            np.random.normal(1.5, 0.2, len(high_drift_data)),  # 30% get much richer
            np.random.normal(0.8, 0.1, len(high_drift_data))   # 70% get poorer
        )
        high_drift_data['income'] = high_drift_data['income'] * income_multiplier
        high_drift_data['income'] = np.maximum(high_drift_data['income'], 15000)
    
    # Significant BMI changes: Health crisis or fitness trend
    if 'bmi' in high_drift_data.columns:
        bmi_multiplier = np.random.normal(1.25, 0.15, len(high_drift_data))
        high_drift_data['bmi'] = high_drift_data['bmi'] * bmi_multiplier
        high_drift_data['bmi'] = np.clip(high_drift_data['bmi'], 15, 55)
    
    # Healthcare system changes: Telemedicine boom
    if 'visits_last_year' in high_drift_data.columns:
        visit_multiplier = np.random.normal(0.6, 0.2, len(high_drift_data))  # Fewer in-person visits
        high_drift_data['visits_last_year'] = (high_drift_data['visits_last_year'] * visit_multiplier).round().astype(int)
        high_drift_data['visits_last_year'] = np.maximum(high_drift_data['visits_last_year'], 0)
    
    # Provider quality shift
    if 'provider_quality' in high_drift_data.columns:
        quality_shift = np.random.normal(0.5, 0.3, len(high_drift_data))
        high_drift_data['provider_quality'] = high_drift_data['provider_quality'] + quality_shift
        high_drift_data['provider_quality'] = np.clip(high_drift_data['provider_quality'], 1, 5)
    
    # Major regional shift
    if 'region' in high_drift_data.columns:
        # Simulate migration to specific regions
        regions = high_drift_data['region'].unique()
        if len(regions) > 1:
            # 60% move to the first region
            target_region = regions[0]
            change_mask = np.random.random(len(high_drift_data)) < 0.6
            high_drift_data.loc[change_mask, 'region'] = target_region
    
    print(f"✅ Generated high drift data: {len(high_drift_data)} rows")
    return high_drift_data

def create_data_quality_issues_data(baseline_df, n_samples=5000):
    """
    Create deployment data with various data quality issues
    """
    print("🔧 Generating data with quality issues...")
    
    quality_data = baseline_df.sample(n=min(n_samples, len(baseline_df)), random_state=789).copy()
    
    # Introduce missing values (5-15% for different columns)
    missing_candidates = ['income', 'bmi', 'ldl', 'hba1c', 'provider_quality']
    for col in missing_candidates:
        if col in quality_data.columns:
            missing_rate = np.random.uniform(0.05, 0.15)  # 5-15% missing
            missing_indices = np.random.choice(
                quality_data.index, 
                size=int(len(quality_data) * missing_rate), 
                replace=False
            )
            quality_data.loc[missing_indices, col] = np.nan
    
    # Introduce outliers
    outlier_candidates = ['age', 'bmi', 'income', 'visits_last_year']
    for col in outlier_candidates:
        if col in quality_data.columns:
            outlier_rate = 0.02  # 2% outliers
            outlier_indices = np.random.choice(
                quality_data.index,
                size=int(len(quality_data) * outlier_rate),
                replace=False
            )
            
            if col == 'age':
                quality_data.loc[outlier_indices, col] = np.random.choice([150, 200, -5], len(outlier_indices))
            elif col == 'bmi':
                quality_data.loc[outlier_indices, col] = np.random.choice([80, 100, -10], len(outlier_indices))
            elif col == 'income':
                quality_data.loc[outlier_indices, col] = np.random.choice([1000000, 5000000, -1000], len(outlier_indices))
            elif col == 'visits_last_year':
                quality_data.loc[outlier_indices, col] = np.random.choice([100, 200, -5], len(outlier_indices))
    
    # Add duplicate rows (2% duplication)
    n_duplicates = int(len(quality_data) * 0.02)
    duplicate_indices = np.random.choice(quality_data.index, size=n_duplicates, replace=False)
    duplicate_rows = quality_data.loc[duplicate_indices].copy()
    quality_data = pd.concat([quality_data, duplicate_rows], ignore_index=True)
    
    # Introduce inconsistent categorical values
    if 'sex' in quality_data.columns:
        # Add some inconsistent entries
        inconsistent_indices = np.random.choice(quality_data.index, size=10, replace=False)
        quality_data.loc[inconsistent_indices, 'sex'] = np.random.choice(['M', 'F', 'Unknown'], 10)
    
    print(f"✅ Generated quality issues data: {len(quality_data)} rows")
    return quality_data

def create_performance_degradation_data(baseline_df, n_samples=5000):
    """
    Create data that will cause model performance degradation
    """
    print("📉 Generating model performance degradation data...")
    
    perf_data = baseline_df.sample(n=min(n_samples, len(baseline_df)), random_state=999).copy()
    
    # Shift key model features to cause prediction errors
    model_features = ['visits_last_year', 'chronic_count', 'ldl', 'income', 'hba1c', 
                     'bmi', 'provider_quality', 'systolic_bp', 'diastolic_bp', 
                     'risk_score', 'days_hospitalized_last_3yrs', 'policy_term_years']
    
    for feature in model_features:
        if feature in perf_data.columns:
            if feature in ['visits_last_year', 'chronic_count']:
                # Categorical-like features: shift distribution
                shift_multiplier = np.random.normal(1.5, 0.3, len(perf_data))
                perf_data[feature] = (perf_data[feature] * shift_multiplier).round().astype(int)
                perf_data[feature] = np.maximum(perf_data[feature], 0)
            else:
                # Continuous features: add systematic bias
                bias = np.random.normal(0.3 * perf_data[feature].std(), 0.1 * perf_data[feature].std(), len(perf_data))
                perf_data[feature] = perf_data[feature] + bias
                
                # Ensure reasonable bounds
                if feature in ['bmi']:
                    perf_data[feature] = np.clip(perf_data[feature], 15, 60)
                elif feature in ['ldl', 'hba1c']:
                    perf_data[feature] = np.maximum(perf_data[feature], 0)
    
    print(f"✅ Generated performance degradation data: {len(perf_data)} rows")
    return perf_data

def save_deployment_datasets(datasets_dict, output_dir='deployment_data'):
    """Save all generated datasets to files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")
    
    for name, df in datasets_dict.items():
        filename = f"{output_dir}/{name}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Saved {name}: {filename} ({len(df)} rows)")

def generate_documentation(output_dir='deployment_data'):
    """Generate documentation for the test datasets"""
    doc_content = """# Deployment Test Datasets

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
"""
    
    doc_path = f"{output_dir}/README.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    print(f"📝 Generated documentation: {doc_path}")

def main():
    """Main function to generate all deployment test datasets"""
    print("🏗️ Medical Insurance Deployment Data Generator")
    print("=" * 50)
    
    # Load baseline data
    baseline_df = load_baseline_data()
    if baseline_df is None:
        return
    
    # Generate different drift scenarios
    datasets = {}
    
    # 1. Stable data (minimal drift)
    datasets['stable_deployment_data'] = create_stable_deployment_data(baseline_df)
    
    # 2. Moderate drift
    datasets['moderate_drift_data'] = create_moderate_drift_data(baseline_df)
    
    # 3. High drift
    datasets['high_drift_data'] = create_high_drift_data(baseline_df)
    
    # 4. Data quality issues
    datasets['data_quality_issues'] = create_data_quality_issues_data(baseline_df)
    
    # 5. Model performance degradation
    datasets['performance_degradation_data'] = create_performance_degradation_data(baseline_df)
    
    # Save all datasets
    save_deployment_datasets(datasets)
    
    # Generate documentation
    generate_documentation()
    
    print("\n" + "=" * 50)
    print("✅ Successfully generated all deployment test datasets!")
    print("📁 Files saved in: deployment_data/")
    print("📝 Documentation: deployment_data/README.md")
    print("\n🚀 Ready to test the monitoring dashboard!")
    print("   Upload any CSV file from deployment_data/ to see drift detection in action.")

if __name__ == "__main__":
    main()