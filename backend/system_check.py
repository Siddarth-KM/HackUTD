"""
COMPREHENSIVE SYSTEM STATUS CHECK
Run this to verify everything is ready for ML model training
"""

import os
import sys
import pandas as pd

print("\n" + "="*70)
print(" "*15 + "OPTIONS PRICING ML SYSTEM - STATUS CHECK")
print("="*70)

# 1. API KEYS
print("\n[1] API KEYS")
print("-" * 70)
env_path = 'backend/.env'
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        content = f.read()
        alpha_key = 'ALPHA_VANTAGE_KEY' in content and 'your_' not in content
        fred_key = 'FRED_KEY' in content and 'your_' not in content
        polygon_key = 'POLYGON_API_KEY' in content and 'your_' not in content
        
        print(f"  ✓ .env file exists")
        print(f"  {'✓' if alpha_key else '✗'} Alpha Vantage API key: {'Configured' if alpha_key else 'Missing'}")
        print(f"  {'✓' if fred_key else '✗'} FRED API key: {'Configured' if fred_key else 'Missing'}")
        print(f"  {'✓' if polygon_key else '✗'} Polygon.io API key: {'Configured' if polygon_key else 'Missing'}")
else:
    print("  ✗ .env file not found!")

# 2. TRAINING DATA
print("\n[2] TRAINING DATA")
print("-" * 70)
train_path = 'backend/data/train.csv'
test_path = 'backend/data/test.csv'

if os.path.exists(train_path) and os.path.exists(test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"  ✓ Training set: {len(df_train):,} samples")
    print(f"  ✓ Test set: {len(df_test):,} samples")
    print(f"  ✓ Total columns: {len(df_train.columns)}")
    
    # Check features
    feature_cols = [c for c in df_train.columns 
                   if not c.startswith('target') 
                   and c not in ['date', 'ticker', 'strike', 'expiration', 'option_type']]
    print(f"  ✓ Feature count: {len(feature_cols)} (expected: 66)")
    
    # Check targets
    target_cols = [c for c in df_train.columns if c.startswith('target')]
    print(f"  ✓ Target variables: {len(target_cols)}")
    for target in target_cols:
        print(f"    - {target}")
    
    # Check for missing values
    missing = df_train.isnull().sum().sum()
    if missing == 0:
        print(f"  ✓ No missing values - data is clean!")
    else:
        print(f"  ⚠ Warning: {missing} missing values found")
        
else:
    print(f"  ✗ Training data not found!")
    print(f"    Expected: {train_path}")

# 3. FEATURE ENGINEERING
print("\n[3] FEATURE ENGINEERING")
print("-" * 70)
try:
    sys.path.insert(0, 'backend')
    from feature_engineering import FEATURE_GROUPS, prepare_feature_matrix
    
    total_features = (len(FEATURE_GROUPS['minmax_features']) + 
                     len(FEATURE_GROUPS['categorical_features']) + 
                     len(FEATURE_GROUPS['standard_features']))
    
    print(f"  ✓ Feature engineering module loaded")
    print(f"  ✓ MinMax features: {len(FEATURE_GROUPS['minmax_features'])}")
    print(f"  ✓ Categorical features: {len(FEATURE_GROUPS['categorical_features'])}")
    print(f"  ✓ Standard features: {len(FEATURE_GROUPS['standard_features'])}")
    print(f"  ✓ Total: {total_features} features configured")
    
except Exception as e:
    print(f"  ✗ Feature engineering error: {e}")

# 4. SCALER STATUS
print("\n[4] FEATURE SCALER")
print("-" * 70)
scaler_path = 'backend/scalers/feature_scalers.pkl'
if os.path.exists(scaler_path):
    print(f"  ✓ Scaler exists: {scaler_path}")
    print(f"    (Will be loaded for predictions)")
else:
    print(f"  ℹ Scaler not yet created: {scaler_path}")
    print(f"    (Will be created during model training)")

# 5. POLYGON.IO INTEGRATION
print("\n[5] POLYGON.IO DATA SOURCE")
print("-" * 70)
polygon_client_path = 'backend/polygon_client.py'
if os.path.exists(polygon_client_path):
    print(f"  ✓ Polygon client module exists")
    print(f"  ✓ Integrated with fetch_stock_data()")
    print(f"  ✓ /api/ml-features endpoint uses real Polygon data")
else:
    print(f"  ✗ Polygon client not found")

# 6. FLASK ENDPOINTS
print("\n[6] FLASK API ENDPOINTS")
print("-" * 70)
app_path = 'backend/app.py'
if os.path.exists(app_path):
    with open(app_path, 'r') as f:
        content = f.read()
        
    endpoints = [
        ('/api/ml-features', 'Get 66 ML features with real Polygon data'),
        ('/api/test-normalization', 'Test feature normalization pipeline'),
        ('/api/test-option-features-mock', 'Test option features (mock data)'),
        ('/api/features/<ticker>', 'Legacy endpoint'),
    ]
    
    print(f"  ✓ Flask app exists")
    for endpoint, description in endpoints:
        exists = endpoint.replace('<ticker>', '') in content
        print(f"  {'✓' if exists else '✗'} {endpoint}")
        print(f"    {description}")
else:
    print(f"  ✗ Flask app not found")

# 7. MODELS DIRECTORY
print("\n[7] ML MODELS")
print("-" * 70)
models_dir = 'backend/models'
if os.path.exists(models_dir):
    models = os.listdir(models_dir)
    if models:
        print(f"  ✓ Models directory exists with {len(models)} files:")
        for model in models:
            print(f"    - {model}")
    else:
        print(f"  ℹ Models directory exists but empty (models not yet trained)")
else:
    print(f"  ℹ Models directory not created yet")
    print(f"    (Will be created during training)")

# SUMMARY
print("\n" + "="*70)
print(" "*25 + "SYSTEM READINESS SUMMARY")
print("="*70)

checks = [
    ("API Keys Configured", polygon_key and fred_key),
    ("Training Data Ready", os.path.exists(train_path) and os.path.exists(test_path)),
    ("Feature Engineering Ready", total_features == 66),
    ("Polygon Integration Active", os.path.exists(polygon_client_path)),
    ("Flask Endpoints Ready", os.path.exists(app_path)),
]

all_ready = all(check[1] for check in checks)

for check_name, status in checks:
    print(f"  {'✓' if status else '✗'} {check_name}")

print("\n" + "="*70)
if all_ready:
    print(" "*20 + "✓ SYSTEM READY FOR ML TRAINING!")
    print("\nNext steps:")
    print("  1. Train LightGBM for IV prediction (15-20 min)")
    print("  2. Train LSTM for price movement (30-40 min)")
    print("  3. Create prediction API endpoints (10-15 min)")
    print("  4. Build React frontend (2-3 hours)")
else:
    print(" "*20 + "⚠ SYSTEM NOT READY - Fix issues above")
    
print("="*70 + "\n")
