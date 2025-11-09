"""
LightGBM Model Training for Implied Volatility Prediction
Trains a gradient boosting model to predict option IV from 66 features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys
from datetime import datetime

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    prepare_feature_matrix,
    create_feature_scaler,
    normalize_features,
    FEATURE_GROUPS
)

print("="*70)
print(" "*15 + "LIGHTGBM IV PREDICTION - TRAINING")
print("="*70)

# ========================================
# 1. LOAD DATA
# ========================================
print("\n[1] Loading training data...")
train_df = pd.read_csv('backend/data/train.csv')
test_df = pd.read_csv('backend/data/test.csv')

print(f"  Training samples: {len(train_df):,}")
print(f"  Test samples: {len(test_df):,}")

# ========================================
# 2. PREPARE FEATURES AND TARGETS
# ========================================
print("\n[2] Preparing features and targets...")

# Define feature columns (exclude metadata and targets)
exclude_cols = ['date', 'ticker', 'strike', 'expiration', 'option_type', 
                'target_next_day_return', 'target_actual_iv', 
                'target_price_direction', 'target_iv_category']

feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"  Feature count: {len(feature_cols)}")
print(f"  Target variable: target_actual_iv (implied volatility)")

# Extract features and target
X_train = train_df[feature_cols]
y_train = train_df['target_actual_iv']

X_test = test_df[feature_cols]
y_test = test_df['target_actual_iv']

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train range: {y_train.min():.3f} to {y_train.max():.3f}")

# ========================================
# 3. NORMALIZE FEATURES
# ========================================
print("\n[3] Normalizing features...")

# Create feature matrix
train_matrix = prepare_feature_matrix(X_train.to_dict('records'), return_dataframe=True)
test_matrix = prepare_feature_matrix(X_test.to_dict('records'), return_dataframe=True)

# Create and fit scaler on training data
scaler_dict = create_feature_scaler(train_matrix)

# Normalize both sets
X_train_normalized = normalize_features(train_matrix, scaler_dict, return_dataframe=False)
X_test_normalized = normalize_features(test_matrix, scaler_dict, return_dataframe=False)

print(f"  Normalized train shape: {X_train_normalized.shape}")
print(f"  Normalized test shape: {X_test_normalized.shape}")
print(f"  Scaler saved to: backend/scalers/feature_scalers.pkl")

# ========================================
# 4. TRAIN LIGHTGBM MODEL
# ========================================
print("\n[4] Training LightGBM model...")

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_normalized, label=y_train)
test_data = lgb.Dataset(X_test_normalized, label=y_test, reference=train_data)

# LightGBM parameters optimized for IV prediction
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'min_child_samples': 20,
    'max_depth': 7,
}

print(f"  Parameters: {params}")
print(f"  Training with early stopping...")

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

print(f"  ✓ Training complete!")
print(f"  Best iteration: {model.best_iteration}")

# ========================================
# 5. EVALUATE MODEL
# ========================================
print("\n[5] Evaluating model performance...")

# Predictions
y_train_pred = model.predict(X_train_normalized, num_iteration=model.best_iteration)
y_test_pred = model.predict(X_test_normalized, num_iteration=model.best_iteration)

# Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n  TRAINING SET:")
print(f"    MAE:  {train_mae:.4f} (±{train_mae*100:.2f}% IV points)")
print(f"    RMSE: {train_rmse:.4f}")
print(f"    R²:   {train_r2:.4f}")

print(f"\n  TEST SET:")
print(f"    MAE:  {test_mae:.4f} (±{test_mae*100:.2f}% IV points)")
print(f"    RMSE: {test_rmse:.4f}")
print(f"    R²:   {test_r2:.4f}")

# Calculate mean percentage error
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
print(f"    MAPE: {mape_test:.2f}%")

# ========================================
# 6. FEATURE IMPORTANCE
# ========================================
print("\n[6] Feature importance analysis...")

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
})
importance_df = importance_df.sort_values('importance', ascending=False)

print(f"\n  Top 10 most important features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:8.0f}")

# ========================================
# 7. SAVE MODEL
# ========================================
print("\n[7] Saving model...")

# Create models directory
models_dir = 'backend/models'
os.makedirs(models_dir, exist_ok=True)

# Save LightGBM model
model_path = os.path.join(models_dir, 'lightgbm_iv_predictor.txt')
model.save_model(model_path)
print(f"  ✓ LightGBM model saved: {model_path}")

# Save as joblib for easier loading
joblib_path = os.path.join(models_dir, 'lightgbm_iv_predictor.pkl')
joblib.dump(model, joblib_path)
print(f"  ✓ Joblib model saved: {joblib_path}")

# Save feature importance
importance_path = os.path.join(models_dir, 'feature_importance_iv.csv')
importance_df.to_csv(importance_path, index=False)
print(f"  ✓ Feature importance saved: {importance_path}")

# Save metadata
metadata = {
    'model_type': 'LightGBM Regressor',
    'target': 'target_actual_iv',
    'features': feature_cols,
    'n_features': len(feature_cols),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'test_mape': float(mape_test),
    'best_iteration': model.best_iteration,
    'created_at': datetime.now().isoformat(),
    'parameters': params
}

metadata_path = os.path.join(models_dir, 'lightgbm_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"  ✓ Metadata saved: {metadata_path}")

# ========================================
# 8. SAMPLE PREDICTIONS
# ========================================
print("\n[8] Sample predictions on test set:")
print(f"\n  {'Actual IV':>10s}  {'Predicted':>10s}  {'Error':>10s}  {'% Error':>10s}")
print("  " + "-"*46)

sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_test_pred[idx]
    error = predicted - actual
    pct_error = (error / actual) * 100
    
    print(f"  {actual:10.4f}  {predicted:10.4f}  {error:10.4f}  {pct_error:9.2f}%")

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*70)
print(" "*20 + "TRAINING COMPLETE!")
print("="*70)

print(f"\n  Model Performance:")
print(f"    Test MAE: {test_mae:.4f} (±{test_mae*100:.2f}% IV points)")
print(f"    Test R²:  {test_r2:.4f}")
print(f"    Test MAPE: {mape_test:.2f}%")

print(f"\n  Files created:")
print(f"    - {model_path}")
print(f"    - {joblib_path}")
print(f"    - {importance_path}")
print(f"    - {metadata_path}")
print(f"    - backend/scalers/feature_scalers.pkl")

print(f"\n  Model ready for predictions!")
print(f"  Next step: Create /api/predict-iv endpoint in Flask")

print("\n" + "="*70)
