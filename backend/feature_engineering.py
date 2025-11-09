"""
Feature Engineering Module for Options Pricing ML Models
Handles feature matrix preparation, normalization, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os
from datetime import datetime


# Feature categorization for different scaling strategies
FEATURE_GROUPS = {
    'minmax_features': [
        'rsi',           # 0-100
        'stoch_k',       # 0-100
        'stoch_d',       # 0-100
        'bb_position',   # 0-1
        'hv_percentile', # 0-1
    ],
    
    'categorical_features': [
        'is_call',                    # 0 or 1
        'earnings_before_expiration', # 0 or 1
        'hv_regime',                  # 0 or 1
        'time_regime',                # 0, 1, or 2
        'sector_code',                # 0-10
        'vix_regime',                 # 0-12
    ],
    
    'standard_features': [
        # Prices
        'open', 'high', 'low', 'close',
        
        # Volume
        'volume', 'volume_change', 'volume_ma_20', 'volume_ratio',
        
        # Technical Indicators
        'sma_10', 'sma_20', 'sma_50',
        'ema_10', 'ema_20',
        'price_vs_sma10', 'price_vs_sma20',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr', 'atr_pct',
        'adx',
        'momentum_10', 'roc_10',
        'returns_1d', 'returns_5d', 'returns_10d',
        
        # Historical Volatility
        'hv_30', 'hv_60', 'hv_90',
        'daily_volatility',
        'hv_slope',
        'hv_acceleration',
        'hv_z_score',
        
        # Option Features
        'moneyness',
        'itm_amount', 'itm_percent',
        'otm_distance',
        'dte',
        'time_to_expiration',
        'theta_multiplier',
        'days_to_earnings',
        'earnings_risk',
        'delta', 'gamma', 'vega', 'theta', 'rho',
        'vol_skew',
        'vix_ratio',
        'vix',
        'risk_free_rate',
    ]
}


def prepare_feature_matrix(features_list, return_dataframe=True):
    """
    Convert list of feature dictionaries into a structured matrix
    
    Args:
        features_list: List of dicts (each dict is ml_features from API)
                      OR single dict for single prediction
        return_dataframe: If True, return pandas DataFrame. If False, return numpy array
    
    Returns:
        pd.DataFrame or np.ndarray: Feature matrix (n_samples, 66)
    
    Example:
        # Single prediction
        features = {...}  # From /api/ml-features-mock
        df = prepare_feature_matrix(features)
        
        # Multiple samples
        features_list = [features1, features2, ...]
        df = prepare_feature_matrix(features_list)
    """
    # Handle single dict input
    if isinstance(features_list, dict):
        features_list = [features_list]
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Ensure all expected features are present
    all_features = (FEATURE_GROUPS['minmax_features'] + 
                   FEATURE_GROUPS['categorical_features'] + 
                   FEATURE_GROUPS['standard_features'])
    
    missing_features = set(all_features) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features as NaN
        for feat in missing_features:
            df[feat] = np.nan
    
    # Sort columns alphabetically for consistency
    df = df[sorted(df.columns)]
    
    # Handle missing values
    if df.isnull().any().any():
        print("Warning: NaN values detected. Filling with 0.")
        df = df.fillna(0)
    
    print(f"Feature matrix shape: {df.shape}")
    print(f"Features: {len(df.columns)}")
    
    if return_dataframe:
        return df
    else:
        return df.values


def create_feature_scaler(training_data, save_path='backend/scalers/feature_scalers.pkl'):
    """
    Create and fit scalers on training data
    
    Args:
        training_data: pd.DataFrame with training features (n_samples, 66)
        save_path: Path to save fitted scalers
    
    Returns:
        dict: {
            'minmax_scaler': fitted MinMaxScaler,
            'standard_scaler': fitted StandardScaler,
            'feature_groups': feature group definitions,
            'feature_order': list of feature names in order,
            'created_at': timestamp
        }
    
    Example:
        # Collect training data
        training_df = prepare_feature_matrix(historical_features_list)
        
        # Fit scalers
        scaler_dict = create_feature_scaler(training_df)
        
        # Scalers are automatically saved to disk
    """
    if isinstance(training_data, np.ndarray):
        raise ValueError("training_data must be a pandas DataFrame with column names")
    
    print(f"Fitting scalers on {len(training_data)} samples...")
    
    # Separate features by group
    minmax_features = [f for f in FEATURE_GROUPS['minmax_features'] if f in training_data.columns]
    standard_features = [f for f in FEATURE_GROUPS['standard_features'] if f in training_data.columns]
    categorical_features = [f for f in FEATURE_GROUPS['categorical_features'] if f in training_data.columns]
    
    # Initialize scalers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    # Fit MinMaxScaler on bounded features
    if minmax_features:
        minmax_scaler.fit(training_data[minmax_features])
        print(f"MinMaxScaler fitted on {len(minmax_features)} features")
    
    # Fit StandardScaler on standard features
    if standard_features:
        standard_scaler.fit(training_data[standard_features])
        print(f"StandardScaler fitted on {len(standard_features)} features")
    
    # Build scaler dictionary
    scaler_dict = {
        'minmax_scaler': minmax_scaler,
        'standard_scaler': standard_scaler,
        'feature_groups': {
            'minmax_features': minmax_features,
            'standard_features': standard_features,
            'categorical_features': categorical_features
        },
        'feature_order': sorted(training_data.columns.tolist()),
        'created_at': datetime.now().isoformat(),
        'n_samples': len(training_data)
    }
    
    # Save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
    print(f"Scalers saved to: {save_path}")
    
    return scaler_dict


def load_feature_scaler(load_path='backend/scalers/feature_scalers.pkl'):
    """
    Load pre-fitted scalers from disk
    
    Args:
        load_path: Path to saved scalers
    
    Returns:
        dict: Scaler dictionary
    
    Example:
        scaler_dict = load_feature_scaler()
        normalized_df = normalize_features(features_df, scaler_dict)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Scaler file not found: {load_path}")
    
    with open(load_path, 'rb') as f:
        scaler_dict = pickle.load(f)
    
    print(f"Loaded scalers created at: {scaler_dict['created_at']}")
    print(f"Trained on {scaler_dict['n_samples']} samples")
    
    return scaler_dict


def normalize_features(features, scaler_dict, return_dataframe=True):
    """
    Normalize features using pre-fitted scalers
    
    Args:
        features: pd.DataFrame, dict, or list of dicts (raw features from API)
        scaler_dict: Fitted scalers from create_feature_scaler() or load_feature_scaler()
        return_dataframe: If True, return DataFrame. If False, return numpy array
    
    Returns:
        pd.DataFrame or np.ndarray: Normalized features (same shape as input)
    
    Example:
        # Load scalers
        scaler_dict = load_feature_scaler()
        
        # Normalize single prediction
        raw_features = {...}  # From /api/ml-features-mock
        normalized_df = normalize_features(raw_features, scaler_dict)
        
        # Normalize multiple samples
        normalized_df = normalize_features(test_df, scaler_dict)
    """
    # Convert to DataFrame if needed
    if isinstance(features, (dict, list)):
        df = prepare_feature_matrix(features, return_dataframe=True)
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise ValueError("features must be dict, list of dicts, or DataFrame")
    
    # Ensure feature order matches training
    if 'feature_order' in scaler_dict:
        missing = set(scaler_dict['feature_order']) - set(df.columns)
        if missing:
            print(f"Warning: Missing features: {missing}. Filling with 0.")
            for feat in missing:
                df[feat] = 0
        df = df[scaler_dict['feature_order']]
    
    # Get feature groups
    minmax_features = scaler_dict['feature_groups']['minmax_features']
    standard_features = scaler_dict['feature_groups']['standard_features']
    categorical_features = scaler_dict['feature_groups']['categorical_features']
    
    # Create copy for normalization
    normalized_df = df.copy()
    
    # Apply MinMaxScaler to bounded features
    if minmax_features:
        normalized_df[minmax_features] = scaler_dict['minmax_scaler'].transform(
            df[minmax_features]
        )
    
    # Apply StandardScaler to standard features
    if standard_features:
        normalized_df[standard_features] = scaler_dict['standard_scaler'].transform(
            df[standard_features]
        )
    
    # Categorical features remain unchanged
    # (they're already in normalized_df)
    
    print(f"Normalized {len(normalized_df)} samples with {len(normalized_df.columns)} features")
    
    if return_dataframe:
        return normalized_df
    else:
        return normalized_df.values


def inverse_transform_features(normalized_features, scaler_dict):
    """
    Convert normalized features back to original scale (for interpretation)
    
    Args:
        normalized_features: pd.DataFrame or np.ndarray (normalized)
        scaler_dict: Fitted scalers
    
    Returns:
        pd.DataFrame: Features in original scale
    
    Example:
        # Useful for debugging or explaining model predictions
        original_df = inverse_transform_features(normalized_df, scaler_dict)
    """
    if isinstance(normalized_features, np.ndarray):
        if 'feature_order' not in scaler_dict:
            raise ValueError("Cannot inverse transform array without feature_order")
        df = pd.DataFrame(normalized_features, columns=scaler_dict['feature_order'])
    else:
        df = normalized_features.copy()
    
    minmax_features = scaler_dict['feature_groups']['minmax_features']
    standard_features = scaler_dict['feature_groups']['standard_features']
    
    original_df = df.copy()
    
    if minmax_features:
        original_df[minmax_features] = scaler_dict['minmax_scaler'].inverse_transform(
            df[minmax_features]
        )
    
    if standard_features:
        original_df[standard_features] = scaler_dict['standard_scaler'].inverse_transform(
            df[standard_features]
        )
    
    return original_df


# Example usage and testing
if __name__ == "__main__":
    """
    Test the feature engineering pipeline with mock data
    """
    print("="*60)
    print("Testing Feature Engineering Pipeline")
    print("="*60)
    
    # Example: Create mock training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random features (this would come from your API in production)
    mock_training_data = []
    for i in range(n_samples):
        features = {
            # OHLCV
            'open': np.random.uniform(250, 280),
            'high': np.random.uniform(250, 280),
            'low': np.random.uniform(250, 280),
            'close': np.random.uniform(250, 280),
            'volume': np.random.randint(50_000_000, 150_000_000),
            
            # Technical indicators (subset for demo)
            'rsi': np.random.uniform(30, 70),
            'macd': np.random.uniform(-5, 5),
            'sma_10': np.random.uniform(250, 280),
            'stoch_k': np.random.uniform(0, 100),
            'stoch_d': np.random.uniform(0, 100),
            
            # Option features
            'delta': np.random.uniform(0.3, 0.7),
            'gamma': np.random.uniform(0.01, 0.03),
            'theta': np.random.uniform(-100, -20),
            'vega': np.random.uniform(0.2, 0.4),
            'rho': np.random.uniform(0, 0.2),
            'moneyness': np.random.uniform(0.95, 1.05),
            'dte': np.random.randint(7, 90),
            'is_call': np.random.choice([0, 1]),
            'vix_regime': np.random.randint(0, 13),
            
            # Add more features to reach 66...
            # (truncated for brevity)
        }
        mock_training_data.append(features)
    
    print(f"\n1. Created {n_samples} mock training samples")
    
    # Step 1: Prepare feature matrix
    training_df = prepare_feature_matrix(mock_training_data)
    print(f"\n2. Feature matrix created: {training_df.shape}")
    print(f"   Columns: {list(training_df.columns)[:5]}...")
    
    # Step 2: Create and fit scalers
    scaler_dict = create_feature_scaler(training_df)
    print(f"\n3. Scalers fitted and saved")
    
    # Step 3: Normalize features
    normalized_df = normalize_features(training_df, scaler_dict)
    print(f"\n4. Features normalized: {normalized_df.shape}")
    print(f"\n   Sample normalized values (first 3 features):")
    print(normalized_df.iloc[0, :3])
    
    # Step 4: Test single prediction
    single_feature = mock_training_data[0]
    single_normalized = normalize_features(single_feature, scaler_dict)
    print(f"\n5. Single prediction normalized: {single_normalized.shape}")
    
    # Step 5: Test inverse transform
    original_df = inverse_transform_features(normalized_df.iloc[:5], scaler_dict)
    print(f"\n6. Inverse transform test:")
    print(f"   Original rsi: {training_df.iloc[0]['rsi']:.2f}")
    print(f"   Normalized rsi: {normalized_df.iloc[0]['rsi']:.4f}")
    print(f"   Inverse rsi: {original_df.iloc[0]['rsi']:.2f}")
    
    print("\n" + "="*60)
    print("Feature Engineering Pipeline Test Complete!")
    print("="*60)
