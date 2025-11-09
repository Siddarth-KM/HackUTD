"""
Fill Target Variables for Training Data
Calculates actual targets for ML model training:
1. Next-day stock price return (for price prediction model)
2. Synthetic IV based on moneyness and time (for IV prediction model)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def calculate_target_variables(df):
    """
    Fill target variables in the training dataset
    
    Args:
        df: DataFrame with training data
    
    Returns:
        DataFrame with filled target variables
    """
    print("Calculating target variables...")
    print(f"Initial samples: {len(df)}")
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # ========================================
    # Target 1: Next-Day Stock Price Return
    # ========================================
    print("\n1. Calculating next-day returns...")
    
    # Create a lookup table of unique dates and prices per ticker
    # (since each date has multiple option rows with same close price)
    date_prices = df.groupby(['ticker', 'date'])['close'].first().reset_index()
    date_prices = date_prices.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # For each ticker, get the next date's close price
    date_prices['next_day_close'] = date_prices.groupby('ticker')['close'].shift(-1)
    date_prices['next_date'] = date_prices.groupby('ticker')['date'].shift(-1)
    
    # Merge back to main dataframe
    df = df.merge(
        date_prices[['ticker', 'date', 'next_day_close', 'next_date']], 
        on=['ticker', 'date'],
        how='left'
    )
    
    # Calculate return: (next_close - current_close) / current_close
    df['target_next_day_return'] = (df['next_day_close'] - df['close']) / df['close']
    
    # Remove rows without next-day data (last date for each ticker)
    before_drop = len(df)
    df = df.dropna(subset=['target_next_day_return'])
    print(f"   Dropped {before_drop - len(df)} samples without next-day data")
    print(f"   Next-day return range: {df['target_next_day_return'].min():.4f} to {df['target_next_day_return'].max():.4f}")
    print(f"   Mean return: {df['target_next_day_return'].mean():.4f}")
    
    # Clean up temporary columns
    df = df.drop(columns=['next_day_close', 'next_date'])
    
    # ========================================
    # Target 2: Implied Volatility (Synthetic)
    # ========================================
    print("\n2. Calculating synthetic implied volatility...")
    
    # For training purposes, we'll create synthetic IV based on real relationships:
    # - IV increases with distance from ATM (volatility smile)
    # - IV decreases as expiration approaches (term structure)
    # - IV correlates with historical volatility
    # - IV has randomness to simulate market conditions
    
    # Base IV from historical volatility
    base_iv = df['hv_30'].values
    
    # Moneyness effect (volatility smile)
    # OTM options have higher IV than ATM
    moneyness_distance = np.abs(df['moneyness'] - 1.0)  # Distance from ATM
    smile_factor = 1.0 + (moneyness_distance * 0.5)  # 50% increase for 10% OTM
    
    # Time effect (term structure)
    # Longer DTE generally has lower IV (mean reversion)
    time_factor = 1.0 - (df['dte'] / 365) * 0.1  # -10% IV for 1 year out
    time_factor = np.clip(time_factor, 0.7, 1.0)  # Cap the adjustment
    
    # VIX effect (market volatility regime)
    # Higher VIX = higher IV across all options
    vix_factor = df['vix'] / 20.0  # Normalize around VIX=20
    
    # Put-Call skew (puts tend to have higher IV)
    skew_factor = np.where(df['is_call'] == 0, 1.05, 1.0)  # Puts 5% higher
    
    # Combine all factors
    synthetic_iv = base_iv * smile_factor * time_factor * vix_factor * skew_factor
    
    # Add realistic noise (±10% random variation)
    np.random.seed(42)  # Reproducible
    noise = np.random.normal(1.0, 0.1, len(df))
    synthetic_iv = synthetic_iv * noise
    
    # Clip to reasonable range (5% to 150%)
    synthetic_iv = np.clip(synthetic_iv, 0.05, 1.5)
    
    df['target_actual_iv'] = synthetic_iv
    
    print(f"   IV range: {df['target_actual_iv'].min():.4f} to {df['target_actual_iv'].max():.4f}")
    print(f"   Mean IV: {df['target_actual_iv'].mean():.4f}")
    print(f"   Median IV: {df['target_actual_iv'].median():.4f}")
    
    # ========================================
    # Create Classification Targets
    # ========================================
    print("\n3. Creating classification targets...")
    
    # Price direction (for classification model)
    # 0 = down, 1 = neutral, 2 = up
    df['target_price_direction'] = np.where(
        df['target_next_day_return'] < -0.005, 0,  # Down > 0.5%
        np.where(df['target_next_day_return'] > 0.005, 2, 1)  # Up > 0.5%, else neutral
    )
    
    direction_counts = df['target_price_direction'].value_counts().sort_index()
    print(f"   Price direction distribution:")
    print(f"     Down (0): {direction_counts.get(0, 0)} samples")
    print(f"     Neutral (1): {direction_counts.get(1, 0)} samples")
    print(f"     Up (2): {direction_counts.get(2, 0)} samples")
    
    # IV category (for binning)
    # 0 = low (<20%), 1 = medium (20-40%), 2 = high (>40%)
    df['target_iv_category'] = np.where(
        df['target_actual_iv'] < 0.20, 0,
        np.where(df['target_actual_iv'] < 0.40, 1, 2)
    )
    
    iv_counts = df['target_iv_category'].value_counts().sort_index()
    print(f"   IV category distribution:")
    print(f"     Low (0): {iv_counts.get(0, 0)} samples")
    print(f"     Medium (1): {iv_counts.get(1, 0)} samples")
    print(f"     High (2): {iv_counts.get(2, 0)} samples")
    
    print(f"\nFinal samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    return df


def create_train_test_split(df, test_size=0.2):
    """
    Split data into train and test sets
    Uses time-based split (last 20% of dates for testing)
    
    Args:
        df: DataFrame with filled targets
        test_size: Fraction for test set (default 0.2)
    
    Returns:
        train_df, test_df
    """
    print("\n" + "="*60)
    print("Creating train/test split...")
    print("="*60)
    
    # Get unique dates and sort
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    
    # Split point
    split_idx = int(n_dates * (1 - test_size))
    split_date = unique_dates[split_idx]
    
    # Split data
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nSplit date: {split_date}")
    print(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, test_df


def main():
    """
    Main execution: Load data, calculate targets, split, and save
    """
    print("="*60)
    print("Training Data Target Variable Preparation")
    print("="*60)
    
    # Load raw training data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    input_file = os.path.join(data_dir, 'training_data.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Run collect_training_data.py first")
        return
    
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Calculate target variables
    df_with_targets = calculate_target_variables(df)
    
    # Save full dataset with targets
    output_file = os.path.join(data_dir, 'training_data_with_targets.csv')
    df_with_targets.to_csv(output_file, index=False)
    print(f"\n✓ Saved complete dataset to: {output_file}")
    print(f"  Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df_with_targets)
    
    # Save train and test sets
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✓ Saved training set to: {train_file}")
    print(f"  Size: {os.path.getsize(train_file) / 1024 / 1024:.2f} MB")
    print(f"\n✓ Saved test set to: {test_file}")
    print(f"  Size: {os.path.getsize(test_file) / 1024 / 1024:.2f} MB")
    
    # Show sample statistics
    print("\n" + "="*60)
    print("Sample Statistics")
    print("="*60)
    
    print("\nFeature columns available for ML:")
    feature_cols = [col for col in df_with_targets.columns 
                   if col not in ['date', 'ticker', 'strike', 'expiration', 'option_type', 
                                  'target_next_day_return', 'target_actual_iv', 
                                  'target_price_direction', 'target_iv_category']]
    print(f"  {len(feature_cols)} features: {', '.join(feature_cols[:10])}...")
    
    print("\nTarget variables:")
    print(f"  target_next_day_return (regression)")
    print(f"  target_actual_iv (regression)")
    print(f"  target_price_direction (classification)")
    print(f"  target_iv_category (classification)")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nReady for ML model training!")
    print("\nNext steps:")
    print("  1. train_df = pd.read_csv('backend/data/train.csv')")
    print("  2. Normalize features using feature_engineering.py")
    print("  3. Train LightGBM for IV prediction")
    print("  4. Train LSTM for price movement prediction")


if __name__ == "__main__":
    main()
