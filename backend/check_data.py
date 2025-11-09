import pandas as pd

# Check training data
df_train = pd.read_csv('backend/data/train.csv')
df_test = pd.read_csv('backend/data/test.csv')

print("="*60)
print("TRAINING DATA VERIFICATION")
print("="*60)

print(f"\nTrain set: {len(df_train)} samples, {len(df_train.columns)} columns")
print(f"Test set: {len(df_test)} samples")

# Feature columns
feature_cols = [c for c in df_train.columns 
                if not c.startswith('target') 
                and c not in ['date', 'ticker', 'strike', 'expiration', 'option_type']]

print(f"\nFeature count: {len(feature_cols)}")
print(f"First 10 features: {feature_cols[:10]}")

# Target columns
target_cols = [c for c in df_train.columns if c.startswith('target')]
print(f"\nTarget variables: {target_cols}")

# Target statistics
print(f"\ntarget_actual_iv:")
print(f"  Range: {df_train['target_actual_iv'].min():.3f} to {df_train['target_actual_iv'].max():.3f}")
print(f"  Mean: {df_train['target_actual_iv'].mean():.3f}")

print(f"\ntarget_next_day_return:")
print(f"  Range: {df_train['target_next_day_return'].min():.4f} to {df_train['target_next_day_return'].max():.4f}")
print(f"  Mean: {df_train['target_next_day_return'].mean():.4f}")

# Check for missing values
print(f"\nMissing values:")
missing = df_train.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("  None - data is clean!")

print("\n" + "="*60)
