import pandas as pd

print("="*60)
print("VERIFYING CORRECTED TARGET VARIABLES")
print("="*60)

train = pd.read_csv('backend/data/train.csv')
test = pd.read_csv('backend/data/test.csv')

print('\nTRAIN DATA:')
print(f'  Samples: {len(train):,}')
print(f'  Date range: {train["date"].min()} to {train["date"].max()}')
print(f'  Unique dates: {train["date"].nunique()}')
print(f'\n  target_next_day_return stats:')
print(train['target_next_day_return'].describe())
print(f'\n  Non-zero returns: {(train["target_next_day_return"] != 0).sum():,} / {len(train):,} ({(train["target_next_day_return"] != 0).sum() / len(train) * 100:.1f}%)')

print('\n\nTEST DATA:')
print(f'  Samples: {len(test):,}')
print(f'  Date range: {test["date"].min()} to {test["date"].max()}')
print(f'  Unique dates: {test["date"].nunique()}')
print(f'\n  target_next_day_return stats:')
print(test['target_next_day_return'].describe())
print(f'\n  Non-zero returns: {(test["target_next_day_return"] != 0).sum():,} / {len(test):,} ({(test["target_next_day_return"] != 0).sum() / len(test) * 100:.1f}%)')

print('\n\nSAMPLE ACTUAL RETURNS (first 20 unique dates):')
sample = train.groupby(['ticker', 'date'])['target_next_day_return'].first().reset_index()
sample = sample.sort_values(['ticker', 'date']).head(20)
print(sample[['ticker', 'date', 'target_next_day_return']])

print("\n" + "="*60)
if (train["target_next_day_return"] != 0).sum() / len(train) > 0.5:
    print("✅ DATA LOOKS GOOD - Ready for LSTM training!")
else:
    print("❌ WARNING - Still too many zeros!")
print("="*60)
