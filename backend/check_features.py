import sys
sys.path.insert(0, 'backend')

from feature_engineering import FEATURE_GROUPS

print("="*60)
print("FEATURE ENGINEERING CONFIGURATION")
print("="*60)

print(f"\nFeature groups:")
print(f"  MinMax features: {len(FEATURE_GROUPS['minmax_features'])}")
print(f"    {FEATURE_GROUPS['minmax_features']}")

print(f"\n  Categorical features: {len(FEATURE_GROUPS['categorical_features'])}")
print(f"    {FEATURE_GROUPS['categorical_features']}")

print(f"\n  Standard features: {len(FEATURE_GROUPS['standard_features'])}")
print(f"    {FEATURE_GROUPS['standard_features'][:10]}... (showing first 10)")

total = (len(FEATURE_GROUPS['minmax_features']) + 
         len(FEATURE_GROUPS['categorical_features']) + 
         len(FEATURE_GROUPS['standard_features']))
print(f"\n  Total expected features: {total}")

print("\n" + "="*60)
