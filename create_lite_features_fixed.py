# -*- coding: utf-8 -*-
"""
Create new lite features from fixed features (no leakage)
"""
import pandas as pd

# Load feature importance from fixed model
feat_imp = pd.read_csv('results/metrics/feature_importance_final.csv')

print("="*60)
print("Creating NEW Lite Features (No Leakage)")
print("="*60)

# Top features by importance
print("\nTop 15 features from fixed RF model:")
print(feat_imp.head(15).to_string(index=False))

# Select lite features - focus on most important + diversity
lite_features = [
    # Top lag features
    'Total_Energy_kWh_lag_1',      # 89.89% importance
    'Total_Energy_kWh_lag_24',     # 1.30% importance
    'Total_Energy_kWh_lag_2',      # 0.10% importance
    
    # Time features
    'hour',                         # 2.66% importance
    'hour_sin',                     # 0.94% importance
    'hour_cos',                     # 0.21% importance
    'day_of_week',
    'is_weekend',
    
    # Rolling features (best performing)
    'Total_Energy_kWh_rolling_std_24',   # 0.26% importance
    'Total_Energy_kWh_rolling_mean_24',  # 0.19% importance
    
    # Weather features (top 2)
    'Temperature',
    'GHI',  # Global Horizontal Irradiance
]

print(f"\n\nSelected {len(lite_features)} lite features:")
for i, feat in enumerate(lite_features, 1):
    imp = feat_imp[feat_imp['feature'] == feat]['importance'].values
    imp_val = imp[0] if len(imp) > 0 else 0
    print(f"{i:2}. {feat:<40} ({imp_val*100:6.3f}%)")

# Calculate total importance coverage
total_imp = feat_imp[feat_imp['feature'].isin(lite_features)]['importance'].sum()
print(f"\nTotal importance coverage: {total_imp*100:.2f}%")

# Save new lite features
with open('results/models/lite_features_fixed.txt', 'w') as f:
    for feat in lite_features:
        f.write(f"{feat}\n")

print(f"\nSaved to: results/models/lite_features_fixed.txt")

# Verify no leakage features
leakage_keywords = ['Electricity_kWh', 'HotWater', 'SpaceCooling', 'SpaceHeating', 
                     '_J', 'diff_']
has_leakage = False
for feat in lite_features:
    for keyword in leakage_keywords:
        if keyword in feat and 'lag_' not in feat and 'rolling_' not in feat:
            print(f"\nWARNING: Potential leakage in {feat}")
            has_leakage = True

if not has_leakage:
    print("\n✓ No leakage features detected in lite set")

print("="*60)
