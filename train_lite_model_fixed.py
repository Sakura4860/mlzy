# -*- coding: utf-8 -*-
"""
Retrain RF Lite with FIXED features (no leakage)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json

print("="*60)
print("Retraining RF Lite with FIXED features")
print("="*60)

# Load clean data (122 features)
X_train = pd.read_csv('results/metrics/X_train_final.csv')
X_test = pd.read_csv('results/metrics/X_test_final.csv')
y_train = pd.read_csv('results/metrics/y_train.csv')['Total_Energy_kWh']
y_test = pd.read_csv('results/metrics/y_test.csv')['Total_Energy_kWh']

print(f"\nOriginal data shape:")
print(f"  Train: {X_train.shape}")
print(f"  Test:  {X_test.shape}")

# Load new lite features
with open('results/models/lite_features_fixed.txt', 'r') as f:
    lite_features = [line.strip() for line in f if line.strip()]

print(f"\nLite features ({len(lite_features)}):")
for i, feat in enumerate(lite_features, 1):
    print(f"{i:2}. {feat}")

# Extract lite features
X_train_lite = X_train[lite_features]
X_test_lite = X_test[lite_features]

print(f"\nLite data shape:")
print(f"  Train: {X_train_lite.shape}")
print(f"  Test:  {X_test_lite.shape}")

# Train Random Forest Lite
print("\n" + "="*60)
print("Training Random Forest (Lite, FIXED)")
print("="*60)

rf_lite = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_lite.fit(X_train_lite, y_train)
y_pred = rf_lite.predict(X_test_lite)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\nRF Lite Performance (FIXED - No Leakage):")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R2:   {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# Save model
model_path = 'results/models/random_forest_lite_fixed.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_lite, f)
print(f"\nModel saved: {model_path}")

# Compare with original lite (with leakage)
print("\n" + "="*60)
print("Comparison: Lite with Leakage vs Fixed Lite")
print("="*60)

comparison = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2', 'MAPE'],
    'Lite with Leakage': [0.4651, 0.2330, 0.9977, 1.00],
    'Lite FIXED (No Leakage)': [rmse, mae, r2, mape]
})

comparison['Difference'] = (comparison['Lite FIXED (No Leakage)'] - 
                            comparison['Lite with Leakage'])

print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv('results/metrics/lite_model_before_after_fix.csv', index=False)
print("\nSaved: results/metrics/lite_model_before_after_fix.csv")

# Feature importance for lite model
feat_imp_lite = pd.DataFrame({
    'feature': lite_features,
    'importance': rf_lite.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n" + "="*60)
print("Feature Importance (Lite FIXED)")
print("="*60)
print(feat_imp_lite.to_string(index=False))

feat_imp_lite.to_csv('results/metrics/feature_importance_lite_fixed.csv', index=False)

# Compare with full model (122 features)
print("\n" + "="*60)
print("Comparison: Full Model vs Lite FIXED")
print("="*60)

# Load full model metrics
full_metrics = pd.read_csv('results/metrics/model_comparison_final.csv', index_col=0)
rf_full_r2 = full_metrics.loc['RF', 'R2']

print(f"\nRandom Forest Comparison:")
print(f"  Full Model (122 features): R2 = {rf_full_r2:.4f}")
print(f"  Lite FIXED (12 features):  R2 = {r2:.4f}")
print(f"  Performance drop: {(rf_full_r2 - r2)*100:.2f}%")
print(f"  Feature reduction: {(122-12)/122*100:.1f}% fewer features")

print("\n" + "="*60)
print("✓ RF Lite retrained with CLEAN features (no leakage)")
print("="*60)
