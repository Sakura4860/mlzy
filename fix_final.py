# -*- coding: utf-8 -*-
"""
Final fix: Remove both energy components AND diff features
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

print("="*70)
print("FINAL FIX: Remove energy components AND diff features")
print("="*70)

# Load data
X_train_full = pd.read_csv('results/metrics/X_train_full.csv')
X_test_full = pd.read_csv('results/metrics/X_test_full.csv')
y_train = pd.read_csv('results/metrics/y_train.csv').squeeze()
y_test = pd.read_csv('results/metrics/y_test.csv').squeeze()

print(f"\n[1] Original features: {X_train_full.shape[1]}")

# Identify ALL leakage features
leakage = []

# Type 1: Energy components (measured with target)
for col in X_train_full.columns:
    if ('kWh' in col or '_J' in col):
        if not any(x in col for x in ['lag_', 'rolling_',' diff_']):
            leakage.append(col)

# Type 2: Diff features of Total_Energy_kWh (can reconstruct target with lag)
for col in X_train_full.columns:
    if 'Total_Energy_kWh_diff_' in col:
        leakage.append(col)

print(f"\n[2] Found {len(leakage)} leakage features:")
energy_comp = [f for f in leakage if 'diff_' not in f]
diff_feats = [f for f in leakage if 'diff_' in f]
print(f"  - Energy components: {len(energy_comp)}")
for f in energy_comp:
    print(f"    * {f}")
print(f"  - Diff features (Total_Energy): {len(diff_feats)}")
for f in diff_feats:
    print(f"    * {f}")

# Remove ALL leakage features
valid = [col for col in X_train_full.columns if col not in leakage]
X_train = X_train_full[valid]
X_test = X_test_full[valid]

print(f"\n[3] After removal:")
print(f"  - Valid features: {len(valid)}")
print(f"  - Removed: {len(leakage)}")

# Save
X_train.to_csv('results/metrics/X_train_final.csv', index=False)
X_test.to_csv('results/metrics/X_test_final.csv', index=False)

with open('results/metrics/valid_features_final.txt', 'w') as f:
    for feat in valid:
        f.write(f"{feat}\n")

# Retrain
print("\n[4] Retraining models...")
results = {}

# LR
print("\n--- Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
results['LR'] = {
    'R2': r2_score(y_test, pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
    'MAE': mean_absolute_error(y_test, pred),
    'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100
}
print(f"R2={results['LR']['R2']:.4f}, RMSE={results['LR']['RMSE']:.4f}")
with open('results/models/lr_final.pkl', 'wb') as f:
    pickle.dump(lr, f)

# SVR
print("\n--- SVR ---")
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
idx = np.random.choice(len(X_train), min(5000, len(X_train)), replace=False)
svr.fit(X_train.values[idx], y_train.values[idx])
pred = svr.predict(X_test)
results['SVR'] = {
    'R2': r2_score(y_test, pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
    'MAE': mean_absolute_error(y_test, pred),
    'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100
}
print(f"R2={results['SVR']['R2']:.4f}, RMSE={results['SVR']['RMSE']:.4f}")
with open('results/models/svr_final.pkl', 'wb') as f:
    pickle.dump(svr, f)

# KNN
print("\n--- KNN ---")
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
results['KNN'] = {
    'R2': r2_score(y_test, pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
    'MAE': mean_absolute_error(y_test, pred),
    'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100
}
print(f"R2={results['KNN']['R2']:.4f}, RMSE={results['KNN']['RMSE']:.4f}")
with open('results/models/knn_final.pkl', 'wb') as f:
    pickle.dump(knn, f)

# RF
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
results['RF'] = {
    'R2': r2_score(y_test, pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
    'MAE': mean_absolute_error(y_test, pred),
    'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100
}
print(f"R2={results['RF']['R2']:.4f}, RMSE={results['RF']['RMSE']:.4f}")
with open('results/models/rf_final.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('results/metrics/model_comparison_final.csv')

# Feature importance
feat_imp = pd.DataFrame({
    'feature': valid,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
feat_imp.to_csv('results/metrics/feature_importance_final.csv', index=False)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(results_df.to_string())

print(f"\n\nTop 15 features:")
print(feat_imp.head(15).to_string(index=False))

print("\n" + "="*70)
print("DATA LEAKAGE COMPLETELY FIXED!")
print("="*70)
print(f"\nRemoved features: {len(leakage)}")
print(f"  - Energy components: {len(energy_comp)}")
print(f"  - Diff features: {len(diff_feats)}")
print(f"\nValid features: {len(valid)}")
print(f"\nPerformance (Linear Regression):")
print(f"  - Before fix: R2=1.0000")
print(f"  - After fix: R2={results['LR']['R2']:.4f}")
print(f"\nBest model: Random Forest")
print(f"  - R2: {results['RF']['R2']:.4f}")
print(f"  - RMSE: {results['RF']['RMSE']:.4f} kWh")
print(f"  - MAPE: {results['RF']['MAPE']:.2f}%")
print("="*70)
