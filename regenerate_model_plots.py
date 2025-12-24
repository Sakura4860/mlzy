# -*- coding: utf-8 -*-
"""
Regenerate prediction and residual plots with FIXED models (no leakage)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig_dir = Path('results/figures')
models_dir = Path('results/models')
metrics_dir = Path('results/metrics')

print("="*60)
print("Regenerating Prediction & Residual Plots (FIXED Models)")
print("="*60)

# Load clean data
print("\nLoading clean data (122 features, no leakage)...")
X_test = pd.read_csv(metrics_dir / 'X_test_final.csv')
y_test = pd.read_csv(metrics_dir / 'y_test.csv')['Total_Energy_kWh'].values

print(f"  Test set: {X_test.shape}")
print(f"  Features: 122 (no leakage)")

# Load lite features for RF Lite
with open(models_dir / 'lite_features_fixed.txt', 'r') as f:
    lite_features = [line.strip() for line in f if line.strip()]
X_test_lite = X_test[lite_features]

print(f"  Lite features: {len(lite_features)}")

# Models to visualize
models_to_plot = {
    'Linear Regression': (models_dir / 'lr_final.pkl', X_test, 'LR'),
    'KNN': (models_dir / 'knn_final.pkl', X_test, 'KNN'),
    'SVR': (models_dir / 'svr_final.pkl', X_test, 'SVR'),
    'Random Forest': (models_dir / 'rf_final.pkl', X_test, 'RF Full (122 feat)'),
    'RF Lite Fixed': (models_dir / 'random_forest_lite_fixed.pkl', X_test_lite, 'RF Lite (12 feat)'),
}

# Generate plots for each model
for model_name, (model_path, X_data, display_name) in models_to_plot.items():
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    y_pred = model.predict(X_data)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # ========================================
    # 1. Prediction Plot
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot subset for clarity (first 200 points)
    n_points = min(200, len(y_test))
    indices = range(n_points)
    
    ax.plot(indices, y_test[:n_points], 'o-', label='Actual', 
            color='#2E86AB', alpha=0.7, markersize=4, linewidth=1.5)
    ax.plot(indices, y_pred[:n_points], 's--', label='Predicted', 
            color='#A23B72', alpha=0.7, markersize=4, linewidth=1.5)
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Energy (kWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'{display_name} - Predictions vs Actual (FIXED, No Leakage)\n'
                 f'R²={r2:.4f}, RMSE={rmse:.2f} kWh, MAPE={mape:.2f}%',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add data leakage fixed annotation
    ax.text(0.98, 0.02, '✓ Data Leakage Fixed', 
            transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save with original filename format for compatibility
    save_name = model_name if model_name != 'RF Lite Fixed' else 'RF_Lite_Fixed'
    plt.savefig(fig_dir / f'{save_name}_predictions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_name}_predictions.png")
    plt.close()
    
    # ========================================
    # 2. Residual Plot
    # ========================================
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{display_name} - Residual Analysis (FIXED, No Leakage)',
                 fontsize=14, fontweight='bold')
    
    # Residual scatter
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='#F18F01')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Predicted Values (kWh)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals (kWh)', fontsize=11, fontweight='bold')
    ax.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add metrics text
    ax.text(0.05, 0.95, f'Mean Error: {np.mean(residuals):.4f}\n'
                        f'Std Error: {np.std(residuals):.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=50, color='#C73E1D', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Residuals (kWh)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add normal distribution overlay
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax2 = ax.twinx()
    ax2.plot(x, p, 'k--', linewidth=2, label='Normal Fit')
    ax2.set_ylabel('Probability Density', fontsize=10)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'{save_name}_residuals.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_name}_residuals.png")
    plt.close()

# ========================================
# Generate combined comparison plot
# ========================================
print(f"\n{'='*60}")
print("Creating Combined Model Comparison Plot")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Predictions Comparison (FIXED - No Data Leakage)', 
             fontsize=16, fontweight='bold')

plot_idx = 0
n_points = min(150, len(y_test))
indices = range(n_points)

for model_name, (model_path, X_data, display_name) in models_to_plot.items():
    ax = axes[plot_idx // 3, plot_idx % 3]
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_data)
    
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    ax.plot(indices, y_test[:n_points], 'o-', label='Actual', 
            color='#2E86AB', alpha=0.6, markersize=3, linewidth=1)
    ax.plot(indices, y_pred[:n_points], 's--', label='Predicted', 
            color='#A23B72', alpha=0.6, markersize=3, linewidth=1)
    
    ax.set_title(f'{display_name}\nR²={r2:.4f}, MAPE={mape:.2f}%',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample', fontsize=9)
    ax.set_ylabel('Energy (kWh)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1

# Remove empty subplot
if plot_idx < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig(fig_dir / 'all_models_predictions_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: all_models_predictions_comparison.png")
plt.close()

print("\n" + "="*60)
print("✓ All prediction and residual plots regenerated!")
print("="*60)
print("\nGenerated Files:")
print("  Predictions:")
for model_name in models_to_plot.keys():
    save_name = model_name if model_name != 'RF Lite Fixed' else 'RF_Lite_Fixed'
    print(f"    - {save_name}_predictions.png")
print("\n  Residuals:")
for model_name in models_to_plot.keys():
    save_name = model_name if model_name != 'RF Lite Fixed' else 'RF_Lite_Fixed'
    print(f"    - {save_name}_residuals.png")
print("\n  Combined:")
print("    - all_models_predictions_comparison.png")
print("="*60)
