# -*- coding: utf-8 -*-
"""
Update visualizations with fixed data (no leakage)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figures directory
fig_dir = Path('results/figures')
fig_dir.mkdir(exist_ok=True, parents=True)

print("="*60)
print("Updating Visualizations (Data Leakage Fixed)")
print("="*60)

# ============================================================
# 1. Model Comparison Chart (with RF Lite Fixed)
# ============================================================
print("\n1. Creating Model Comparison Chart...")

models = {
    'RF Lite\n(Fixed)': {'R2': 0.9338, 'RMSE': 2.4864, 'MAE': 1.7149, 'MAPE': 7.31},
    'RF Full\n(122 feat)': {'R2': 0.9264, 'RMSE': 2.6214, 'MAE': 1.8523, 'MAPE': 8.15},
    'Improved\nLSTM': {'R2': 0.8895, 'RMSE': 3.1847, 'MAE': 2.3494, 'MAPE': 9.99},
    'Linear\nRegression': {'R2': 0.8739, 'RMSE': 3.4312, 'MAE': 2.5027, 'MAPE': 10.40},
    'Baseline\nLSTM': {'R2': 0.7954, 'RMSE': 4.3555, 'MAE': 3.3094, 'MAPE': 13.43},
    'SVR': {'R2': 0.7100, 'RMSE': 5.2039, 'MAE': 4.0406, 'MAPE': 15.68},
    'KNN': {'R2': 0.6773, 'RMSE': 5.4894, 'MAE': 3.8724, 'MAPE': 16.66},
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison (Data Leakage Fixed)', 
             fontsize=16, fontweight='bold')

# R² Score
ax = axes[0, 0]
r2_values = [m['R2'] for m in models.values()]
colors = ['#FF6B6B' if v == max(r2_values) else '#4ECDC4' if v == sorted(r2_values)[-2] else '#95E1D3' 
          for v in r2_values]
bars = ax.barh(list(models.keys()), r2_values, color=colors)
ax.set_xlabel('R² Score', fontweight='bold')
ax.set_title('R² Score (Higher is Better)', fontweight='bold')
ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.3, label='Excellent (0.9)')
ax.legend()
for i, v in enumerate(r2_values):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# RMSE
ax = axes[0, 1]
rmse_values = [m['RMSE'] for m in models.values()]
colors = ['#FF6B6B' if v == min(rmse_values) else '#4ECDC4' if v == sorted(rmse_values)[1] else '#95E1D3' 
          for v in rmse_values]
bars = ax.barh(list(models.keys()), rmse_values, color=colors)
ax.set_xlabel('RMSE (kWh)', fontweight='bold')
ax.set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
for i, v in enumerate(rmse_values):
    ax.text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')

# MAE
ax = axes[1, 0]
mae_values = [m['MAE'] for m in models.values()]
colors = ['#FF6B6B' if v == min(mae_values) else '#4ECDC4' if v == sorted(mae_values)[1] else '#95E1D3' 
          for v in mae_values]
bars = ax.barh(list(models.keys()), mae_values, color=colors)
ax.set_xlabel('MAE (kWh)', fontweight='bold')
ax.set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
for i, v in enumerate(mae_values):
    ax.text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')

# MAPE
ax = axes[1, 1]
mape_values = [m['MAPE'] for m in models.values()]
colors = ['#FF6B6B' if v == min(mape_values) else '#4ECDC4' if v == sorted(mape_values)[1] else '#95E1D3' 
          for v in mape_values]
bars = ax.barh(list(models.keys()), mape_values, color=colors)
ax.set_xlabel('MAPE (%)', fontweight='bold')
ax.set_title('Mean Absolute Percentage Error (Lower is Better)', fontweight='bold')
ax.axvline(x=5, color='green', linestyle='--', alpha=0.3, label='Excellent (<5%)')
ax.axvline(x=10, color='orange', linestyle='--', alpha=0.3, label='Good (<10%)')
ax.legend()
for i, v in enumerate(mape_values):
    ax.text(v + 0.3, i, f'{v:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'model_comparison_fixed.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_dir / 'model_comparison_fixed.png'}")
plt.close()

# ============================================================
# 2. Feature Importance (Lite Fixed)
# ============================================================
print("\n2. Creating Feature Importance Chart (Lite Fixed)...")

feat_imp = pd.read_csv('results/metrics/feature_importance_lite_fixed.csv')
feat_imp_sorted = feat_imp.sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp_sorted)))
bars = ax.barh(feat_imp_sorted['feature'], feat_imp_sorted['importance'] * 100, color=colors)

ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - RF Lite Fixed (12 Features, No Leakage)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add values
for i, (feat, imp) in enumerate(zip(feat_imp_sorted['feature'], feat_imp_sorted['importance'])):
    ax.text(imp * 100 + 1, i, f'{imp*100:.2f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / 'feature_importance_lite_fixed.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_dir / 'feature_importance_lite_fixed.png'}")
plt.close()

# ============================================================
# 3. Data Leakage Fix Comparison
# ============================================================
print("\n3. Creating Data Leakage Fix Comparison...")

fix_comparison = pd.read_csv('results/metrics/lite_model_before_after_fix.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RF Lite: Before vs After Data Leakage Fix', 
             fontsize=16, fontweight='bold')

metrics = fix_comparison['Metric'].tolist()
before = fix_comparison['Lite with Leakage'].tolist()
after = fix_comparison['Lite FIXED (No Leakage)'].tolist()

# Before vs After bars
ax = axes[0]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, before, width, label='With Leakage (Invalid)', 
               color='#FF6B6B', alpha=0.7)
bars2 = ax.bar(x + width/2, after, width, label='Fixed (Valid)', 
               color='#4ECDC4', alpha=0.7)

ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Metric Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# Percentage change
ax = axes[1]
pct_change = [(after[i] - before[i]) / before[i] * 100 if before[i] != 0 else 0 
              for i in range(len(metrics))]
colors_pct = ['green' if x < 0 else 'red' for x in pct_change]
bars = ax.barh(metrics, pct_change, color=colors_pct, alpha=0.7)

ax.set_xlabel('Change (%)', fontweight='bold')
ax.set_title('Performance Change After Fix', fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(pct_change):
    ax.text(v + 10 if v > 0 else v - 10, i, f'{v:+.1f}%', 
            va='center', ha='left' if v > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'data_leakage_fix_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_dir / 'data_leakage_fix_comparison.png'}")
plt.close()

# ============================================================
# 4. Feature Count vs Performance
# ============================================================
print("\n4. Creating Feature Count vs Performance Chart...")

feature_performance = {
    '12 feat\n(Lite Fixed)': {'features': 12, 'r2': 0.9338, 'mape': 7.31, 'label': 'Best!'},
    '122 feat\n(Full)': {'features': 122, 'r2': 0.9264, 'mape': 8.15, 'label': ''},
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, data in feature_performance.items():
    size = 500 if 'Best' in data['label'] else 300
    marker = '*' if 'Best' in data['label'] else 'o'
    color = '#FF6B6B' if 'Best' in data['label'] else '#4ECDC4'
    
    ax.scatter(data['features'], data['r2'], s=size, marker=marker, 
              color=color, alpha=0.7, edgecolors='black', linewidth=2,
              label=name)
    
    # Add annotation
    ax.annotate(f"{name}\nR²={data['r2']:.4f}\nMAPE={data['mape']:.2f}%",
                xy=(data['features'], data['r2']),
                xytext=(20, 20) if data['features'] == 12 else (-80, -20),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color=color, lw=2),
                fontweight='bold')

ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Quality > Feature Quantity\n(12 Selected Features Outperform 122 Features!)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 140)
ax.set_ylim(0.92, 0.94)

# Add conclusion text
ax.text(65, 0.9355, '★ Occam\'s Razor Principle ★\n"Simpler is Better"', 
        fontsize=12, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.8', fc='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(fig_dir / 'feature_count_vs_performance.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_dir / 'feature_count_vs_performance.png'}")
plt.close()

# ============================================================
# 5. Full Model Performance Evolution
# ============================================================
print("\n5. Creating Performance Evolution Chart...")

evolution_data = {
    'Linear Regression': {
        'With Leakage': {'R2': 1.0000, 'MAPE': 0.01},
        'Fixed': {'R2': 0.8739, 'MAPE': 10.40}
    },
    'Random Forest': {
        'With Leakage': {'R2': 0.9973, 'MAPE': 1.15},
        'Fixed': {'R2': 0.9264, 'MAPE': 8.15}
    },
    'RF Lite': {
        'With Leakage': {'R2': 0.9977, 'MAPE': 1.00},
        'Fixed': {'R2': 0.9338, 'MAPE': 7.31}
    },
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance: Data Leakage vs Fixed', 
             fontsize=16, fontweight='bold')

models_list = list(evolution_data.keys())
x = np.arange(len(models_list))
width = 0.35

# R² comparison
ax = axes[0]
r2_leak = [evolution_data[m]['With Leakage']['R2'] for m in models_list]
r2_fixed = [evolution_data[m]['Fixed']['R2'] for m in models_list]

bars1 = ax.bar(x - width/2, r2_leak, width, label='With Leakage (Invalid)', 
              color='#FF6B6B', alpha=0.7)
bars2 = ax.bar(x + width/2, r2_fixed, width, label='Fixed (Valid)', 
              color='#4ECDC4', alpha=0.7)

ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('R² Score Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
ax.set_ylim(0.85, 1.01)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# MAPE comparison
ax = axes[1]
mape_leak = [evolution_data[m]['With Leakage']['MAPE'] for m in models_list]
mape_fixed = [evolution_data[m]['Fixed']['MAPE'] for m in models_list]

bars1 = ax.bar(x - width/2, mape_leak, width, label='With Leakage (Too Good)', 
              color='#FF6B6B', alpha=0.7)
bars2 = ax.bar(x + width/2, mape_fixed, width, label='Fixed (Realistic)', 
              color='#4ECDC4', alpha=0.7)

ax.set_ylabel('MAPE (%)', fontweight='bold')
ax.set_title('MAPE Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'performance_evolution.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_dir / 'performance_evolution.png'}")
plt.close()

print("\n" + "="*60)
print("✓ All visualizations updated successfully!")
print("="*60)
print("\nGenerated Files:")
print("  1. model_comparison_fixed.png")
print("  2. feature_importance_lite_fixed.png")
print("  3. data_leakage_fix_comparison.png")
print("  4. feature_count_vs_performance.png")
print("  5. performance_evolution.png")
print("="*60)
