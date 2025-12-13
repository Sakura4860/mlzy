"""
为Improved LSTM生成可视化图表
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("生成Improved LSTM可视化图表")
print("="*60)

# 1. 加载数据
print("\n[1] 加载测试数据...")
X_test = pd.read_csv('results/metrics/X_test_full.csv')
y_test = pd.read_csv('results/metrics/y_test.csv').squeeze()

# 加载lite特征列表
with open('results/models/lite_features.txt', 'r') as f:
    lite_features = [line.strip() for line in f if line.strip()]

X_test_lite = X_test[lite_features].values.astype(np.float32)
print(f"测试数据: {len(X_test_lite)} 样本, {len(lite_features)} 特征")

# 2. 加载模型并预测
print("\n[2] 加载Improved LSTM模型...")
from src.models.improved_lstm_model import ImprovedLSTMModel

model = ImprovedLSTMModel(input_size=12, hidden_size=128, num_layers=3, dropout=0.3)
model.load_model('results/models/improved_lstm.pth')

# 获取预测
seq_len = 48
print(f"序列长度: {seq_len}")

# 直接使用模型的predict方法,它会返回正确长度的预测
y_pred = model.predict(X_test_lite, sequence_length=seq_len)
print(f"预测样本数: {len(y_pred)}")

# y_pred比X_test少seq_len个样本,需要对齐y_test
y_test_aligned = y_test.values[seq_len:]
print(f"对齐后的真实值样本数: {len(y_test_aligned)}")

# 确保长度一致
if len(y_pred) != len(y_test_aligned):
    min_len = min(len(y_pred), len(y_test_aligned))
    y_pred = y_pred[:min_len]
    y_test_aligned = y_test_aligned[:min_len]
    print(f"调整后样本数: {min_len}")

# 检查并移除NaN值
valid_mask = ~(np.isnan(y_pred) | np.isnan(y_test_aligned))
if not valid_mask.all():
    print(f"⚠ 发现{(~valid_mask).sum()}个NaN值,将被移除")
    y_pred = y_pred[valid_mask]
    y_test_aligned = y_test_aligned[valid_mask]
    print(f"移除NaN后样本数: {len(y_pred)}")

y_seq = y_test_aligned

# 计算指标
r2 = r2_score(y_seq, y_pred)
rmse = np.sqrt(mean_squared_error(y_seq, y_pred))
mae = mean_absolute_error(y_seq, y_pred)
print(f"\n指标: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

# 3. 创建预测对比图
print("\n[3] 生成预测对比图...")
fig = plt.figure(figsize=(16, 10))

# 子图1: 预测值 vs 真实值 (时间序列)
ax1 = plt.subplot(2, 2, 1)
n_samples = min(500, len(y_seq))
plt.plot(y_seq[:n_samples], label='真实值', alpha=0.7, linewidth=1.5, color='blue')
plt.plot(y_pred[:n_samples], label='预测值', alpha=0.7, linewidth=1.5, color='orange')
plt.xlabel('时间步', fontsize=12)
plt.ylabel('能耗 (kWh)', fontsize=12)
plt.title(f'Improved LSTM - 预测值 vs 真实值 (前{n_samples}个样本)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 子图2: 散点图
ax2 = plt.subplot(2, 2, 2)
plt.scatter(y_seq, y_pred, alpha=0.5, s=20, color='steelblue')
plt.plot([y_seq.min(), y_seq.max()], [y_seq.min(), y_seq.max()], 
         'r--', lw=2, label='完美预测线')
plt.xlabel('真实值 (kWh)', fontsize=12)
plt.ylabel('预测值 (kWh)', fontsize=12)
plt.title('Improved LSTM - 预测散点图', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 添加指标文本
textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# 子图3: 残差分布
ax3 = plt.subplot(2, 2, 3)
residuals = y_seq - y_pred
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('残差 (kWh)', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('Improved LSTM - 残差分布', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零残差线')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 子图4: 误差随时间变化
ax4 = plt.subplot(2, 2, 4)
abs_errors = np.abs(residuals)
plt.plot(abs_errors[:n_samples], alpha=0.7, linewidth=1, color='coral')
plt.xlabel('时间步', fontsize=12)
plt.ylabel('绝对误差 (kWh)', fontsize=12)
plt.title('Improved LSTM - 绝对误差随时间变化', fontsize=14, fontweight='bold')
plt.axhline(y=mae, color='red', linestyle='--', linewidth=2, 
            label=f'平均绝对误差 = {mae:.2f}')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/improved_lstm_predictions.png', dpi=300, bbox_inches='tight')
print('✓ 已保存: results/figures/improved_lstm_predictions.png')
plt.close()

# 4. 创建训练曲线图 (如果有训练历史)
print("\n[4] 生成训练曲线图...")
with open('results/metrics/improved_lstm_metrics.json', 'r') as f:
    metrics_data = json.load(f)

if hasattr(model, 'train_losses') and len(model.train_losses) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(model.train_losses) + 1)
    
    # 训练损失
    ax1.plot(epochs, model.train_losses, 'b-', linewidth=2, label='训练损失', marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失 (MSE)', fontsize=12)
    ax1.set_title('Improved LSTM - 训练损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 验证损失
    if hasattr(model, 'val_losses') and len(model.val_losses) > 0:
        ax2.plot(epochs, model.val_losses, 'r-', linewidth=2, label='验证损失', marker='s', markersize=3)
        best_val = metrics_data.get('best_val_loss', 0)
        ax2.axhline(y=best_val, color='green', linestyle='--', linewidth=2, 
                   label=f'最佳验证损失 = {best_val:.4f}')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('损失 (MSE)', fontsize=12)
        ax2.set_title('Improved LSTM - 验证损失曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/improved_lstm_training_curves.png', dpi=300, bbox_inches='tight')
    print('✓ 已保存: results/figures/improved_lstm_training_curves.png')
    plt.close()
else:
    print("⚠ 模型中没有训练历史数据,跳过训练曲线图")

# 5. 创建baseline LSTM的可视化(如果存在)
print("\n[5] 检查Baseline LSTM...")
baseline_model_path = Path('results/models/lstm.pth')
if baseline_model_path.exists():
    print("生成Baseline LSTM可视化...")
    from src.models.lstm_model import LSTMModel
    
    baseline_model = LSTMModel(input_size=12, hidden_size=64, num_layers=2, dropout=0.2)
    baseline_model.load_model(baseline_model_path)
    
    seq_len_base = 24
    baseline_pred = baseline_model.predict(X_test_lite, sequence_length=seq_len_base)
    y_test_base = y_test.values[seq_len_base:]
    
    # 对齐长度
    min_len_base = min(len(baseline_pred), len(y_test_base))
    baseline_pred = baseline_pred[:min_len_base]
    y_test_base = y_test_base[:min_len_base]
    
    # 移除NaN
    valid_mask_base = ~(np.isnan(baseline_pred) | np.isnan(y_test_base))
    if not valid_mask_base.all():
        print(f"⚠ Baseline发现{(~valid_mask_base).sum()}个NaN值,将被移除")
        baseline_pred = baseline_pred[valid_mask_base]
        y_test_base = y_test_base[valid_mask_base]
    
    # 计算baseline指标
    r2_base = r2_score(y_test_base, baseline_pred)
    rmse_base = np.sqrt(mean_squared_error(y_test_base, baseline_pred))
    mae_base = mean_absolute_error(y_test_base, baseline_pred)
    mape_base = np.mean(np.abs((y_test_base - baseline_pred) / y_test_base)) * 100
    
    # 创建对比图
    fig = plt.figure(figsize=(16, 6))
    
    # Baseline
    ax1 = plt.subplot(1, 2, 1)
    n_show = min(500, len(baseline_pred))
    plt.plot(y_test_base[:n_show], label='真实值', alpha=0.7, linewidth=1.5)
    plt.plot(baseline_pred[:n_show], label='预测值', alpha=0.7, linewidth=1.5)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('能耗 (kWh)', fontsize=12)
    plt.title(f'Baseline LSTM (R²={r2_base:.4f}, MAPE={mape_base:.2f}%)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Improved
    ax2 = plt.subplot(1, 2, 2)
    n_show_imp = min(500, len(y_seq))
    plt.plot(y_seq[:n_show_imp], label='真实值', alpha=0.7, linewidth=1.5)
    plt.plot(y_pred[:n_show_imp], label='预测值', alpha=0.7, linewidth=1.5)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('能耗 (kWh)', fontsize=12)
    mape = np.mean(np.abs((y_seq - y_pred) / y_seq)) * 100
    plt.title(f'Improved LSTM (R²={r2:.4f}, MAPE={mape:.2f}%)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/lstm_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ 已保存: results/figures/lstm_comparison.png')
    plt.close()
else:
    print("⚠ Baseline LSTM模型不存在,跳过对比图")

print("\n" + "="*60)
print("所有LSTM可视化图表已生成!")
print("="*60)
print("\n生成的文件:")
print("  1. results/figures/improved_lstm_predictions.png - 预测对比图(4子图)")
print("  2. results/figures/improved_lstm_training_curves.png - 训练曲线")
print("  3. results/figures/lstm_comparison.png - Baseline vs Improved对比")
