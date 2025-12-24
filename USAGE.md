# 项目使用指南 (最终修复版)

**项目状态**: ✅ 数据泄露已修复，所有模型使用合法特征  
**推荐模型**: RF Lite (12特征, R²=0.9338, MAPE=7.31%)  
**最后更新**: 2025-12-24

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整实验流程

#### 方法A: 使用最新修复版本 (推荐)
```bash
# 1. 使用修复后的122合法特征训练所有模型
python fix_final.py

# 2. 创建12特征精简集合（无泄露）
python create_lite_features_fixed.py

# 3. 训练RF Lite模型（最佳性能）
python train_lite_model_fixed.py

# 4. 训练改进版LSTM
python train_improved_lstm.py

# 5. 生成所有可视化图表
python update_visualizations.py
python regenerate_model_plots.py
```

#### 方法B: 使用原始脚本（⚠️ 包含数据泄露）
```bash
python main.py  # 仅用于教学演示，不可用于生产
```

### 3. 使用Jupyter Notebook交互式探索
```bash
jupyter notebook
```

打开 [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) 进行数据探索。

---

## 项目结构详解

### 核心模块

#### 1. `src/config.py`
- 所有配置参数集中管理
- 可修改模型超参数、文件路径等

#### 2. `src/data_loader.py`
- 加载天气数据和建筑能耗数据
- 合并数据并确保时间对齐

#### 3. `src/preprocessing.py`
- 异常值检测和处理（IQR方法、Z-score方法）
- 缺失值填充（线性插值）
- 数据归一化（Min-Max Scaler）

#### 4. `src/feature_engineering.py`
- 创建滞后特征（1,2,3,24小时前）✅ 合法
- 创建滚动统计特征（shift后rolling）✅ 修复
- 创建时间特征（小时、星期、季节等）✅ 合法
- ~~创建差分特征~~ ❌ 已移除（数据泄露）
- ~~使用能耗组成部分~~ ❌ 已移除（数据泄露）

#### 5. `src/models/`
- `linear_regression.py`: 线性回归基线模型
- `svr_model.py`: 支持向量回归
- `knn_model.py`: K近邻回归
- `random_forest.py`: 随机森林（可输出特征重要性）
- `improved_lstm_model.py`: 改进版LSTM（3层+注意力）
- `lstm_model.py`: LSTM Baseline

#### 6. `src/evaluation.py`
- 模型评估（RMSE, MAE, R², MAPE）
- 模型对比
- 计算改进百分比

#### 7. `src/visualization.py`
- 预测对比图
- 残差图
- 特征重要性图
- 模型比较图
- 训练历史图

---

## 模型性能总览 (修复后)

| 模型 | RMSE | MAE | R² | MAPE(%) | 特征数 | 推荐 |
|------|------|-----|-----|---------|--------|------|
| **RF Lite** 🏆 | **2.49** | **1.71** | **0.9338** | **7.31** | **12** | ✅ 生产部署 |
| RF Full | 2.62 | 1.85 | 0.9264 | 8.15 | 122 | ✅ 研究对比 |
| Improved LSTM | 3.18 | 2.35 | 0.8895 | 9.99 | 12 | ✅ 深度学习实验 |
| Linear Regression | 3.43 | 2.50 | 0.8739 | 10.40 | 122 | ✅ 基线模型 |
| Baseline LSTM | 4.36 | 3.31 | 0.7954 | 13.43 | 12 | ⚠️ 仅供对比 |
| SVR | 5.20 | 4.04 | 0.7100 | 15.68 | 122 | ⚠️ 性能较差 |
| KNN | 5.49 | 3.87 | 0.6773 | 16.66 | 122 | ⚠️ 性能较差 |

---

## 自定义实验

### 修改目标变量（不推荐，可能引入泄露）
```python
# src/config.py
TARGET_COLUMN = 'Total_Energy_kWh'  # 保持默认，其他可能泄露
```

### 调整RF Lite参数
```python
# train_lite_model_fixed.py
rf_params = {
    'n_estimators': 200,      # 从100增加到200
    'max_depth': 20,          # 从15增加到20
    'min_samples_split': 5,   # 从10减少到5
    ...
}
```

### 修改LSTM参数
```python
# train_improved_lstm.py
model = ImprovedLSTMModel(
    input_size=12,
    hidden_size=256,          # 从128增加到256
    num_layers=4,             # 从3增加到4
    dropout=0.4               # 从0.3增加到0.4
)
```

---

## 结果文件说明

运行完成后，`results/` 目录包含：

### `figures/` - 图表 (19个PNG文件)
**预测对比图**:
- `Linear_Regression_predictions.png`
- `SVR_predictions.png`
- `KNN_predictions.png`
- `Random_Forest_predictions.png`
- `Random_Forest_Lite_predictions.png`
- `LSTM_Baseline_predictions.png`
- `Improved_LSTM_predictions.png`

**残差分析图**:
- `Linear_Regression_residuals.png`
- `SVR_residuals.png`
- `KNN_residuals.png`
- `Random_Forest_residuals.png`
- `Random_Forest_Lite_residuals.png`
- `LSTM_Baseline_residuals.png`
- `Improved_LSTM_residuals.png`

**对比分析图**:
- `model_comparison_r2.png` - R²对比
- `model_comparison_mape.png` - MAPE对比
- `model_comparison_rmse.png` - RMSE对比
- `feature_comparison_chart.png` - 特征数对比
- `feature_importance_lite_fixed.png` - 12特征重要性
- `feature_importance_final.png` - 122特征重要性

### `metrics/` - 评估指标
- `model_comparison_final.csv`: 所有模型性能对比（修复后）
- `feature_importance_final.csv`: 122特征重要性列表
- `feature_importance_lite_fixed.csv`: 12特征重要性
- `improved_lstm_metrics.csv/json`: LSTM详细指标
- `X_train_final.csv`, `X_test_final.csv`: 特征数据（122列）
- `y_train.csv`, `y_test.csv`: 目标变量
- `valid_features_final.txt`: 122个合法特征列表

### `models/` - 保存的模型
**最终模型** (无泄露):
- `lr_final.pkl` - Linear Regression
- `svr_final.pkl` - SVR
- `knn_final.pkl` - KNN
- `rf_final.pkl` - Random Forest Full
- `random_forest_lite_fixed.pkl` - **RF Lite 🏆 最佳**
- `improved_lstm.pth` - Improved LSTM
- `lite_features_fixed.txt` - RF Lite使用的12个特征列表

### `experiment_summary.json`
实验概要信息（修复前版本，仅供参考）

---

## 数据泄露说明 ⚠️

### 已识别并移除的泄露特征 (12个)

**1. 能耗组成部分** (10个特征):
```python
# 这些是Total_Energy_kWh的直接组成部分，相关性>0.98
Electricity_kWh, Electricity_J
HotWater_kWh, HotWater_J
SpaceHeating_kWh, SpaceHeating_J
SpaceCooling_kWh, SpaceCooling_J
HotWater_SpaceHeating_kWh, HotWater_SpaceHeating_J
```

**2. 差分特征** (2个特征):
```python
# y[t] = diff_1 + lag_1 可完美重构目标
Total_Energy_kWh_diff_1
Total_Energy_kWh_diff_24
```

### 合法的特征类型 ✅

**滞后特征** (使用shift):
```python
lag_1 = df['Total_Energy_kWh'].shift(1)    # ✅ 使用历史值
lag_24 = df['Total_Energy_kWh'].shift(24)  # ✅ 昨天同时刻
```

**滚动特征** (shift后rolling):
```python
rolling_mean = df['col'].shift(1).rolling(3).mean()  # ✅ 仅历史数据
```

---

## 报告撰写建议 (7000字课程设计)

### 1. 背景介绍 (1500字)
- 能源管理系统的重要性
- 建筑能耗预测的应用场景
- 瑞士数据集描述
- 研究目标和意义
- **新增**: 数据泄露问题的普遍性及重要性

### 2. 机器学习方法介绍 (1500字)
- 线性回归原理
- SVR与核函数
- KNN的相似性度量
- 随机森林的集成学习
- LSTM的时间序列建模能力
- **新增**: 特征工程中的数据泄露风险

### 3. 方法具体实施 (2500字) ⬆️ 增加篇幅
- 数据预处理流程（截图）
- 特征工程策略
- **重点**: 数据泄露的识别与修复过程
- 模型训练过程
- 超参数设置
- **新增**: 特征精简策略（122→12特征）

### 4. 实验结果 (1500字)
- 各模型性能对比表（修复前后）
- 可视化图表分析
- 特征重要性分析
- **新增**: RF Lite优于RF Full的原因分析
- **新增**: LSTM改进效果分析

### 5. 总结与讨论 (500字)
- 最佳模型及原因（RF Lite）
- **重点**: 数据泄露的经验教训
- **重点**: 特征质量>特征数量的启示
- 实际应用价值
- 未来改进方向

---

## 常见问题

### Q1: 为什么12特征优于122特征？
A: 
- 精选的12特征覆盖95.71%总重要性
- 减少冗余特征，降低过拟合风险
- 特征质量>特征数量，符合奥卡姆剃刀原则

### Q2: 如何确保没有数据泄露？
A: 
- 所有特征必须在预测时刻之前可获得
- 不使用目标的同期测量值或其直接函数
- 检查特征-目标相关性（<0.95为正常）
- 验证R²不应达到1.0或接近1.0（除非真实数据）

### Q3: LSTM为什么不如RF？
A: 
- 数据量有限（8k样本，LSTM需要10k+）
- lag_1特征已包含大部分信息
- 任务相对简单，线性趋势为主
- RF的集成学习更适合这类任务

### Q4: 如何使用GPU加速LSTM？
A: 
确保安装了CUDA版本的PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

LSTM会自动检测并使用GPU。

### Q5: 如何在生产环境部署？
A:
```python
# 1. 加载RF Lite模型
import joblib
model = joblib.load('results/models/random_forest_lite_fixed.pkl')

# 2. 准备12个特征
features = ['Total_Energy_kWh_lag_1', 'Total_Energy_kWh_lag_24', ...]

# 3. 预测
prediction = model.predict(X_new[features])
```

---

## 参考文档

- [README.md](README.md) - 项目总览
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - 完成状态
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - 结果摘要
- [DATA_LEAKAGE_FIX_COMPLETE.md](DATA_LEAKAGE_FIX_COMPLETE.md) - 数据泄露修复详情
- [EXPERIMENT_LOG_v2.md](EXPERIMENT_LOG_v2.md) - 重整后的实验日志

---

**祝实验顺利！如有问题请参考文档或查看源代码注释。**

```python
'lstm': {
    'hidden_size': 256,    # 增加隐藏层大小
    'num_layers': 3,       # 增加层数
    'epochs': 100,         # 增加训练轮数
    ...
}
```

## 结果文件说明

运行完成后，`results/` 目录包含：

### `figures/` - 图表
- `*_predictions.png`: 各模型的预测对比图
- `*_residuals.png`: 残差分析图
- `model_comparison.png`: 模型性能对比图
- `feature_importance.png`: 特征重要性图
- `lstm_training_history.png`: LSTM训练曲线

### `metrics/` - 评估指标
- `model_comparison.csv`: 模型性能对比表
- `model_improvement.csv`: 相对基准模型的改进
- `feature_importance.csv`: 特征重要性列表

### `models/` - 保存的模型
- `Linear Regression.pkl`
- `SVR.pkl`
- `KNN.pkl`
- `Random Forest.pkl`
- `LSTM.pth`

### `experiment_summary.json`
实验概要信息

## 报告撰写建议

根据课程要求（7000字），建议章节分配：

### 1. 背景介绍 (1500字)
- 能源管理系统的重要性
- 建筑能耗预测的应用场景
- 瑞士数据集描述
- 研究目标和意义

### 2. 机器学习方法介绍 (1500字)
- 线性回归原理
- SVR与核函数
- KNN的相似性度量
- 随机森林的集成学习
- LSTM的时间序列建模能力

### 3. 方法具体实施 (2000字)
- 数据预处理流程（截图）
- 特征工程策略
- 模型训练过程
- 超参数设置

### 4. 实验结果 (1500字)
- 各模型性能对比表
- 可视化图表分析
- 特征重要性分析
- LSTM优势分析

### 5. 总结与讨论 (500字)
- 最佳模型及原因
- 实际应用价值
- 未来改进方向

## 常见问题

### Q1: SVR训练太慢怎么办？
A: `main.py`中已经对SVR使用了采样训练（5000样本）。如果还是太慢，可以：
- 减少样本量
- 改用线性核 `kernel='linear'`
- 或直接注释掉SVR部分

### Q2: LSTM准确率不高？
A: 尝试：
- 增加训练轮数（epochs）
- 调整序列长度（sequence_length）
- 增加隐藏层大小（hidden_size）
- 使用更多特征

### Q3: 如何加快实验速度？
A: 
- 减少数据量（只用前几个月）
- 减少特征（删除部分滑动窗口特征）
- 减少LSTM训练轮数
- 使用GPU训练LSTM

### Q4: 如何使用GPU？
A: 
确保安装了CUDA版本的PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

LSTM会自动检测并使用GPU。

## 联系与支持

如有问题，请参考：
- 课程要求文档
- 参考论文
- Python官方文档
- Stack Overflow

祝实验顺利！
