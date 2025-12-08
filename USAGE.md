# 项目实施指南

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整实验
```bash
python main.py
```

这将执行以下步骤：
- 加载和合并数据
- 数据预处理
- 特征工程
- 训练5个模型（LR, SVR, KNN, RF, LSTM）
- 评估和比较模型
- 生成可视化图表
- 保存模型和结果

### 3. 使用Jupyter Notebook交互式探索
```bash
jupyter notebook
```

打开 `notebooks/01_data_exploration.ipynb` 进行数据探索。

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
- 创建滞后特征（1,2,3,6,12,24小时前）
- 创建滑动窗口统计特征
- 创建时间特征（小时、星期、季节等）
- 创建交互特征

#### 5. `src/models/`
- `linear_regression.py`: 线性回归基准模型
- `svr_model.py`: 支持向量回归
- `knn_model.py`: K近邻回归
- `random_forest.py`: 随机森林（可输出特征重要性）
- `lstm_model.py`: LSTM深度学习模型

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

## 自定义实验

### 修改建筑类型
编辑 `src/config.py`:
```python
SELECTED_BUILDINGS = ['Schools']  # 或 ['Restaurants', 'Shops']
```

### 修改预测目标
编辑 `src/config.py`:
```python
TARGET_COLUMN = 'SpaceHeating_kWh'  # 预测供暖能耗
# 或 'SpaceCooling_kWh'  # 预测制冷能耗
```

### 调整模型参数
编辑 `src/config.py` 中的 `MODEL_PARAMS` 字典。

例如，增加随机森林的树数量：
```python
'random_forest': {
    'n_estimators': 200,  # 从100增加到200
    ...
}
```

### 修改LSTM参数
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
