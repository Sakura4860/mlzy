"""
配置文件
包含项目的所有配置参数
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
FIGURES_DIR = RESULTS_DIR / 'figures'
METRICS_DIR = RESULTS_DIR / 'metrics'

# 数据文件路径
WEATHER_FILE = DATA_DIR / 'WEATHER_DATA_ZURICH_2020_2019.xlsx'
BUILDING_FILES = {
    'Hospitals': DATA_DIR / 'Hospitals_1991_2000_Full_retrofit.xlsx',
    'Restaurants': DATA_DIR / 'Restaurants_1991_2000_Full_retrofit.xlsx',
    'Schools': DATA_DIR / 'Schools_2010_2015_Full_retrofit.xlsx',
    'Shops': DATA_DIR / 'Shops_1991_2000_Full_retrofit.xlsx'
}

# 数据处理参数
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 时间序列通常使用后20%作为测试集
VALIDATION_SIZE = 0.1  # 从训练集中分出10%作为验证集

# 特征工程参数
LAG_FEATURES = [1, 2, 3, 6, 12, 24]  # 滞后特征：1,2,3,6,12,24小时前
ROLLING_WINDOWS = [3, 6, 12, 24]  # 滑动窗口：3,6,12,24小时

# 预测目标
TARGET_COLUMN = 'Total_Energy_kWh'  # 预测总能耗
# 可选目标：'SpaceHeating_kWh', 'SpaceCooling_kWh', 'Electricity_kWh'

# 模型参数
MODEL_PARAMS = {
    'linear_regression': {
        'fit_intercept': True
        # normalize参数在scikit-learn 1.0+中已被移除
    },
    'svr': {
        'kernel': 'rbf',
        'C': 100.0,
        'gamma': 'scale',
        'epsilon': 0.1
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'distance',
        'algorithm': 'auto'
    },
    'random_forest': {
        'n_estimators': 50,  # 减少树数量以加快训练（从100降到50）
        'max_depth': 15,  # 减少树深度
        'min_samples_split': 10,  # 增大最小分裂样本数
        'min_samples_leaf': 5,  # 增大叶节点最小样本数
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'lstm': {
        'hidden_size': 64,  # 降低隐藏层大小加快训练
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 128,  # 增大batch加快训练
        'epochs': 20,  # 减少训练轮数（从50降到20）
        'learning_rate': 0.001,
        'sequence_length': 24  # 使用过去24小时的数据预测下一小时
    }
}

# 评估指标
METRICS = ['RMSE', 'MAE', 'R2', 'MAPE']

# 可视化参数
FIG_SIZE = (12, 6)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# 建筑类型列表
BUILDING_TYPES = list(BUILDING_FILES.keys())

# 选择要预测的建筑类型（可以选择一个或多个）
SELECTED_BUILDINGS = ['Hospitals']  # 先从医院开始，可以改为['Schools', 'Shops']等

# 日志配置
LOG_LEVEL = 'INFO'
