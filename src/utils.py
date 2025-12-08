"""
工具函数模块
包含常用的辅助函数
"""
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import logging

def setup_logging(log_level='INFO'):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def ensure_dir(directory: Path):
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    directory.mkdir(parents=True, exist_ok=True)

def save_json(data: Dict[Any, Any], filepath: Path):
    """
    保存字典到JSON文件
    
    Args:
        data: 要保存的字典
        filepath: 文件路径
    """
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath: Path) -> Dict[Any, Any]:
    """
    从JSON文件加载字典
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_model(model, filepath: Path):
    """
    保存模型到文件
    
    Args:
        model: 模型对象
        filepath: 文件路径
    """
    ensure_dir(filepath.parent)
    joblib.dump(model, filepath)

def load_model(filepath: Path):
    """
    从文件加载模型
    
    Args:
        filepath: 文件路径
        
    Returns:
        模型对象
    """
    return joblib.load(filepath)

def create_time_features(df: pd.DataFrame, datetime_col: str = 'DateTime') -> pd.DataFrame:
    """
    从时间列创建时间特征
    
    Args:
        df: 数据框
        datetime_col: 时间列名称
        
    Returns:
        添加了时间特征的数据框
    """
    df = df.copy()
    
    # 确保datetime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], format='%m/%d  %H:%M:%S')
    
    # 提取时间特征
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['day_of_month'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 季节特征（北半球）
    df['season'] = df['month'].apply(lambda x: 
        0 if x in [12, 1, 2] else  # 冬季
        1 if x in [3, 4, 5] else   # 春季
        2 if x in [6, 7, 8] else   # 夏季
        3                           # 秋季
    )
    
    # 周期性编码（用于捕捉周期性模式）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def split_time_series(df: pd.DataFrame, test_size: float = 0.2, 
                      val_size: float = 0.1) -> tuple:
    """
    时间序列数据分割（不能随机打乱）
    
    Args:
        df: 数据框
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中分出）
        
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    return train_df, val_df, test_df

def print_data_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    打印数据集信息
    
    Args:
        df: 数据框
        name: 数据集名称
    """
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"{'='*50}\n")

def get_feature_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """
    获取特征列（排除指定列）
    
    Args:
        df: 数据框
        exclude: 要排除的列名列表
        
    Returns:
        特征列名列表
    """
    return [col for col in df.columns if col not in exclude]
