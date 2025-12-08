"""
数据预处理模块
负责数据清洗、异常值处理、缺失值填充和归一化
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, List, Tuple
import logging

from utils import setup_logging

logger = setup_logging()

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.scalers = {}
        self.statistics = {}
        
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str], 
                           factor: float = 1.5) -> pd.DataFrame:
        """
        使用IQR方法检测异常值
        
        Args:
            df: 数据框
            columns: 要检测的列
            factor: IQR倍数因子
            
        Returns:
            标记了异常值的数据框
        """
        df = df.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask |= col_outliers
            
            n_outliers = col_outliers.sum()
            if n_outliers > 0:
                logger.info(f"{col}: {n_outliers} outliers detected "
                          f"(range: [{lower_bound:.2f}, {upper_bound:.2f}])")
        
        df['is_outlier'] = outlier_mask
        return df
    
    def detect_outliers_zscore(self, df: pd.DataFrame, columns: List[str], 
                              threshold: float = 3) -> pd.DataFrame:
        """
        使用Z-score方法检测异常值
        
        Args:
            df: 数据框
            columns: 要检测的列
            threshold: Z-score阈值
            
        Returns:
            标记了异常值的数据框
        """
        df = df.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > threshold
            outlier_mask |= col_outliers
            
            n_outliers = col_outliers.sum()
            if n_outliers > 0:
                logger.info(f"{col}: {n_outliers} outliers detected (Z-score > {threshold})")
        
        df['is_outlier'] = outlier_mask
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'clip', 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: 数据框
            method: 处理方法 ('clip', 'remove', 'interpolate')
            columns: 要处理的列
            
        Returns:
            处理后的数据框
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'clip':
            # 截断到合理范围
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
                
        elif method == 'remove':
            # 移除异常值行
            if 'is_outlier' in df.columns:
                n_before = len(df)
                df = df[~df['is_outlier']].copy()
                logger.info(f"Removed {n_before - len(df)} outlier rows")
                
        elif method == 'interpolate':
            # 插值填充
            if 'is_outlier' in df.columns:
                for col in columns:
                    df.loc[df['is_outlier'], col] = np.nan
                df[columns] = df[columns].interpolate(method='linear')
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据框
            method: 处理方法 ('interpolate', 'ffill', 'bfill', 'mean', 'drop')
            columns: 要处理的列
            
        Returns:
            处理后的数据框
        """
        df = df.copy()
        
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()
        
        if not columns:
            logger.info("No missing values found")
            return df
        
        n_missing = df[columns].isnull().sum().sum()
        logger.info(f"Handling {n_missing} missing values using method: {method}")
        
        if method == 'interpolate':
            # 线性插值（适合时间序列）
            df[columns] = df[columns].interpolate(method='linear', limit_direction='both')
            
        elif method == 'ffill':
            # 前向填充
            df[columns] = df[columns].fillna(method='ffill')
            
        elif method == 'bfill':
            # 后向填充
            df[columns] = df[columns].fillna(method='bfill')
            
        elif method == 'mean':
            # 均值填充
            df[columns] = df[columns].fillna(df[columns].mean())
            
        elif method == 'drop':
            # 删除包含缺失值的行
            df = df.dropna(subset=columns)
        
        # 检查是否还有缺失值
        remaining = df[columns].isnull().sum().sum()
        if remaining > 0:
            logger.warning(f"Still have {remaining} missing values after handling")
        else:
            logger.info("All missing values handled successfully")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str],
                          method: str = 'minmax', 
                          fit: bool = True) -> pd.DataFrame:
        """
        特征归一化
        
        Args:
            df: 数据框
            columns: 要归一化的列
            method: 归一化方法 ('minmax', 'standard')
            fit: 是否拟合scaler（训练集True，测试集False）
            
        Returns:
            归一化后的数据框
        """
        df = df.copy()
        
        if method == 'minmax':
            if fit:
                self.scalers[method] = MinMaxScaler()
                df[columns] = self.scalers[method].fit_transform(df[columns])
                logger.info(f"Fitted MinMaxScaler on {len(columns)} columns")
            else:
                if method not in self.scalers:
                    raise ValueError("Scaler not fitted yet. Set fit=True first.")
                df[columns] = self.scalers[method].transform(df[columns])
                
        elif method == 'standard':
            if fit:
                self.scalers[method] = StandardScaler()
                df[columns] = self.scalers[method].fit_transform(df[columns])
                logger.info(f"Fitted StandardScaler on {len(columns)} columns")
            else:
                if method not in self.scalers:
                    raise ValueError("Scaler not fitted yet. Set fit=True first.")
                df[columns] = self.scalers[method].transform(df[columns])
        
        return df
    
    def inverse_transform(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        逆归一化
        
        Args:
            data: 归一化后的数据
            method: 归一化方法
            
        Returns:
            原始尺度的数据
        """
        if method not in self.scalers:
            raise ValueError(f"Scaler '{method}' not found")
        
        return self.scalers[method].inverse_transform(data)
    
    def compute_statistics(self, df: pd.DataFrame, columns: List[str]) -> dict:
        """
        计算统计信息
        
        Args:
            df: 数据框
            columns: 列名列表
            
        Returns:
            统计信息字典
        """
        stats = {}
        for col in columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            }
        
        self.statistics = stats
        return stats
    
    def full_preprocessing(self, df: pd.DataFrame, 
                          target_column: str,
                          fit: bool = True) -> pd.DataFrame:
        """
        完整的预处理流程
        
        Args:
            df: 数据框
            target_column: 目标列
            fit: 是否拟合（训练集True，测试集False）
            
        Returns:
            预处理后的数据框
        """
        logger.info("Starting full preprocessing pipeline...")
        
        # 1. 处理缺失值
        df = self.handle_missing_values(df, method='interpolate')
        
        # 2. 检测并处理异常值（仅对能耗相关列）
        energy_columns = [col for col in df.columns 
                         if 'kWh' in col or 'Energy' in col]
        if energy_columns:
            df = self.detect_outliers_iqr(df, energy_columns, factor=2.0)
            df = self.handle_outliers(df, method='clip', columns=energy_columns)
            if 'is_outlier' in df.columns:
                df = df.drop('is_outlier', axis=1)
        
        # 3. 归一化（排除分类列和日期列）
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['DateTime', 'is_outlier']
        normalize_cols = [col for col in numeric_columns 
                         if col not in exclude_cols and col != target_column]
        
        if normalize_cols:
            df = self.normalize_features(df, normalize_cols, method='minmax', fit=fit)
        
        # 4. 计算统计信息
        if fit:
            self.compute_statistics(df, numeric_columns)
        
        logger.info("Preprocessing completed")
        return df

def main():
    """测试预处理功能"""
    from data_loader import DataLoader
    
    # 加载数据
    loader = DataLoader()
    df = loader.merge_weather_building('Hospitals')
    
    print("\nOriginal data:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # 预处理
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.full_preprocessing(df, target_column='Total_Energy_kWh')
    
    print("\nProcessed data:")
    print(df_processed.head())
    print(f"Shape: {df_processed.shape}")
    print(f"Missing values:\n{df_processed.isnull().sum()}")

if __name__ == '__main__':
    main()
