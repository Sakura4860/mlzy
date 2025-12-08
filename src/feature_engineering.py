"""
特征工程模块
创建滞后特征、滑动窗口特征、时间特征等
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from config import LAG_FEATURES, ROLLING_WINDOWS
from utils import setup_logging, create_time_features

logger = setup_logging()

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.feature_names = []
        
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                          lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 数据框
            columns: 要创建滞后特征的列
            lags: 滞后期数列表
            
        Returns:
            添加了滞后特征的数据框
        """
        if lags is None:
            lags = LAG_FEATURES
        
        df = df.copy()
        
        logger.info(f"Creating lag features for {len(columns)} columns with lags: {lags}")
        
        for col in columns:
            for lag in lags:
                feature_name = f'{col}_lag_{lag}'
                df[feature_name] = df[col].shift(lag)
                self.feature_names.append(feature_name)
        
        logger.info(f"Created {len(columns) * len(lags)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str],
                              windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        创建滑动窗口统计特征
        
        Args:
            df: 数据框
            columns: 要创建滑动特征的列
            windows: 窗口大小列表
            
        Returns:
            添加了滑动窗口特征的数据框
        """
        if windows is None:
            windows = ROLLING_WINDOWS
        
        df = df.copy()
        
        logger.info(f"Creating rolling features for {len(columns)} columns with windows: {windows}")
        
        for col in columns:
            for window in windows:
                # 滑动平均 (使用shift(1)避免数据泄露)
                feature_name = f'{col}_rolling_mean_{window}'
                df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
                self.feature_names.append(feature_name)
                
                # 滑动最大值
                feature_name = f'{col}_rolling_max_{window}'
                df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).max()
                self.feature_names.append(feature_name)
                
                # 滑动最小值
                feature_name = f'{col}_rolling_min_{window}'
                df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).min()
                self.feature_names.append(feature_name)
                
                # 滑动标准差
                feature_name = f'{col}_rolling_std_{window}'
                df[feature_name] = df[col].shift(1).rolling(window=window, min_periods=1).std()
                self.feature_names.append(feature_name)
        
        logger.info(f"Created {len(columns) * len(windows) * 4} rolling features")
        return df
    
    def create_diff_features(self, df: pd.DataFrame, columns: List[str],
                           periods: List[int] = [1, 24]) -> pd.DataFrame:
        """
        创建差分特征（捕捉变化趋势）
        
        Args:
            df: 数据框
            columns: 要创建差分特征的列
            periods: 差分周期列表
            
        Returns:
            添加了差分特征的数据框
        """
        df = df.copy()
        
        logger.info(f"Creating diff features for {len(columns)} columns with periods: {periods}")
        
        for col in columns:
            for period in periods:
                feature_name = f'{col}_diff_{period}'
                df[feature_name] = df[col].diff(periods=period)
                self.feature_names.append(feature_name)
        
        logger.info(f"Created {len(columns) * len(periods)} diff features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   col_pairs: List[tuple]) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            df: 数据框
            col_pairs: 列对列表，如[('Temperature', 'hour'), ...]
            
        Returns:
            添加了交互特征的数据框
        """
        df = df.copy()
        
        logger.info(f"Creating {len(col_pairs)} interaction features")
        
        for col1, col2 in col_pairs:
            if col1 in df.columns and col2 in df.columns:
                # 乘法交互
                feature_name = f'{col1}_x_{col2}'
                df[feature_name] = df[col1] * df[col2]
                self.feature_names.append(feature_name)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                   columns: List[str],
                                   window: int = 24) -> pd.DataFrame:
        """
        创建统计特征
        
        Args:
            df: 数据框
            columns: 列名列表
            window: 统计窗口大小
            
        Returns:
            添加了统计特征的数据框
        """
        df = df.copy()
        
        logger.info(f"Creating statistical features with window={window}")
        
        for col in columns:
            # 变异系数
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            feature_name = f'{col}_cv_{window}'
            df[feature_name] = rolling_std / (rolling_mean + 1e-8)
            self.feature_names.append(feature_name)
            
            # 峰度
            feature_name = f'{col}_skew_{window}'
            df[feature_name] = df[col].rolling(window=window, min_periods=1).skew()
            self.feature_names.append(feature_name)
        
        logger.info(f"Created {len(columns) * 2} statistical features")
        return df
    
    def create_all_features(self, df: pd.DataFrame, 
                           target_column: str,
                           weather_columns: List[str] = None,
                           energy_columns: List[str] = None) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            df: 数据框
            target_column: 目标列
            weather_columns: 天气列
            energy_columns: 能耗列（用于创建滞后特征）
            
        Returns:
            添加了所有特征的数据框
        """
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # 1. 创建时间特征
        df = create_time_features(df, datetime_col='DateTime')
        logger.info("Time features created")
        
        # 2. 默认天气列
        if weather_columns is None:
            weather_columns = ['Temperature', 'DNI', 'DIF', 'GHI', 'WindSpeed']
            weather_columns = [col for col in weather_columns if col in df.columns]
        
        # 3. 默认能耗列（只对目标变量创建滞后特征）
        if energy_columns is None:
            energy_columns = [target_column]
        
        # 4. 创建滞后特征（主要用于目标变量）
        df = self.create_lag_features(df, energy_columns, lags=LAG_FEATURES)
        
        # 5. 创建滑动窗口特征（天气和目标变量）
        rolling_cols = weather_columns + [target_column]
        df = self.create_rolling_features(df, rolling_cols, windows=ROLLING_WINDOWS)
        
        # 6. 创建差分特征（捕捉趋势）
        df = self.create_diff_features(df, [target_column], periods=[1, 24])
        
        # 7. 创建交互特征（温度与时间）
        if 'Temperature' in df.columns and 'hour' in df.columns:
            interaction_pairs = [
                ('Temperature', 'hour'),
                ('Temperature', 'is_weekend'),
            ]
            if 'WindSpeed' in df.columns:
                interaction_pairs.append(('Temperature', 'WindSpeed'))
            
            df = self.create_interaction_features(df, interaction_pairs)
        
        # 8. 删除因滞后和滑动窗口产生的初始NaN行
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_after = len(df)
        logger.info(f"Dropped {n_before - n_after} rows with NaN values")
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, 
                         exclude: List[str]) -> List[str]:
        """
        获取特征列名
        
        Args:
            df: 数据框
            exclude: 要排除的列
            
        Returns:
            特征列名列表
        """
        return [col for col in df.columns if col not in exclude]
    
    def select_features_by_importance(self, df: pd.DataFrame, 
                                     feature_cols: List[str],
                                     target_col: str,
                                     top_k: int = 50) -> List[str]:
        """
        基于随机森林选择最重要的k个特征
        
        Args:
            df: 数据框
            feature_cols: 特征列
            target_col: 目标列
            top_k: 选择前k个特征
            
        Returns:
            选择的特征列表
        """
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info(f"Selecting top {top_k} features from {len(feature_cols)} features")
        
        X = df[feature_cols]
        y = df[target_col]
        
        # 训练随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 获取特征重要性
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = importances.head(top_k)['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 10 features: {selected_features[:10]}")
        
        return selected_features

def main():
    """测试特征工程功能"""
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    
    # 加载和预处理数据
    loader = DataLoader()
    df = loader.merge_weather_building('Hospitals')
    
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df, method='interpolate')
    
    print("\nOriginal data:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 特征工程
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df, target_column='Total_Energy_kWh')
    
    print("\nAfter feature engineering:")
    print(df_features.head())
    print(f"Shape: {df_features.shape}")
    print(f"\nNumber of columns: {len(df_features.columns)}")
    
    # 获取特征列
    exclude_cols = ['DateTime', 'Building_type', 'Construction_period', 
                   'Retrofit_scenario', 'Total_Energy_kWh']
    feature_cols = engineer.get_feature_names(df_features, exclude_cols)
    print(f"\nNumber of feature columns: {len(feature_cols)}")

if __name__ == '__main__':
    main()
