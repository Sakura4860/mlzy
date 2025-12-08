"""
数据加载模块
负责加载和合并天气数据和建筑能耗数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging

from config import WEATHER_FILE, BUILDING_FILES, SELECTED_BUILDINGS
from utils import setup_logging

logger = setup_logging()

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        """初始化数据加载器"""
        self.weather_data = None
        self.building_data = {}
        self.merged_data = None
        
    def load_weather_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        加载天气数据
        
        Args:
            filepath: 天气数据文件路径，默认使用config中的路径
            
        Returns:
            天气数据DataFrame
        """
        if filepath is None:
            filepath = WEATHER_FILE
            
        logger.info(f"Loading weather data from {filepath}")
        
        # 读取数据
        df = pd.read_excel(filepath)
        
        # 清理数据：删除无用列
        columns_to_keep = ['TIME[h]', 'DNI[kW/m2]', 'DIF[kW/m2]', 
                          'GHI[kW/m2]', 'WS[m/s]', 'TAMB[C]']
        df = df[columns_to_keep].copy()
        
        # 重命名列
        df.columns = ['Hour', 'DNI', 'DIF', 'GHI', 'WindSpeed', 'Temperature']
        
        # 添加时间索引（假设从1月1日0点开始）
        df['DateTime'] = pd.date_range(start='2019-01-01', periods=len(df), freq='h')
        
        logger.info(f"Weather data loaded: {df.shape}")
        self.weather_data = df
        return df
    
    def load_building_data(self, building_type: str, 
                          filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        加载建筑能耗数据
        
        Args:
            building_type: 建筑类型（Hospitals, Restaurants, Schools, Shops）
            filepath: 建筑数据文件路径，默认使用config中的路径
            
        Returns:
            建筑能耗数据DataFrame
        """
        if filepath is None:
            filepath = BUILDING_FILES.get(building_type)
            if filepath is None:
                raise ValueError(f"Unknown building type: {building_type}")
        
        logger.info(f"Loading {building_type} data from {filepath}")
        
        # 读取数据
        df = pd.read_excel(filepath)
        
        # 转换时间格式（处理24:00:00的情况）
        # 有些数据用24:00:00表示第二天00:00:00
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d  %H:%M:%S', errors='coerce')
        
        # 如果有NaT（转换失败的），用索引生成时间
        if df['DateTime'].isna().any():
            logger.warning(f"Found {df['DateTime'].isna().sum()} invalid datetime entries, generating from index")
            df['DateTime'] = pd.date_range(start='2019-01-01', periods=len(df), freq='h')
        
        # 确保有完整的年份（8760小时）
        if len(df) != 8760:
            logger.warning(f"{building_type} data has {len(df)} rows, expected 8760")
        
        logger.info(f"{building_type} data loaded: {df.shape}")
        self.building_data[building_type] = df
        return df
    
    def load_all_buildings(self, building_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有指定的建筑类型数据
        
        Args:
            building_types: 建筑类型列表，默认使用config中的SELECTED_BUILDINGS
            
        Returns:
            建筑类型到数据框的字典
        """
        if building_types is None:
            building_types = SELECTED_BUILDINGS
        
        logger.info(f"Loading building data for: {building_types}")
        
        for building_type in building_types:
            self.load_building_data(building_type)
        
        return self.building_data
    
    def merge_weather_building(self, building_type: str) -> pd.DataFrame:
        """
        合并天气数据和建筑能耗数据
        
        Args:
            building_type: 建筑类型
            
        Returns:
            合并后的DataFrame
        """
        if self.weather_data is None:
            logger.info("Weather data not loaded, loading now...")
            self.load_weather_data()
        
        if building_type not in self.building_data:
            logger.info(f"{building_type} data not loaded, loading now...")
            self.load_building_data(building_type)
        
        weather_df = self.weather_data.copy()
        building_df = self.building_data[building_type].copy()
        
        # 合并数据（基于Hour列）
        # 为weather_data添加hour列用于合并
        weather_df['Hour_idx'] = weather_df['Hour']
        building_df['Hour_idx'] = building_df.index % 8760 + 1
        
        # 使用DateTime合并更准确
        merged_df = pd.merge(
            building_df,
            weather_df[['Hour_idx', 'DNI', 'DIF', 'GHI', 'WindSpeed', 'Temperature']],
            on='Hour_idx',
            how='left'
        )
        
        # 删除临时列
        merged_df = merged_df.drop('Hour_idx', axis=1)
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        logger.info(f"Columns: {merged_df.columns.tolist()}")
        
        self.merged_data = merged_df
        return merged_df
    
    def get_merged_data(self, building_type: str, reload: bool = False) -> pd.DataFrame:
        """
        获取合并后的数据（带缓存）
        
        Args:
            building_type: 建筑类型
            reload: 是否重新加载
            
        Returns:
            合并后的DataFrame
        """
        if self.merged_data is None or reload:
            return self.merge_weather_building(building_type)
        return self.merged_data
    
    def save_merged_data(self, filepath: Path, building_type: str):
        """
        保存合并后的数据
        
        Args:
            filepath: 保存路径
            building_type: 建筑类型
        """
        if self.merged_data is None:
            self.merge_weather_building(building_type)
        
        logger.info(f"Saving merged data to {filepath}")
        self.merged_data.to_csv(filepath, index=False)
        logger.info("Merged data saved successfully")

def main():
    """测试数据加载功能"""
    loader = DataLoader()
    
    # 加载天气数据
    weather_df = loader.load_weather_data()
    print("\nWeather Data:")
    print(weather_df.head())
    print(f"Shape: {weather_df.shape}")
    
    # 加载建筑数据
    building_df = loader.load_building_data('Hospitals')
    print("\nBuilding Data:")
    print(building_df.head())
    print(f"Shape: {building_df.shape}")
    
    # 合并数据
    merged_df = loader.merge_weather_building('Hospitals')
    print("\nMerged Data:")
    print(merged_df.head())
    print(f"Shape: {merged_df.shape}")
    print(f"\nColumns: {merged_df.columns.tolist()}")
    
    # 检查缺失值
    print(f"\nMissing values:\n{merged_df.isnull().sum()}")

if __name__ == '__main__':
    main()
