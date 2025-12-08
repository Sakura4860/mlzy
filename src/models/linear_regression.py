"""
线性回归模型
作为基准模型（Baseline）
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class LinearRegressionModel:
    """线性回归模型类"""
    
    def __init__(self, **kwargs):
        """
        初始化线性回归模型
        
        Args:
            **kwargs: sklearn LinearRegression参数
        """
        self.model = LinearRegression(**kwargs)
        self.model_name = 'Linear Regression'
        logger.info(f"Initialized {self.model_name}")
    
    def train(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training completed")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征
            
        Returns:
            预测值
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征
            y: 真实标签
            
        Returns:
            评估指标字典
        """
        y_pred = self.predict(X)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred),
            'MAPE': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """
        获取特征系数（线性回归的权重）
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        import pandas as pd
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_
        })
        importance['abs_coefficient'] = np.abs(importance['coefficient'])
        importance = importance.sort_values('abs_coefficient', ascending=False)
        
        return importance
