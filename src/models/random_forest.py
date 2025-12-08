"""
随机森林模型
集成学习方法，可以输出特征重要性
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class RandomForestModel:
    """随机森林回归模型类"""
    
    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, n_jobs=-1, **kwargs):
        """
        初始化随机森林模型
        
        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            random_state: 随机种子
            n_jobs: 并行任务数
            **kwargs: 其他RandomForest参数
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        self.model_name = 'Random Forest'
        logger.info(f"Initialized {self.model_name} with n_estimators={n_estimators}")
    
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
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        return importance
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'n_features': self.model.n_features_in_
        }
        
        return info
