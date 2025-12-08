"""
K近邻回归模型
基于"相似的天气和时间会有相似的能耗"假设
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class KNNModel:
    """K近邻回归模型类"""
    
    def __init__(self, n_neighbors=5, weights='distance', algorithm='auto', **kwargs):
        """
        初始化KNN模型
        
        Args:
            n_neighbors: 近邻数量
            weights: 权重方式 ('uniform', 'distance')
            algorithm: 算法 ('auto', 'ball_tree', 'kd_tree', 'brute')
            **kwargs: 其他KNN参数
        """
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            **kwargs
        )
        self.model_name = 'K-Nearest Neighbors'
        logger.info(f"Initialized {self.model_name} with n_neighbors={n_neighbors}")
    
    def train(self, X_train, y_train):
        """
        训练模型（KNN实际上是存储训练数据）
        
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
    
    def get_neighbors(self, X, n_neighbors=None):
        """
        获取最近邻
        
        Args:
            X: 特征
            n_neighbors: 近邻数量
            
        Returns:
            距离和索引
        """
        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors
        
        distances, indices = self.model.kneighbors(X, n_neighbors=n_neighbors)
        return distances, indices
