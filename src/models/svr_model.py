"""
支持向量回归模型
使用RBF核函数处理非线性关系
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class SVRModel:
    """支持向量回归模型类"""
    
    def __init__(self, kernel='rbf', C=100.0, gamma='scale', epsilon=0.1, **kwargs):
        """
        初始化SVR模型
        
        Args:
            kernel: 核函数类型
            C: 正则化参数
            gamma: 核系数
            epsilon: epsilon-tube宽度
            **kwargs: 其他SVR参数
        """
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, **kwargs)
        self.model_name = 'Support Vector Regression'
        logger.info(f"Initialized {self.model_name} with kernel={kernel}, C={C}")
    
    def train(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        logger.info(f"Training {self.model_name}...")
        logger.info("Note: SVR training may take several minutes on large datasets")
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
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'n_support': self.model.n_support_,
            'support_vectors_shape': self.model.support_vectors_.shape,
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma,
            'epsilon': self.model.epsilon
        }
        
        return info
