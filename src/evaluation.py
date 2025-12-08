"""
模型评估模块
提供统一的模型评估和比较功能
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name, **kwargs):
        """
        评估单个模型
        
        Args:
            model: 模型对象
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称
            **kwargs: 额外参数（如LSTM的sequence_length）
            
        Returns:
            评估指标字典
        """
        logger.info(f"Evaluating {model_name}...")
        
        metrics = model.evaluate(X_test, y_test, **kwargs)
        self.results[model_name] = metrics
        
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        比较所有模型
        
        Returns:
            比较结果DataFrame
        """
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(self.results).T
        
        # 添加排名
        for metric in df.columns:
            if metric in ['RMSE', 'MAE', 'MAPE']:
                # 越小越好
                df[f'{metric}_rank'] = df[metric].rank()
            else:
                # 越大越好（R2）
                df[f'{metric}_rank'] = df[metric].rank(ascending=False)
        
        # 计算平均排名
        rank_cols = [col for col in df.columns if '_rank' in col]
        df['avg_rank'] = df[rank_cols].mean(axis=1)
        
        # 排序
        df = df.sort_values('avg_rank')
        
        logger.info("\n" + "="*60)
        logger.info("Model Comparison Results")
        logger.info("="*60)
        logger.info(f"\n{df.to_string()}")
        logger.info("="*60)
        
        return df
    
    def get_best_model(self, metric='RMSE') -> str:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳模型名称
        """
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results).T
        
        if metric in ['RMSE', 'MAE', 'MAPE']:
            best_model = df[metric].idxmin()
        else:
            best_model = df[metric].idxmax()
        
        logger.info(f"Best model based on {metric}: {best_model}")
        return best_model
    
    def compute_improvement(self, baseline_model='Linear Regression'):
        """
        计算相对于基准模型的改进
        
        Args:
            baseline_model: 基准模型名称
            
        Returns:
            改进百分比DataFrame
        """
        if baseline_model not in self.results:
            logger.warning(f"Baseline model {baseline_model} not found")
            return None
        
        baseline_metrics = self.results[baseline_model]
        improvements = {}
        
        for model_name, metrics in self.results.items():
            if model_name == baseline_model:
                continue
            
            model_improvements = {}
            for metric in metrics:
                baseline_value = baseline_metrics[metric]
                current_value = metrics[metric]
                
                if metric in ['RMSE', 'MAE', 'MAPE']:
                    # 越小越好
                    improvement = (baseline_value - current_value) / baseline_value * 100
                else:
                    # 越大越好（R2）
                    improvement = (current_value - baseline_value) / abs(baseline_value) * 100
                
                model_improvements[f'{metric}_improvement'] = improvement
            
            improvements[model_name] = model_improvements
        
        df = pd.DataFrame(improvements).T
        
        logger.info(f"\nImprovement over {baseline_model} (%):")
        logger.info(f"\n{df.to_string()}")
        
        return df
    
    def export_results(self, filepath):
        """
        导出评估结果
        
        Args:
            filepath: 导出文件路径
        """
        df = pd.DataFrame(self.results).T
        df.to_csv(filepath)
        logger.info(f"Results exported to {filepath}")

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    
    return metrics
