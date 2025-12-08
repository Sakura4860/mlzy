"""
可视化模块
生成各种分析图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class Visualizer:
    """可视化器"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid', figsize=(12, 6), dpi=300):
        """
        初始化可视化器
        
        Args:
            style: 绘图风格
            figsize: 图表大小
            dpi: 分辨率
        """
        plt.style.use(style)
        self.figsize = figsize
        self.dpi = dpi
        sns.set_palette("husl")
    
    def plot_predictions(self, y_true, y_pred, model_name, 
                        save_path=None, show=True):
        """
        绘制真实值vs预测值对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
            show: 是否显示图表
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # 时间序列对比
        axes[0].plot(y_true, label='True Values', alpha=0.7, linewidth=1)
        axes[0].plot(y_pred, label='Predictions', alpha=0.7, linewidth=1)
        axes[0].set_xlabel('Time (hours)')
        axes[0].set_ylabel('Energy Consumption (kWh)')
        axes[0].set_title(f'{model_name}: Time Series Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 散点图
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # 绘制理想线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', label='Ideal', linewidth=2)
        
        axes[1].set_xlabel('True Values (kWh)')
        axes[1].set_ylabel('Predictions (kWh)')
        axes[1].set_title(f'{model_name}: Scatter Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_residuals(self, y_true, y_pred, model_name, 
                      save_path=None, show=True):
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
            show: 是否显示图表
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # 残差分布
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residuals (kWh)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name}: Residual Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # 残差散点图
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predictions (kWh)')
        axes[1].set_ylabel('Residuals (kWh)')
        axes[1].set_title(f'{model_name}: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, importance_df, top_k=20, 
                               save_path=None, show=True):
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            top_k: 显示前k个特征
            save_path: 保存路径
            show: 是否显示图表
        """
        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        
        top_features = importance_df.head(top_k)
        
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_k} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, results_df, save_path=None, show=True):
        """
        绘制模型对比图
        
        Args:
            results_df: 结果DataFrame
            save_path: 保存路径
            show: 是否显示图表
        """
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
            
            data = results_df[metric].sort_values(ascending=(metric != 'R2'))
            
            axes[idx].barh(range(len(data)), data.values)
            axes[idx].set_yticks(range(len(data)))
            axes[idx].set_yticklabels(data.index)
            axes[idx].set_xlabel(metric)
            axes[idx].set_title(f'Model Comparison: {metric}')
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # 标注数值
            for i, v in enumerate(data.values):
                axes[idx].text(v, i, f' {v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_history(self, train_losses, val_losses=None, 
                             save_path=None, show=True):
        """
        绘制训练历史（用于LSTM）
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 保存路径
            show: 是否显示图表
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('LSTM Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_matrix(self, df, columns=None, 
                               save_path=None, show=True):
        """
        绘制相关性矩阵
        
        Args:
            df: 数据框
            columns: 列名列表
            save_path: 保存路径
            show: 是否显示图表
        """
        if columns is not None:
            df = df[columns]
        
        plt.figure(figsize=(12, 10))
        
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_data_distribution(self, df, columns, 
                              save_path=None, show=True):
        """
        绘制数据分布图
        
        Args:
            df: 数据框
            columns: 列名列表
            save_path: 保存路径
            show: 是否显示图表
        """
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if col not in df.columns:
                continue
            
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_time_series(self, df, columns, datetime_col='DateTime',
                        save_path=None, show=True):
        """
        绘制时间序列图
        
        Args:
            df: 数据框
            columns: 列名列表
            datetime_col: 时间列
            save_path: 保存路径
            show: 是否显示图表
        """
        plt.figure(figsize=(14, len(columns) * 3))
        
        for idx, col in enumerate(columns, 1):
            if col not in df.columns:
                continue
            
            plt.subplot(len(columns), 1, idx)
            plt.plot(df[datetime_col], df[col], linewidth=1, alpha=0.8)
            plt.xlabel('Time')
            plt.ylabel(col)
            plt.title(f'Time Series: {col}')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
