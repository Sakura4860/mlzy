"""
LSTM模型
长短期记忆网络，用于处理时间序列的长期依赖关系
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, X, y):
        """
        初始化数据集
        
        Args:
            X: 特征 (samples, sequence_length, features)
            y: 标签 (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMNetwork(nn.Module):
    """LSTM网络"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """
        初始化LSTM网络
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入 (batch, sequence, features)
            
        Returns:
            输出 (batch, 1)
        """
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output

class LSTMModel:
    """LSTM模型类"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 batch_size=64, epochs=50, learning_rate=0.001, device=None):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            device: 设备 ('cpu', 'cuda')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = LSTMNetwork(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.model_name = 'LSTM'
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Initialized {self.model_name} on device: {self.device}")
        logger.info(f"Model parameters: hidden_size={hidden_size}, num_layers={num_layers}")
    
    def create_sequences(self, X, y, sequence_length=24):
        """
        创建时间序列序列
        
        Args:
            X: 特征 (samples, features)
            y: 标签 (samples,)
            sequence_length: 序列长度
            
        Returns:
            序列特征和标签
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, sequence_length=24):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            sequence_length: 序列长度
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"Creating sequences with length={sequence_length}")
        
        # 创建序列
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, sequence_length)
        logger.info(f"Training sequences shape: {X_train_seq.shape}")
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 验证集
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, sequence_length)
            val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = self.criterion(outputs.squeeze(), y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.epochs}] "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f}")
        
        logger.info(f"{self.model_name} training completed")
    
    def predict(self, X, sequence_length=24):
        """
        预测
        
        Args:
            X: 特征
            sequence_length: 序列长度
            
        Returns:
            预测值
        """
        self.model.eval()
        
        # 创建序列
        X_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
        
        if len(X_seq) == 0:
            raise ValueError(f"Input length {len(X)} is too short for sequence_length {sequence_length}")
        
        X_seq = np.array(X_seq)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()
        
        # 为前sequence_length个点填充NaN（因为无法预测）
        full_predictions = np.full(len(X), np.nan)
        full_predictions[sequence_length:] = predictions
        
        return full_predictions
    
    def evaluate(self, X, y, sequence_length=24):
        """
        评估模型
        
        Args:
            X: 特征
            y: 真实标签
            sequence_length: 序列长度
            
        Returns:
            评估指标字典
        """
        y_pred = self.predict(X, sequence_length)
        
        # 去除NaN值
        mask = ~np.isnan(y_pred)
        y_true = y[mask]
        y_pred = y_pred[mask]
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        return metrics
    
    def save_model(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Model loaded from {filepath}")
