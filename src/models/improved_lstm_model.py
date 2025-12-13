"""
Improved LSTM model with attention mechanism, better regularization and training strategies.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttentionLayer(nn.Module):
    """注意力层 - 帮助模型关注重要的时间步"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            weighted context: (batch, hidden_size)
        """
        # 计算注意力权重
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # 归一化
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context, attention_weights


class ImprovedLSTMNetwork(nn.Module):
    """改进的LSTM网络 - 添加注意力、残差连接和层归一化"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(ImprovedLSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入投影层（可选，帮助特征学习）
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # 单向，因为是时间序列预测
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 注意力机制
        self.attention = AttentionLayer(hidden_size)
        
        # 输出层（更深的网络）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            output: (batch, 1)
        """
        batch_size = x.size(0)
        
        # 输入投影
        x_proj = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # LSTM层
        lstm_out, _ = self.lstm(x_proj)  # (batch, seq_len, hidden_size)
        
        # 层归一化
        lstm_out = self.layer_norm(lstm_out)
        
        # 注意力机制
        context, attention_weights = self.attention(lstm_out)  # (batch, hidden_size)
        
        # 输出层
        output = self.fc(context)  # (batch, 1)
        
        return output


class ImprovedLSTMModel:
    """改进的LSTM模型类"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3,
                 batch_size=64, epochs=100, learning_rate=0.001, 
                 weight_decay=1e-5, patience=15, device=None):
        """
        初始化改进的LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            batch_size: 批次大小
            epochs: 最大训练轮数
            learning_rate: 初始学习率
            weight_decay: 权重衰减（L2正则化）
            patience: 早停耐心值
            device: 设备 ('cpu', 'cuda')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = ImprovedLSTMNetwork(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.model_name = 'ImprovedLSTM'
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器 - 验证集不改善时降低学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Initialized {self.model_name} on device: {self.device}")
        logger.info(f"Model parameters: hidden={hidden_size}, layers={num_layers}, dropout={dropout}")
        logger.info(f"Training: lr={learning_rate}, weight_decay={weight_decay}, patience={patience}")
    
    def create_sequences(self, X, y, sequence_length=48):
        """创建时间序列序列"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, sequence_length=48):
        """
        训练模型（带早停和学习率调度）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            sequence_length: 序列长度
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"Sequence length: {sequence_length}")
        
        # 创建序列
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, sequence_length)
        logger.info(f"Training sequences: {X_train_seq.shape}")
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 验证集
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, sequence_length)
            val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            logger.info(f"Validation sequences: {X_val_seq.shape}")
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
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
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # 保存最佳模型状态
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.patience_counter += 1
                
                # 打印进度
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.epochs}] "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Best Val: {self.best_val_loss:.4f}, Patience: {self.patience_counter}/{self.patience}"
                    )
                
                # 早停
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    # 恢复最佳模型
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f}")
        
        logger.info(f"{self.model_name} training completed")
        if val_loader is not None:
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def predict(self, X, sequence_length=48):
        """预测"""
        self.model.eval()
        
        # 创建序列
        X_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
        
        if len(X_seq) == 0:
            raise ValueError(f"Input length {len(X)} is too short for sequence_length {sequence_length}")
        
        X_seq = np.array(X_seq)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()
        
        # 为前sequence_length个点填充NaN
        full_predictions = np.full(len(X), np.nan)
        full_predictions[sequence_length:] = predictions
        
        return full_predictions
    
    def evaluate(self, X, y, sequence_length=48):
        """评估模型"""
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
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Model loaded from {filepath}")
