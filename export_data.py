"""
导出训练数据用于特征精简实验
"""
import pandas as pd
import joblib
from pathlib import Path
from src.config import RESULTS_DIR, MODELS_DIR
from src.utils import setup_logging

logger = setup_logging()

def export_train_data():
    """从已保存的模型和scaler重新生成训练数据"""
    logger.info("Exporting training data for lite model experiment...")
    
    # 创建metrics目录
    metrics_dir = RESULTS_DIR / 'metrics'
    metrics_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载原始数据并重新处理
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
    from src.feature_engineering import FeatureEngineer
    from src.config import TARGET_COLUMN
    
    # 加载数据
    loader = DataLoader()
    df = loader.load_and_merge()
    
    # 预处理
    preprocessor = Preprocessor()
    df = preprocessor.preprocess(df)
    
    # 特征工程
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_all_features(df, target_column=TARGET_COLUMN)
    
    # 划分数据
    train_size = int(0.72 * len(df))
    val_size = int(0.08 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # 提取特征和目标
    exclude_cols = ['DateTime', 'Building_type', 'Construction_period', 
                    'Retrofit_scenario', TARGET_COLUMN]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]
    
    # 加载scaler并归一化
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 保存数据
    logger.info(f"Saving data to {metrics_dir}")
    X_train_scaled.to_csv(metrics_dir / 'X_train_full.csv', index=False)
    X_test_scaled.to_csv(metrics_dir / 'X_test_full.csv', index=False)
    y_train.to_csv(metrics_dir / 'y_train.csv', index=False)
    y_test.to_csv(metrics_dir / 'y_test.csv', index=False)
    
    logger.info(f"✅ Data exported successfully!")
    logger.info(f"   X_train: {X_train_scaled.shape}")
    logger.info(f"   X_test: {X_test_scaled.shape}")
    logger.info(f"   Features: {len(feature_cols)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    export_train_data()
