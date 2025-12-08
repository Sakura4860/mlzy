"""
主运行脚本
完整的机器学习实验流程
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime

# 导入自定义模块
from src.config import *
from src.utils import setup_logging, ensure_dir, save_json, print_data_info, split_time_series
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import LinearRegressionModel, SVRModel, KNNModel, RandomForestModel, LSTMModel
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer

# 设置日志
logger = setup_logging(LOG_LEVEL)

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Building Energy Consumption Prediction")
    logger.info("Machine Learning Course Project")
    logger.info("="*60)
    
    # 确保输出目录存在
    for dir_path in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
        ensure_dir(dir_path)
    
    # ========== 1. 数据加载 ==========
    logger.info("\n[Step 1] Loading Data...")
    loader = DataLoader()
    
    # 选择建筑类型
    building_type = SELECTED_BUILDINGS[0]
    logger.info(f"Selected building type: {building_type}")
    
    # 加载和合并数据
    df = loader.merge_weather_building(building_type)
    print_data_info(df, f"{building_type} Raw Data")
    
    # ========== 2. 数据预处理 ==========
    logger.info("\n[Step 2] Data Preprocessing...")
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    df = preprocessor.handle_missing_values(df, method='interpolate')
    
    # 检测异常值
    energy_cols = [col for col in df.columns if 'kWh' in col]
    df = preprocessor.detect_outliers_iqr(df, energy_cols, factor=2.0)
    df = preprocessor.handle_outliers(df, method='clip', columns=energy_cols)
    if 'is_outlier' in df.columns:
        df = df.drop('is_outlier', axis=1)
    
    logger.info(f"Data after preprocessing: {df.shape}")
    
    # ========== 3. 特征工程 ==========
    logger.info("\n[Step 3] Feature Engineering...")
    engineer = FeatureEngineer()
    
    df_features = engineer.create_all_features(df, target_column=TARGET_COLUMN)
    logger.info(f"Data after feature engineering: {df_features.shape}")
    
    # 获取特征列
    exclude_cols = ['DateTime', 'Building_type', 'Construction_period', 
                   'Retrofit_scenario', TARGET_COLUMN]
    exclude_cols = [col for col in exclude_cols if col in df_features.columns]
    feature_cols = engineer.get_feature_names(df_features, exclude_cols)
    
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # ========== 4. 数据分割 ==========
    logger.info("\n[Step 4] Splitting Data...")
    train_df, val_df, test_df = split_time_series(df_features, TEST_SIZE, VALIDATION_SIZE)
    
    logger.info(f"Train set: {train_df.shape}")
    logger.info(f"Validation set: {val_df.shape}")
    logger.info(f"Test set: {test_df.shape}")
    
    # 准备训练和测试数据
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COLUMN].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COLUMN].values
    
    # 归一化特征
    logger.info("Normalizing features...")
    X_train_scaled = preprocessor.normalize_features(
        pd.DataFrame(X_train, columns=feature_cols), 
        feature_cols, 
        method='minmax', 
        fit=True
    ).values
    
    X_val_scaled = preprocessor.normalize_features(
        pd.DataFrame(X_val, columns=feature_cols), 
        feature_cols, 
        method='minmax', 
        fit=False
    ).values
    
    X_test_scaled = preprocessor.normalize_features(
        pd.DataFrame(X_test, columns=feature_cols), 
        feature_cols, 
        method='minmax', 
        fit=False
    ).values
    
    # ========== 5. 模型训练 ==========
    logger.info("\n[Step 5] Training Models...")
    
    models = {}
    predictions = {}
    
    # 5.1 线性回归
    logger.info("\n--- Training Linear Regression ---")
    lr_model = LinearRegressionModel(**MODEL_PARAMS['linear_regression'])
    lr_model.train(X_train_scaled, y_train)
    models['Linear Regression'] = lr_model
    predictions['Linear Regression'] = lr_model.predict(X_test_scaled)
    
    # 5.2 SVR（可选：数据量大时可能很慢）
    logger.info("\n--- Training SVR ---")
    # 对于大数据集，可以采样训练
    sample_size = min(5000, len(X_train_scaled))
    sample_idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    svr_model = SVRModel(**MODEL_PARAMS['svr'])
    svr_model.train(X_train_scaled[sample_idx], y_train[sample_idx])
    models['SVR'] = svr_model
    predictions['SVR'] = svr_model.predict(X_test_scaled)
    
    # 5.3 KNN
    logger.info("\n--- Training KNN ---")
    knn_model = KNNModel(**MODEL_PARAMS['knn'])
    knn_model.train(X_train_scaled, y_train)
    models['KNN'] = knn_model
    predictions['KNN'] = knn_model.predict(X_test_scaled)
    
    # 5.4 随机森林
    logger.info("\n--- Training Random Forest ---")
    rf_model = RandomForestModel(**MODEL_PARAMS['random_forest'])
    rf_model.train(X_train_scaled, y_train)
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_model.predict(X_test_scaled)
    
    logger.info("\n[Step 5 Complete] 4 baseline models trained successfully!")
    logger.info("LSTM will be trained separately if needed.")
    
    # 保存训练数据用于特征精简实验
    logger.info("\nSaving training data for feature reduction experiments...")
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(METRICS_DIR / 'X_train_full.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(METRICS_DIR / 'X_test_full.csv', index=False)
    pd.Series(y_train).to_csv(METRICS_DIR / 'y_train.csv', index=False, header=['Total_Energy_kWh'])
    pd.Series(y_test).to_csv(METRICS_DIR / 'y_test.csv', index=False, header=['Total_Energy_kWh'])
    
    # 5.5 LSTM（可选：需要较长训练时间，单独运行）
    # logger.info("\n--- Training LSTM ---")
    # sequence_length = MODEL_PARAMS['lstm']['sequence_length']
    # lstm_model = LSTMModel(
    #     input_size=len(feature_cols),
    #     hidden_size=MODEL_PARAMS['lstm']['hidden_size'],
    #     num_layers=MODEL_PARAMS['lstm']['num_layers'],
    #     dropout=MODEL_PARAMS['lstm']['dropout'],
    #     batch_size=MODEL_PARAMS['lstm']['batch_size'],
    #     epochs=MODEL_PARAMS['lstm']['epochs'],
    #     learning_rate=MODEL_PARAMS['lstm']['learning_rate']
    # )
    # lstm_model.train(X_train_scaled, y_train, X_val_scaled, y_val, sequence_length)
    # models['LSTM'] = lstm_model
    # predictions['LSTM'] = lstm_model.predict(X_test_scaled, sequence_length)
    
    # ========== 6. 模型评估 ==========
    logger.info("\n[Step 6] Evaluating Models...")
    evaluator = ModelEvaluator()
    
    # 评估传统模型
    for model_name in ['Linear Regression', 'SVR', 'KNN', 'Random Forest']:
        evaluator.evaluate_model(models[model_name], X_test_scaled, y_test, model_name)
    
    # 评估LSTM（需要特殊处理NaN）- 单独训练LSTM后取消注释
    # lstm_pred = predictions['LSTM']
    # mask = ~np.isnan(lstm_pred)
    # lstm_metrics = evaluator.evaluate_model(
    #     models['LSTM'], 
    #     X_test_scaled, 
    #     y_test, 
    #     'LSTM',
    #     sequence_length=sequence_length
    # )
    
    # 模型比较
    comparison_df = evaluator.compare_models()
    
    # 计算改进
    improvement_df = evaluator.compute_improvement(baseline_model='Linear Regression')
    
    # 导出结果
    comparison_df.to_csv(METRICS_DIR / 'model_comparison.csv')
    if improvement_df is not None:
        improvement_df.to_csv(METRICS_DIR / 'model_improvement.csv')
    
    # ========== 7. 可视化 ==========
    logger.info("\n[Step 7] Generating Visualizations...")
    visualizer = Visualizer()
    
    # 7.1 每个模型的预测对比
    for model_name, y_pred in predictions.items():
        # LSTM特殊处理 - 单独训练LSTM后取消注释
        # if model_name == 'LSTM':
        #     mask = ~np.isnan(y_pred)
        #     y_true_plot = y_test[mask]
        #     y_pred_plot = y_pred[mask]
        # else:
        #     y_true_plot = y_test
        #     y_pred_plot = y_pred
        y_true_plot = y_test
        y_pred_plot = y_pred
        
        # 预测对比图
        visualizer.plot_predictions(
            y_true_plot, 
            y_pred_plot, 
            model_name,
            save_path=FIGURES_DIR / f'{model_name}_predictions.png',
            show=False
        )
        
        # 残差图
        visualizer.plot_residuals(
            y_true_plot, 
            y_pred_plot, 
            model_name,
            save_path=FIGURES_DIR / f'{model_name}_residuals.png',
            show=False
        )
    
    # 7.2 模型对比图
    visualizer.plot_model_comparison(
        comparison_df,
        save_path=FIGURES_DIR / 'model_comparison.png',
        show=False
    )
    
    # 7.3 特征重要性（随机森林）
    importance_df = rf_model.get_feature_importance(feature_cols)
    visualizer.plot_feature_importance(
        importance_df,
        top_k=20,
        save_path=FIGURES_DIR / 'feature_importance.png',
        show=False
    )
    importance_df.to_csv(METRICS_DIR / 'feature_importance.csv', index=False)
    
    # 7.4 LSTM训练历史 - 单独训练LSTM后取消注释
    # visualizer.plot_training_history(
    #     lstm_model.train_losses,
    #     lstm_model.val_losses,
    #     save_path=FIGURES_DIR / 'lstm_training_history.png',
    #     show=False
    # )
    
    # ========== 8. 保存模型 ==========
    logger.info("\n[Step 8] Saving Models...")
    from src.utils import save_model
    
    for model_name, model in models.items():
        if model_name == 'LSTM':
            model.save_model(MODELS_DIR / f'{model_name}.pth')
        else:
            save_model(model, MODELS_DIR / f'{model_name}.pkl')
    
    logger.info(f"Models saved to {MODELS_DIR}")
    
    # ========== 9. 生成报告摘要 ==========
    logger.info("\n[Step 9] Generating Report Summary...")
    
    best_model = evaluator.get_best_model(metric='RMSE')
    
    summary = {
        'project': 'Building Energy Consumption Prediction',
        'building_type': building_type,
        'target': TARGET_COLUMN,
        'n_samples': len(df_features),
        'n_features': len(feature_cols),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'models': list(models.keys()),
        'best_model': best_model,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_json(summary, RESULTS_DIR / 'experiment_summary.json')
    
    logger.info("\n" + "="*60)
    logger.info("Experiment Completed Successfully!")
    logger.info(f"Best Model: {best_model}")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("="*60)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)
