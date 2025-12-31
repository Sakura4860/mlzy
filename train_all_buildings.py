"""
虚拟园区能耗预测 - 训练所有建筑类型模型

目标：为每种建筑类型训练独立的RF Lite模型（12特征）
建筑类型：Hospitals, Restaurants, Schools, Shops
用途：构建虚拟园区，预测总能耗，计算所需光伏规模
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime

# 导入现有模块
import sys
sys.path.append('src')
from feature_engineering import FeatureEngineer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
MODELS_DIR = RESULTS_DIR / 'models' / 'virtual_park'
METRICS_DIR = RESULTS_DIR / 'metrics' / 'virtual_park'

# 创建目录
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# 建筑类型定义
BUILDING_TYPES = {
    'Hospitals': 'Hospitals_1991_2000_Full_retrofit.xlsx',
    'Restaurants': 'Restaurants_1991_2000_Full_retrofit.xlsx',
    'Schools': 'Schools_2010_2015_Full_retrofit.xlsx',
    'Shops': 'Shops_1991_2000_Full_retrofit.xlsx'
}

# RF Lite 最优参数（来自之前的训练结果）
RF_LITE_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': 42,
    'n_jobs': -1
}

# 精选的12个特征（无数据泄露）
LITE_FEATURES = [
    'Total_Energy_kWh_lag_1',
    'Total_Energy_kWh_lag_24',
    'Total_Energy_kWh_lag_2',
    'hour',
    'hour_sin',
    'hour_cos',
    'day_of_week',
    'is_weekend',
    'Total_Energy_kWh_rolling_std_24',
    'Total_Energy_kWh_rolling_mean_24',
    'Temperature',
    'GHI'
]

def load_weather_data():
    """加载天气数据"""
    weather_file = DATA_DIR / 'WEATHER_DATA_ZURICH_2020_2019.xlsx'
    print(f"加载天气数据: {weather_file}")
    
    df_weather = pd.read_excel(weather_file)
    
    # 检查列名
    # 原始列名: TIME[h], DNI[kW/m2], DIF[kW/m2], GHI[kW/m2], WS[m/s], TAMB[C]
    column_mapping = {
        'TIME[h]': 'hour_index',  # 1-8760的序号
        'TAMB[C]': 'Temperature',
        'WS[m/s]': 'Wind',
        'DNI[kW/m2]': 'DNI',
        'DIF[kW/m2]': 'DIF',
        'GHI[kW/m2]': 'GHI'
    }
    
    # 只保留需要的列
    available_columns = [col for col in column_mapping.keys() if col in df_weather.columns]
    df_weather = df_weather[available_columns]
    df_weather = df_weather.rename(columns=column_mapping)
    
    # 创建datetime（2019年，从1月1日0时开始）
    from datetime import datetime, timedelta
    start_date = datetime(2019, 1, 1, 0, 0, 0)
    df_weather['datetime'] = [start_date + timedelta(hours=i) for i in range(len(df_weather))]
    
    # 添加默认湿度（如果没有）
    if 'Humidity' not in df_weather.columns:
        df_weather['Humidity'] = 0.5  # 默认50%
    
    return df_weather

def load_building_data(building_type):
    """加载建筑能耗数据"""
    building_file = DATA_DIR / BUILDING_TYPES[building_type]
    print(f"加载建筑数据: {building_file}")
    
    df_building = pd.read_excel(building_file)
    
    # 确保有datetime列（检查DateTime或time）
    if 'DateTime' in df_building.columns:
        df_building = df_building.rename(columns={'DateTime': 'datetime'})
    elif 'time' in df_building.columns:
        df_building = df_building.rename(columns={'time': 'datetime'})
    
    # 处理日期格式（可能没有年份）
    # 格式: "01/01  01:00:00" -> "2019-01-01 01:00:00"
    if df_building['datetime'].dtype == 'object':
        # 添加年份2019
        df_building['datetime'] = '2019/' + df_building['datetime'].astype(str)
    
    df_building['datetime'] = pd.to_datetime(df_building['datetime'], format='%Y/%m/%d  %H:%M:%S', errors='coerce')
    
    # 移除无法解析的日期
    df_building = df_building.dropna(subset=['datetime'])
    
    return df_building

def merge_data(df_weather, df_building):
    """合并天气和建筑数据"""
    df = pd.merge(df_weather, df_building, on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 移除datetime后的缺失值
    df = df.dropna()
    
    # 重命名为DateTime供特征工程使用
    df = df.rename(columns={'datetime': 'DateTime'})
    
    return df

def train_test_split_timeseries(df, train_ratio=0.72, val_ratio=0.08):
    """时间序列数据分割"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def calculate_mape(y_true, y_pred):
    """计算MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_building_model(building_type):
    """为特定建筑类型训练RF Lite模型"""
    
    print(f"\n{'='*80}")
    print(f"开始训练建筑类型: {building_type}")
    print(f"{'='*80}")
    
    # 1. 加载数据
    df_weather = load_weather_data()
    df_building = load_building_data(building_type)
    df = merge_data(df_weather, df_building)
    
    print(f"合并后数据形状: {df.shape}")
    print(f"目标变量统计:")
    print(df['Total_Energy_kWh'].describe())
    
    # 2. 特征工程
    print("\n执行特征工程...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df.copy(), target_column='Total_Energy_kWh')
    
    # 移除泄露特征
    leakage_features = [
        'Electricity_kWh', 'Electricity_J',
        'HotWater_kWh', 'HotWater_J',
        'SpaceHeating_kWh', 'SpaceHeating_J',
        'SpaceCooling_kWh', 'SpaceCooling_J',
        'HotWater_SpaceHeating_kWh', 'HotWater_SpaceHeating_J',
        'Total_Energy_kWh_diff_1', 'Total_Energy_kWh_diff_24'
    ]
    
    existing_leakage = [f for f in leakage_features if f in df_features.columns]
    if existing_leakage:
        df_features = df_features.drop(columns=existing_leakage)
        print(f"移除泄露特征: {len(existing_leakage)}个")
    
    # 移除缺失值
    df_features = df_features.dropna()
    print(f"特征工程后数据形状: {df_features.shape}")
    
    # 3. 数据分割
    train_df, val_df, test_df = train_test_split_timeseries(df_features)
    
    print(f"\n数据分割:")
    print(f"  训练集: {len(train_df)} 样本 ({len(train_df)/len(df_features)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 样本 ({len(val_df)/len(df_features)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 样本 ({len(test_df)/len(df_features)*100:.1f}%)")
    
    # 保存datetime用于后续分析
    train_datetime = train_df['DateTime'].copy() if 'DateTime' in train_df.columns else None
    test_datetime = test_df['DateTime'].copy() if 'DateTime' in test_df.columns else None
    
    # 4. 准备训练数据（仅使用12个精选特征）
    # 检查特征是否存在
    missing_features = [f for f in LITE_FEATURES if f not in df_features.columns]
    if missing_features:
        print(f"\n⚠️ 缺失特征: {missing_features}")
        # 尝试查找替代特征
        available_features = [f for f in LITE_FEATURES if f in df_features.columns]
        print(f"可用特征: {len(available_features)}/{len(LITE_FEATURES)}")
        
        if len(available_features) < 8:
            raise ValueError(f"可用特征不足，无法训练模型")
        
        features_to_use = available_features
    else:
        features_to_use = LITE_FEATURES
    
    print(f"\n使用特征: {len(features_to_use)}个")
    print(features_to_use)
    
    X_train = train_df[features_to_use]
    X_val = val_df[features_to_use]
    X_test = test_df[features_to_use]
    
    y_train = train_df['Total_Energy_kWh']
    y_val = val_df['Total_Energy_kWh']
    y_test = test_df['Total_Energy_kWh']
    
    # 5. 归一化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. 训练RF Lite模型
    print(f"\n开始训练Random Forest Lite...")
    print(f"参数: {RF_LITE_PARAMS}")
    
    model = RandomForestRegressor(**RF_LITE_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    # 7. 评估模型
    print(f"\n模型评估:")
    
    # 训练集
    y_train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = calculate_mape(y_train, y_train_pred)
    
    print(f"  训练集: R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
    
    # 验证集
    y_val_pred = model.predict(X_val_scaled)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mape = calculate_mape(y_val, y_val_pred)
    
    print(f"  验证集: R2={val_r2:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, MAPE={val_mape:.2f}%")
    
    # 测试集
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = calculate_mape(y_test, y_test_pred)
    
    print(f"  测试集: R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")
    
    # 8. 特征重要性
    feature_importance = pd.DataFrame({
        'feature': features_to_use,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 特征重要性:")
    print(feature_importance.head())
    
    # 9. 保存模型和相关文件
    model_file = MODELS_DIR / f'rf_lite_{building_type.lower()}.pkl'
    scaler_file = MODELS_DIR / f'scaler_{building_type.lower()}.pkl'
    features_file = MODELS_DIR / f'features_{building_type.lower()}.txt'
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    with open(features_file, 'w') as f:
        f.write('\n'.join(features_to_use))
    
    print(f"\n模型已保存: {model_file}")
    print(f"归一化器已保存: {scaler_file}")
    print(f"特征列表已保存: {features_file}")
    
    # 10. 保存评估指标
    metrics = {
        'building_type': building_type,
        'model': 'RF_Lite',
        'n_features': len(features_to_use),
        'features': features_to_use,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train': {
            'r2': float(train_r2),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'mape': float(train_mape)
        },
        'val': {
            'r2': float(val_r2),
            'rmse': float(val_rmse),
            'mae': float(val_mae),
            'mape': float(val_mape)
        },
        'test': {
            'r2': float(test_r2),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'mape': float(test_mape)
        },
        'feature_importance': feature_importance.to_dict('records'),
        'energy_stats': {
            'mean_kwh': float(df['Total_Energy_kWh'].mean()),
            'std_kwh': float(df['Total_Energy_kWh'].std()),
            'min_kwh': float(df['Total_Energy_kWh'].min()),
            'max_kwh': float(df['Total_Energy_kWh'].max())
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metrics_file = METRICS_DIR / f'metrics_{building_type.lower()}.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"指标已保存: {metrics_file}")
    
    return metrics

def main():
    """主函数：训练所有建筑类型的模型"""
    
    print("="*80)
    print("虚拟园区能耗预测系统 - 模型训练")
    print("="*80)
    print(f"\n建筑类型: {list(BUILDING_TYPES.keys())}")
    print(f"模型: Random Forest Lite (12特征)")
    print(f"保存路径: {MODELS_DIR}")
    print()
    
    all_metrics = {}
    
    # 训练每种建筑类型
    for building_type in BUILDING_TYPES.keys():
        try:
            metrics = train_building_model(building_type)
            all_metrics[building_type] = metrics
            print(f"\n[SUCCESS] {building_type} 模型训练完成!")
            
        except Exception as e:
            print(f"\n[FAILED] {building_type} 模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    print(f"\n{'='*80}")
    print("训练汇总")
    print(f"{'='*80}\n")
    
    summary_data = []
    for building_type, metrics in all_metrics.items():
        summary_data.append({
            'Building': building_type,
            'R2': metrics['test']['r2'],
            'RMSE': metrics['test']['rmse'],
            'MAE': metrics['test']['mae'],
            'MAPE_pct': metrics['test']['mape'],
            'Features': metrics['n_features'],
            'Mean_kWh': metrics['energy_stats']['mean_kwh'],
            'Std_kWh': metrics['energy_stats']['std_kwh']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('R2', ascending=False)
    
    print(summary_df.to_string(index=False))
    
    # 保存汇总CSV
    summary_file = METRICS_DIR / 'all_buildings_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n汇总结果已保存: {summary_file}")
    
    # 保存完整指标JSON
    all_metrics_file = METRICS_DIR / 'all_buildings_metrics.json'
    with open(all_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"完整指标已保存: {all_metrics_file}")
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] 所有建筑类型模型训练完成!")
    print(f"{'='*80}")
    print(f"\n模型文件: {MODELS_DIR}")
    print(f"指标文件: {METRICS_DIR}")
    print(f"\n下一步: 运行 predict_virtual_park.py 预测园区总能耗")

if __name__ == '__main__':
    main()
