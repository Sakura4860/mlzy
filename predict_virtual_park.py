"""
虚拟园区能耗预测与光伏规模计算

目标：
1. 使用训练好的模型预测各类建筑的能耗
2. 计算整个园区的总能耗
3. 基于总能耗计算所需光伏装机容量
4. 生成可视化报告
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置
RESULTS_DIR = Path('results')
MODELS_DIR = RESULTS_DIR / 'models' / 'virtual_park'
METRICS_DIR = RESULTS_DIR / 'metrics' / 'virtual_park'
FIGURES_DIR = RESULTS_DIR / 'figures' / 'virtual_park'
PARK_DIR = RESULTS_DIR / 'virtual_park'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PARK_DIR.mkdir(parents=True, exist_ok=True)

# 园区配置（每种建筑的数量）
PARK_CONFIG = {
    'Hospitals': 2,      # 2栋医院
    'Restaurants': 5,    # 5个餐厅
    'Schools': 3,        # 3所学校
    'Shops': 8           # 8个商店
}

# 光伏系统参数
PV_PARAMS = {
    'panel_efficiency': 0.20,      # 光伏板效率 20%
    'system_loss': 0.15,           # 系统损失 15%
    'capacity_factor': 0.15,       # 容量系数 15% (瑞士年均)
    'peak_sun_hours': 3.5,         # 瑞士年均峰值日照小时数
    'panel_area_per_kw': 5.0,      # 每kW需要约5平方米
    'cost_per_kw': 1500,           # 每kW安装成本（瑞士法郎）
    'lifetime_years': 25,          # 光伏系统寿命
    'degradation_rate': 0.005      # 年衰减率 0.5%
}

def load_model_and_scaler(building_type):
    """加载模型和归一化器"""
    model_file = MODELS_DIR / f'rf_lite_{building_type.lower()}.pkl'
    scaler_file = MODELS_DIR / f'scaler_{building_type.lower()}.pkl'
    features_file = MODELS_DIR / f'features_{building_type.lower()}.txt'
    
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    with open(features_file, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, scaler, features

def load_building_predictions(building_type):
    """加载或生成建筑预测数据"""
    
    # 检查是否已有预测文件
    pred_file = PARK_DIR / f'predictions_{building_type.lower()}.csv'
    
    if pred_file.exists():
        print(f"加载已有预测: {building_type}")
        return pd.read_csv(pred_file)
    
    print(f"生成新预测: {building_type}")
    
    # 加载模型
    model, scaler, features = load_model_and_scaler(building_type)
    
    # 加载测试数据（使用训练时的数据）
    # 这里简化处理，使用完整数据集的后20%作为预测期
    import sys
    sys.path.append('src')
    from feature_engineering import FeatureEngineer
    
    # 加载数据
    DATA_DIR = Path('data')
    building_files = {
        'Hospitals': 'Hospitals_1991_2000_Full_retrofit.xlsx',
        'Restaurants': 'Restaurants_1991_2000_Full_retrofit.xlsx',
        'Schools': 'Schools_2010_2015_Full_retrofit.xlsx',
        'Shops': 'Shops_1991_2000_Full_retrofit.xlsx'
    }
    
    # 加载天气数据
    df_weather = pd.read_excel(DATA_DIR / 'WEATHER_DATA_ZURICH_2020_2019.xlsx')
    
    # 检查列名并重命名
    column_mapping = {
        'TIME[h]': 'hour_index',
        'TAMB[C]': 'Temperature',
        'WS[m/s]': 'Wind',
        'DNI[kW/m2]': 'DNI',
        'DIF[kW/m2]': 'DIF',
        'GHI[kW/m2]': 'GHI'
    }
    
    available_columns = [col for col in column_mapping.keys() if col in df_weather.columns]
    df_weather = df_weather[available_columns]
    df_weather = df_weather.rename(columns=column_mapping)
    
    # 创建datetime
    from datetime import datetime, timedelta
    start_date = datetime(2019, 1, 1, 0, 0, 0)
    df_weather['datetime'] = [start_date + timedelta(hours=i) for i in range(len(df_weather))]
    
    if 'Humidity' not in df_weather.columns:
        df_weather['Humidity'] = 0.5
    
    # 加载建筑数据
    df_building = pd.read_excel(DATA_DIR / building_files[building_type])
    if 'DateTime' in df_building.columns:
        df_building = df_building.rename(columns={'DateTime': 'datetime'})
    elif 'time' in df_building.columns:
        df_building = df_building.rename(columns={'time': 'datetime'})
    
    # 处理日期格式
    if df_building['datetime'].dtype == 'object':
        df_building['datetime'] = '2019/' + df_building['datetime'].astype(str)
    df_building['datetime'] = pd.to_datetime(df_building['datetime'], format='%Y/%m/%d  %H:%M:%S', errors='coerce')
    df_building = df_building.dropna(subset=['datetime'])
    
    # 合并
    df = pd.merge(df_weather, df_building, on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.dropna()
    
    # 重命名为DateTime供特征工程使用
    df = df.rename(columns={'datetime': 'DateTime'})
    
    # 特征工程
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
    
    df_features = df_features.dropna()
    
    # 取后20%作为预测期
    test_start = int(len(df_features) * 0.8)
    test_df = df_features.iloc[test_start:].copy()
    
    # 准备特征
    available_features = [f for f in features if f in test_df.columns]
    X_test = test_df[available_features]
    y_test = test_df['Total_Energy_kWh']
    
    # 预测
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # 创建结果DataFrame
    result = pd.DataFrame({
        'datetime': test_df['DateTime'].values,
        'actual_kWh': y_test.values,
        'predicted_kWh': y_pred,
        'hour': test_df['hour'].values if 'hour' in test_df.columns else pd.to_datetime(test_df['DateTime']).dt.hour
    })
    
    # 保存
    result.to_csv(pred_file, index=False)
    print(f"预测已保存: {pred_file}")
    
    return result

def calculate_park_total_energy():
    """计算园区总能耗"""
    
    print("\n" + "="*80)
    print("虚拟园区总能耗计算")
    print("="*80)
    
    print("\n园区配置:")
    for building, count in PARK_CONFIG.items():
        print(f"  {building}: {count}栋")
    
    # 加载所有建筑的预测
    all_predictions = {}
    for building_type in PARK_CONFIG.keys():
        df_pred = load_building_predictions(building_type)
        all_predictions[building_type] = df_pred
    
    # 计算总能耗
    # 假设所有建筑的时间序列对齐
    base_datetime = all_predictions[list(PARK_CONFIG.keys())[0]]['datetime']
    
    park_energy = pd.DataFrame({
        'datetime': base_datetime
    })
    
    # 每种建筑乘以数量
    total_predicted = np.zeros(len(base_datetime))
    total_actual = np.zeros(len(base_datetime))
    
    building_contributions = {}
    
    for building_type, count in PARK_CONFIG.items():
        df_pred = all_predictions[building_type]
        
        # 确保长度一致（取最小长度）
        min_len = min(len(total_predicted), len(df_pred))
        
        predicted_contribution = df_pred['predicted_kWh'].values[:min_len] * count
        actual_contribution = df_pred['actual_kWh'].values[:min_len] * count
        
        total_predicted[:min_len] += predicted_contribution
        total_actual[:min_len] += actual_contribution
        
        building_contributions[building_type] = {
            'predicted_kWh': predicted_contribution,
            'actual_kWh': actual_contribution,
            'count': count
        }
    
    park_energy['total_predicted_kWh'] = total_predicted
    park_energy['total_actual_kWh'] = total_actual
    park_energy['hour'] = pd.to_datetime(park_energy['datetime']).dt.hour
    park_energy['date'] = pd.to_datetime(park_energy['datetime']).dt.date
    
    # 保存园区总能耗
    park_file = PARK_DIR / 'park_total_energy.csv'
    park_energy.to_csv(park_file, index=False)
    print(f"\n园区总能耗已保存: {park_file}")
    
    # 统计信息
    print("\n园区能耗统计 (预测值):")
    print(f"  总时长: {len(park_energy)} 小时")
    print(f"  平均功率: {park_energy['total_predicted_kWh'].mean():.2f} kWh/h")
    print(f"  峰值功率: {park_energy['total_predicted_kWh'].max():.2f} kWh/h")
    print(f"  最低功率: {park_energy['total_predicted_kWh'].min():.2f} kWh/h")
    print(f"  日均能耗: {park_energy['total_predicted_kWh'].sum() / (len(park_energy)/24):.2f} kWh/天")
    print(f"  年均能耗: {park_energy['total_predicted_kWh'].sum() / (len(park_energy)/8760):.2f} kWh/年")
    
    return park_energy, building_contributions

def calculate_pv_capacity(park_energy):
    """计算所需光伏容量"""
    
    print("\n" + "="*80)
    print("光伏系统规模计算")
    print("="*80)
    
    # 年总能耗（kWh）
    hours_in_data = len(park_energy)
    annual_energy_kwh = park_energy['total_predicted_kWh'].sum() / hours_in_data * 8760
    
    print(f"\n年总能耗: {annual_energy_kwh:,.0f} kWh")
    
    # 方法1: 基于年能耗和容量系数
    # PV年发电量 = 装机容量(kW) × 8760(h) × 容量系数
    capacity_method1 = annual_energy_kwh / (8760 * PV_PARAMS['capacity_factor'])
    
    print(f"\n方法1: 基于容量系数({PV_PARAMS['capacity_factor']*100}%)")
    print(f"  所需装机容量: {capacity_method1:,.0f} kW ({capacity_method1/1000:.2f} MW)")
    
    # 方法2: 基于峰值功率
    peak_power_kw = park_energy['total_predicted_kWh'].max()
    capacity_method2 = peak_power_kw * 1.2  # 增加20%余量
    
    print(f"\n方法2: 基于峰值功率 + 20%余量")
    print(f"  园区峰值功率: {peak_power_kw:.2f} kW")
    print(f"  所需装机容量: {capacity_method2:,.0f} kW ({capacity_method2/1000:.2f} MW)")
    
    # 方法3: 基于日均峰值日照时数
    # 日均能耗 / 日均峰值日照时数
    daily_energy = park_energy.groupby('date')['total_predicted_kWh'].sum().mean()
    capacity_method3 = daily_energy / PV_PARAMS['peak_sun_hours']
    
    print(f"\n方法3: 基于峰值日照时数({PV_PARAMS['peak_sun_hours']}h/天)")
    print(f"  日均能耗: {daily_energy:.2f} kWh")
    print(f"  所需装机容量: {capacity_method3:,.0f} kW ({capacity_method3/1000:.2f} MW)")
    
    # 推荐方案：取方法1和方法3的较大值（更保守）
    recommended_capacity = max(capacity_method1, capacity_method3)
    
    print(f"\n{'='*80}")
    print(f"推荐装机容量: {recommended_capacity:,.0f} kW ({recommended_capacity/1000:.2f} MW)")
    print(f"{'='*80}")
    
    # 系统规模和成本
    panel_area = recommended_capacity * PV_PARAMS['panel_area_per_kw']
    installation_cost = recommended_capacity * PV_PARAMS['cost_per_kw']
    
    print(f"\n光伏系统详细参数:")
    print(f"  装机容量: {recommended_capacity:,.0f} kW")
    print(f"  光伏板面积: {panel_area:,.0f} m² ({panel_area/10000:.2f} 公顷)")
    print(f"  安装成本: {installation_cost:,.0f} CHF ({installation_cost/1e6:.2f} 百万瑞士法郎)")
    print(f"  年发电量: {recommended_capacity * 8760 * PV_PARAMS['capacity_factor']:,.0f} kWh")
    print(f"  自给率: {(recommended_capacity * 8760 * PV_PARAMS['capacity_factor'] / annual_energy_kwh * 100):.1f}%")
    print(f"  设计寿命: {PV_PARAMS['lifetime_years']} 年")
    
    # 保存结果
    pv_results = {
        'park_annual_energy_kwh': float(annual_energy_kwh),
        'peak_power_kw': float(peak_power_kw),
        'daily_average_kwh': float(daily_energy),
        'calculation_methods': {
            'method1_capacity_factor': {
                'capacity_kw': float(capacity_method1),
                'description': f'基于容量系数{PV_PARAMS["capacity_factor"]*100}%'
            },
            'method2_peak_power': {
                'capacity_kw': float(capacity_method2),
                'description': '基于峰值功率 + 20%余量'
            },
            'method3_peak_sun_hours': {
                'capacity_kw': float(capacity_method3),
                'description': f'基于峰值日照时数{PV_PARAMS["peak_sun_hours"]}h'
            }
        },
        'recommended': {
            'capacity_kw': float(recommended_capacity),
            'capacity_mw': float(recommended_capacity/1000),
            'panel_area_m2': float(panel_area),
            'panel_area_hectare': float(panel_area/10000),
            'installation_cost_chf': float(installation_cost),
            'installation_cost_million_chf': float(installation_cost/1e6),
            'annual_generation_kwh': float(recommended_capacity * 8760 * PV_PARAMS['capacity_factor']),
            'self_sufficiency_rate': float(recommended_capacity * 8760 * PV_PARAMS['capacity_factor'] / annual_energy_kwh * 100),
            'lifetime_years': PV_PARAMS['lifetime_years']
        },
        'pv_parameters': PV_PARAMS,
        'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    pv_file = PARK_DIR / 'pv_system_design.json'
    with open(pv_file, 'w', encoding='utf-8') as f:
        json.dump(pv_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n光伏系统设计已保存: {pv_file}")
    
    return pv_results

def visualize_park_energy(park_energy, building_contributions, pv_results):
    """可视化园区能耗和光伏系统"""
    
    print("\n生成可视化图表...")
    
    # 图1: 园区总能耗时间序列
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 子图1: 预测 vs 实际
    ax1 = axes[0]
    park_energy_sample = park_energy.iloc[:24*7]  # 显示一周数据
    ax1.plot(park_energy_sample['datetime'], park_energy_sample['total_actual_kWh'], 
             label='实际能耗', alpha=0.7, linewidth=2)
    ax1.plot(park_energy_sample['datetime'], park_energy_sample['total_predicted_kWh'], 
             label='预测能耗', alpha=0.7, linewidth=2, linestyle='--')
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('能耗 (kWh)', fontsize=12)
    ax1.set_title('虚拟园区总能耗 - 一周示例', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 子图2: 日均负荷曲线
    ax2 = axes[1]
    hourly_avg = park_energy.groupby('hour')['total_predicted_kWh'].mean()
    ax2.bar(hourly_avg.index, hourly_avg.values, alpha=0.7, color='steelblue')
    ax2.set_xlabel('小时', fontsize=12)
    ax2.set_ylabel('平均能耗 (kWh)', fontsize=12)
    ax2.set_title('园区日均负荷曲线', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_file1 = FIGURES_DIR / 'park_energy_analysis.png'
    plt.savefig(fig_file1, dpi=300, bbox_inches='tight')
    print(f"保存图表: {fig_file1}")
    plt.close()
    
    # 图2: 各建筑类型能耗占比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 饼图: 能耗占比
    ax1 = axes[0]
    building_totals = {}
    for building, data in building_contributions.items():
        building_totals[building] = data['predicted_kWh'].sum()
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax1.pie(
        building_totals.values(), 
        labels=[f"{k}\n({v}栋)" for k, v in PARK_CONFIG.items()],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    ax1.set_title('各建筑类型能耗占比', fontsize=14, fontweight='bold')
    
    # 柱状图: 绝对能耗
    ax2 = axes[1]
    buildings = list(building_totals.keys())
    energies = [building_totals[b]/1000 for b in buildings]  # 转换为MWh
    bars = ax2.bar(buildings, energies, color=colors, alpha=0.7)
    ax2.set_ylabel('总能耗 (MWh)', fontsize=12)
    ax2.set_title('各建筑类型总能耗', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_file2 = FIGURES_DIR / 'building_energy_distribution.png'
    plt.savefig(fig_file2, dpi=300, bbox_inches='tight')
    print(f"保存图表: {fig_file2}")
    plt.close()
    
    # 图3: 光伏系统可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 容量计算方法对比
    ax1 = axes[0, 0]
    methods = ['方法1\n容量系数', '方法2\n峰值功率', '方法3\n日照时数', '推荐\n容量']
    capacities = [
        pv_results['calculation_methods']['method1_capacity_factor']['capacity_kw']/1000,
        pv_results['calculation_methods']['method2_peak_power']['capacity_kw']/1000,
        pv_results['calculation_methods']['method3_peak_sun_hours']['capacity_kw']/1000,
        pv_results['recommended']['capacity_mw']
    ]
    bars = ax1.bar(methods, capacities, color=['skyblue', 'lightgreen', 'salmon', 'gold'], alpha=0.8)
    bars[-1].set_edgecolor('red')
    bars[-1].set_linewidth(3)
    ax1.set_ylabel('装机容量 (MW)', fontsize=12)
    ax1.set_title('光伏容量计算方法对比', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子图2: 成本分析
    ax2 = axes[0, 1]
    cost_data = {
        '装机容量\n(MW)': pv_results['recommended']['capacity_mw'],
        '光伏板面积\n(公顷)': pv_results['recommended']['panel_area_hectare'],
        '安装成本\n(百万CHF)': pv_results['recommended']['installation_cost_million_chf']
    }
    bars = ax2.bar(cost_data.keys(), cost_data.values(), 
                   color=['steelblue', 'forestgreen', 'coral'], alpha=0.8)
    ax2.set_title('光伏系统规模与成本', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子图3: 能耗与发电对比
    ax3 = axes[1, 0]
    annual_demand = pv_results['park_annual_energy_kwh'] / 1000  # MWh
    annual_generation = pv_results['recommended']['annual_generation_kwh'] / 1000  # MWh
    categories = ['年总需求', '光伏年发电']
    values = [annual_demand, annual_generation]
    bars = ax3.bar(categories, values, color=['orangered', 'limegreen'], alpha=0.8)
    ax3.set_ylabel('能量 (MWh)', fontsize=12)
    ax3.set_title('能耗需求 vs 光伏发电', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子图4: 自给率饼图
    ax4 = axes[1, 1]
    self_sufficiency = pv_results['recommended']['self_sufficiency_rate']
    grid_dependency = 100 - self_sufficiency
    sizes = [self_sufficiency, grid_dependency]
    labels = [f'光伏自给\n{self_sufficiency:.1f}%', f'电网补充\n{grid_dependency:.1f}%']
    colors_pie = ['gold', 'lightgray']
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    ax4.set_title('光伏自给率', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_file3 = FIGURES_DIR / 'pv_system_design.png'
    plt.savefig(fig_file3, dpi=300, bbox_inches='tight')
    print(f"保存图表: {fig_file3}")
    plt.close()
    
    print(f"\n所有图表已保存至: {FIGURES_DIR}")

def generate_report():
    """生成完整报告"""
    
    print("\n" + "="*80)
    print("生成虚拟园区能耗与光伏系统报告")
    print("="*80)
    
    # 计算园区总能耗
    park_energy, building_contributions = calculate_park_total_energy()
    
    # 计算光伏容量
    pv_results = calculate_pv_capacity(park_energy)
    
    # 生成可视化
    visualize_park_energy(park_energy, building_contributions, pv_results)
    
    # 生成文字报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("虚拟园区能耗预测与光伏系统设计报告")
    report_lines.append("="*80)
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_lines.append("\n一、园区配置")
    report_lines.append("-" * 80)
    total_buildings = sum(PARK_CONFIG.values())
    report_lines.append(f"总建筑数: {total_buildings}栋\n")
    for building, count in PARK_CONFIG.items():
        report_lines.append(f"  - {building}: {count}栋")
    
    report_lines.append("\n二、园区能耗预测")
    report_lines.append("-" * 80)
    report_lines.append(f"预测时长: {len(park_energy)} 小时")
    report_lines.append(f"平均功率: {park_energy['total_predicted_kWh'].mean():.2f} kWh/h")
    report_lines.append(f"峰值功率: {park_energy['total_predicted_kWh'].max():.2f} kWh/h")
    report_lines.append(f"日均能耗: {park_energy.groupby('date')['total_predicted_kWh'].sum().mean():.2f} kWh/天")
    report_lines.append(f"年均能耗: {pv_results['park_annual_energy_kwh']:,.0f} kWh/年")
    
    report_lines.append("\n各建筑类型能耗占比:")
    building_totals = {}
    for building, data in building_contributions.items():
        building_totals[building] = data['predicted_kWh'].sum()
    total_energy = sum(building_totals.values())
    for building, energy in sorted(building_totals.items(), key=lambda x: -x[1]):
        percentage = energy / total_energy * 100
        report_lines.append(f"  - {building}: {energy/1000:.2f} MWh ({percentage:.1f}%)")
    
    report_lines.append("\n三、光伏系统设计")
    report_lines.append("-" * 80)
    report_lines.append(f"推荐装机容量: {pv_results['recommended']['capacity_kw']:,.0f} kW ({pv_results['recommended']['capacity_mw']:.2f} MW)")
    report_lines.append(f"光伏板面积: {pv_results['recommended']['panel_area_m2']:,.0f} m² ({pv_results['recommended']['panel_area_hectare']:.2f} 公顷)")
    report_lines.append(f"安装成本: {pv_results['recommended']['installation_cost_chf']:,.0f} CHF ({pv_results['recommended']['installation_cost_million_chf']:.2f} 百万瑞士法郎)")
    report_lines.append(f"年发电量: {pv_results['recommended']['annual_generation_kwh']:,.0f} kWh")
    report_lines.append(f"光伏自给率: {pv_results['recommended']['self_sufficiency_rate']:.1f}%")
    report_lines.append(f"设计寿命: {pv_results['recommended']['lifetime_years']} 年")
    
    report_lines.append("\n光伏系统参数:")
    report_lines.append(f"  - 光伏板效率: {PV_PARAMS['panel_efficiency']*100}%")
    report_lines.append(f"  - 系统损失: {PV_PARAMS['system_loss']*100}%")
    report_lines.append(f"  - 容量系数: {PV_PARAMS['capacity_factor']*100}%")
    report_lines.append(f"  - 峰值日照时数: {PV_PARAMS['peak_sun_hours']} 小时/天")
    report_lines.append(f"  - 年衰减率: {PV_PARAMS['degradation_rate']*100}%")
    
    report_lines.append("\n四、经济分析")
    report_lines.append("-" * 80)
    levelized_cost = pv_results['recommended']['installation_cost_chf'] / (pv_results['recommended']['annual_generation_kwh'] * pv_results['recommended']['lifetime_years'])
    report_lines.append(f"平准化度电成本 (LCOE): {levelized_cost:.4f} CHF/kWh")
    
    # 假设电网电价
    grid_price = 0.20  # CHF/kWh
    annual_savings = pv_results['recommended']['annual_generation_kwh'] * grid_price
    payback_years = pv_results['recommended']['installation_cost_chf'] / annual_savings
    report_lines.append(f"年节省电费 (假设0.20 CHF/kWh): {annual_savings:,.0f} CHF")
    report_lines.append(f"静态回收期: {payback_years:.1f} 年")
    
    report_lines.append("\n五、结论与建议")
    report_lines.append("-" * 80)
    report_lines.append("1. 推荐安装 {:.2f} MW 光伏系统，可满足园区 {:.1f}% 的电力需求".format(
        pv_results['recommended']['capacity_mw'],
        pv_results['recommended']['self_sufficiency_rate']
    ))
    report_lines.append("2. 光伏板需占地约 {:.2f} 公顷，建议安装在建筑屋顶和停车场棚顶".format(
        pv_results['recommended']['panel_area_hectare']
    ))
    report_lines.append("3. 初期投资约 {:.2f} 百万瑞士法郎，预计 {:.1f} 年回本".format(
        pv_results['recommended']['installation_cost_million_chf'],
        payback_years
    ))
    report_lines.append("4. 建议配置储能系统（容量约为光伏装机的30%）以提高自给率")
    report_lines.append("5. 建议实施需求响应机制，在光伏高峰发电时段调整可控负荷")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("报告生成完毕")
    report_lines.append("="*80)
    
    # 保存报告
    report_file = PARK_DIR / 'virtual_park_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n完整报告已保存: {report_file}")
    
    # 打印报告
    print('\n'.join(report_lines))

if __name__ == '__main__':
    generate_report()
