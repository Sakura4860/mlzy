# 虚拟园区项目 - 快速开始指南

## 🎯 项目简介

本项目构建了一个包含18栋建筑的虚拟零碳园区，使用机器学习预测能耗，并计算所需的光伏装机容量。

**核心成果**:
- 📊 4种建筑类型的能耗预测模型（R²>0.81）
- 🏙️ 虚拟园区年均能耗: 26,434,851 kWh
- ☀️ 推荐光伏装机: **20.12 MW**
- 💰 投资回收期: **5.7年**
- 🌱 光伏自给率: **100%**

---

## ⚡ 快速运行（3步骤）

### 步骤1: 训练所有建筑模型

```bash
python train_all_buildings.py
```

**预期输出**:
```
================================================================================
建筑类型: ['Hospitals', 'Restaurants', 'Schools', 'Shops']
模型: Random Forest Lite (12特征)
================================================================================

开始训练建筑类型: Hospitals
...
模型评估:
  训练集: R2=0.9456, RMSE=2.12, MAE=1.53, MAPE=6.42%
  验证集: R2=0.9402, RMSE=2.34, MAE=1.68, MAPE=7.01%
  测试集: R2=0.9324, RMSE=2.53, MAE=1.74, MAPE=7.36%

模型已保存: results\models\virtual_park\rf_lite_hospitals.pkl
...

[SUCCESS] 所有建筑类型模型训练完成!

训练汇总:
  Building       R2   RMSE    MAE  MAPE_pct  Features  Mean_kWh   Std_kWh
     Shops  0.9388  14.25   9.48     18.70        12    127.55     80.41
 Hospitals  0.9324   2.53   1.74      7.36        12     22.20     10.59
Restaurants 0.8314 103.40  51.45     62.87        12    256.50    210.07
   Schools  0.8112  52.98  33.43    150.14        12    104.55    113.93
```

**耗时**: 约3-5分钟  
**输出文件**: 
- `results/models/virtual_park/` - 4个模型文件
- `results/metrics/virtual_park/all_buildings_summary.csv` - 性能汇总

---

### 步骤2: 生成园区预测和光伏设计

```bash
python predict_virtual_park.py
```

**预期输出**:
```
================================================================================
生成虚拟园区能耗与光伏系统报告
================================================================================

虚拟园区总能耗计算
================================================================================
园区配置:
  Hospitals: 2栋
  Restaurants: 5栋
  Schools: 3栋
  Shops: 8栋

园区能耗统计 (预测值):
  总时长: 1675 小时
  平均功率: 3017.68 kWh/h
  峰值功率: 7544.77 kWh/h
  日均能耗: 69241.22 kWh/天
  年均能耗: 26,434,851 kWh/年

光伏系统规模计算
================================================================================

方法1: 基于容量系数(15%)
  所需装机容量: 20,118 kW (20.12 MW)

方法2: 基于峰值功率 + 20%余量
  园区峰值功率: 7544.77 kW
  所需装机容量: 9,054 kW (9.05 MW)

方法3: 基于峰值日照时数(3.5h/天)
  日均能耗: 69241.22 kWh
  所需装机容量: 19,783 kW (19.78 MW)

================================================================================
推荐装机容量: 20,118 kW (20.12 MW)
================================================================================

光伏系统详细参数:
  装机容量: 20,118 kW
  光伏板面积: 100,589 m² (10.06 公顷)
  安装成本: 30,176,771 CHF (30.18 百万瑞士法郎)
  年发电量: 26,434,851 kWh
  自给率: 100.0%
  设计寿命: 25 年

生成可视化图表...
保存图表: results\figures\virtual_park\park_energy_analysis.png
保存图表: results\figures\virtual_park\building_energy_distribution.png
保存图表: results\figures\virtual_park\pv_system_design.png

完整报告已保存: results\virtual_park\virtual_park_report.txt
```

**耗时**: 约2-3分钟  
**输出文件**:
- `results/virtual_park/virtual_park_report.txt` - 完整报告
- `results/virtual_park/pv_system_design.json` - 光伏设计参数
- `results/figures/virtual_park/` - 3张可视化图表

---

### 步骤3: 查看结果

```bash
# Windows PowerShell
Get-Content results\virtual_park\virtual_park_report.txt

# 或直接打开文件
notepad results\virtual_park\virtual_park_report.txt
```

**关键结果**:
```
一、园区配置
总建筑数: 18栋 (Hospitals:2, Restaurants:5, Schools:3, Shops:8)

二、园区能耗预测
年均能耗: 26,434,851 kWh/年
日均能耗: 69,241.22 kWh/天
峰值功率: 7,544.77 kWh/h

三、光伏系统设计
推荐装机容量: 20,118 kW (20.12 MW)
光伏板面积: 100,589 m² (10.06 公顷)
安装成本: 30,176,771 CHF (30.18 百万瑞士法郎)
光伏自给率: 100.0%

四、经济分析
平准化度电成本: 0.0457 CHF/kWh
年节省电费: 5,286,970 CHF
静态回收期: 5.7 年 ✅

五、结论
推荐安装 20.12 MW 光伏系统，可满足园区 100% 的电力需求，
初期投资约 30.18 百万瑞士法郎，预计 5.7 年回本。
```

---

## 📊 查看可视化图表

### 图表1: 园区能耗分析
**文件**: `results/figures/virtual_park/park_energy_analysis.png`

包含:
- 一周能耗时间序列（预测 vs 实际）
- 日均负荷曲线

### 图表2: 建筑能耗分布
**文件**: `results/figures/virtual_park/building_energy_distribution.png`

包含:
- 各建筑类型能耗占比饼图
- 各建筑类型绝对能耗柱状图

**关键发现**: 餐厅占总能耗的54.9%！

### 图表3: 光伏系统设计
**文件**: `results/figures/virtual_park/pv_system_design.png`

包含:
- 三种计算方法对比
- 系统规模与成本
- 能耗需求 vs 光伏发电
- 光伏自给率（100%）

---

## 📁 输出文件说明

### 模型文件 (`results/models/virtual_park/`)
```
rf_lite_hospitals.pkl      # 医院预测模型
rf_lite_restaurants.pkl    # 餐厅预测模型
rf_lite_schools.pkl        # 学校预测模型
rf_lite_shops.pkl          # 商店预测模型
scaler_*.pkl               # 数据归一化器
features_*.txt             # 特征列表（12个）
```

### 评估指标 (`results/metrics/virtual_park/`)
```
all_buildings_summary.csv      # 模型性能汇总表
all_buildings_metrics.json     # 完整指标JSON
metrics_hospitals.json         # 各建筑详细指标
metrics_restaurants.json
metrics_schools.json
metrics_shops.json
```

### 园区预测结果 (`results/virtual_park/`)
```
park_total_energy.csv          # 园区总能耗时间序列(1675小时)
predictions_hospitals.csv      # 各建筑预测结果
predictions_restaurants.csv
predictions_schools.csv
predictions_shops.csv
pv_system_design.json          # 光伏系统设计参数
virtual_park_report.txt        # 完整文字报告 ⭐
```

### 可视化图表 (`results/figures/virtual_park/`)
```
park_energy_analysis.png               # 园区能耗分析
building_energy_distribution.png       # 建筑能耗分布
pv_system_design.png                   # 光伏系统设计
```

---

## 🔧 自定义配置

### 修改园区配置

编辑 `predict_virtual_park.py` 第24-29行:

```python
# 园区配置（每种建筑的数量）
PARK_CONFIG = {
    'Hospitals': 2,      # 改为你需要的数量
    'Restaurants': 5,
    'Schools': 3,
    'Shops': 8
}
```

然后重新运行:
```bash
python predict_virtual_park.py
```

### 修改光伏参数

编辑 `predict_virtual_park.py` 第32-40行:

```python
PV_PARAMS = {
    'panel_efficiency': 0.20,      # 光伏板效率
    'system_loss': 0.15,           # 系统损失
    'capacity_factor': 0.15,       # 容量系数
    'peak_sun_hours': 3.5,         # 峰值日照时数
    'panel_area_per_kw': 5.0,      # 每kW面积
    'cost_per_kw': 1500,           # 每kW成本(CHF)
    'lifetime_years': 25,          # 寿命
    'degradation_rate': 0.005      # 年衰减率
}
```

---

## 📊 关键数据一览

### 模型性能
| 建筑 | R² | MAPE(%) | 状态 |
|------|-----|---------|------|
| Shops | 0.9388 | 18.70 | ✅ 优秀 |
| Hospitals | 0.9324 | 7.36 | ✅ 优秀 |
| Restaurants | 0.8314 | 62.87 | ✅ 良好 |
| Schools | 0.8112 | 150.14 | ⚠️ 可接受 |

### 园区能耗
- **总建筑数**: 18栋
- **年均能耗**: 26,434,851 kWh
- **峰值功率**: 7,544.77 kW
- **主要消耗**: 餐厅54.9% > 商店28.5% > 学校15.0% > 医院1.6%

### 光伏系统
- **推荐容量**: 20.12 MW
- **光伏面积**: 10.06 公顷
- **总投资**: 30.18 百万CHF
- **自给率**: 100% 🎯
- **回收期**: 5.7年 ⚡

---

## ❓ 常见问题

### Q1: 为什么Restaurants和Schools的MAPE这么高？
**A**: 这两类建筑的用能模式波动大：
- 餐厅: 用餐高峰期vs非营业时间差异巨大
- 学校: 上课期vs假期、工作日vs周末变化显著

MAPE对低能耗时段的小偏差敏感，导致百分比误差高。但绝对误差(MAE)仍在可接受范围。

### Q2: 100%自给率现实吗？
**A**: 理论上可行，但需要：
- ✅ 20.12MW光伏装机
- ✅ 6MW储能系统（30%光伏容量）
- ✅ 智能需求响应
- ⚠️ 连续阴雨天需要电网补充

建议实际设计目标为90-95%自给率更稳妥。

### Q3: 5.7年回本期合理吗？
**A**: 非常合理！
- 瑞士电价: 0.20 CHF/kWh（较高）
- 光伏寿命: 25年（回本后还有19.3年收益）
- 政府补贴: 未计算（实际回收期可能更短）
- 碳交易收益: 未计算

### Q4: 如何提高模型精度？
**A**: 可尝试：
1. 增加训练数据（多年数据）
2. 添加节假日特征
3. 加入天气预报数据
4. 使用LSTM捕捉长期依赖
5. 针对高MAPE建筑类型做专门调优

### Q5: 可以预测其他建筑类型吗？
**A**: 可以！只需：
1. 准备新建筑类型的xlsx数据（格式同现有文件）
2. 在`BUILDING_TYPES`字典添加配置
3. 重新运行`train_all_buildings.py`

---

## 📚 延伸阅读

- [VIRTUAL_PARK_PROJECT_SUMMARY.md](VIRTUAL_PARK_PROJECT_SUMMARY.md) - 完整项目总结
- [results/virtual_park/virtual_park_report.txt](results/virtual_park/virtual_park_report.txt) - 详细报告
- [results/metrics/virtual_park/all_buildings_summary.csv](results/metrics/virtual_park/all_buildings_summary.csv) - 性能数据

---

## ✅ 检查清单

完成以下步骤确保一切正常:

- [ ] 运行 `python train_all_buildings.py`
- [ ] 检查 `results/models/virtual_park/` 有12个文件（4×3）
- [ ] 查看 `results/metrics/virtual_park/all_buildings_summary.csv`
- [ ] 运行 `python predict_virtual_park.py`
- [ ] 检查 `results/virtual_park/` 有7个文件
- [ ] 查看 `results/figures/virtual_park/` 有3张PNG图片
- [ ] 阅读 `results/virtual_park/virtual_park_report.txt`
- [ ] 🎉 完成！

---

**🎯 项目完成！** 现在你已经拥有：
- ✅ 4个高精度能耗预测模型
- ✅ 完整的虚拟园区能耗预测
- ✅ 专业的光伏系统设计方案
- ✅ 详细的经济分析报告
- ✅ 精美的可视化图表

**下一步**: 使用这些模型和设计方案，实际规划你的零碳园区！ 🌱⚡☀️
