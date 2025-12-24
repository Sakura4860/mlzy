# 项目文档导航

**最后更新**: 2025-12-24  
**项目状态**: ✅ 数据泄露已修复，所有文档已更新

---

## 📖 核心文档（必读）

### 1. [README.md](README.md) 🏠
**用途**: 项目总览和快速入门  
**内容**: 
- 项目简介和目标
- 数据集描述
- 核心成果（RF Lite R²=0.9338）
- 快速开始指南
- 7个模型性能对比

**适合**: 首次了解项目的用户

---

### 2. [USAGE.md](USAGE.md) 🔧
**用途**: 详细使用指南  
**内容**:
- 完整实验流程（含修复版）
- 模型训练命令
- 自定义实验方法
- 结果文件说明
- 数据泄露说明
- 报告撰写建议（7000字课程设计）
- 常见问题解答

**适合**: 运行实验和撰写报告的用户

---

### 3. [experiment_log.md](experiment_log.md) 📊
**用途**: 完整实验日志（重新整理版）  
**内容**:
- 项目全流程记录（数据加载→特征工程→模型训练→修复）
- 数据泄露发现和修复过程
- 所有技术细节和代码片段
- 9章系统性文档（6000+行）
- 特征工程策略详解
- 模型对比分析
- 最终结论和建议

**适合**: 深入了解实验细节，撰写报告需要引用技术内容

---

### 4. [PROJECT_STATUS.md](PROJECT_STATUS.md) ✅
**用途**: 项目完成状态和检查清单  
**内容**:
- 完成时间：2025-12-24
- 任务完成清单
- 最佳模型：RF Lite
- 核心成果和关键发现
- 数据泄露修复说明

**适合**: 快速查看项目完成情况

---

### 5. [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) 📈
**用途**: 结果摘要和对比分析  
**内容**:
- 所有模型性能对比表
- 修复前后对比
- 特征重要性分析
- RF Lite vs RF Full对比
- 可视化图表列表（19个PNG）
- 文件结构说明

**适合**: 快速查找性能指标和对比结果

---

## 📚 专题文档

### 6. [DATA_LEAKAGE_FIX_COMPLETE.md](DATA_LEAKAGE_FIX_COMPLETE.md) 🚨
**用途**: 数据泄露完整修复报告  
**内容**:
- 2类数据泄露详解（能耗组成部分、差分重构）
- 修复脚本说明（fix_final.py）
- 修复前后性能对比
- Linear Regression R²: 1.0→0.8739
- 12个泄露特征列表

**适合**: 了解数据泄露问题和修复方法

---

### 7. [FEATURE_REDUCTION_REPORT.md](FEATURE_REDUCTION_REPORT.md) ⚠️
**状态**: 历史文档（包含数据泄露）  
**用途**: 特征精简实验记录  
**内容**:
- 134→12特征精简过程（⚠️ 旧版，包含泄露）
- 特征重要性分析
- 已标注"数据泄露警告"

**适合**: 了解特征精简思路（需注意数据泄露标注）

---

### 8. [DATA_LEAKAGE_FIX.md](DATA_LEAKAGE_FIX.md) ⚠️
**状态**: 历史文档（第一阶段修复，不完整）  
**用途**: 第一次数据泄露修复记录（2025-12-08）  
**内容**:
- 仅修复了滚动特征时间窗口泄露
- 未发现能耗组成部分和差分重构泄露
- 已标注"不完整修复"

**适合**: 了解数据泄露发现的历程

---

## 🛠️ 环境与配置

### 9. [VENV_GUIDE.md](VENV_GUIDE.md) 🐍
**用途**: 虚拟环境配置指南  
**内容**:
- 虚拟环境激活方法（Windows/CMD/PowerShell）
- 已安装包列表
- VSCode集成
- Jupyter Notebook内核配置
- 常见问题解答

**适合**: 环境配置和安装依赖

---

### 10. [requirements.txt](requirements.txt) 📦
**用途**: Python依赖包列表  
**关键包**:
- numpy 2.3.5
- pandas 2.3.3
- scikit-learn 1.7.2
- torch 2.9.1+cpu
- matplotlib 3.10.7

---

## 📄 参考材料

### 11. [论文经验.md](论文经验.md) 📖
**用途**: 论文撰写经验分享  
**内容**: 学术写作技巧和注意事项

### 12. [方案参考.md](方案参考.md) 💡
**用途**: 项目方案参考  
**内容**: 初始设计思路和备选方案

### 13. MinerU转换文件
- `MinerU_1-s2.0-S0378778823010812-main__20251208024251.md`
- `MinerU_机器学习_课程设计_v2__20251208024305.md`

**用途**: 源文献和课程要求（MinerU工具转换的PDF）

---

## 🗂️ 文档使用流程建议

### 场景1: 快速了解项目（5分钟）
```
README.md → PROJECT_STATUS.md → RESULTS_SUMMARY.md
```

### 场景2: 运行实验（30分钟）
```
VENV_GUIDE.md → USAGE.md → 运行脚本 → RESULTS_SUMMARY.md
```

### 场景3: 撰写课程报告（2-3小时）
```
README.md → experiment_log.md → DATA_LEAKAGE_FIX_COMPLETE.md 
→ RESULTS_SUMMARY.md → USAGE.md (报告撰写建议)
```

### 场景4: 深入研究数据泄露（1小时）
```
DATA_LEAKAGE_FIX.md (第一阶段) 
→ DATA_LEAKAGE_FIX_COMPLETE.md (完整修复)
→ experiment_log.md (第5章)
```

### 场景5: 理解特征工程（1小时）
```
experiment_log.md (第4章) 
→ FEATURE_REDUCTION_REPORT.md (注意泄露警告)
→ results/metrics/feature_importance_lite_fixed.csv
```

---

## 📊 文件状态标识

| 标识 | 含义 | 示例 |
|------|------|------|
| ✅ | 最新版本，可直接使用 | README.md |
| ⚠️ | 历史文档，包含数据泄露或过时信息 | FEATURE_REDUCTION_REPORT.md |
| 🏠 | 入口文档，首选阅读 | README.md |
| 🔧 | 实践指南，操作性强 | USAGE.md |
| 📊 | 数据和结果文档 | experiment_log.md |
| 🐍 | 环境配置文档 | VENV_GUIDE.md |
| 📖 | 参考材料 | 论文经验.md |

---

## 🔍 快速查找指南

### 查找性能指标
- 📍 [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - 完整对比表
- 📍 [README.md](README.md) - 核心结果
- 📍 [results/metrics/model_comparison_final.csv](results/metrics/model_comparison_final.csv) - CSV数据

### 查找最佳模型
- 📍 [PROJECT_STATUS.md](PROJECT_STATUS.md) - 最佳模型声明
- 📍 [USAGE.md](USAGE.md) - 模型性能总览
- 📍 [results/models/random_forest_lite_fixed.pkl](results/models/random_forest_lite_fixed.pkl) - 模型文件

### 查找数据泄露说明
- 📍 [DATA_LEAKAGE_FIX_COMPLETE.md](DATA_LEAKAGE_FIX_COMPLETE.md) - 完整修复报告
- 📍 [experiment_log.md](experiment_log.md) (第5章) - 详细分析
- 📍 [USAGE.md](USAGE.md) - 泄露类型简介

### 查找特征列表
- 📍 [results/models/lite_features_fixed.txt](results/models/lite_features_fixed.txt) - 12特征
- 📍 [results/metrics/valid_features_final.txt](results/metrics/valid_features_final.txt) - 122特征
- 📍 [results/metrics/feature_importance_lite_fixed.csv](results/metrics/feature_importance_lite_fixed.csv) - 特征重要性

### 查找可视化图表
- 📍 [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - 图表列表和说明
- 📍 [results/figures/](results/figures/) - 19个PNG文件
- 📍 [USAGE.md](USAGE.md) - 图表分类说明

---

## 📞 联系与支持

遇到问题请按以下顺序查找：
1. 本文档索引
2. [USAGE.md](USAGE.md) 常见问题章节
3. [experiment_log.md](experiment_log.md) 详细技术说明
4. 源代码注释

**项目完成状态**: ✅ 所有任务完成，文档最新  
**最后验证**: 2025-12-24
