import pandas as pd
import numpy as np

# 模拟简单数据
data = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# 问题1: 当前的滚动特征（包含当前值）
data['rolling_mean_3_current'] = data['value'].rolling(window=3, min_periods=1).mean()

# 正确的滚动特征（不包含当前值）
data['rolling_mean_3_correct'] = data['value'].shift(1).rolling(window=3, min_periods=1).mean()

print('数据泄露检查:')
print(data)
print('\n当索引=5时:')
print(f'真实值: {data.loc[5, \"value\"]}')
print(f'错误的rolling_mean (包含当前): {data.loc[5, \"rolling_mean_3_current\"]}')
print(f'正确的rolling_mean (不包含当前): {data.loc[5, \"rolling_mean_3_correct\"]}')
print(f'\n相关性分析:')
print(f'错误方法与真实值相关性: {data[\"value\"].corr(data[\"rolling_mean_3_current\"]):.4f}')
print(f'正确方法与真实值相关性: {data[\"value\"].corr(data[\"rolling_mean_3_correct\"]):.4f}')
