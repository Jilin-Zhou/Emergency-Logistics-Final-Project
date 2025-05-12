import pandas as pd

# 读取数据集
df = pd.read_csv('public_emdat_1900.csv')

# 筛选Total Affected不为空的行
df_filtered = df[df['Total Affected'].notna()]

# 保存为新的CSV文件
df_filtered.to_csv('flood_CHA.csv', index=False)

print("筛选完成并保存为flood_CHA.csv")
