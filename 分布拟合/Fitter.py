import pandas as pd
import numpy as np
from fitter import Fitter, get_common_distributions, get_distributions
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

FILE_PATH = 'flood_CHA.csv'
COLUMN_TO_FIT = 'Total Affected'
PERCENTILE_THRESHOLD = 0.95


df = pd.read_csv(FILE_PATH)
print(f"成功加载文件: {FILE_PATH}")

data_series = df[COLUMN_TO_FIT].copy()
data_series_numeric = pd.to_numeric(data_series, errors='coerce')

nan_count = data_series_numeric.isnull().sum()
if nan_count > 0:
    data_series_numeric.dropna(inplace=True)

if data_series_numeric.empty:
    exit()

original_count = len(data_series_numeric)
data_positive = data_series_numeric[data_series_numeric > 0]
removed_zeros_negatives = original_count - len(data_positive)

if data_positive.empty:
   exit()

print(f"\n用于计算百分位数的正数据点数量: {len(data_positive)}")

if len(data_positive) < 2:
    data_to_fit = data_positive.copy()
else:
    cutoff_value = data_positive.quantile(PERCENTILE_THRESHOLD)
    data_to_fit = data_positive[data_positive <= cutoff_value].copy()
    num_removed_by_percentile = len(data_positive) - len(data_to_fit)

if data_to_fit.empty:
    exit()

data_to_fit /= 1e4
print(f"筛选后数据的描述:\n{data_to_fit.describe()}")

# all_distributions = get_distributions()
all_distributions = ['invgauss']
f_filtered = Fitter(data_to_fit,
                    distributions=all_distributions,
                    timeout=120)

print("\n开始对筛选后的数据进行拟合...")
try:
    f_filtered.fit()
except Exception as e:
    print(f"对筛选后数据进行拟合时发生错误: {e}")
    print("可能是由于数据特性或所选分布不适用。")
    exit()


# 图像输出
summary_df_filtered = f_filtered.summary(plot=True)
plt.xlabel(f"受灾人数(万人)")
plt.ylabel("频数/概率密度")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.savefig('拟合结果.png',dpi=900)
plt.show()

print(f"\n--- 对筛选后数据的拟合统计量 (按 sumsquare_error 升序排列) ---")
print(summary_df_filtered)

print(f"\n--- 对筛选后数据的最佳拟合分布 ---")
best_fit_sse_filtered = f_filtered.get_best(method='sumsquare_error')
print(f"按 sumsquare_error: {best_fit_sse_filtered}")

try:
    best_fit_aic_filtered = f_filtered.get_best(method='aic')
    print(f"按 AIC: {best_fit_aic_filtered}")
except KeyError:
    print("警告: 无法基于 AIC 获取筛选后数据的最佳拟合。")

try:
    best_fit_bic_filtered = f_filtered.get_best(method='bic')
    print(f"按 BIC: {best_fit_bic_filtered}")
except KeyError:
    print("警告: 无法基于 BIC 获取筛选后数据的最佳拟合。")

if best_fit_sse_filtered:
    best_dist_name_filtered = list(best_fit_sse_filtered.keys())[0]
    best_params_filtered = best_fit_sse_filtered[best_dist_name_filtered]
    print(f"\n筛选后数据的最佳拟合分布 '{best_dist_name_filtered}' 的参数: {best_params_filtered}")
