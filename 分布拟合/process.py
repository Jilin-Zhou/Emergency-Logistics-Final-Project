# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from fitter import Fitter, get_common_distributions
# from scipy import stats
# # Load the dataset
# df = pd.read_csv('flood_CHA.csv')
# # Convert the 'Total Affected' column to numeric, handling potential errors
# df['Total Affected'] = pd.to_numeric(df['Total Affected'], errors='coerce')
# # Remove rows with NaN in 'Total Affected' after coercion
# df.dropna(subset=['Total Affected'], inplace=True)
# # Filter out values of 0 as they don't represent affected people
# df = df[df['Total Affected'] > 0]
# # Select the raw data for fitting (no log transform)
# data = df['Total Affected'].values
# # Get a list of common distributions
# # You can choose 'common', 'popular', or 'all'
# # 'common' is a good starting point
# distributions_to_try = get_common_distributions()
# # Initialize Fitter with the raw data and common distributions
# # Increase timeout and use parallel processing for potentially faster fitting
# f = Fitter(data,
#            distributions=distributions_to_try,
#            timeout=30, # Increased timeout, may need more depending on data size and distributions
#            )
# # Fit the distributions
# print("Starting distribution fitting for raw data...")
# f.fit()
# print("Fitting complete.")
# # Print the summary of the best fits
# print("\nFitter Summary (Best Fits):")
# print(f.summary())
# # Get the best fitted distribution
# # 'sumsquare_error' is the default method for get_best()
# best_fit = f.get_best()
# print("\nBest fitted distribution parameters (based on sumsquare_error):")
# print(best_fit)
#
#
# # --- Visualize the best fitted distribution ---
# # Create figure and axes
# plt.figure(figsize=(16, 9)) # Increased figure size for better visibility
# # Plot histogram of raw data
# # Use many bins to capture details, especially at the lower end
# sns.histplot(data, bins=200, kde=False, stat='density', label='Raw Data Histogram', color='skyblue', edgecolor='black')
# # Add KDE for a smoother representation
# sns.kdeplot(data, color='blue', linestyle='-', label='Raw Data KDE', linewidth=2)
# # Get the best fitted distribution name and parameters
# best_dist_name = list(best_fit.keys())[0]
# best_dist_params_dict = best_fit[best_dist_name]
# # Get the distribution object from scipy.stats
# try:
#     best_dist = getattr(stats, best_dist_name)
# except AttributeError:
#     print(f"Error: Distribution '{best_dist_name}' not found in scipy.stats.")
#     best_dist = None
# if best_dist:
#     try:
#         pdf_func = best_dist.pdf
#     except AttributeError:
#         print(f"Error: PDF function not found for distribution '{best_dist_name}'.")
#         pdf_func = None
#     if pdf_func:
#         # Generate x values for plotting the PDF
#         # Start from a value > 0, as some distributions are only defined for positive values
#         # Extend the range slightly beyond the max data value to see the tail behavior
#         x_raw = np.linspace(max(1, data.min()), data.max() * 1.1, 1000)
#         # Attempt keyword arguments
#         try:
#             plt.plot(x_raw, pdf_func(x_raw, **best_dist_params_dict), label=f'Best Fit: {best_dist_name}\nParams: {np.round(list(best_dist_params_dict.values()), 3)}', color='red', linestyle='--', linewidth=2)
#         except TypeError as e:
#              print(f"Error plotting PDF for {best_dist_name}: {e}")
#              print("Attempting to pass parameters as a tuple instead... (Requires knowing parameter order)")
#              # Fallback to tuple passing - requires manual knowledge
#              print(f"Manual parameter tuple passing for {best_dist_name} visualization is not implemented in this example.")
# plt.title('Fitted Distribution to Raw Total Affected People')
# plt.xlabel('Total Affected People')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # Set x-axis limits. Limiting the upper bound can make the main distribution clearer.
# # Consider limiting to a percentile if the max value is extremely large
# # plt.xlim(data.min(), np.percentile(data, 99)) # Example: limit to 99th percentile
# # Option 1: Linear X-axis (default)
# plt.xlim(data.min(), data.max() * 1.1) # Extend slightly beyond max for PDF visualization
# # Option 2: Logarithmic X-axis (Recommended for long-tailed data visualization)
# # This will make the long tail more visible and compress the high density area at low values.
# # Be careful with log scale if your data includes 0 or values very close to 0 (though we filtered 0s).
# # plt.xscale('log')
# # plt.xlim(max(1, data.min()), data.max() * 1.1) # Adjust xlim for log scale, ensure minimum is > 0
# plt.tight_layout() # Adjust layout to prevent labels overlapping
# plt.show()



import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
import matplotlib.pyplot as plt
import warnings

# 忽略一些fitter和scipy可能产生的运行时警告，例如拟合不佳或除零等
warnings.filterwarnings("ignore")

# --- 配置 ---
FILE_PATH = 'flood_CHA.csv'
COLUMN_TO_FIT = 'Total Affected'

# --- 1. 加载数据 ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"成功加载文件: {FILE_PATH}")
except FileNotFoundError:
    print(f"错误: 文件 '{FILE_PATH}' 未找到。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载文件时发生错误: {e}")
    exit()

# --- 2. 数据准备 ---
if COLUMN_TO_FIT not in df.columns:
    print(f"错误: 列 '{COLUMN_TO_FIT}' 在CSV文件中未找到。")
    print(f"可用的列有: {df.columns.tolist()}")
    exit()

# 提取目标列
data_series = df[COLUMN_TO_FIT].copy() # 使用 .copy() 避免 SettingWithCopyWarning

# 转换为数值型，无法转换的将变为NaN
data_series_numeric = pd.to_numeric(data_series, errors='coerce')

# 移除NaN值
nan_count = data_series_numeric.isnull().sum()
if nan_count > 0:
    print(f"警告: 在 '{COLUMN_TO_FIT}' 列中发现 {nan_count} 个非数值或空值，已将其移除。")
    data_series_numeric.dropna(inplace=True)

# 检查数据是否为空
if data_series_numeric.empty:
    print(f"错误: 在 '{COLUMN_TO_FIT}' 列中没有有效的数值数据进行拟合。")
    exit()

# 许多分布要求数据为正数。
# "Total Affected" 理论上不应为负，但可能为0。
# 一些分布 (如 lognorm, gamma) 对0值敏感或不支持。
# 为了更广泛的分布拟合，通常移除0值。
original_count = len(data_series_numeric)
data_to_fit = data_series_numeric[data_series_numeric > 0]
removed_zeros_negatives = original_count - len(data_to_fit)
if removed_zeros_negatives > 0:
    print(f"警告: 移除了 {removed_zeros_negatives} 个零值或负值，以便进行分布拟合。")

if data_to_fit.empty:
    print(f"错误: 在 '{COLUMN_TO_FIT}' 列中移除零值/负值后，没有有效的正数数据进行拟合。")
    exit()

print(f"\n准备对 '{COLUMN_TO_FIT}' 列中的 {len(data_to_fit)} 个数据点进行拟合。")
print(f"数据描述:\n{data_to_fit.describe()}")

# --- 3. 使用 Fitter进行分布拟合 ---

# 选择要尝试的分布
# common_distributions = get_common_distributions() # 常用分布
all_distributions = get_distributions() # SciPy中所有可用分布 (可能非常耗时)
# 或者自定义列表
# distributions_to_try = ['gamma', 'lognorm', 'expon', 'weibull_min', 'weibull_max', 'norm', 'pareto', 'genextreme', 'burr']
# 确保这些分布在您的scipy版本中可用，get_common_distributions() 更安全
# distributions_to_try = get_common_distributions() + ['burr', 'pareto']
# distributions_to_try = list(set(distributions_to_try)) # 移除重复项

# print(f"\n将尝试拟合以下分布: {distributions_to_try}")

# 创建Fitter实例
# timeout参数为每种分布的拟合设置超时时间（秒）
f = Fitter(data_to_fit,
           distributions=all_distributions,
           timeout=120) # 例如，每种分布最多尝试2分钟

# 执行拟合
print("\n开始拟合分布...")
try:
    f.fit()
except Exception as e:
    print(f"拟合过程中发生错误: {e}")
    print("可能是由于数据特性或所选分布不适用。")
    exit()

# --- 4. 显示结果 ---
print("\n--- 拟合结果摘要 ---")
# summary()方法会绘制直方图和拟合的PDF，并返回一个包含拟合统计量的DataFrame
# 默认按 'sumsquare_error' 排序
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
summary_df = f.summary(plot=True)
plt.suptitle(f"对 '{COLUMN_TO_FIT}' 列的分布拟合", y=1.02)
plt.xlabel(COLUMN_TO_FIT)
plt.ylabel("频数/概率密度")
plt.show() # 确保图形显示

print("\n--- 拟合统计量 (按 sumsquare_error 升序排列) ---")
print(summary_df)

print("\n--- 最佳拟合分布 (基于 sumsquare_error) ---")
best_fit_sse = f.get_best(method='sumsquare_error')
print(f"按 sumsquare_error: {best_fit_sse}")

# 你也可以获取基于 AIC 或 BIC 的最佳拟合
try:
    best_fit_aic = f.get_best(method='aic')
    print(f"按 AIC: {best_fit_aic}")
except KeyError:
    print("警告: 无法基于 AIC 获取最佳拟合 (可能因为所有分布的AIC都无效)。")


try:
    best_fit_bic = f.get_best(method='bic')
    print(f"按 BIC: {best_fit_bic}")
except KeyError:
    print("警告: 无法基于 BIC 获取最佳拟合 (可能因为所有分布的BIC都无效)。")


print("\n说明:")
print("- sumsquare_error (SSE): 拟合值与实际值之间的平方差之和，越小越好。")
print("- AIC (Akaike Information Criterion): 考虑拟合优度和模型复杂度的指标，越小越好。")
print("- BIC (Bayesian Information Criterion): 类似于AIC，但对模型参数数量的惩罚更大，越小越好。")
print("- KStest (Kolmogorov-Smirnov test): 检验数据是否来自特定分布的统计检验。")
print("  - 'ks_stat' 是KS统计量。")
print("  - 'ks_pvalue' 是p值。如果p值大于显著性水平 (如0.05)，则不能拒绝数据来自该分布的原假设。")
print("  (注意: fitter的summary可能不直接显示ks_pvalue，但其内部会使用它)")

# 打印最佳分布的参数
if best_fit_sse:
    best_dist_name = list(best_fit_sse.keys())[0]
    best_params = best_fit_sse[best_dist_name]
    print(f"\n最佳拟合分布 '{best_dist_name}' 的参数: {best_params}")
