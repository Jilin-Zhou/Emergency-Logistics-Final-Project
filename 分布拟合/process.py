import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# 1. 读取数据集
try:
    df = pd.read_csv('flood_CHA.csv')
except FileNotFoundError:
    print("Error: flood_CHA.csv not found.")
    exit()

# 2. 提取关键数据并清洗
df_flood = df.copy()
df_flood = df_flood.dropna(subset=['Total Affected']).copy()
df_flood['Total Affected'] = df_flood['Total Affected'].astype(float)

# 转换为万人
df_flood['Total Affected'] = df_flood['Total Affected'] / 10000

# 过滤掉受灾人数为0的事件（如果您的关注点是大规模洪水事件）
df_flood = df_flood[df_flood['Total Affected'] > 0].copy()

# 3. 分组与计数（创建直方图）
# 选择合适的binning策略，可以根据数据范围和图2的binning进行调整
# 根据图2，受灾人数大致在0到25000万人之间，且bin宽度不固定，但大致可以观察到一些区间。
# 我们可以尝试使用hist函数自动计算bin，或者手动指定bin的边界。
# 考虑到您提供了图2，我们尽量模拟其视觉效果，虽然精确复现可能需要原始binning数据。
# 这里我们尝试使用hist函数，并根据图2的范围设定bins。
hist_counts, bin_edges, patches = plt.hist(df_flood['Total Affected'], bins='auto', alpha=0.7, edgecolor='black', label='实际发生次数')
# 计算bin的中心和宽度
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths = np.diff(bin_edges)
plt.close() # 关闭自动生成的直方图，我们稍后会重新绘制
# 4. 可视化直方图 (使用plt.bar以更好地控制绘制)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解决坐标轴负号显示问题
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_counts, width=bin_widths * 0.8, alpha=0.7, edgecolor='black', label='实际发生次数') # 调整width以避免柱子之间挨得太近
plt.xlabel('受灾人数/万人', fontsize=12)
plt.ylabel('次数/次', fontsize=12)
plt.title('受灾人数和发生次数关系直方图', fontsize=14)
plt.grid(axis='y', linestyle='--')
plt.show()


# 5. 分布拟合 (广义帕累托分布 GPD)
# GPD通常用于拟合超过某个阈值的数据。这里我们直接对所有数据进行拟合，
# 模拟图2的整体分布拟合。在实际应用中，通常会设定一个阈值。

# 极大似然估计 (MLE)
try:
    # 使用scipy.stats.genpareto进行拟合，它对应于GPD的scipy实现。
    # genpareto的参数为 c (shape), loc (location), scale (scale)
    # GPD通常定义为 G(y) = 1 - (1 + xi*y/sigma)^(-1/xi) for y >= 0, where y = x - mu
    # 在scipy中，genpareto的CDF是 F(x, c) = 1 - (1 + c * (x - loc)/scale)^(-1/c)
    # 所以 c 对应 xi, loc 对应 mu, scale 对应 sigma
    c_mle, loc_mle, scale_mle = st.genpareto.fit(df_flood['Total Affected'], floc=0) # 假设loc为0
    print(f"MLE拟合参数 (c, loc, scale): {c_mle}, {loc_mle}, {scale_mle}")

    # 绘制MLE拟合曲线 (概率密度函数 PDF)
    x_range = np.linspace(0, df_flood['Total Affected'].max(), 200)
    pdf_mle = st.genpareto.pdf(x_range, c_mle, loc=loc_mle, scale=scale_mle) * len(df_flood) * (x_range[1] - x_range[0]) / bin_widths[0] # 乘以总事件数和平均bin宽度进行缩放，近似频次
    # 更准确的做法是计算每个bin的理论频次
    pdf_mle_binned = st.genpareto.pdf(bin_centers, c_mle, loc=loc_mle, scale=scale_mle) * len(df_flood) * bin_widths[0] # 乘以总事件数和第一个bin的宽度进行缩放

    plt.plot(x_range, st.genpareto.pdf(x_range, c_mle, loc=loc_mle, scale=scale_mle) * len(df_flood) * bin_widths[0], 'k-', lw=2, label='GPD-MLE 拟合曲线') # 乘以总事件数和平均bin宽度进行缩放，近似频次
    plt.show()
except Exception as e:
    print(f"MLE拟合失败: {e}")
#
#
# # 矩估计 (MM) - 对于GPD，MM的实现稍微复杂一些，scipy.stats.genpareto.fit默认使用MLE。
# # 如果需要精确的MM拟合，需要手动实现。考虑到图2只提供了曲线，我们在此不进行精确的MM拟合，
# # 但可以说明MM的原理是匹配样本矩和理论矩。
# # 对于GPD，MM参数估计通常涉及解非线性方程组，比较复杂。
# # 这里我们可以简单展示图2中虚线的大致形态，或者跳过MM的精确实现。
# # 为了模拟图2，我们可能需要根据图2的虚线形态进行一些猜测或者使用其他简化方法。
# # 由于没有提供图2的MM参数，我们无法直接绘制。
# # 如果您有图2的MM拟合参数，我可以帮您绘制。
# # 暂时跳过MM的精确实现。
#
# plt.legend()
# plt.show()
#
# # 6. 简要分析拟合结果
# print("\n拟合结果简要分析:")
# print("直方图显示，受灾人数较低的洪水事件发生次数最多，随着受灾人数的增加，发生次数迅速减少，呈现明显的长尾分布特征。")
# print("GPD-MLE 拟合曲线大致捕捉了这一分布趋势。GPD 是处理极值数据的常用模型，适用于描述洪水等极端事件的受灾人数分布。")
# print("具体的拟合参数（c, loc, scale）反映了分布的形态、位置和尺度。")
# if 'c_mle' in locals():
#     print(f"MLE拟合的形状参数 c = {c_mle:.4f}。一般来说，c > 0 表示重尾分布，c = 0 表示指数分布，c < 0 表示有限右端点分布。对于洪水受灾人数这类极端事件，通常期望 c > 0。")
# # 如果您对矩估计法感兴趣，我可以提供更多关于其原理的信息，或者如果您能提供图2的MM拟合参数，我可以绘制其曲线。

