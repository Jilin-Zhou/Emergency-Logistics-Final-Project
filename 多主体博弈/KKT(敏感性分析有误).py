import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
import pandas as pd
from matplotlib.gridspec import GridSpec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 1. 定义应急物资需求概率分布函数

# 均匀分布
class UniformDistribution:
    def __init__(self, U):
        self.U = U

    def pdf(self, x):
        if isinstance(x, (int, float)):
            if 0 <= x <= self.U:
                return 1 / self.U
            else:
                return 0
        else:  # numpy array
            result = np.zeros_like(x, dtype=float)
            mask = (0 <= x) & (x <= self.U)
            result[mask] = 1 / self.U
            return result

    def cdf(self, x):
        if isinstance(x, (int, float)):
            if x < 0:
                return 0
            elif 0 <= x <= self.U:
                return x / self.U
            else:
                return 1
        else:  # numpy array
            result = np.zeros_like(x, dtype=float)
            mask1 = x < 0
            mask2 = (0 <= x) & (x <= self.U)
            mask3 = x > self.U
            result[mask1] = 0
            result[mask2] = x[mask2] / self.U
            result[mask3] = 1
            return result

    def inverse_cdf(self, p):
        if 0 <= p <= 1:
            return p * self.U
        else:
            raise ValueError("概率值必须在[0,1]区间内")


# 广义帕累托分布
class GeneralizedParetoDistribution:
    def __init__(self, k, sigma, theta=0):
        self.k = k
        self.sigma = sigma
        self.theta = theta

    def pdf(self, x):
        if isinstance(x, (int, float)):
            if x < self.theta:
                return 0
            else:
                if self.k == 0:
                    return (1 / self.sigma) * np.exp(-(x - self.theta) / self.sigma)
                else:
                    return (1 / self.sigma) * (1 + self.k * (x - self.theta) / self.sigma) ** (-1 - 1 / self.k)
        else:  # numpy array
            result = np.zeros_like(x, dtype=float)
            mask = x >= self.theta
            if self.k == 0:
                result[mask] = (1 / self.sigma) * np.exp(-(x[mask] - self.theta) / self.sigma)
            else:
                result[mask] = (1 / self.sigma) * (1 + self.k * (x[mask] - self.theta) / self.sigma) ** (
                            -1 - 1 / self.k)
            return result

    def cdf(self, x):
        if isinstance(x, (int, float)):
            if x < self.theta:
                return 0
            else:
                if self.k == 0:
                    return 1 - np.exp(-(x - self.theta) / self.sigma)
                else:
                    return 1 - (1 + self.k * (x - self.theta) / self.sigma) ** (-1 / self.k)
        else:  # numpy array
            result = np.zeros_like(x, dtype=float)
            mask = x >= self.theta
            if self.k == 0:
                result[mask] = 1 - np.exp(-(x[mask] - self.theta) / self.sigma)
            else:
                result[mask] = 1 - (1 + self.k * (x[mask] - self.theta) / self.sigma) ** (-1 / self.k)
            return result

    def inverse_cdf(self, p):
        if 0 <= p <= 1:
            if self.k == 0:
                return self.theta - self.sigma * np.log(1 - p)
            else:
                return self.theta + (self.sigma / self.k) * ((1 - p) ** (-self.k) - 1)
        else:
            raise ValueError("概率值必须在[0,1]区间内")


# 2. 实现政府利润函数
def government_profit(Q, q, dist, alpha, v, p1, c1, p2, s, m, U):
    """
    计算政府利润

    参数:
    Q: 政府实物储备量
    q: 企业实物储备量
    dist: 概率分布函数对象
    alpha: 灾害发生概率
    v: 单位物资残值
    p1: 灾害前物资单价
    c1: 政府单位物资储存成本
    p2: 企业单位物资代储收入
    s: 企业单位物资使用补贴
    m: 灾害后物资市场单价
    U: 最大物资需求量

    返回:
    政府利润
    """
    # 灾害未发生的利润
    profit_no_disaster = v * Q - (p1 + c1) * Q - p2 * q

    # 灾害发生的利润
    # 积分1: 0到Q
    def integrand1(x):
        return (v * (Q - x) - (p1 + c1) * Q - p2 * q) * dist.pdf(x)

    # 积分2: Q到Q+q
    def integrand2(x):
        return (-(p1 + c1) * Q - p2 * q - s * (x - Q)) * dist.pdf(x)

    # 积分3: Q+q到U
    def integrand3(x):
        return (-(p1 + c1) * Q - p2 * q - s * q - m * (x - Q - q)) * dist.pdf(x)

    # 数值积分
    integral1 = integrate.quad(integrand1, 0, Q)[0] if Q > 0 else 0
    integral2 = integrate.quad(integrand2, Q, Q + q)[0] if q > 0 else 0
    integral3 = integrate.quad(integrand3, Q + q, U)[0]

    profit_disaster = integral1 + integral2 + integral3

    # 总利润
    total_profit = (1 - alpha) * profit_no_disaster + alpha * profit_disaster

    return total_profit


# 3. 使用KKT条件求解最优储备量
def optimal_reserve_quantities(dist, alpha, v, p1, c1, p2, s, m):
    """
    计算不同储备决策下的最优储备量

    参数与government_profit函数相同

    返回:
    各种决策下的最优储备量
    """
    # 决策1：无政府实物储备，无企业实物储备
    Q1, q1 = 0, 0

    # 决策2：有政府实物储备，无企业实物储备
    try:
        # Q2 = dist.inverse_cdf(1 - ((1 - alpha) * v + alpha * m - p1 - c1) / (alpha * (m - v)))
        Q2 = dist.inverse_cdf(((1 - alpha) * v + alpha * m - p1 - c1) / (alpha * (m - v)))
    except:
        Q2 = 0
    q2 = 0

    # 决策3：无政府实物储备，有企业实物储备
    Q3 = 0
    try:
        q3 = dist.inverse_cdf(1 + p2 / (alpha * (s - m)))
    except:
        q3 = 0

    # 决策4：有政府实物储备，有企业实物储备
    try:
        Q4 = dist.inverse_cdf(1 + (-v + p1 + c1 - p2) / (alpha * (v - s)))
    except:
        Q4 = 0

    try:
        q4 = dist.inverse_cdf(1 + p2 / (alpha * (s - m))) - dist.inverse_cdf(
            1 + (-v + p1 + c1 - p2) / (alpha * (v - s)))
    except:
        q4 = 0

    if q4 < 0:
        q4 = 0

    return {
        'decision1': (Q1, q1),
        'decision2': (Q2, q2),
        'decision3': (Q3, q3),
        'decision4': (Q4, q4)
    }


# 4. 案例计算和敏感性分析
def case_calculation(dist_type='GPD'):
    """
    进行案例计算

    参数:
    dist_type: 分布类型，'GPD'为广义帕累托分布，'UD'为均匀分布

    返回:
    最优储备量和储备比例
    """
    # 参数设置
    v = 100  # 单位物资残值
    p1 = 200  # 灾害前物资单价
    m = 500  # 灾害后应急物资市场单价
    alpha = 1  # 灾害发生概率
    e = 400  # 企业单位物资加急生产成本
    p2 = 170  # 企业单位物资代储收入
    c2 = 300  # 企业单位物资储存成本
    s = 180  # 企业单位物资使用补贴
    c1 = 120  # 政府单位物资储存成本
    T = 5551  # 应急物资总需求量（万件）

    # 创建分布对象
    if dist_type == 'GPD':
        k = 1.26089
        sigma = 1059.03
        dist = GeneralizedParetoDistribution(k, sigma)
    else:  # 'UD'
        dist = UniformDistribution(T)

    # 计算最优储备量
    optimal_reserves = optimal_reserve_quantities(dist, alpha, v, p1, c1, p2, s, m)

    # 决策4的结果
    Q4, q4 = optimal_reserves['decision4']
    p4 = T - Q4 - q4  # 企业生产能力储备量

    # 计算比例
    total = Q4 + q4 + p4
    Q4_ratio = Q4 / total
    q4_ratio = q4 / total
    p4_ratio = p4 / total

    return {
        'Q': Q4,
        'q': q4,
        'p': p4,
        'Q_ratio': Q4_ratio,
        'q_ratio': q4_ratio,
        'p_ratio': p4_ratio
    }


# 敏感性分析函数
def sensitivity_analysis(parameter_name, values, dist_type='GPD'):
    """
    进行敏感性分析

    参数:
    parameter_name: 参数名称
    values: 参数值列表
    dist_type: 分布类型

    返回:
    储备量随参数变化的结果
    """
    # 基本参数
    base_params = {
        'v': 100,
        'p1': 200,
        'm': 500,
        'alpha': 1,
        'e': 400,
        'p2': 170,
        'c2': 300,
        's': 180,
        'c1': 120,
        'T': 5551
    }

    # 创建分布对象
    if dist_type == 'GPD':
        k = 1.26089
        sigma = 1059.03
        dist = GeneralizedParetoDistribution(k, sigma)
    else:  # 'UD'
        dist = UniformDistribution(base_params['T'])

    results = []

    for value in values:
        # 更新参数
        params = base_params.copy()
        params[parameter_name] = value

        # 计算最优储备量
        optimal_reserves = optimal_reserve_quantities(
            dist, params['alpha'], params['v'], params['p1'],
            params['c1'], params['p2'], params['s'], params['m']
        )

        # 决策4的结果
        Q4, q4 = optimal_reserves['decision4']
        p4 = params['T'] - Q4 - q4  # 企业生产能力储备量

        results.append({'value': value, 'Q': Q4, 'q': q4, 'p': p4})

    return results


# 5. 可视化结果
def plot_case_results():
    """绘制案例计算结果"""
    # 计算结果
    gpd_results = case_calculation('GPD')
    ud_results = case_calculation('UD')

    # 创建数据框
    data = {
        '储备方式': ['政府最优实物储备量Q*', '企业最优实物储备量q*', '企业最优生产能力储备量p*', '总储备量T'],
        'GPD储备量/万件': [gpd_results['Q'], gpd_results['q'], gpd_results['p'],
                           gpd_results['Q'] + gpd_results['q'] + gpd_results['p']],
        'GPD占比': [gpd_results['Q_ratio'], gpd_results['q_ratio'], gpd_results['p_ratio'], 1.0],
        'UD储备量/万件': [ud_results['Q'], ud_results['q'], ud_results['p'],
                          ud_results['Q'] + ud_results['q'] + ud_results['p']],
        'UD占比': [ud_results['Q_ratio'], ud_results['q_ratio'], ud_results['p_ratio'], 1.0]
    }

    df = pd.DataFrame(data)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置柱状图
    bar_width = 0.35
    index = np.arange(3)  # 只显示前三行（不显示总储备量）

    # 绘制两个分布的储备量对比
    gpd_bars = ax.bar(index - bar_width / 2, df['GPD储备量/万件'][:3], bar_width,
                      label='广义帕累托分布(GPD)', color='skyblue')
    ud_bars = ax.bar(index + bar_width / 2, df['UD储备量/万件'][:3], bar_width,
                     label='均匀分布(UD)', color='lightcoral')

    # 添加文本标签
    for i, bar in enumerate(gpd_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 50,
                f'{height:.1f}\n({df["GPD占比"][i]:.2%})',
                ha='center', va='bottom')

    for i, bar in enumerate(ud_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 50,
                f'{height:.1f}\n({df["UD占比"][i]:.2%})',
                ha='center', va='bottom')

    # 设置图表元素
    ax.set_xlabel('储备方式')
    ax.set_ylabel('储备量/万件')
    ax.set_title('不同概率分布下的最优储备量比较')
    ax.set_xticks(index)
    ax.set_xticklabels(df['储备方式'][:3])
    ax.legend()

    # 显示总储备量
    plt.figtext(0.5, 0.01, f'总储备量: {df["GPD储备量/万件"][3]:.1f}万件', ha='center')

    plt.tight_layout(pad=3)
    return fig


def plot_sensitivity_analysis():
    """绘制敏感性分析结果"""
    # 参数设置
    parameters = {
        'alpha': np.linspace(0.1, 1.0, 10),  # 灾害发生概率
        'v': np.linspace(50, 200, 10),  # 单位物资残值
        's': np.linspace(120, 220, 10),  # 企业单位物资使用补贴
        'p2': np.linspace(120, 220, 10),  # 企业单位物资代储收入
        'p1': np.linspace(150, 250, 10),  # 灾害前物资单价
        'c1': np.linspace(70, 170, 10),  # 政府单位物资储存成本
        'm': np.linspace(300, 600, 10)  # 灾害后市场单价
    }

    # 创建图表
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(7, 2, figure=fig)

    # 参数名称映射
    param_names = {
        'alpha': '灾害发生概率α',
        'v': '单位物资残值v',
        's': '企业单位物资使用补贴s',
        'p2': '企业单位物资代储收入p₂',
        'p1': '灾害前物资单价p₁',
        'c1': '政府单位物资储存成本c₁',
        'm': '灾害后市场单价m'
    }

    # 绘制每个参数的敏感性分析
    for i, (param, values) in enumerate(parameters.items()):
        # 获取两种分布的结果
        gpd_results = sensitivity_analysis(param, values, 'GPD')
        ud_results = sensitivity_analysis(param, values, 'UD')

        # 提取数据
        gpd_Q = [r['Q'] for r in gpd_results]
        gpd_q = [r['q'] for r in gpd_results]
        gpd_p = [r['p'] for r in gpd_results]

        ud_Q = [r['Q'] for r in ud_results]
        ud_q = [r['q'] for r in ud_results]
        ud_p = [r['p'] for r in ud_results]

        # 绘制广义帕累托分布的结果
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(values, gpd_Q, 'b-', marker='o', label='政府实物储备量Q*')
        ax1.plot(values, gpd_q, 'r-', marker='s', label='企业实物储备量q*')
        ax1.plot(values, gpd_p, 'g-', marker='^', label='企业生产能力储备量p*')
        ax1.set_title(f'广义帕累托分布: {param_names[param]}的影响')
        ax1.set_xlabel(param_names[param])
        ax1.set_ylabel('储备量/万件')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 绘制均匀分布的结果
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(values, ud_Q, 'b-', marker='o', label='政府实物储备量Q*')
        ax2.plot(values, ud_q, 'r-', marker='s', label='企业实物储备量q*')
        ax2.plot(values, ud_p, 'g-', marker='^', label='企业生产能力储备量p*')
        ax2.set_title(f'均匀分布: {param_names[param]}的影响')
        ax2.set_xlabel(param_names[param])
        ax2.set_ylabel('储备量/万件')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


# 主函数
def main():
    # 案例计算结果
    fig1 = plot_case_results()
    fig1.savefig('case_results.png', dpi=300, bbox_inches='tight')

    # 敏感性分析结果
    fig2 = plot_sensitivity_analysis()
    fig2.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
