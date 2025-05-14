import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt


# 定义概率分布类
class Distribution:
    """概率分布基类"""

    def pdf(self, x):
        """概率密度函数"""
        pass

    def cdf(self, x):
        """累积分布函数"""
        pass


class UniformDistribution(Distribution):
    """均匀分布"""

    def __init__(self, a, b):
        self.a = a  # 下限
        self.b = b  # 上限

    def pdf(self, x):
        if isinstance(x, (int, float)):
            return 1 / (self.b - self.a) if self.a <= x <= self.b else 0
        else:  # 处理数组输入
            result = np.zeros_like(x, dtype=float)
            mask = (x >= self.a) & (x <= self.b)
            result[mask] = 1 / (self.b - self.a)
            return result

    def cdf(self, x):
        if isinstance(x, (int, float)):
            if x < self.a:
                return 0
            elif x > self.b:
                return 1
            else:
                return (x - self.a) / (self.b - self.a)
        else:  # 处理数组输入
            result = np.zeros_like(x, dtype=float)
            mask1 = (x < self.a)
            mask2 = (x > self.b)
            mask3 = ~(mask1 | mask2)
            result[mask2] = 1
            result[mask3] = (x[mask3] - self.a) / (self.b - self.a)
            return result


class GPDistribution(Distribution):
    """广义帕累托分布"""

    def __init__(self, k, sigma, mu):
        self.k = k  # 形状参数
        self.sigma = sigma  # 尺度参数
        self.mu = mu  # 位置参数

    def pdf(self, x):
        if isinstance(x, (int, float)):
            if x < self.mu:
                return 0
            else:
                z = (x - self.mu) / self.sigma
                return (1 / self.sigma) * (1 + self.k * z) ** (-1 - 1 / self.k)
        else:  # 处理数组输入
            result = np.zeros_like(x, dtype=float)
            mask = (x >= self.mu)
            z = np.zeros_like(x, dtype=float)
            z[mask] = (x[mask] - self.mu) / self.sigma
            result[mask] = (1 / self.sigma) * (1 + self.k * z[mask]) ** (-1 - 1 / self.k)
            return result

    def cdf(self, x):
        if isinstance(x, (int, float)):
            if x < self.mu:
                return 0
            else:
                z = (x - self.mu) / self.sigma
                return 1 - (1 + self.k * z) ** (-1 / self.k)
        else:  # 处理数组输入
            result = np.zeros_like(x, dtype=float)
            mask = (x >= self.mu)
            z = np.zeros_like(x, dtype=float)
            z[mask] = (x[mask] - self.mu) / self.sigma
            result[mask] = 1 - (1 + self.k * z[mask]) ** (-1 / self.k)
            return result


class EmergencyModel:
    """应急物资储备决策模型"""

    def __init__(self, alpha, v, p1, c1, p2, s, m, distribution):
        """
        初始化模型参数

        参数:
        alpha - 灾害发生概率
        v - 单位物资残值
        p1 - 灾害前物资单价
        c1 - 政府单位物资储存成本
        p2 - 企业单位物资代储收入
        s - 企业单位物资使用补贴
        m - 灾害后物资市场单价
        distribution - 需求分布对象
        """
        self.alpha = alpha
        self.v = v
        self.p1 = p1
        self.c1 = c1
        self.p2 = p2
        self.s = s
        self.m = m
        self.dist = distribution

    def profit_case1(self, x, Q, q):
        """需求量 0 <= x <= Q 时的利润"""
        return self.v * (Q - x) - (self.p1 + self.c1) * Q - self.p2 * q

    def profit_case2(self, x, Q, q):
        """需求量 Q < x <= Q+q 时的利润"""
        return -(self.p1 + self.c1) * Q - self.p2 * q - self.s * (x - Q)

    def profit_case3(self, x, Q, q):
        """需求量 x > Q+q 时的利润"""
        return -(self.p1 + self.c1) * Q - self.p2 * q - self.s * q - self.m * (x - Q - q)

    def expected_profit_disaster(self, Q, q):
        """灾害发生时的期望利润"""
        # 使用数值积分计算三种情况下的期望利润

        # 情况1：0 <= x <= Q
        if Q > 0:
            integral1, _ = integrate.quad(
                lambda x: self.profit_case1(x, Q, q) * self.dist.pdf(x),
                0, Q
            )
        else:
            integral1 = 0

        # 情况2：Q < x <= Q+q
        if q > 0:
            integral2, _ = integrate.quad(
                lambda x: self.profit_case2(x, Q, q) * self.dist.pdf(x),
                Q, Q + q
            )
        else:
            integral2 = 0

        # 情况3：x > Q+q
        # 使用截断上限进行积分，因为无穷大可能导致积分不收敛
        upper_limit = Q + q + 100000000 # 选择一个足够大的上限值

        # 如果使用的是均匀分布，可以直接使用分布的上限
        if isinstance(self.dist, UniformDistribution):
            upper_limit = min(upper_limit, self.dist.b)

        integral3, _ = integrate.quad(
            lambda x: self.profit_case3(x, Q, q) * self.dist.pdf(x),
            Q + q, upper_limit
        )

        return integral1 + integral2 + integral3

    def government_profit(self, Qq):
        """政府总利润函数"""
        Q, q = Qq
        # 如果Q或q为负，返回大正值作为惩罚
        if Q < 0 or q < 0:
            return 1e10

        # 灾害未发生时的利润
        no_disaster_profit = (1 - self.alpha) * (self.v * Q - (self.p1 + self.c1) * Q - self.p2 * q)

        # 灾害发生时的期望利润
        disaster_profit = self.alpha * self.expected_profit_disaster(Q, q)

        # 总利润
        total_profit = no_disaster_profit + disaster_profit

        # 因为minimize是求最小值，所以返回负的利润
        return -total_profit

    def solve(self, initial_guess=None, multi_start=True, n_starts=5):
        """
        求解最优储备量

        参数:
        initial_guess - 初始猜测值
        multi_start - 是否使用多起点优化策略
        n_starts - 多起点优化时的起点数量

        返回:
        (Q*, q*) - 最优政府储备量和企业储备量
        profit - 最大利润值
        """
        if initial_guess is None:
            initial_guess = [500, 500]

        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # Q >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]}  # q >= 0
        ]

        if multi_start:
            # 多起点优化
            results = []

            # 生成随机起点
            np.random.seed(1)  # 设置随机种子以便结果可重复
            initial_points = np.random.rand(n_starts, 2) * 5551

            # 对每个起点进行优化
            for point in initial_points:
                try:
                    result = optimize.minimize(
                        self.government_profit,
                        point,
                        method='SLSQP',
                        constraints=constraints,
                        options={'ftol': 1e-8, 'disp': False, 'maxiter': 5000}
                    )
                    if result.success:
                        results.append((result.x, -result.fun, result.success))
                except Exception as e:
                    print(f"优化失败，起点: {point}, 错误: {e}")

            if not results:
                raise ValueError("所有优化尝试都失败了，请检查模型参数和积分设置")

            # 选择最优解
            results.sort(key=lambda x: x[1], reverse=True)  # 按利润降序排序
            best_result = results[0]

            Q_star, q_star = best_result[0]
            max_profit = best_result[1]

            return (Q_star, q_star), max_profit, results
        else:
            # 单起点优化
            result = optimize.minimize(
                self.government_profit,
                initial_guess,
                method='SLSQP',
                constraints=constraints,
                options={'ftol': 1e-8, 'disp': True, 'maxiter': 5000}
            )

            Q_star, q_star = result.x
            max_profit = -result.fun

            return (Q_star, q_star), max_profit, result


def perform_sensitivity_analysis(model_class, base_params, param_name, param_values, dist_name, T):
    """
    执行敏感性分析并返回不同参数值下的最优储备量

    参数:
    model_class - 模型类
    base_params - 基本参数字典
    param_name - 要分析的参数名称
    param_values - 参数取值列表
    dist_name - 分布名称("GP"或"Uniform")
    T - 总需求量

    返回:
    Q_values - 政府实物储备量列表
    q_values - 企业实物储备量列表
    p_values - 企业生产能力储备量列表
    """
    Q_values = []
    q_values = []
    p_values = []

    for value in param_values:
        # 复制基本参数并更新待分析参数
        params = base_params.copy()
        params[param_name] = value

        # 创建适当的分布
        if dist_name == "GP":
            dist = GPDistribution(1.26089, 1059.03, 0)
        else:  # Uniform
            dist = UniformDistribution(0, T)

        # 创建并求解模型
        model = EmergencyModel(
            params['alpha'], params['v'], params['p1'], params['c1'],
            params['p2'], params['s'], params['m'], dist
        )

        try:
            optimal, _, _ = model.solve(
                initial_guess=[T * 0.3, T * 0.1],  # 基于总需求的初始猜测
                multi_start=True,
                n_starts=10
            )

            Q_star, q_star = optimal
            p_star = T - Q_star - q_star  # 企业生产能力储备 = 总需求 - 政府储备 - 企业储备

            Q_values.append(Q_star)
            q_values.append(q_star)
            p_values.append(p_star)

            print(f"{dist_name}分布, {param_name}={value}: Q*={Q_star:.2f}, q*={q_star:.2f}, p*={p_star:.2f}")

        except Exception as e:
            print(f"求解失败 ({dist_name}分布, {param_name}={value}): {e}")
            # 添加None作为占位符，表示求解失败点
            Q_values.append(None)
            q_values.append(None)
            p_values.append(None)

    return Q_values, q_values, p_values


def plot_sensitivity(param_name, param_values, gp_results, uniform_results, param_label, filename, T):
    """
    绘制敏感性分析图

    参数:
    param_name - 参数名称
    param_values - 参数值列表
    gp_results - GP分布结果 (Q, q, p)
    uniform_results - 均匀分布结果 (Q, q, p)
    param_label - 图表中显示的参数标签
    filename - 保存的文件名
    T - 总需求量，用于归一化
    """
    # 提取结果
    gp_Q, gp_q, gp_p = gp_results
    uniform_Q, uniform_q, uniform_p = uniform_results

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 设置布局和标题
    plt.suptitle(f'敏感性分析: {param_label}对最优储备量的影响', fontsize=16)

    # 广义帕累托分布子图
    plt.subplot(2, 1, 1)

    # 非空值的索引
    valid_indices = [i for i, v in enumerate(gp_Q) if v is not None]
    valid_params = [param_values[i] for i in valid_indices]

    # 绘制GP分布结果
    if valid_indices:
        valid_gp_Q = [gp_Q[i] for i in valid_indices]
        valid_gp_q = [gp_q[i] for i in valid_indices]
        valid_gp_p = [gp_p[i] for i in valid_indices]

        plt.plot(valid_params, valid_gp_Q, 'ro-', linewidth=2, markersize=8, label='政府实物储备量 (Q*)')
        plt.plot(valid_params, valid_gp_q, 'bs-', linewidth=2, markersize=8, label='企业实物储备量 (q*)')
        plt.plot(valid_params, valid_gp_p, 'gd-', linewidth=2, markersize=8, label='企业生产能力储备量 (p*)')

    plt.title('广义帕累托分布 (GPD)')
    plt.xlabel(param_label)
    plt.ylabel('储备量 (万件)')
    plt.legend()
    plt.grid(True)

    # 均匀分布子图
    plt.subplot(2, 1, 2)

    # 非空值的索引
    valid_indices = [i for i, v in enumerate(uniform_Q) if v is not None]
    valid_params = [param_values[i] for i in valid_indices]

    # 绘制均匀分布结果
    if valid_indices:
        valid_uniform_Q = [uniform_Q[i] for i in valid_indices]
        valid_uniform_q = [uniform_q[i] for i in valid_indices]
        valid_uniform_p = [uniform_p[i] for i in valid_indices]

        plt.plot(valid_params, valid_uniform_Q, 'ro-', linewidth=2, markersize=8, label='政府实物储备量 (Q*)')
        plt.plot(valid_params, valid_uniform_q, 'bs-', linewidth=2, markersize=8, label='企业实物储备量 (q*)')
        plt.plot(valid_params, valid_uniform_p, 'gd-', linewidth=2, markersize=8, label='企业生产能力储备量 (p*)')

    plt.title('均匀分布 (UD)')
    plt.xlabel(param_label)
    plt.ylabel('储备量 (万件)')
    plt.legend()
    plt.grid(True)

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_stacked_bar(gp_result, uniform_result, T):
    """
    绘制堆叠柱状图比较两种分布方式下的储备结构

    参数:
    gp_result - GP分布的结果 (Q*, q*, p*)
    uniform_result - 均匀分布的结果 (Q*, q*, p*)
    T - 总需求量
    """
    labels = ['广义帕累托分布', '均匀分布']

    # 提取结果
    Q_gp, q_gp, p_gp = gp_result
    Q_uniform, q_uniform, p_uniform = uniform_result

    # 计算比例
    Q_gp_ratio = Q_gp / T
    q_gp_ratio = q_gp / T
    p_gp_ratio = p_gp / T

    Q_uniform_ratio = Q_uniform / T
    q_uniform_ratio = q_uniform / T
    p_uniform_ratio = p_uniform / T

    # 数据准备
    Q_values = [Q_gp_ratio, Q_uniform_ratio]
    q_values = [q_gp_ratio, q_uniform_ratio]
    p_values = [p_gp_ratio, p_uniform_ratio]

    # 绘制堆叠柱状图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 底部位置初始化
    bottoms = [0, 0]

    # 绘制三种储备方式的堆叠柱状图
    p1 = ax.bar(labels, Q_values, label='政府实物储备 (Q*)', color='#3274A1')
    p2 = ax.bar(labels, q_values, bottom=Q_values, label='企业实物储备 (q*)', color='#E1812C')
    p3 = ax.bar(labels, p_values, bottom=[Q_values[i] + q_values[i] for i in range(len(labels))],
                label='企业生产能力储备 (p*)', color='#3A923A')

    # 添加百分比标签
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0.03:  # 仅当比例足够大时显示标签
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_y() + height / 2.,
                        f'{value:.1%}',
                        ha='center', va='center', color='white', fontweight='bold')

    add_labels(p1, Q_values)
    add_labels(p2, q_values)
    add_labels(p3, p_values)

    # 设置图表
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('储备占比')
    ax.set_title('不同概率分布下的最优储备结构对比')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # 在柱状图上方添加总需求量信息
    for i, label in enumerate(labels):
        ax.text(i, 1.02, f'总需求: {T}万件', ha='center')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.tight_layout()
    plt.savefig('储备结构对比.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_sensitivity_analysis():
    """运行完整的敏感性分析"""
    # 模型参数
    T = 5551  # 应急物资总需求量（万件）

    base_params = {
        'alpha': 1,  # 灾害发生概率
        'v': 100,  # 单位物资残值
        'p1': 200,  # 灾害前物资单价
        'c1': 120,  # 政府单位物资储存成本
        'p2': 170,  # 企业单位物资代储收入
        's': 180,  # 企业单位物资使用补贴
        'm': 500,  # 灾害后物资市场单价
    }

    # 记录基本情况的最优解
    print("计算基准解...")
    uniform_dist = UniformDistribution(0, T)
    gp_dist = GPDistribution(1.26089, 1059.03, 0)

    model_uniform = EmergencyModel(
        base_params['alpha'], base_params['v'], base_params['p1'], base_params['c1'],
        base_params['p2'], base_params['s'], base_params['m'], uniform_dist
    )

    model_gp = EmergencyModel(
        base_params['alpha'], base_params['v'], base_params['p1'], base_params['c1'],
        base_params['p2'], base_params['s'], base_params['m'], gp_dist
    )

    optimal_uniform, profit_uniform, _ = model_uniform.solve(
        initial_guess=[T * 0.2, T * 0.1], multi_start=True, n_starts=10
    )

    optimal_gp, profit_gp, _ = model_gp.solve(
        initial_guess=[T * 0.2, T * 0.1], multi_start=True, n_starts=10
    )

    Q_uniform, q_uniform = optimal_uniform
    p_uniform = T - Q_uniform - q_uniform

    Q_gp, q_gp = optimal_gp
    p_gp = T - Q_gp - q_gp

    print(f"\n基准解 - 均匀分布:")
    print(f"政府实物储备量(Q*): {Q_uniform:.2f} ({Q_uniform / T:.2%})")
    print(f"企业实物储备量(q*): {q_uniform:.2f} ({q_uniform / T:.2%})")
    print(f"企业生产能力储备量(p*): {p_uniform:.2f} ({p_uniform / T:.2%})")
    print(f"最大利润: {profit_uniform:.2f}")

    print(f"\n基准解 - 广义帕累托分布:")
    print(f"政府实物储备量(Q*): {Q_gp:.2f} ({Q_gp / T:.2%})")
    print(f"企业实物储备量(q*): {q_gp:.2f} ({q_gp / T:.2%})")
    print(f"企业生产能力储备量(p*): {p_gp:.2f} ({p_gp / T:.2%})")
    print(f"最大利润: {profit_gp:.2f}")

    # 绘制储备结构对比图
    gp_result = (Q_gp, q_gp, p_gp)
    uniform_result = (Q_uniform, q_uniform, p_uniform)
    plot_stacked_bar(gp_result, uniform_result, T)

    # 敏感性分析参数设置
    sensitivity_params = [
        # 参数名, 参数值范围, 图表标签, 文件名
        ('alpha', [0.8, 0.84, 0.88, 0.92, 0.96, 1.0], '灾害发生概率 (α)', 'sensitivity_alpha.png'),
        ('v', [75, 80, 85, 90, 95, 100], '单位物资残值 (v)', 'sensitivity_v.png'),
        ('s', [160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190], '企业单位物资使用补贴 (s)', 'sensitivity_s.png'),
        ('p2', [150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170], '企业单位物资代储收入 (p2)', 'sensitivity_p2.png'),
        ('p1', [200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220], '灾害前物资单价 (p1)', 'sensitivity_p1.png'),
        ('c1', [120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140], '政府单位物资储存成本 (c1)', 'sensitivity_c1.png'),
        ('m', [500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550], '灾害后市场单价 (m)', 'sensitivity_m.png'),
    ]

    # 执行敏感性分析
    for param_name, param_values, param_label, filename in sensitivity_params:
        print(f"\n执行敏感性分析: {param_label}")

        # GP分布敏感性分析
        gp_results = perform_sensitivity_analysis(
            EmergencyModel, base_params, param_name, param_values, "GP", T
        )

        # 均匀分布敏感性分析
        uniform_results = perform_sensitivity_analysis(
            EmergencyModel, base_params, param_name, param_values, "Uniform", T
        )

        # 绘制并保存图表
        plot_sensitivity(param_name, param_values, gp_results, uniform_results, param_label, filename, T)

        print(f"完成 {param_label} 敏感性分析并保存图表到 {filename}")

# 主函数：应用模型求解案例
def main():
    # # 模型参数
    # v = 100  # 单位物资残值
    # p1 = 200  # 灾害前物资单价
    # m = 500  # 灾害后应急物资市场单价
    # alpha = 1  # 灾害发生概率
    # e = 400  # 企业单位物资加急生产成本
    # p2 = 170  # 企业单位物资代储收入
    # c2 = 300  # 企业单位物资储存成本
    # s = 180  # 企业单位物资使用补贴
    # c1 = 120  # 政府单位物资储存成本
    # T = 5551  # 应急物资总需求量（万件）
    #
    # uniform_dist = UniformDistribution(0, T)
    #
    #
    # # 广义帕累托参数
    # gp_dist = GPDistribution(1.26089, 1059.03, 0)
    #
    # # 均匀分布
    # model_uniform = EmergencyModel(alpha, v, p1, c1, p2, s, m, uniform_dist)
    #
    # # 求解最优储备量
    # print("使用均匀分布求解...")
    # try:
    #     optimal_uniform, profit_uniform, results_uniform = model_uniform.solve(
    #         initial_guess=[2000, 1000],  # 给一个更合理的初始猜测
    #         multi_start=True,
    #         n_starts=50
    #     )
    #
    #     print(f"最优政府储备量(Q*): {optimal_uniform[0]:.4f}")
    #     print(f"最优企业储备量(q*): {optimal_uniform[1]:.4f}")
    #     print(f"最大利润: {profit_uniform:.4f}")
    #
    #
    # except Exception as e:
    #     print(f"均匀分布模型求解失败: {e}")
    #
    # # 创建模型实例 - 使用广义帕累托分布
    # try:
    #     model_gp = EmergencyModel(alpha, v, p1, c1, p2, s, m, gp_dist)
    #
    #     # 求解最优储备量
    #     print("\n使用广义帕累托分布求解...")
    #     optimal_gp, profit_gp, results_gp = model_gp.solve(
    #         initial_guess=[100, 100],
    #         multi_start=True,
    #         n_starts=50
    #     )
    #
    #     print(f"最优政府储备量(Q*): {optimal_gp[0]:.4f}")
    #     print(f"最优企业储备量(q*): {optimal_gp[1]:.4f}")
    #     print(f"最大利润: {profit_gp:.4f}")
    #
    #
    # except Exception as e:
    #     print(f"广义帕累托分布模型求解失败: {e}")

    # 运行敏感性分析
    print("开始敏感性分析...")
    run_sensitivity_analysis()


if __name__ == "__main__":
    main()
