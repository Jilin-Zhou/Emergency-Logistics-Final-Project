import numpy as np
from matplotlib import font_manager, rcParams
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import scienceplots


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


class InverseGaussianDistribution(Distribution):
    """反高斯分布"""

    def __init__(self, mu, loc, scale):
        self.mu = mu  # 均值参数
        self.loc = loc  # 位置参数
        self.scale = scale  # 尺度参数

    def pdf(self, x):
        """概率密度函数"""
        if isinstance(x, (int, float)):
            # 调整位置参数
            x_adj = x - self.loc
            if x_adj <= 0:  # 反高斯分布只在正实数上有定义
                return 0
            else:
                # 反高斯分布的PDF公式
                lambda_param = self.scale
                mu = self.mu
                return np.sqrt(lambda_param / (2 * np.pi * x_adj ** 3)) * np.exp(
                    -lambda_param * (x_adj - mu) ** 2 / (2 * mu ** 2 * x_adj)
                )
        else:  # 处理数组输入
            result = np.zeros_like(x, dtype=float)
            x_adj = x - self.loc
            mask = (x_adj > 0)  # 只在正实数上有定义

            lambda_param = self.scale
            mu = self.mu

            # 使用向量化操作计算PDF
            valid_x = x_adj[mask]
            result[mask] = np.sqrt(lambda_param / (2 * np.pi * valid_x ** 3)) * np.exp(
                -lambda_param * (valid_x - mu) ** 2 / (2 * mu ** 2 * valid_x)
            )
            return result

    def cdf(self, x):
        """累积分布函数"""
        if isinstance(x, (int, float)):
            # 调整位置参数
            x_adj = x - self.loc
            if x_adj <= 0:
                return 0
            else:
                # 反高斯分布的CDF公式
                lambda_param = self.scale
                mu = self.mu

                # 计算CDF的两个部分
                term1 = norm.cdf(np.sqrt(lambda_param / x_adj) * (x_adj / mu - 1))
                term2 = np.exp(2 * lambda_param / mu) * norm.cdf(-np.sqrt(lambda_param / x_adj) * (x_adj / mu + 1))

                return term1 + term2
        else:  # 处理数组输入
            result = np.zeros_like(x, dtype=float)
            x_adj = x - self.loc
            mask = (x_adj > 0)

            lambda_param = self.scale
            mu = self.mu

            # 使用向量化操作计算CDF
            valid_x = x_adj[mask]
            term1 = norm.cdf(np.sqrt(lambda_param / valid_x) * (valid_x / mu - 1))
            term2 = np.exp(2 * lambda_param / mu) * norm.cdf(-np.sqrt(lambda_param / valid_x) * (valid_x / mu + 1))

            result[mask] = term1 + term2
            return result


class EmergencyModel:
    """应急物资储备决策模型"""

    def __init__(self, alpha, v, p1, c1, p2, s, m, e, lam, distribution):
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
        lam - 企业捐赠系数
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
        self.Qj = (lam**2) * m * ((m - e)**2) / (4 * (e**2))

    def profit_case1(self, x, Q, q):
        """需求量 0 <= x <= Q 时的利润"""
        return self.v * (Q - x) - (self.p1 + self.c1) * Q - self.p2 * q

    def profit_case2(self, x, Q, q):
        """需求量 Q < x <= Q+q 时的利润"""
        return -(self.p1 + self.c1) * Q - self.p2 * q - self.s * (x - Q)

    def profit_case3(self, x, Q, q):
        """需求量 Q+q < x <= Q+q+Qj 时的利润"""
        return -(self.p1 + self.c1) * Q - self.p2 * q - self.s * q

    def profit_case4(self, x, Q, q):
        """需求量 x > Q+q+Qj 时的利润"""
        return -(self.p1 + self.c1) * Q - self.p2 * q - self.s * q - self.m * (x - Q - q - self.Qj)

    def expected_profit_disaster(self, Q, q):
        """灾害发生时的期望利润"""
        # 使用数值积分计算三种情况下的期望利润
        Qj = self.Qj
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

        # 情况3：Q+q < x <= Q+q+Qj
        if Qj > 0:
            integral3, _ = integrate.quad(
                lambda x: self.profit_case3(x, Q, q) * self.dist.pdf(x),
                Q + q, Q + q + Qj
            )
        else:
            integral3 = 0

        # 情况4：x > Q+q+Qj
        # upper_limit取T就行了
        upper_limit = 180
        if isinstance(self.dist, UniformDistribution):
            # print(self.dist.b)
            upper_limit = self.dist.b
        integral4, _ = integrate.quad(
            lambda x: self.profit_case4(x, Q, q) * self.dist.pdf(x),
            Q + q + Qj, upper_limit
        )

        return integral1 + integral2 + integral3 + integral4

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
            initial_guess = [5, 5]

        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # Q >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # q >= 0
            {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},
            # {'type': 'ineq', 'fun': lambda x: 180 - x[0] - x[1]}
        ]

        if multi_start:
            # 多起点优化
            results = []

            # 生成随机起点
            np.random.seed(1)  # 设置随机种子以便结果可重复
            initial_points = np.random.rand(n_starts, 2) * 25

            # 对每个起点进行优化
            for point in initial_points:
                try:
                    result = optimize.minimize(
                        self.government_profit,
                        point,
                        method='SLSQP',
                        constraints=constraints,
                        options={'ftol': 1e-8, 'disp': False, 'maxiter': 50000}
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

            return (Q_star, q_star, self.Qj), max_profit, results
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

            return (Q_star, q_star, self.Qj), max_profit, result


def calculate_enterprise_profit(Q, q, Qj, params, distribution):
    """
    计算企业的利润

    参数:
    Q - 政府实物储备量
    q - 企业实物储备量
    Qj - 企业生产能力储备量（由参数计算得到）
    params - 包含所有模型参数的字典
    distribution - 需求分布对象

    返回:
    企业利润值
    """
    # 提取参数
    alpha = params['alpha']  # 灾害发生概率
    v = params['v']  # 单位物资残值
    p2 = params['p2']  # 企业单位物资代储收入
    c2 = params['c2']  # 企业单位物资储存成本
    s = params['s']  # 企业单位物资使用补贴
    m = params['m']  # 灾害后物资市场单价
    e = params['e']  # 企业单位物资加急生产成本
    lam = params['lam']  # 企业捐赠系数

    # 灾害未发生时的企业利润
    no_disaster_profit = (v + p2 - c2) * q

    # 灾害发生时的企业期望利润
    # 情况1：0 <= x <= Q
    if Q > 0:
        integral1, _ = integrate.quad(
            lambda x: ((v + p2 - c2) * q) * distribution.pdf(x),
            0, Q
        )
    else:
        integral1 = 0

    # 情况2：Q < x <= Q+q
    if q > 0:
        integral2, _ = integrate.quad(
            lambda x: (s * (x - Q) + v * (Q + q - x) + (p2 - c2) * q) * distribution.pdf(x),
            Q, Q + q
        )
    else:
        integral2 = 0

    # 情况3：Q+q < x <= Q+q+Qj
    if Qj > 0:
        integral3, _ = integrate.quad(
            lambda x: (s * q + (p2 - c2) * q + lam * (m - e) * np.sqrt(Qj) / np.sqrt(m) - e * Qj) * distribution.pdf(x),
            Q + q, Q + q + Qj
        )
    else:
        integral3 = 0

    # 情况4：x > Q+q+Qj
    # 截断上限积分，设定一个最大值即可
    upper_limit = 180
    if isinstance(distribution, UniformDistribution):
        upper_limit = distribution.b

    integral4, _ = integrate.quad(
        lambda x: (s * q + (p2 - c2) * q + lam * (m - e) * np.sqrt(Qj) / np.sqrt(m) - e * Qj +
                   (m - e) * (x - Q - q - Qj)) * distribution.pdf(x),
        Q + q + Qj, upper_limit
    )

    # 灾害发生时的期望利润
    disaster_profit = integral1 + integral2 + integral3 + integral4

    # 总期望利润
    total_profit = (1 - alpha) * no_disaster_profit + alpha * disaster_profit

    return total_profit


def perform_sensitivity_analysis(model_class, base_params, param_name, param_values, dist_name, T_total_demand):
    Q_values = []
    q_values = []
    p_values = []
    Qj_values = []
    for value in param_values:
        params = base_params.copy()
        params[param_name] = value

        current_T_for_dist = T_total_demand  # Use the overall T for uniform dist range
        if param_name == 'T' and dist_name == "Uniform":  # If T itself is changing for Uniform
            current_T_for_dist = value
        if dist_name == "InvGauss":
            dist = InverseGaussianDistribution(40.69, -0.97, 4.87)
        else:
            dist = UniformDistribution(0, current_T_for_dist)
        model = EmergencyModel(
            params['alpha'], params['v'], params['p1'], params['c1'],
            params['p2'], params['s'], params['m'], params['e'], params['lam'], dist
        )
        try:
            initial_Q_guess = current_T_for_dist * 0.2 if current_T_for_dist > 0 else 1
            initial_q_guess = current_T_for_dist * 0.1 if current_T_for_dist > 0 else 1

            optimal, _, _ = model.solve(
                initial_guess=[initial_Q_guess, initial_q_guess],
                multi_start=True,
                n_starts=10
            )
            Q_star, q_star, Qj_star = optimal
            p_star = T_total_demand - Q_star - q_star - Qj_star

            Q_values.append(Q_star)
            q_values.append(q_star)
            p_values.append(p_star)
            Qj_values.append(Qj_star)
            # print(f"{dist_name}分布, {param_name}={value}: Q*={Q_star:.2f}, q*={q_star:.2f}, Qj*={Qj_star:.2f}, p*(residual)={p_star:.2f}")
        except Exception as e:
            # print(f"求解失败 ({dist_name}分布, {param_name}={value}): {e}")
            Q_values.append(None)
            q_values.append(None)
            p_values.append(None)
            Qj_values.append(None)
    return Q_values, q_values, p_values, Qj_values


def plot_sensitivity(param_name, param_values, uniform_results, invgauss_results, param_label, filename, T):    
    plt.style.use(['science', 'no-latex'])
    font_path = "tnw+simsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.grid(True)
    uniform_Q, uniform_q, uniform_p, uniform_Qj = uniform_results
    invgauss_Q, invgauss_q, invgauss_p, invgauss_Qj = invgauss_results

    plt.figure(figsize=(8, 5))
    # 均匀分布子图
    plt.subplot(2, 1, 1)
    valid_indices_uniform = [i for i, v in enumerate(uniform_Q) if v is not None]
    valid_params_uniform = [param_values[i] for i in valid_indices_uniform]

    if valid_indices_uniform:
        valid_uniform_Q = [uniform_Q[i] for i in valid_indices_uniform]
        valid_uniform_q = [uniform_q[i] for i in valid_indices_uniform]
        valid_uniform_p_residual = [uniform_p[i] for i in valid_indices_uniform]  # p is residual
        valid_uniform_Qj = [uniform_Qj[i] for i in valid_indices_uniform]
        plt.plot(valid_params_uniform, valid_uniform_Q, 'ro-', linewidth=2, markersize=8, label=r'政府实物储备量 ($Q^*$)')
        plt.plot(valid_params_uniform, valid_uniform_q, 'bs-', linewidth=2, markersize=8, label=r'企业实物储备量 ($q^*$)')
        plt.plot(valid_params_uniform, valid_uniform_Qj, 'mo-', linewidth=2, markersize=8,
                 label=r'企业捐赠 ($Q^*_j$)')  # New Qj line
        plt.plot(valid_params_uniform, valid_uniform_p_residual, 'gd-', linewidth=2, markersize=8,
                 label=r'企业生产能力储备 ($p^*$)')  # p is residual

    plt.title('均匀分布 (UD)')
    plt.xlabel(param_label)
    plt.ylabel('储备量 (万件)')

    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True)
    # 反高斯分布子图
    plt.subplot(2, 1, 2)
    valid_indices_invgauss = [i for i, v in enumerate(invgauss_Q) if v is not None]
    valid_params_invgauss = [param_values[i] for i in valid_indices_invgauss]
    if valid_indices_invgauss:
        valid_invgauss_Q = [invgauss_Q[i] for i in valid_indices_invgauss]
        valid_invgauss_q = [invgauss_q[i] for i in valid_indices_invgauss]
        valid_invgauss_p_residual = [invgauss_p[i] for i in valid_indices_invgauss]  # p is residual
        valid_invgauss_Qj = [invgauss_Qj[i] for i in valid_indices_invgauss]
        plt.plot(valid_params_invgauss, valid_invgauss_Q, 'ro-', linewidth=2, markersize=8, label=r'政府实物储备量 ($Q^*$)')
        plt.plot(valid_params_invgauss, valid_invgauss_q, 'bs-', linewidth=2, markersize=8, label=r'企业实物储备量 ($q^*$)')
        plt.plot(valid_params_invgauss, valid_invgauss_Qj, 'mo-', linewidth=2, markersize=8,
                 label=r'企业捐赠 ($Q^*_j$)')  # New Qj line
        plt.plot(valid_params_invgauss, valid_invgauss_p_residual, 'gd-', linewidth=2, markersize=8,
                 label=r'企业生产能力储备 ($p^*$)')  # p is residual
    plt.title('反高斯分布')
    plt.xlabel(param_label)
    plt.ylabel('储备量 (万件)')

    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Reduced DPI for faster save during testing
    # plt.show()


def plot_stacked_bar(uniform_Q_q_Qj_result, invgauss_Q_q_Qj_result, T):
    labels = ['均匀分布', '反高斯分布']

    Q_uniform, q_uniform, Qj_uniform = uniform_Q_q_Qj_result
    Q_invgauss, q_invgauss, Qj_invgauss = invgauss_Q_q_Qj_result

    p_uniform = T-Qj_uniform-Q_uniform-q_uniform
    p_invgauss = T-Qj_invgauss-Qj_invgauss-q_invgauss

    T_adj = T if T > 0 else 1
    Q_uniform_ratio = Q_uniform / T_adj
    q_uniform_ratio = q_uniform / T_adj
    Qj_uniform_ratio = Qj_uniform / T_adj
    p_uniform_ratio = p_uniform / T_adj
    Q_invgauss_ratio = Q_invgauss / T_adj
    q_invgauss_ratio = q_invgauss / T_adj
    Qj_invgauss_ratio = Qj_invgauss / T_adj
    p_invgauss_ratio = p_invgauss / T_adj

    Q_values_ratio = [Q_uniform_ratio, Q_invgauss_ratio]
    q_values_ratio = [q_uniform_ratio, q_invgauss_ratio]
    Qj_values_ratio = [Qj_uniform_ratio, Qj_invgauss_ratio]
    p_values_ratio = [p_uniform_ratio, p_invgauss_ratio]

    fig, ax = plt.subplots(figsize=(5, 4))

    plt.style.use(['science','no-latex'])  # 使用scienceplots样式

    from mplfonts import use_font
    from matplotlib import font_manager
    from matplotlib import rcParams
    font_path = "tnw+simsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = 'sans-serif' # 使用字体中的无衬线体
    rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    rcParams['axes.unicode_minus'] = False # 使坐标轴刻度标签正常显示正负号

    bar_width = 0.5  # Adjust bar width if needed

    p1 = ax.bar(labels, Q_values_ratio, width=bar_width, label='政府实物储备 (Q*)', color='#3274A1')
    bottom_q = Q_values_ratio
    p2 = ax.bar(labels, q_values_ratio, width=bar_width, bottom=bottom_q, label='企业实物储备 (q*)', color='#E1812C')
    bottom_Qj = [bottom_q[i] + q_values_ratio[i] for i in range(len(labels))]
    p3 = ax.bar(labels, Qj_values_ratio, width=bar_width, bottom=bottom_Qj,
                label=r'企业捐赠($Q_j^*$)', color='#3A923A')  # Updated label for Qj
    bottom_p = [bottom_Qj[i] + Qj_values_ratio[i] for i in range(len(labels))]
    p4 = ax.bar(labels, p_values_ratio, width=bar_width, bottom=bottom_p, label='企业生产能力储备',color='#25C4EC')

    def add_labels(bars, values_ratio):
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            value = values_ratio[bar_idx]  # Use the original ratio for the label
            if height > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_y() + height / 2.,
                        f'{value:.1%}',  # Display the ratio as percentage
                        ha='center', va='center', color='white', fontweight='bold')

    add_labels(p1, Q_values_ratio)
    add_labels(p2, q_values_ratio)
    add_labels(p3, Qj_values_ratio)  # Add labels for Qj bar
    add_labels(p4, p_values_ratio)

    ax.set_ylim(0, max(1.0, max(
        bottom_Qj[i] + Qj_values_ratio[i] for i in range(len(labels))) * 1.1))  # Adjust y-limit if sum > 1
    ax.set_ylabel('储备占比 (占总需求T)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    for i, label in enumerate(labels):
        ax.text(i, ax.get_ylim()[1] * 1.02, f'总需求 T: {T} 万件', ha='center')
    plt.style.use(['ieee', 'muted', 'no-latex'])
    font_path = "tnw+simsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('储备结构对比.png', dpi=300, bbox_inches='tight')
    # plt.show()


def run_sensitivity_analysis():
    T_total_demand = 20
    base_params = {
        'alpha': 1,
        'v': 150,
        'p1': 220,
        'c1': 120,
        'p2': 170,
        's': 180,
        'm': 500,
        'e': 400,
        'lam': 0.2,
    }

    print("计算基准解...")
    uniform_dist = UniformDistribution(0, T_total_demand)
    invgauss_dist = InverseGaussianDistribution(40.69, -0.97, 4.87)  # mu, loc, scale

    model_uniform = EmergencyModel(
        base_params['alpha'], base_params['v'], base_params['p1'], base_params['c1'],
        base_params['p2'], base_params['s'], base_params['m'], base_params['e'], base_params['lam'], uniform_dist
    )
    model_invgauss = EmergencyModel(
        base_params['alpha'], base_params['v'], base_params['p1'], base_params['c1'],
        base_params['p2'], base_params['s'], base_params['m'], base_params['e'], base_params['lam'], invgauss_dist
    )

    try:
        optimal_uniform, profit_uniform, _ = model_uniform.solve(
            initial_guess=[T_total_demand * 0.2, T_total_demand * 0.1], multi_start=True, n_starts=10
        )
        Q_uniform, q_uniform, Qj_uniform = optimal_uniform
        p_uniform_residual = T_total_demand - Q_uniform - q_uniform - Qj_uniform # Residual
        print(f"\n基准解 - 均匀分布 (T={T_total_demand}):")
        print(f"政府实物储备量(Q*): {Q_uniform:.2f} ({Q_uniform / T_total_demand:.2%})")
        print(f"企业实物储备量(q*): {q_uniform:.2f} ({q_uniform / T_total_demand:.2%})")
        print(f"企业捐赠(Qj*): {Qj_uniform:.2f} ({Qj_uniform / T_total_demand:.2%})")
        print(f"企业产能储备(p*): {p_uniform_residual:.2f} ({p_uniform_residual / T_total_demand:.2%})")
        print(f"最大利润: {profit_uniform:.2f}")
        uniform_result_stacked = (Q_uniform, q_uniform, Qj_uniform,)
    except Exception as e:
        print(f"基准解 - 均匀分布求解失败: {e}")
        uniform_result_stacked = (0, 0, 0)  # Default for plot if fails
    try:
        optimal_invgauss, profit_invgauss, _ = model_invgauss.solve(
            initial_guess=[T_total_demand * 0.2, T_total_demand * 0.1], multi_start=True, n_starts=10
        )
        Q_invgauss, q_invgauss, Qj_invgauss = optimal_invgauss
        p_invgauss_residual = T_total_demand - Q_invgauss - q_invgauss - Qj_invgauss # Residual
        print(f"\n基准解 - 反高斯分布 (参照 T={T_total_demand} for ratios):")
        print(f"政府实物储备量(Q*): {Q_invgauss:.2f} ({Q_invgauss / T_total_demand:.2%})")
        print(f"企业实物储备量(q*): {q_invgauss:.2f} ({q_invgauss / T_total_demand:.2%})")
        print(f"企业捐赠(Qj*): {Qj_invgauss:.2f} ({Qj_invgauss / T_total_demand:.2%})")
        print(f"企业产能储备(p*): {p_invgauss_residual:.2f} ({p_invgauss_residual / T_total_demand:.2%})")
        print(f"最大利润: {profit_invgauss:.2f}")
        invgauss_result_stacked = (Q_invgauss, q_invgauss, Qj_invgauss)
    except Exception as e:
        print(f"基准解 - 反高斯分布求解失败: {e}")
        invgauss_result_stacked = (0, 0, 0)  # Default for plot if fails

    plot_stacked_bar(uniform_result_stacked, invgauss_result_stacked, T_total_demand)

    sensitivity_params = [
        ('alpha', [0.8, 0.84, 0.88, 0.92, 0.96, 1.0], r'灾害发生概率 ($\alpha$)', 'sensitivity_alpha.png'),
        ('v', [120, 130, 140, 150, 160, 170], r'单位物资残值 ($v$)', 'sensitivity_v.png'),
        ('s', [150, 160, 170, 180, 190, 200], r'企业单位物资使用补贴 ($s$)', 'sensitivity_s.png'),
        ('p2', [150, 160, 170, 180, 190], r'企业单位物资代储收入 ($p_2$)', 'sensitivity_p2.png'),
        ('p1', [200, 210, 220, 230, 240, 250], r'灾害前物资单价 ($p_1$)', 'sensitivity_p1.png'),
        ('c1', [100, 110, 120, 130, 140, 150], r'政府单位物资储存成本 ($c_1$)', 'sensitivity_c1.png'),
        ('m', [450, 470, 490, 500, 510, 530, 550], r'灾害后市场单价 ($m$)', 'sensitivity_m.png'),
        ('lam', [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2], r'企业捐赠系数 ($λ$)', 'sensitivity_lam.png'),
        # ($'T', [15, 18, 20, 22, 25], '应急物资总需求量 ($T$)', 'sensitivity_T.png') # Example for T sensitivity
    ]

    for param_name, param_values, param_label, filename in sensitivity_params:
        print(f"\n执行敏感性分析: {param_label}")

        current_T_for_analysis = T_total_demand
        if param_name == 'T':
            pass
        uniform_results_sens = perform_sensitivity_analysis(
            EmergencyModel, base_params, param_name, param_values, "Uniform", current_T_for_analysis
        )
        invgauss_results_sens = perform_sensitivity_analysis(
            EmergencyModel, base_params, param_name, param_values, "InvGauss", current_T_for_analysis
        )

        plot_sensitivity(param_name, param_values, uniform_results_sens, invgauss_results_sens,
                         param_label, filename, current_T_for_analysis)  # Pass T for context if needed by plot
        print(f"完成 {param_label} 敏感性分析并保存图表到 {filename}")


# 主函数：应用模型求解案例
def main():
    # 模型参数
    v = 150  # 单位物资残值
    p1 = 220  # 灾害前物资单价
    m = 500  # 灾害后应急物资市场单价
    alpha = 1  # 灾害发生概率
    e = 400  # 企业单位物资加急生产成本
    p2 = 170  # 企业单位物资代储收入
    c2 = 300  # 企业单位物资储存成本
    s = 180  # 企业单位物资使用补贴
    c1 = 120  # 政府单位物资储存成本
    lam = 0.2
    T = 20  # 应急物资总需求量（万件）

    params = {
        'alpha': alpha,
        'v': v,
        'p1': p1,
        'c1': c1,
        'p2': p2,
        'c2': c2,
        's': s,
        'm': m,
        'e': e,
        'lam': lam
    }

    uniform_dist = UniformDistribution(0, T)

    model_uniform = EmergencyModel(alpha, v, p1, c1, p2, s, m, e, lam, uniform_dist)

    # 求解最优储备量
    print("使用均匀分布求解...")
    try:
        optimal_uniform, profit_uniform, results_uniform = model_uniform.solve(
            initial_guess=[5, 5],  # 给一个更合理的初始猜测
            multi_start=True,
            n_starts=50
        )
        enterprise_profit = calculate_enterprise_profit(optimal_uniform[0], optimal_uniform[1], optimal_uniform[2], params, uniform_dist)

        print(f"最优政府储备量(Q*): {optimal_uniform[0]:.4f}")
        print(f"最优企业储备量(q*): {optimal_uniform[1]:.4f}")
        print(f"企业捐赠量(p*): {optimal_uniform[2]:.4f}")
        print(f"政府最大利润: {profit_uniform:.4f}")
        print(f"企业最大利润:{enterprise_profit:.4f}")


    except Exception as e:
        print(f"均匀分布模型求解失败: {e}")


    # 创建反高斯分布
    invgauss_dist = InverseGaussianDistribution(
            mu=40.69,
            loc=-0.97,
            scale=4.87
    )
    # 创建模型实例 - 使用反高斯分布

    model_invgauss = EmergencyModel(alpha, v, p1, c1, p2, s, m, e, lam, invgauss_dist)
    # 求解最优储备量
    print("\n使用反高斯分布求解...")
    try:
        optimal_invgauss, profit_invgauss, results_invgauss = model_invgauss.solve(
            initial_guess=[5, 5],
            multi_start=True,
            n_starts=10
        )
        enterprise_profit = calculate_enterprise_profit(optimal_invgauss[0], optimal_invgauss[1], optimal_invgauss[2], params, invgauss_dist)
        print(f"最优政府储备量(Q*): {optimal_invgauss[0]:.4f}")
        print(f"最优企业储备量(q*): {optimal_invgauss[1]:.4f}")
        print(f"企业捐赠量(p*): {optimal_invgauss[2]:.4f}")
        print(f"政府最大利润: {profit_invgauss:.4f}")
        print(f"企业最大利润:{enterprise_profit:.4f}")
    except Exception as e:
        print(f"反高斯分布模型求解失败: {e}")

    print("开始敏感性分析...")
    run_sensitivity_analysis()


if __name__ == "__main__":
    main()
