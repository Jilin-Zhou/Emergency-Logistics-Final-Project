import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import norm


# 定义概率分布类
class Distribution:
    """概率分布基类"""

    def pdf(self, x):
        """概率密度函数"""
        pass

    def cdf(self, x):
        """累积分布函数"""
        pass
def calculate_expected_value(distribution_instance: Distribution, lower_bound: float = -np.inf, upper_bound: float = np.inf) -> float:
    """
    通过数值积分计算给定概率分布实例的期望值。
    参数:
    distribution_instance (Distribution): 实现了 pdf 方法的概率分布实例。
    lower_bound (float): 积分下限，默认为负无穷。
    upper_bound (float): 积分上限，默认为正无穷。
    返回:
    float: 计算得到的期望值。
    """
    # 被积函数 x * pdf(x)
    integrand = lambda x: x * distribution_instance.pdf(x)
    
    # 使用 scipy.integrate.quad进行数值积分
    # quad 返回一个元组 (积分结果, 估计误差)
    expected_value, integration_error = integrate.quad(integrand, lower_bound, upper_bound)
    
    # 此处可根据需要处理 integration_error
    # print(f"数值积分误差估计: {integration_error}")
    
    return expected_value

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

        upper_limit = 20
        if isinstance(self.dist, UniformDistribution):
            upper_limit = min(upper_limit, self.dist.b)
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
            {'type': 'ineq', 'fun': lambda x: 20 - x[0] - x[1] - self.Qj}
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
        
    def solve_with_no_q(self, initial_guess=None, multi_start=True, n_starts=5):
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
            {'type': 'eq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: 20 - x[0] - x[1] - self.Qj}
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

            
            Q_star = Q_star + q_star
            q_star = 0
            max_profit = self.government_profit([Q_star, q_star])
            return (Q_star, q_star, self.Qj), max_profit, result


    def solve_signle(self):
        Q = calculate_expected_value(self.dist)
        q = 0
        max_profit = self.government_profit([Q, q])
        return (Q, q, self.Qj), max_profit


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
    upper_limit = 20
    if isinstance(distribution, UniformDistribution):
        upper_limit = min(upper_limit, distribution.b)

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
    T = 15  # 应急物资总需求量（万件），作为积分上界

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


    uniform_dist = UniformDistribution(0, T)

    model_uniform = EmergencyModel(alpha, v, p1, c1, p2, s, m, e, 0, uniform_dist)
    # 求解最优储备量
    print("使用均匀分布求解...")
    try:
        optimal_uniform, profit_uniform, results_uniform = model_uniform.solve(
            initial_guess=[5, 5],  # 给一个更合理的初始猜测
            multi_start=True,
            n_starts=50
        )
        enterprise_profit = calculate_enterprise_profit(optimal_uniform[0], optimal_uniform[1], optimal_uniform[2], params, uniform_dist)

        print(f"不考虑捐赠最优政府储备量(Q*): {optimal_uniform[0]:.4f}")
        print(f"不考虑捐赠最优企业储备量(q*): {optimal_uniform[1]:.4f}")
        print(f"不考虑捐赠企业捐赠量(p*): {optimal_uniform[2]:.4f}")
        print(f"不考虑捐赠政府最大利润: {profit_uniform:.4f}")
        print(f"不考虑捐赠企业最大利润:{enterprise_profit:.4f}")


    except Exception as e:
        print(f"不考虑捐赠均匀分布模型求解失败: {e}")


    # 创建反高斯分布
    invgauss_dist = InverseGaussianDistribution(
            mu=40.69,
            loc=-0.97,
            scale=4.87
    )
    # 创建模型实例 - 使用反高斯分布

    model_invgauss = EmergencyModel(alpha, v, p1, c1, p2, s, m, e, 0, invgauss_dist)
    # 求解最优储备量
    print("\n使用反高斯分布求解...")
    try:
        optimal_invgauss, profit_invgauss, results_invgauss = model_invgauss.solve(
            initial_guess=[5, 5],
            multi_start=True,
            n_starts=10
        )
        enterprise_profit = calculate_enterprise_profit(optimal_invgauss[0], optimal_invgauss[1], optimal_invgauss[2], params, invgauss_dist)
        print(f"不考虑捐赠最优政府储备量(Q*): {optimal_invgauss[0]:.4f}")
        print(f"不考虑捐赠最优企业储备量(q*): {optimal_invgauss[1]:.4f}")
        print(f"不考虑捐赠企业捐赠量(p*): {optimal_invgauss[2]:.4f}")
        print(f"不考虑捐赠政府最大利润: {profit_invgauss:.4f}")
        print(f"不考虑捐赠企业最大利润:{enterprise_profit:.4f}")
    except Exception as e:
        print(f"不考虑捐赠反高斯分布模型求解失败: {e}")

    uniform_dist = UniformDistribution(0, T)

    model_uniform = EmergencyModel(alpha, v, p1, c1, p2, s, m, e, lam, uniform_dist)
    # 求解最优储备量
    print("使用均匀分布求解...")
    try:
        optimal_uniform, profit_uniform, results_uniform = model_uniform.solve_with_no_q(
            initial_guess=[5, 5],  # 给一个更合理的初始猜测
            multi_start=True,
            n_starts=50
        )
        enterprise_profit = calculate_enterprise_profit(optimal_uniform[0], optimal_uniform[1], optimal_uniform[2], params, uniform_dist)

        print(f"不考虑企业代储最优政府储备量(Q*): {optimal_uniform[0]:.4f}")
        print(f"不考虑企业代储最优企业储备量(q*): {optimal_uniform[1]:.4f}")
        print(f"不考虑企业代储企业捐赠量(p*): {optimal_uniform[2]:.4f}")
        print(f"不考虑企业代储政府最大利润: {profit_uniform:.4f}")
        print(f"不考虑企业代储企业最大利润:{enterprise_profit:.4f}")


    except Exception as e:
        print(f"不考虑企业代储均匀分布模型求解失败: {e}")


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
        optimal_invgauss, profit_invgauss, results_invgauss = model_invgauss.solve_with_no_q(
            initial_guess=[5, 5],
            multi_start=True,
            n_starts=10
        )
        enterprise_profit = calculate_enterprise_profit(optimal_invgauss[0], optimal_invgauss[1], optimal_invgauss[2], params, invgauss_dist)
        print(f"不考虑企业代储最优政府储备量(Q*): {optimal_invgauss[0]:.4f}")
        print(f"不考虑企业代储最优企业储备量(q*): {optimal_invgauss[1]:.4f}")
        print(f"不考虑企业代储企业捐赠量(p*): {optimal_invgauss[2]:.4f}")
        print(f"不考虑企业代储政府最大利润: {profit_invgauss:.4f}")
        print(f"不考虑企业代储企业最大利润:{enterprise_profit:.4f}")
    except Exception as e:
        print(f"不考虑企业代储反高斯分布模型求解失败: {e}")

    uniform_dist = UniformDistribution(0, T)

    model_uniform = EmergencyModel(alpha, v, p1, c1, p2, s, m, e, lam, uniform_dist)

    # 求解最优储备量
    print("使用均匀分布求解...")
    try:
        optimal_uniform, profit_uniform = model_uniform.solve_signle(
        )
        enterprise_profit = calculate_enterprise_profit(optimal_uniform[0], optimal_uniform[1], optimal_uniform[2], params, uniform_dist)

        print(f"不考虑企业代储和企业生产最优政府储备量(Q*): {optimal_uniform[0]:.4f}")
        print(f"不考虑企业代储和企业生产最优企业储备量(q*): {optimal_uniform[1]:.4f}")
        print(f"不考虑企业代储和企业生产企业捐赠量(p*): {optimal_uniform[2]:.4f}")
        print(f"不考虑企业代储和企业生产政府最大利润: {profit_uniform:.4f}")
        print(f"不考虑企业代储和企业生产企业最大利润:{enterprise_profit:.4f}")


    except Exception as e:
        print(f"不考虑企业代储和企业生产均匀分布模型求解失败: {e}")


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
        optimal_invgauss, profit_invgauss = model_invgauss.solve_signle(
    
        )
        enterprise_profit = calculate_enterprise_profit(optimal_invgauss[0], optimal_invgauss[1], optimal_invgauss[2], params, invgauss_dist)
        print(f"不考虑企业代储和企业生产最优政府储备量(Q*): {optimal_invgauss[0]:.4f}")
        print(f"不考虑企业代储和企业生产最优企业储备量(q*): {optimal_invgauss[1]:.4f}")
        print(f"不考虑企业代储和企业生产企业捐赠量(p*): {optimal_invgauss[2]:.4f}")
        print(f"不考虑企业代储和企业生产政府最大利润: {profit_invgauss:.4f}")
        print(f"不考虑企业代储和企业生产企业最大利润:{enterprise_profit:.4f}")
    except Exception as e:
        print(f"不考虑企业代储和企业生产反高斯分布模型求解失败: {e}")

if __name__ == "__main__":
    main()
