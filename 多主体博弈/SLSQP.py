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


# 主函数：应用模型求解案例
def main():
    # 模型参数
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

    uniform_dist = UniformDistribution(0, T)


    # 广义帕累托参数
    gp_dist = GPDistribution(1.26089, 1059.03, 0)

    # 均匀分布
    model_uniform = EmergencyModel(alpha, v, p1, c1, p2, s, m, uniform_dist)

    # 求解最优储备量
    print("使用均匀分布求解...")
    try:
        optimal_uniform, profit_uniform, results_uniform = model_uniform.solve(
            initial_guess=[2000, 1000],  # 给一个更合理的初始猜测
            multi_start=True,
            n_starts=50
        )

        print(f"最优政府储备量(Q*): {optimal_uniform[0]:.4f}")
        print(f"最优企业储备量(q*): {optimal_uniform[1]:.4f}")
        print(f"最大利润: {profit_uniform:.4f}")

        # # 可视化不同储备量下的利润(针对均匀分布)
        # Q_range = np.linspace(20, 50, 20)
        # q_range = np.linspace(0, 30, 20)
        # Q_grid, q_grid = np.meshgrid(Q_range, q_range)
        # profit_grid = np.zeros_like(Q_grid)
        #
        # for i in range(Q_grid.shape[0]):
        #     for j in range(Q_grid.shape[1]):
        #         profit_grid[i, j] = -model_uniform.government_profit([Q_grid[i, j], q_grid[i, j]])
        #
        # plt.figure(figsize=(10, 8))
        # contour = plt.contourf(Q_grid, q_grid, profit_grid, 50, cmap='viridis')
        # plt.colorbar(contour, label='Profit')
        # plt.plot(optimal_uniform[0], optimal_uniform[1], 'ro', markersize=10, label='Optimal Point')
        # plt.xlabel('Government Reserve (Q)')
        # plt.ylabel('Enterprise Reserve (q)')
        # plt.title('Profit Landscape (Uniform Distribution)')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('profit_landscape_uniform.png')
        # plt.show()

    except Exception as e:
        print(f"均匀分布模型求解失败: {e}")

    # 创建模型实例 - 使用广义帕累托分布
    try:
        model_gp = EmergencyModel(alpha, v, p1, c1, p2, s, m, gp_dist)

        # 求解最优储备量
        print("\n使用广义帕累托分布求解...")
        optimal_gp, profit_gp, results_gp = model_gp.solve(
            initial_guess=[100, 100],
            multi_start=True,
            n_starts=50
        )

        print(f"最优政府储备量(Q*): {optimal_gp[0]:.4f}")
        print(f"最优企业储备量(q*): {optimal_gp[1]:.4f}")
        print(f"最大利润: {profit_gp:.4f}")

        # # 可视化不同储备量下的利润(针对广义帕累托分布)
        # Q_range = np.linspace(20, 50, 20)
        # q_range = np.linspace(0, 30, 20)
        # Q_grid, q_grid = np.meshgrid(Q_range, q_range)
        # profit_grid = np.zeros_like(Q_grid)
        #
        # for i in range(Q_grid.shape[0]):
        #     for j in range(Q_grid.shape[1]):
        #         profit_grid[i, j] = -model_gp.government_profit([Q_grid[i, j], q_grid[i, j]])
        #
        # plt.figure(figsize=(10, 8))
        # contour = plt.contourf(Q_grid, q_grid, profit_grid, 50, cmap='viridis')
        # plt.colorbar(contour, label='Profit')
        # plt.plot(optimal_gp[0], optimal_gp[1], 'ro', markersize=10, label='Optimal Point')
        # plt.xlabel('Government Reserve (Q)')
        # plt.ylabel('Enterprise Reserve (q)')
        # plt.title('Profit Landscape (Generalized Pareto Distribution)')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('profit_landscape_gp.png')
        # plt.show()

    except Exception as e:
        print(f"广义帕累托分布模型求解失败: {e}")


if __name__ == "__main__":
    main()
