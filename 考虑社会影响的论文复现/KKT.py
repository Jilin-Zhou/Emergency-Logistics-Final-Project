import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# 参数设置
w = 12  # 政府单位物资采购价格
c = 9  # 企业单位物资生产成本
m = 70  # 单位缺货成本
U = 100000  # 应急物资最大需求量
v = 1  # 单位物资剩余价值
mu = 20000  # 需求量分布的均值


# 需求分布函数（指数分布）
def F(x):
    return 1 - np.exp(-x / mu)


def F_inverse(y):
    return -mu * np.log(1 - y)


# 1. 政府单独储备模式
def government_only_model():
    # 政府最佳储备量
    Q0_star = F_inverse((m - w) / (m - v))
    # 供应链最佳储备量
    Qs0_star = F_inverse((m - c) / (m - v))

    # 政府成本
    def government_cost(Q):
        term1 = w * Q
        term2 = m * (U - Q)
        term3 = m * mu * np.exp(-Q / mu)
        term4 = v * (Q - mu + mu * np.exp(-Q / mu))
        return term1 + term2 - term3 - term4

    # 企业利润
    def enterprise_profit(Q):
        return (w - c) * Q

    # 供应链总成本
    def supply_chain_cost(Q):
        term1 = c * Q
        term2 = m * (U - Q)
        term3 = m * mu * np.exp(-Q / mu)
        term4 = v * (Q - mu + mu * np.exp(-Q / mu))
        return term1 + term2 - term3 - term4

    gov_cost = government_cost(Q0_star)
    ent_profit = enterprise_profit(Q0_star)
    sc_cost = supply_chain_cost(Q0_star)

    return {
        'Q0_star': Q0_star,
        'Qs0_star': Qs0_star,
        'government_cost': gov_cost,
        'enterprise_profit': ent_profit,
        'supply_chain_cost': sc_cost
    }


# 2. 政企联合储备模式
def joint_reserve_model(o):
    # 供应链协调的期权执行价格
    e = m - ((m - v) / (c - v)) * o

    # 政府最佳储备量
    Q1_star = F_inverse((o + e - w) / (e - v))

    # 企业最佳期权储备量
    q1_star = F_inverse((o + e - c) / (e - v)) - Q1_star

    # 供应链最佳储备量
    Qs1_star = Q1_star + q1_star

    # 政府成本
    def government_cost(Q1, q1):
        term1 = w * Q1 + o * q1
        term2 = e * q1
        term3 = m * (U - Q1 - q1)
        term4 = m * mu * np.exp(-(Q1 + q1) / mu)
        term5 = v * (Q1 - mu + mu * np.exp(-Q1 / mu))
        return term1 + term2 + term3 - term4 - term5

    # 企业利润
    def enterprise_profit(Q1, q1):
        term1 = (w - c) * Q1
        term2 = (o + e - c) * q1
        term3 = (e - v) * mu * (np.exp(-Q1 / mu) - np.exp(-(Q1 + q1) / mu))
        return term1 + term2 - term3

    # 供应链总成本
    def supply_chain_cost(Q1, q1):
        term1 = c * (Q1 + q1)
        term2 = m * (U - Q1 - q1)
        term3 = m * mu * np.exp(-(Q1 + q1) / mu)
        term4 = v * ((Q1 + q1) - mu + mu * np.exp(-(Q1 + q1) / mu))
        return term1 + term2 - term3 - term4

    gov_cost = government_cost(Q1_star, q1_star)
    ent_profit = enterprise_profit(Q1_star, q1_star)
    sc_cost = supply_chain_cost(Q1_star, q1_star)

    return {
        'e': e,
        'Q1_star': Q1_star,
        'q1_star': q1_star,
        'Qs1_star': Qs1_star,
        'government_cost': gov_cost,
        'enterprise_profit': ent_profit,
        'supply_chain_cost': sc_cost
    }


# 3. 考虑企业社会责任的政企联合储备模式
def csr_joint_reserve_model(o, j, lam, p):
    # 供应链协调的期权执行价格
    e = m - ((m - v) / (c - v)) * o

    # 政府最佳储备量
    Q2_star = F_inverse((o + e - w) / (e - v)) - j

    # 企业最佳期权储备量
    q2_star = F_inverse((o + e - c) / (e - v)) - Q2_star - j

    # 供应链最佳储备量
    Qs2_star = Q2_star + q2_star + j

    # 企业利润变化
    def enterprise_profit_change(j):
        return lam * (p - c) * np.sqrt(j * w) - w * j

    # 政府成本变化
    def government_cost_change(j):
        return w * j

    # 供应链收益变化
    def supply_chain_benefit_change(j):
        return lam * (p - c) * np.sqrt(j * w)

    ent_profit_change = enterprise_profit_change(j)
    gov_cost_change = government_cost_change(j)
    sc_benefit_change = supply_chain_benefit_change(j)

    return {
        'e': e,
        'Q2_star': Q2_star,
        'q2_star': q2_star,
        'j': j,
        'Qs2_star': Qs2_star,
        'enterprise_profit_change': ent_profit_change,
        'government_cost_change': gov_cost_change,
        'supply_chain_benefit_change': sc_benefit_change
    }


# 计算企业最佳捐赠量
def optimal_donation(lam, p, c, w):
    return (lam ** 2 * (p - c) ** 2) / (4 * w)


# 主函数
def main():
    # 1. 政府单独储备模式
    gov_only_results = government_only_model()
    print("政府单独储备模式结果:")
    print(f"政府最佳储备量: {gov_only_results['Q0_star']:.1f}")
    print(f"供应链最佳储备量: {gov_only_results['Qs0_star']:.1f}")
    print(f"政府成本: {gov_only_results['government_cost']:.1f}")
    print(f"企业利润: {gov_only_results['enterprise_profit']:.1f}")
    print(f"供应链总成本: {gov_only_results['supply_chain_cost']:.1f}")
    print()

    # 2. 政企联合储备模式
    option_fees = [1, 2, 3, 4, 5, 6, 7]
    joint_results = []

    print("政企联合储备模式结果:")
    print("期权费用\t执行价格\t政府储备量\t企业储备量\t总储备量\t供应链总成本")

    for o in option_fees:
        result = joint_reserve_model(o)
        joint_results.append(result)
        print(
            f"{o}\t{result['e']:.3f}\t{result['Q1_star']:.1f}\t{result['q1_star']:.1f}\t{result['Qs1_star']:.1f}\t{result['supply_chain_cost']:.1f}")

    print()

    # 3. 考虑企业社会责任的政企联合储备模式
    lam = 30  # 捐赠成本影响系数
    p = 25  # 物资市场售价
    o = 4  # 期权费用

    # 计算最佳捐赠量
    j_optimal = optimal_donation(lam, p, c, w)
    print(f"企业最佳捐赠量: {j_optimal:.1f}")

    print()
    print("不同捐赠量下的联合储备模型收益变化:")
    print("捐赠量\t政府储备量\t企业储备量\t企业增加利润\t政府减少成本\t供应链增加收益")

    donation_amounts = [2400, 3200, 4000, 4800, 5600, 6400, 7200, 8000]
    for j in donation_amounts:
        csr_result = csr_joint_reserve_model(o, j, lam, p)
        print(
            f"{j}\t{csr_result['Q2_star']:.1f}\t{csr_result['q2_star']:.1f}\t{csr_result['enterprise_profit_change']:.2f}\t{csr_result['government_cost_change']:.1f}\t{csr_result['supply_chain_benefit_change']:.2f}")


# 执行主函数
if __name__ == "__main__":
    main()
