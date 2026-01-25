"""
Module 01 - Bellman方程

Bellman方程是RL的数学核心，它描述了价值函数之间的递归关系。

核心概念：
    1. 状态价值函数 V(s): 从状态s开始的期望回报
    2. 动作价值函数 Q(s,a): 在状态s执行动作a后的期望回报
    3. Bellman期望方程: 给定策略下的价值递归
    4. Bellman最优方程: 最优策略下的价值递归

关键公式：

    V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')]

    Q^π(s,a) = R(s,a) + γ * Σ_{s'} P(s'|s,a) * Σ_{a'} π(a'|s') * Q^π(s',a')

    V*(s) = max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s')]

    Q*(s,a) = R(s,a) + γ * Σ_{s'} P(s'|s,a) * max_{a'} Q*(s',a')

为什么重要？
    - Bellman方程是几乎所有RL算法的理论基础
    - 动态规划直接求解Bellman方程
    - TD学习用采样估计Bellman方程
    - 策略梯度优化由Q值指导

与大模型/具身智能的关联：
    - PPO的Critic学习V(s)或Q(s,a)
    - SAC同时学习Q和策略
    - RLHF中的奖励模型本质是在估计Q值
"""

import numpy as np
from typing import Tuple, Dict, Optional
import sys
sys.path.append("../..")


# ==================== 价值函数 ====================

def explain_value_functions():
    """解释价值函数"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                         价值函数详解                              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  状态价值函数 V^π(s)                                              ║
    ║  ═══════════════════                                             ║
    ║  "从状态s开始，遵循策略π，期望获得多少回报？"                         ║
    ║                                                                  ║
    ║  V^π(s) = E_π[G_t | S_t = s]                                     ║
    ║         = E_π[r_t + γ*r_{t+1} + γ²*r_{t+2} + ... | S_t = s]      ║
    ║                                                                  ║
    ║  直觉：V(s)高的状态是"好位置"，我们希望到达这些状态                   ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  动作价值函数 Q^π(s,a)                                            ║
    ║  ══════════════════════                                          ║
    ║  "在状态s执行动作a，然后遵循策略π，期望获得多少回报？"                 ║
    ║                                                                  ║
    ║  Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]                          ║
    ║                                                                  ║
    ║  直觉：Q(s,a)高的动作是"好选择"，我们应该执行这些动作                 ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  V和Q的关系                                                       ║
    ║  ══════════                                                      ║
    ║  V^π(s) = Σ_a π(a|s) * Q^π(s,a)                                  ║
    ║         = 所有动作Q值的加权平均（权重是策略概率）                     ║
    ║                                                                  ║
    ║  Q^π(s,a) = R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')              ║
    ║          = 即时奖励 + 折扣后的下一状态价值                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== Bellman方程 ====================

def explain_bellman_equations():
    """解释Bellman方程"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                       Bellman方程详解                             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Bellman期望方程 (给定策略π)                                       ║
    ║  ═════════════════════════════                                   ║
    ║                                                                  ║
    ║  V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')] ║
    ║           ↑              ↑           ↑                  ↑        ║
    ║        选动作的概率    即时奖励     转移概率         下一状态价值    ║
    ║                                                                  ║
    ║  直觉：当前状态的价值 = 期望即时奖励 + 折扣后的期望未来价值            ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Bellman最优方程 (最优策略π*)                                      ║
    ║  ═════════════════════════════                                   ║
    ║                                                                  ║
    ║  V*(s) = max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s')]          ║
    ║          ↑                                                       ║
    ║       选择最优动作                                                 ║
    ║                                                                  ║
    ║  Q*(s,a) = R(s,a) + γ * Σ_{s'} P(s'|s,a) * max_{a'} Q*(s',a')    ║
    ║                                               ↑                  ║
    ║                                          下一步选最优              ║
    ║                                                                  ║
    ║  直觉：最优价值 = 选择最好的动作能获得的价值                          ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  "期望"方程 vs "最优"方程                                          ║
    ║  ═════════════════════════                                       ║
    ║  - 期望方程：评估给定策略的好坏 → 策略评估                           ║
    ║  - 最优方程：找到最好的策略 → 策略优化                               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 策略评估 ====================

def policy_evaluation(
    env,
    policy: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000
) -> np.ndarray:
    """
    策略评估：计算给定策略的状态价值函数

    使用迭代法求解Bellman期望方程：
        V_{k+1}(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V_k(s')]

    Args:
        env: 环境（需要有P转移矩阵）
        policy: 策略 π[s, a] = P(a|s)
        gamma: 折扣因子
        theta: 收敛阈值
        max_iterations: 最大迭代次数

    Returns:
        V: 状态价值函数
    """
    n_states = env.n_states
    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        delta = 0  # 跟踪最大变化量

        for s in range(n_states):
            v = V[s]  # 保存旧值

            # 计算新值：Bellman期望方程
            new_v = 0
            for a in range(env.n_actions):
                # 获取转移信息
                for prob, next_state, reward, done in env.P[s][a]:
                    # V(s) = Σ_a π(a|s) * [R + γ * V(s')]
                    new_v += policy[s, a] * prob * (reward + gamma * V[next_state] * (1 - done))

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        # 检查收敛
        if delta < theta:
            print(f"策略评估在第 {iteration + 1} 次迭代后收敛")
            break

    return V


# ==================== 价值迭代 ====================

def value_iteration(
    env,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    价值迭代：直接求解Bellman最优方程

    V_{k+1}(s) = max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V_k(s')]

    这是动态规划方法，需要完整的环境模型（P和R）。

    Args:
        env: 环境
        gamma: 折扣因子
        theta: 收敛阈值
        max_iterations: 最大迭代次数

    Returns:
        V: 最优状态价值函数
        policy: 最优策略
    """
    n_states = env.n_states
    n_actions = env.n_actions
    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        delta = 0

        for s in range(n_states):
            v = V[s]

            # 计算每个动作的Q值
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_state] * (1 - done))

            # 取最大Q值作为新的V值
            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            print(f"价值迭代在第 {iteration + 1} 次迭代后收敛")
            break

    # 从V提取最优策略
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state] * (1 - done))

        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0

    return V, policy


# ==================== Q值计算 ====================

def compute_q_from_v(env, V: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    从V函数计算Q函数

    Q(s,a) = R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')

    Args:
        env: 环境
        V: 状态价值函数
        gamma: 折扣因子

    Returns:
        Q: 动作价值函数 Q[s, a]
    """
    Q = np.zeros((env.n_states, env.n_actions))

    for s in range(env.n_states):
        for a in range(env.n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                Q[s, a] += prob * (reward + gamma * V[next_state] * (1 - done))

    return Q


# ==================== 优势函数 ====================

def explain_advantage_function():
    """解释优势函数"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                       优势函数 (Advantage Function)               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  定义：A^π(s,a) = Q^π(s,a) - V^π(s)                               ║
    ║                                                                  ║
    ║  含义："在状态s，动作a比平均水平好多少？"                             ║
    ║                                                                  ║
    ║  - A(s,a) > 0: 动作a比平均更好，应该增加其概率                       ║
    ║  - A(s,a) < 0: 动作a比平均更差，应该减少其概率                       ║
    ║  - A(s,a) = 0: 动作a与平均持平                                     ║
    ║                                                                  ║
    ║  为什么重要？                                                      ║
    ║  ═══════════                                                     ║
    ║  1. 策略梯度中用A代替Q可以减少方差                                   ║
    ║     ∇J = E[∇log π(a|s) * A(s,a)]  比用Q方差更小                   ║
    ║                                                                  ║
    ║  2. PPO和A2C都使用优势函数                                         ║
    ║                                                                  ║
    ║  3. GAE (Generalized Advantage Estimation) 是优势函数的改进估计     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


def compute_advantage(Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    计算优势函数

    A(s,a) = Q(s,a) - V(s)

    Args:
        Q: Q函数 Q[s, a]
        V: V函数 V[s]

    Returns:
        A: 优势函数 A[s, a]
    """
    n_states, n_actions = Q.shape
    A = np.zeros((n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            A[s, a] = Q[s, a] - V[s]

    return A


# ==================== 可视化 ====================

def visualize_values(env, V: np.ndarray, Q: Optional[np.ndarray] = None):
    """可视化价值函数"""
    print("\n状态价值函数 V(s):")
    print("-" * 50)

    # 显示V值网格
    for row in range(env.size):
        for col in range(env.size):
            state = row * env.size + col
            print(f"{V[state]:7.2f}", end=" ")
        print()

    if Q is not None:
        print("\n动作价值函数 Q(s,a) 示例 (状态0):")
        print("-" * 50)
        for a in range(env.n_actions):
            print(f"  {env.ACTION_NAMES[a]}: Q(0, {a}) = {Q[0, a]:.2f}")


# ==================== 演示 ====================

if __name__ == "__main__":
    from envs.gridworld import GridWorld

    print("=" * 70)
    print("Module 01: Bellman方程")
    print("=" * 70)

    # 1. 价值函数概念
    print("\n" + "=" * 70)
    print("1. 价值函数概念")
    explain_value_functions()

    # 2. Bellman方程
    print("\n" + "=" * 70)
    print("2. Bellman方程")
    explain_bellman_equations()

    # 3. 创建环境并演示
    print("\n" + "=" * 70)
    print("3. GridWorld上的价值迭代")
    print("-" * 70)

    env = GridWorld(size=4)
    print(f"\n环境: {env.size}x{env.size} GridWorld")
    print(f"目标: 从(0,0)到达({env.size-1},{env.size-1})")

    # 价值迭代
    print("\n执行价值迭代...")
    V_optimal, policy_optimal = value_iteration(env, gamma=0.99)

    # 显示结果
    print("\n最优价值函数 V*(s):")
    visualize_values(env, V_optimal)

    # 计算Q值
    Q_optimal = compute_q_from_v(env, V_optimal, gamma=0.99)
    print("\n最优Q值 (状态0):")
    for a in range(env.n_actions):
        print(f"  Q*(0, {env.ACTION_NAMES[a]}) = {Q_optimal[0, a]:.2f}")

    # 4. 优势函数
    print("\n" + "=" * 70)
    print("4. 优势函数")
    explain_advantage_function()

    # 计算优势函数
    A = compute_advantage(Q_optimal, V_optimal)
    print("\n优势函数 A(s,a) (状态0):")
    for a in range(env.n_actions):
        print(f"  A(0, {env.ACTION_NAMES[a]}) = {A[0, a]:.2f}", end="")
        if A[0, a] > 0:
            print(" ← 比平均好")
        elif A[0, a] < 0:
            print(" ← 比平均差")
        else:
            print(" ← 平均水平")

    # 5. 策略评估
    print("\n" + "=" * 70)
    print("5. 策略评估：比较随机策略和最优策略")
    print("-" * 70)

    # 随机策略
    random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    print("\n评估随机策略...")
    V_random = policy_evaluation(env, random_policy, gamma=0.99)

    print(f"\n随机策略的V(起点): {V_random[0]:.2f}")
    print(f"最优策略的V(起点): {V_optimal[0]:.2f}")
    print(f"改进: {V_optimal[0] - V_random[0]:.2f}")

    # 6. 展示最优策略
    print("\n" + "=" * 70)
    print("6. 最优策略")
    print("-" * 70)

    print("\n最优策略（每个状态的最优动作）:")
    for row in range(env.size):
        for col in range(env.size):
            state = row * env.size + col
            if state == env.goal_state:
                print("  G  ", end=" ")
            else:
                best_action = np.argmax(policy_optimal[state])
                print(f"  {env.ACTION_NAMES[best_action]}  ", end=" ")
        print()

    print("\n图例: ↑=上, ↓=下, ←=左, →=右, G=目标")

    print("\n" + "=" * 70)
    print("Bellman方程演示完成！")
    print("下一步: 学习 02_value_methods/ 了解TD学习和Q-learning")
    print("=" * 70)
