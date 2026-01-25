"""
Module 01 - MDP核心概念

马尔可夫决策过程 (Markov Decision Process, MDP) 是强化学习的数学基础。

核心概念：
    1. MDP五元组: (S, A, P, R, γ)
    2. 策略 π(a|s): 状态到动作的映射
    3. 轨迹: s0 -> a0 -> r0 -> s1 -> a1 -> r1 -> ...
    4. 回报: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

为什么需要MDP？
    - 提供了一个统一的框架来描述决策问题
    - 允许我们用数学方法分析和求解
    - 所有RL算法都在这个框架下工作

与大模型/具身智能的关联：
    - RLHF: 状态=对话历史, 动作=生成的token, 奖励=人类偏好分数
    - 机器人: 状态=传感器输入, 动作=电机指令, 奖励=任务完成度

学习目标：
    1. 理解MDP的五个核心组件
    2. 区分确定性策略和随机策略
    3. 理解折扣回报的计算
"""

import numpy as np
from typing import Tuple, Dict, List, Callable, Optional


# ==================== MDP五元组 ====================

def explain_mdp_components():
    """
    MDP五元组详解

    MDP = (S, A, P, R, γ)，每个字母代表一个核心组件
    """
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              MDP五元组 (Markov Decision Process)                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  S - 状态空间 (State Space)                                       ║
    ║      所有可能状态的集合                                            ║
    ║      例: 棋盘的所有可能布局，机器人的所有可能位置                     ║
    ║                                                                  ║
    ║  A - 动作空间 (Action Space)                                      ║
    ║      所有可能动作的集合                                            ║
    ║      例: {上,下,左,右}，{买入,卖出,持有}                            ║
    ║                                                                  ║
    ║  P - 转移概率 (Transition Probability)                            ║
    ║      P(s'|s,a) = 在状态s执行动作a后转移到s'的概率                   ║
    ║      确定性环境: P(s'|s,a) = 1 对于某个确定的s'                     ║
    ║      随机性环境: 可能转移到多个状态                                  ║
    ║                                                                  ║
    ║  R - 奖励函数 (Reward Function)                                   ║
    ║      R(s,a,s') 或 R(s,a) = 执行动作后获得的即时奖励                 ║
    ║      这是唯一的"目标信号"，指导Agent学习                            ║
    ║                                                                  ║
    ║  γ - 折扣因子 (Discount Factor)                                   ║
    ║      γ ∈ [0, 1]，控制未来奖励的重要性                              ║
    ║      γ=0: 只看即时奖励（短视）                                     ║
    ║      γ=1: 所有奖励同等重要（可能不收敛）                            ║
    ║      常用: γ=0.99                                                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 策略 ====================

class Policy:
    """
    策略 π(a|s)

    策略定义了Agent在每个状态下如何选择动作。

    两种类型：
        1. 确定性策略 (Deterministic): π(s) = a
           给定状态，总是返回同一个动作

        2. 随机策略 (Stochastic): π(a|s) = P(a|s)
           给定状态，返回动作的概率分布
    """

    def __init__(self, n_states: int, n_actions: int, policy_type: str = 'uniform'):
        """
        初始化策略

        Args:
            n_states: 状态数量
            n_actions: 动作数量
            policy_type: 'uniform'(均匀随机), 'greedy'(贪心), 'custom'
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # 策略表: policy[s, a] = P(a|s)
        if policy_type == 'uniform':
            # 均匀随机策略：每个动作等概率
            self.policy_table = np.ones((n_states, n_actions)) / n_actions
        else:
            self.policy_table = np.zeros((n_states, n_actions))

    def get_action_probs(self, state: int) -> np.ndarray:
        """获取状态s下的动作概率分布"""
        return self.policy_table[state]

    def sample_action(self, state: int) -> int:
        """根据策略采样动作"""
        probs = self.get_action_probs(state)
        return np.random.choice(self.n_actions, p=probs)

    def get_deterministic_action(self, state: int) -> int:
        """获取确定性动作（概率最大的）"""
        return np.argmax(self.policy_table[state])

    def set_action(self, state: int, action: int, prob: float = 1.0):
        """设置某状态下的动作概率"""
        if prob == 1.0:
            # 确定性：清零其他动作
            self.policy_table[state] = 0
        self.policy_table[state, action] = prob

    def make_epsilon_greedy(self, q_values: np.ndarray, epsilon: float = 0.1):
        """
        基于Q值创建ε-greedy策略

        ε-greedy是探索与利用的平衡：
            - 以概率 1-ε 选择最优动作（利用）
            - 以概率 ε 随机选择（探索）
        """
        for s in range(self.n_states):
            best_action = np.argmax(q_values[s])
            for a in range(self.n_actions):
                if a == best_action:
                    self.policy_table[s, a] = 1 - epsilon + epsilon / self.n_actions
                else:
                    self.policy_table[s, a] = epsilon / self.n_actions


# ==================== 轨迹和回报 ====================

def collect_trajectory(
    env,
    policy: Policy,
    max_steps: int = 100
) -> Tuple[List, List, List, float]:
    """
    收集一条轨迹

    轨迹是Agent与环境交互的序列：
        s0 -> a0 -> r0 -> s1 -> a1 -> r1 -> ... -> sT

    Args:
        env: 环境
        policy: 策略
        max_steps: 最大步数

    Returns:
        states: 状态序列
        actions: 动作序列
        rewards: 奖励序列
        total_reward: 总奖励（未折扣）
    """
    states = []
    actions = []
    rewards = []

    state = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        states.append(state)

        # 根据策略选择动作
        action = policy.sample_action(state)
        actions.append(action)

        # 与环境交互
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        total_reward += reward

        state = next_state
        if done:
            break

    return states, actions, rewards, total_reward


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    计算折扣回报

    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        = r_t + γ * G_{t+1}

    折扣的意义：
        1. 数学上保证收敛（无限序列的和有界）
        2. 实际上体现"近期奖励更重要"
        3. γ=0.99 意味着100步后的奖励权重约为0.37

    Args:
        rewards: 奖励序列
        gamma: 折扣因子

    Returns:
        returns: 每个时刻的折扣回报
    """
    returns = []
    G = 0

    # 从后往前计算（更高效）
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return returns


def demonstrate_discount():
    """演示折扣因子的影响"""
    print("\n折扣因子γ的影响：")
    print("-" * 50)

    rewards = [1] * 100  # 每步奖励为1的序列

    for gamma in [0.0, 0.5, 0.9, 0.99, 1.0]:
        returns = compute_returns(rewards, gamma)
        print(f"  γ={gamma:.2f}: G_0 = {returns[0]:.2f}")

    print("\n解释：")
    print("  - γ=0.00: 只看即时奖励，G_0=1")
    print("  - γ=0.50: 未来奖励权重快速衰减")
    print("  - γ=0.99: 常用值，平衡近期和远期")
    print("  - γ=1.00: 所有奖励同等重要（可能不收敛）")


# ==================== 马尔可夫性质 ====================

def explain_markov_property():
    """解释马尔可夫性质"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     马尔可夫性质 (Markov Property)                 ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  "未来只依赖于当前状态，与历史无关"                                   ║
    ║                                                                  ║
    ║  数学表达：                                                        ║
    ║      P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)  ║
    ║                                                                  ║
    ║  直观理解：                                                        ║
    ║      当前状态包含了做决策所需的所有信息                               ║
    ║                                                                  ║
    ║  例子：                                                           ║
    ║      ✓ 国际象棋：棋盘当前布局 = 完整信息                            ║
    ║      ✗ 部分可观测：只能看到部分棋盘 → 需要记忆历史                   ║
    ║                                                                  ║
    ║  为什么重要？                                                      ║
    ║      1. 简化问题：不需要存储完整历史                                 ║
    ║      2. 使Bellman方程成立                                          ║
    ║      3. 大多数RL算法假设马尔可夫性                                   ║
    ║                                                                  ║
    ║  不满足时怎么办？                                                   ║
    ║      - 使用循环神经网络（RNN/LSTM）记忆历史                          ║
    ║      - 扩展状态空间（如包含最近N帧）                                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 实际应用映射 ====================

def show_real_world_examples():
    """展示MDP在实际问题中的应用"""
    examples = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    MDP在实际问题中的应用                           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. 大语言模型对齐 (RLHF)                                          ║
    ║     ┌─────────┬────────────────────────────────────┐            ║
    ║     │ 状态 S  │ 对话历史 + 当前提示                    │            ║
    ║     │ 动作 A  │ 生成下一个token                       │            ║
    ║     │ 奖励 R  │ 人类偏好评分 (reward model)           │            ║
    ║     │ 转移 P  │ 确定性（token确定下一状态）             │            ║
    ║     └─────────┴────────────────────────────────────┘            ║
    ║                                                                  ║
    ║  2. 机器人控制 (具身智能)                                          ║
    ║     ┌─────────┬────────────────────────────────────┐            ║
    ║     │ 状态 S  │ 关节角度、速度、传感器读数              │            ║
    ║     │ 动作 A  │ 电机力矩或目标位置                     │            ║
    ║     │ 奖励 R  │ 任务完成度 - 能耗 - 安全惩罚           │            ║
    ║     │ 转移 P  │ 物理动力学（可能随机）                  │            ║
    ║     └─────────┴────────────────────────────────────┘            ║
    ║                                                                  ║
    ║  3. 游戏AI (如AlphaGo)                                           ║
    ║     ┌─────────┬────────────────────────────────────┐            ║
    ║     │ 状态 S  │ 棋盘布局                              │            ║
    ║     │ 动作 A  │ 落子位置                              │            ║
    ║     │ 奖励 R  │ 胜=+1, 负=-1, 未结束=0                │            ║
    ║     │ 转移 P  │ 确定性（落子确定新状态）                │            ║
    ║     └─────────┴────────────────────────────────────┘            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(examples)


# ==================== 演示 ====================

if __name__ == "__main__":
    import sys
    sys.path.append("../..")
    from envs.gridworld import GridWorld

    print("=" * 70)
    print("Module 01: MDP核心概念")
    print("=" * 70)

    # 1. MDP五元组
    print("\n" + "=" * 70)
    print("1. MDP五元组")
    explain_mdp_components()

    # 2. 马尔可夫性质
    print("\n" + "=" * 70)
    print("2. 马尔可夫性质")
    explain_markov_property()

    # 3. 实际应用
    print("\n" + "=" * 70)
    print("3. 实际应用示例")
    show_real_world_examples()

    # 4. 策略示例
    print("\n" + "=" * 70)
    print("4. 策略示例")
    print("-" * 70)

    # 创建环境
    env = GridWorld(size=4)

    # 均匀随机策略
    uniform_policy = Policy(env.n_states, env.n_actions, 'uniform')
    print("\n均匀随机策略 (每个动作概率相等):")
    print(f"  状态0的动作概率: {uniform_policy.get_action_probs(0)}")

    # 收集轨迹
    print("\n使用随机策略收集轨迹:")
    states, actions, rewards, total = collect_trajectory(env, uniform_policy, max_steps=20)
    print(f"  状态序列: {states[:10]}...")
    print(f"  动作序列: {[env.ACTION_NAMES[a] for a in actions[:10]]}...")
    print(f"  总奖励: {total}")

    # 5. 折扣回报
    print("\n" + "=" * 70)
    print("5. 折扣回报")
    demonstrate_discount()

    # 6. 计算轨迹的折扣回报
    print("\n" + "=" * 70)
    print("6. 轨迹的折扣回报")
    print("-" * 70)

    returns = compute_returns(rewards, gamma=0.99)
    print(f"\n奖励序列: {rewards[:10]}...")
    print(f"折扣回报: {[f'{r:.2f}' for r in returns[:10]]}...")
    print(f"\nG_0 (初始状态的回报): {returns[0]:.2f}")

    print("\n" + "=" * 70)
    print("MDP核心概念演示完成！")
    print("下一步: 学习 bellman_equations.py 了解价值函数和Bellman方程")
    print("=" * 70)
