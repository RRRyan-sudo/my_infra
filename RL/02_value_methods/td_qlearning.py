"""
Module 02 - TD学习与Q-learning

时序差分学习 (Temporal Difference Learning) 是RL的核心算法之一。

核心概念：
    1. TD学习：结合蒙特卡洛和动态规划的优点
    2. TD(0)：最简单的TD方法
    3. Q-learning：off-policy TD控制
    4. SARSA：on-policy TD控制

关键公式：

    TD(0)更新（估计V）:
    V(s) ← V(s) + α * [r + γ*V(s') - V(s)]
                       ↑_________↑
                        TD目标     TD误差

    Q-learning更新（off-policy）:
    Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]
                                 ↑
                            选最优动作

    SARSA更新（on-policy）:
    Q(s,a) ← Q(s,a) + α * [r + γ*Q(s',a') - Q(s,a)]
                               ↑
                          实际执行的动作

为什么重要？
    - "自举"(bootstrapping)思想在Actor-Critic中广泛使用
    - Q-learning是DQN的基础
    - TD误差是优势估计的核心

与大模型/具身智能的关联：
    - Actor-Critic的Critic使用TD学习更新
    - SAC的Q网络用TD学习优化
    - PPO的GAE本质是TD误差的加权和

学习时间：1-2小时（了解即可）
"""

import numpy as np
from typing import Tuple, List, Optional
import sys
sys.path.append("../..")


# ==================== TD学习核心思想 ====================

def explain_td_learning():
    """解释TD学习的核心思想"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TD学习 vs 蒙特卡洛 vs 动态规划                   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  蒙特卡洛 (Monte Carlo)                                           ║
    ║  ════════════════════                                            ║
    ║  - 等到episode结束，用实际回报G_t更新                               ║
    ║  - V(s) ← V(s) + α * [G_t - V(s)]                                ║
    ║  - 优点：无偏估计                                                  ║
    ║  - 缺点：方差大，必须等episode结束                                  ║
    ║                                                                  ║
    ║  动态规划 (Dynamic Programming)                                   ║
    ║  ═══════════════════════════════                                 ║
    ║  - 用完整的环境模型计算期望                                         ║
    ║  - V(s) = Σ P(s'|s,a) * [R + γ*V(s')]                            ║
    ║  - 优点：精确                                                     ║
    ║  - 缺点：需要完整环境模型                                          ║
    ║                                                                  ║
    ║  TD学习 (Temporal Difference)                                     ║
    ║  ═════════════════════════════                                   ║
    ║  - 每一步就更新，用估计值替代实际回报（自举）                          ║
    ║  - V(s) ← V(s) + α * [r + γ*V(s') - V(s)]                        ║
    ║                         ↑_____________↑                          ║
    ║                         TD目标（估计的G_t）                         ║
    ║  - 优点：不需要完整episode，不需要环境模型                           ║
    ║  - 缺点：有偏（因为用估计值）                                       ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  自举 (Bootstrapping) 的含义                                       ║
    ║  ═══════════════════════════                                     ║
    ║  "用一个估计值来更新另一个估计值"                                    ║
    ║                                                                  ║
    ║  TD目标: r + γ*V(s')                                              ║
    ║          ↑      ↑                                                ║
    ║        真实的   估计的                                             ║
    ║                                                                  ║
    ║  这是TD学习的核心思想，也是Actor-Critic的基础                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== TD(0) 实现 ====================

class TD0:
    """
    TD(0) 算法 - 最简单的TD方法

    用于估计给定策略的状态价值函数V(s)

    更新规则:
        V(s) ← V(s) + α * δ
        其中 δ = r + γ*V(s') - V(s)  (TD误差)
    """

    def __init__(
        self,
        n_states: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99
    ):
        """
        Args:
            n_states: 状态数量
            learning_rate: 学习率 α
            gamma: 折扣因子
        """
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = gamma

        # 初始化价值函数
        self.V = np.zeros(n_states)

    def update(self, state: int, reward: float, next_state: int, done: bool) -> float:
        """
        TD(0)更新

        Args:
            state: 当前状态
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止

        Returns:
            td_error: TD误差（用于监控学习）
        """
        # TD目标
        if done:
            td_target = reward  # 终止状态没有未来价值
        else:
            td_target = reward + self.gamma * self.V[next_state]

        # TD误差
        td_error = td_target - self.V[state]

        # 更新V
        self.V[state] += self.lr * td_error

        return td_error


# ==================== Q-learning ====================

class QLearning:
    """
    Q-learning - off-policy TD控制

    核心思想：
        - 学习最优Q函数 Q*
        - 更新时使用 max_a' Q(s',a')，不管实际执行了什么动作
        - 这就是"off-policy"：学习的策略和执行的策略可以不同

    更新规则:
        Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]

    行为策略：ε-greedy（保证探索）
    学习目标：最优策略（greedy）
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """
        Args:
            n_states: 状态数量
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # 初始化Q表
        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state: int) -> int:
        """
        ε-greedy动作选择

        以概率 1-ε 选择最优动作，以概率 ε 随机选择
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Q-learning更新

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止

        Returns:
            td_error: TD误差
        """
        # Q-learning TD目标：使用max（off-policy的关键）
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        # TD误差
        td_error = td_target - self.Q[state, action]

        # 更新Q
        self.Q[state, action] += self.lr * td_error

        return td_error

    def get_policy(self) -> np.ndarray:
        """返回当前贪心策略"""
        return np.argmax(self.Q, axis=1)


# ==================== SARSA ====================

class SARSA:
    """
    SARSA - on-policy TD控制

    名称来源：(S, A, R, S', A') - 更新需要的五元组

    核心思想：
        - 更新时使用实际执行的下一个动作 a'
        - 学习的策略和执行的策略相同
        - 这就是"on-policy"

    更新规则:
        Q(s,a) ← Q(s,a) + α * [r + γ*Q(s',a') - Q(s,a)]
                                    ↑
                              实际执行的a'

    Q-learning vs SARSA:
        - Q-learning: 乐观，假设下一步会选最优
        - SARSA: 保守，考虑实际会选什么（包括探索）
        - SARSA在有风险的环境中更安全
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state: int) -> int:
        """ε-greedy动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool) -> float:
        """
        SARSA更新

        注意：需要next_action（实际执行的下一个动作）

        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            next_state: 下一状态
            next_action: 下一动作（实际执行的）
            done: 是否终止

        Returns:
            td_error: TD误差
        """
        # SARSA TD目标：使用实际的next_action（on-policy的关键）
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

        return td_error

    def get_policy(self) -> np.ndarray:
        """返回当前贪心策略"""
        return np.argmax(self.Q, axis=1)


# ==================== 比较 ====================

def compare_qlearning_sarsa():
    """比较Q-learning和SARSA"""
    comparison = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    Q-learning vs SARSA                           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║                   Q-learning           SARSA                     ║
    ║                   ══════════           ═════                     ║
    ║  类型             off-policy           on-policy                 ║
    ║  TD目标           max_a' Q(s',a')      Q(s',a')                  ║
    ║  学习目标         最优策略             当前策略                    ║
    ║  探索影响         不影响学习           影响学习                    ║
    ║  收敛性           更快                 更稳定                     ║
    ║  安全性           可能过于乐观          更保守                     ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  什么时候用哪个？                                                  ║
    ║  ════════════════                                                ║
    ║  - Q-learning: 离线学习，使用历史数据                              ║
    ║  - SARSA: 在线学习，需要安全保证的场景                              ║
    ║                                                                  ║
    ║  现代深度RL：                                                     ║
    ║  - DQN: Q-learning + 神经网络                                     ║
    ║  - SAC: 类似SARSA的软Q更新                                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(comparison)


# ==================== 训练函数 ====================

def train_agent(
    env,
    agent,
    n_episodes: int = 500,
    max_steps: int = 100,
    verbose: bool = True
) -> List[float]:
    """
    训练Q-learning或SARSA agent

    Args:
        env: 环境
        agent: QLearning或SARSA实例
        n_episodes: 训练episode数
        max_steps: 每个episode最大步数
        verbose: 是否打印进度

    Returns:
        rewards: 每个episode的总奖励
    """
    episode_rewards = []
    is_sarsa = isinstance(agent, SARSA)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        if is_sarsa:
            action = agent.get_action(state)

        for step in range(max_steps):
            if not is_sarsa:
                action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if is_sarsa:
                next_action = agent.get_action(next_state) if not done else 0
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}")

    return episode_rewards


# ==================== 演示 ====================

if __name__ == "__main__":
    from envs.gridworld import GridWorld

    print("=" * 70)
    print("Module 02: TD学习与Q-learning")
    print("=" * 70)

    # 1. TD学习概念
    print("\n" + "=" * 70)
    print("1. TD学习核心思想")
    explain_td_learning()

    # 2. 创建环境
    print("\n" + "=" * 70)
    print("2. 在GridWorld上训练")
    print("-" * 70)

    env = GridWorld(size=4)
    print(f"环境: {env.size}x{env.size} GridWorld")

    # 3. Q-learning训练
    print("\n" + "=" * 70)
    print("3. Q-learning训练")
    print("-" * 70)

    q_agent = QLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1
    )

    q_rewards = train_agent(env, q_agent, n_episodes=500, verbose=True)

    # 4. SARSA训练
    print("\n" + "=" * 70)
    print("4. SARSA训练")
    print("-" * 70)

    sarsa_agent = SARSA(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1
    )

    sarsa_rewards = train_agent(env, sarsa_agent, n_episodes=500, verbose=True)

    # 5. 比较结果
    print("\n" + "=" * 70)
    print("5. 比较Q-learning和SARSA")
    compare_qlearning_sarsa()

    print("\n训练结果比较:")
    print(f"  Q-learning 最后100回合平均奖励: {np.mean(q_rewards[-100:]):.2f}")
    print(f"  SARSA 最后100回合平均奖励: {np.mean(sarsa_rewards[-100:]):.2f}")

    # 6. 显示学到的策略
    print("\n" + "=" * 70)
    print("6. 学到的策略")
    print("-" * 70)

    print("\nQ-learning学到的策略:")
    q_policy = q_agent.get_policy()
    for row in range(env.size):
        for col in range(env.size):
            state = row * env.size + col
            if state == env.goal_state:
                print("  G  ", end=" ")
            else:
                print(f"  {env.ACTION_NAMES[q_policy[state]]}  ", end=" ")
        print()

    # 7. TD误差的意义
    print("\n" + "=" * 70)
    print("7. TD误差的意义")
    print("-" * 70)

    td_explanation = """
    TD误差 δ = r + γ*V(s') - V(s) 的含义：

    - δ > 0: 实际比预期好 → 增加V(s)
    - δ < 0: 实际比预期差 → 减少V(s)
    - δ ≈ 0: 预测准确 → 收敛

    TD误差在现代RL中非常重要：
    - GAE中：A^GAE = Σ (γλ)^t * δ_t
    - PPO中：用GAE计算优势函数
    - Actor-Critic中：Critic的损失 = TD误差²
    """
    print(td_explanation)

    print("\n" + "=" * 70)
    print("TD学习演示完成！")
    print("下一步: 学习 03_policy_gradient/ 了解策略梯度方法")
    print("=" * 70)
