"""
Module 03 - REINFORCE算法

REINFORCE是最简单的策略梯度算法，也叫"蒙特卡洛策略梯度"。

核心概念：
    - 直接优化策略参数θ，使期望回报最大化
    - 使用梯度上升：θ ← θ + α * ∇J(θ)

关键公式：

    目标函数：
    J(θ) = E_π[G_t] = E_π[Σ γ^k * r_{t+k}]

    策略梯度定理：
    ∇J(θ) = E_π[∇log π_θ(a|s) * G_t]
            ↑_______________↑   ↑
           梯度方向告诉我们    回报告诉我们
           如何调整策略       这个动作好不好

    直觉理解：
    - 如果G_t高（好结果），增加π(a|s)（更可能选这个动作）
    - 如果G_t低（坏结果），减少π(a|s)（更少选这个动作）

为什么重要？
    - PPO是REINFORCE的改进版
    - RLHF本质上是策略梯度 + 人类反馈

与大模型的关联：
    - 大模型生成每个token就是在执行策略 π(token|context)
    - RLHF优化这个策略使其符合人类偏好
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple
import sys
sys.path.append("../..")


# ==================== 策略梯度定理 ====================

def explain_policy_gradient():
    """解释策略梯度定理"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                      策略梯度定理                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  目标：找到参数θ使期望回报最大化                                     ║
    ║        max_θ J(θ) = E_π[G_t]                                     ║
    ║                                                                  ║
    ║  问题：如何计算 ∇J(θ)？                                            ║
    ║        期望对θ求导很难，因为π_θ既在概率中又在期望里                    ║
    ║                                                                  ║
    ║  策略梯度定理（巧妙的数学技巧）：                                     ║
    ║        ∇J(θ) = E_π[∇log π_θ(a|s) * Q^π(s,a)]                     ║
    ║                                                                  ║
    ║  推导（log求导技巧）：                                              ║
    ║        ∇π = π * ∇log(π)                                          ║
    ║        因为 ∇log(π) = ∇π / π                                     ║
    ║                                                                  ║
    ║  REINFORCE简化：                                                  ║
    ║        用蒙特卡洛采样的G_t代替Q(s,a)                                ║
    ║        ∇J(θ) ≈ (1/N) * Σ ∇log π_θ(a_i|s_i) * G_i                 ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  为什么叫"log概率"？                                               ║
    ║  ═══════════════════                                             ║
    ║  - 数值稳定：概率很小时，log不会下溢                                 ║
    ║  - 梯度简单：softmax的log导数是 1 - π（很干净）                      ║
    ║  - 信息论：log概率 = 信息量                                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== REINFORCE实现 ====================

class PolicyNetwork(nn.Module):
    """
    简单的策略网络

    输入: 状态 s
    输出: 动作概率分布 π(a|s)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 两层MLP
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 状态，形状 (batch_size, state_dim)

        Returns:
            动作概率，形状 (batch_size, action_dim)
        """
        # x → Linear → ReLU → Linear → Softmax
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class REINFORCE:
    """
    REINFORCE算法（蒙特卡洛策略梯度）

    核心步骤：
    1. 收集一个完整episode的轨迹
    2. 计算每一步的回报G_t
    3. 计算损失 L = -Σ log π(a|s) * G_t
    4. 反向传播更新策略

    注意：
    - 必须等episode结束才能更新（蒙特卡洛）
    - 方差很大（改进方向：加baseline → Actor-Critic）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.gamma = gamma
        self.action_dim = action_dim

        # 策略网络
        self.policy = PolicyNetwork(state_dim, action_dim)

        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 存储轨迹
        self.log_probs = []  # log π(a|s)
        self.rewards = []    # 奖励

    def select_action(self, state: np.ndarray) -> int:
        """
        根据当前策略选择动作

        Args:
            state: 当前状态

        Returns:
            选择的动作
        """
        # NumPy → PyTorch张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 获取动作概率
        probs = self.policy(state_tensor)

        # 创建分类分布并采样
        dist = Categorical(probs)
        action = dist.sample()

        # 保存log概率（用于后续更新）
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward: float):
        """存储奖励"""
        self.rewards.append(reward)

    def compute_returns(self) -> List[float]:
        """
        计算折扣回报

        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        """
        returns = []
        G = 0

        # 从后往前计算
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return returns

    def update(self) -> float:
        """
        策略梯度更新

        损失函数: L = -Σ log π(a|s) * G_t
        (负号是因为我们要最大化期望回报，但优化器是最小化)

        Returns:
            loss值
        """
        # 计算回报
        returns = self.compute_returns()

        # 标准化回报（减少方差的常用技巧）
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算策略梯度损失
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # -log π(a|s) * G
            policy_loss.append(-log_prob * G)

        # 总损失
        loss = torch.stack(policy_loss).sum()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空轨迹存储
        self.log_probs = []
        self.rewards = []

        return loss.item()


# ==================== 训练函数 ====================

def train_reinforce(
    env,
    agent: REINFORCE,
    n_episodes: int = 1000,
    max_steps: int = 500,
    print_interval: int = 100
) -> List[float]:
    """
    训练REINFORCE agent

    Args:
        env: 环境
        agent: REINFORCE实例
        n_episodes: 训练episode数
        max_steps: 每个episode最大步数
        print_interval: 打印间隔

    Returns:
        episode_rewards: 每个episode的总奖励
    """
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 与环境交互
            next_state, reward, done, _ = env.step(action)

            # 存储奖励
            agent.store_reward(reward)

            total_reward += reward
            state = next_state

            if done:
                break

        # Episode结束，更新策略
        loss = agent.update()
        episode_rewards.append(total_reward)

        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}, 损失 = {loss:.4f}")

    return episode_rewards


# ==================== 演示 ====================

if __name__ == "__main__":
    from envs.gridworld import GridWorld

    print("=" * 70)
    print("Module 03: REINFORCE算法")
    print("=" * 70)

    # 1. 策略梯度定理
    print("\n" + "=" * 70)
    print("1. 策略梯度定理")
    explain_policy_gradient()

    # 2. 创建环境和agent
    print("\n" + "=" * 70)
    print("2. 在GridWorld上训练REINFORCE")
    print("-" * 70)

    env = GridWorld(size=4)

    # REINFORCE需要状态是向量形式
    # GridWorld状态是整数，转换为one-hot
    class GridWorldWrapper:
        def __init__(self, env):
            self.env = env
            self.n_states = env.n_states
            self.n_actions = env.n_actions

        def reset(self):
            state = self.env.reset()
            return self._to_onehot(state)

        def step(self, action):
            next_state, reward, done, info = self.env.step(action)
            return self._to_onehot(next_state), reward, done, info

        def _to_onehot(self, state):
            onehot = np.zeros(self.n_states)
            onehot[state] = 1.0
            return onehot

    wrapped_env = GridWorldWrapper(env)

    # 创建agent
    agent = REINFORCE(
        state_dim=env.n_states,
        action_dim=env.n_actions,
        learning_rate=1e-2,
        gamma=0.99
    )

    # 训练
    print("\n开始训练...")
    rewards = train_reinforce(
        wrapped_env,
        agent,
        n_episodes=500,
        print_interval=100
    )

    # 3. 评估学到的策略
    print("\n" + "=" * 70)
    print("3. 评估学到的策略")
    print("-" * 70)

    # 显示每个状态的策略
    print("\n学到的策略 (每个状态的动作概率):")
    agent.policy.eval()

    for state in range(env.n_states):
        state_onehot = np.zeros(env.n_states)
        state_onehot[state] = 1.0

        with torch.no_grad():
            probs = agent.policy(torch.FloatTensor(state_onehot).unsqueeze(0))
            probs = probs.squeeze().numpy()

        if state == env.goal_state:
            continue

        row, col = state // env.size, state % env.size
        best_action = np.argmax(probs)
        print(f"  状态{state} ({row},{col}): 最优动作 = {env.ACTION_NAMES[best_action]}, "
              f"概率 = {probs[best_action]:.2f}")

    # 4. 讨论REINFORCE的问题
    print("\n" + "=" * 70)
    print("4. REINFORCE的问题")
    print("-" * 70)

    problems = """
    REINFORCE的主要问题：

    1. 高方差
       - 用蒙特卡洛采样估计梯度，方差很大
       - 同样的策略可能产生很不同的回报
       - 导致学习不稳定

    2. 必须等episode结束
       - 无法在线学习
       - 对于长episode效率低

    3. 样本效率低
       - 每个episode的数据只用一次
       - 需要大量采样

    改进方向：
    - 加入baseline减少方差 → Actor-Critic
    - 用TD估计代替蒙特卡洛 → GAE
    - 限制策略更新幅度 → PPO
    """
    print(problems)

    print("\n" + "=" * 70)
    print("REINFORCE演示完成！")
    print("下一步: 学习 actor_critic.py 了解如何减少方差")
    print("=" * 70)
