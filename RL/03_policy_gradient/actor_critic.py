"""
Module 03 - Actor-Critic算法

Actor-Critic结合了策略梯度（Actor）和价值函数（Critic）。

核心概念：
    - Actor: 策略网络 π(a|s)，决定如何行动
    - Critic: 价值网络 V(s)，评估状态好坏
    - 用V(s)作为baseline减少方差

关键公式：

    原始策略梯度（REINFORCE）:
    ∇J(θ) = E[∇log π(a|s) * G_t]
    问题：G_t方差很大

    带baseline的策略梯度:
    ∇J(θ) = E[∇log π(a|s) * (G_t - b(s))]
    只要b(s)与动作无关，期望不变，但方差减小

    Actor-Critic:
    ∇J(θ) = E[∇log π(a|s) * A(s,a)]
    其中 A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s) = TD误差

    这就是Actor-Critic的核心思想：
    用TD误差作为优势函数的估计

为什么重要？
    - 是PPO的基础架构
    - SAC也是Actor-Critic
    - RLHF中的策略模型就是Actor

与大模型的关联：
    - RLHF中：Actor = 大模型策略，Critic = 价值头
    - PPO训练时同时更新Actor和Critic
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


# ==================== Actor-Critic解释 ====================

def explain_actor_critic():
    """解释Actor-Critic架构"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    Actor-Critic架构                               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║                       ┌─────────────┐                            ║
    ║           状态 s ───▶ │   Actor     │ ───▶ 动作概率 π(a|s)        ║
    ║              │        │  (策略网络)  │                            ║
    ║              │        └─────────────┘                            ║
    ║              │                                                   ║
    ║              │        ┌─────────────┐                            ║
    ║              └──────▶ │   Critic    │ ───▶ 状态价值 V(s)          ║
    ║                       │  (价值网络)  │                            ║
    ║                       └─────────────┘                            ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Actor的作用：                                                    ║
    ║  ════════════                                                    ║
    ║  - 输出动作的概率分布                                              ║
    ║  - 根据Critic的反馈调整策略                                        ║
    ║  - 目标：最大化期望回报                                            ║
    ║                                                                  ║
    ║  Critic的作用：                                                   ║
    ║  ════════════                                                    ║
    ║  - 估计状态价值V(s)                                               ║
    ║  - 提供baseline减少方差                                           ║
    ║  - 计算TD误差指导Actor学习                                         ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  为什么Critic能减少方差？                                          ║
    ║  ═══════════════════════                                         ║
    ║                                                                  ║
    ║  REINFORCE:  ∇J = E[∇log π * G_t]                                ║
    ║              G_t包含整个轨迹的随机性，方差大                         ║
    ║                                                                  ║
    ║  Actor-Critic: ∇J = E[∇log π * (r + γV(s') - V(s))]              ║
    ║                只涉及一步，V(s)是平滑的估计，方差小                  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== Actor-Critic实现 ====================

class ActorNetwork(nn.Module):
    """Actor网络：输出动作概率"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic网络：估计状态价值"""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    """
    Actor-Critic算法

    与REINFORCE的区别：
    1. 每一步都可以更新（不用等episode结束）
    2. 用TD误差代替回报G_t
    3. 方差更小，学习更稳定

    更新规则：
    - Critic: 最小化 (r + γV(s') - V(s))²，即TD误差
    - Actor: 最大化 log π(a|s) * TD误差
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
        """
        self.gamma = gamma

        # Actor和Critic网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

        # 分别的优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择动作

        Returns:
            action: 动作
            log_prob: log π(a|s)
            value: V(s)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Actor输出动作概率
        probs = self.actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        # Critic输出价值估计
        value = self.critic(state_tensor)

        return action.item(), dist.log_prob(action), value

    def update(
        self,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[float, float]:
        """
        一步更新

        Args:
            log_prob: log π(a|s)
            value: V(s)
            reward: 奖励
            next_state: 下一状态
            done: 是否终止

        Returns:
            actor_loss, critic_loss
        """
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # 计算TD目标
        with torch.no_grad():
            if done:
                next_value = 0
            else:
                next_value = self.critic(next_state_tensor).item()

        td_target = reward + self.gamma * next_value

        # TD误差 = TD目标 - 当前估计
        # 这就是优势函数的单步估计
        td_error = td_target - value.item()

        # ===== 更新Critic =====
        # 损失：(TD目标 - V(s))² = TD误差²
        critic_loss = F.mse_loss(value, torch.tensor([[td_target]]))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== 更新Actor =====
        # 损失：-log π(a|s) * TD误差
        # TD误差 > 0: 这个动作比预期好，增加其概率
        # TD误差 < 0: 这个动作比预期差，减少其概率
        actor_loss = -log_prob * td_error

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


# ==================== 训练函数 ====================

def train_actor_critic(
    env,
    agent: ActorCritic,
    n_episodes: int = 1000,
    max_steps: int = 500,
    print_interval: int = 100
) -> List[float]:
    """
    训练Actor-Critic agent

    注意与REINFORCE的区别：每一步都更新，不用等episode结束
    """
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = agent.select_action(state)

            # 与环境交互
            next_state, reward, done, _ = env.step(action)

            # 立即更新（这是与REINFORCE的关键区别）
            agent.update(log_prob, value, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}")

    return episode_rewards


# ==================== A2C：同步版本 ====================

def explain_a2c():
    """解释A2C（Advantage Actor-Critic）"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              A2C (Advantage Actor-Critic)                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  A2C是Actor-Critic的改进版本：                                     ║
    ║                                                                  ║
    ║  1. 使用优势函数 A(s,a) = Q(s,a) - V(s) 代替TD误差                  ║
    ║     更准确地评估动作的相对好坏                                       ║
    ║                                                                  ║
    ║  2. 批量更新                                                      ║
    ║     收集多步数据后一起更新，更稳定                                    ║
    ║                                                                  ║
    ║  3. 加入熵正则化                                                   ║
    ║     鼓励探索，防止策略过早收敛                                       ║
    ║     L = L_actor + c1 * L_critic - c2 * H(π)                      ║
    ║                                      ↑                           ║
    ║                                    熵奖励                         ║
    ║                                                                  ║
    ║  A2C vs A3C:                                                     ║
    ║  - A2C: 同步更新，一个学习器                                        ║
    ║  - A3C: 异步更新，多个学习器并行                                    ║
    ║                                                                  ║
    ║  A2C → PPO:                                                      ║
    ║  PPO进一步改进了Actor的更新方式（限制更新幅度）                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 演示 ====================

if __name__ == "__main__":
    from envs.gridworld import GridWorld

    print("=" * 70)
    print("Module 03: Actor-Critic算法")
    print("=" * 70)

    # 1. Actor-Critic架构
    print("\n" + "=" * 70)
    print("1. Actor-Critic架构")
    explain_actor_critic()

    # 2. 训练
    print("\n" + "=" * 70)
    print("2. 在GridWorld上训练Actor-Critic")
    print("-" * 70)

    env = GridWorld(size=4)

    # 包装环境（状态转为one-hot）
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
    agent = ActorCritic(
        state_dim=env.n_states,
        action_dim=env.n_actions,
        actor_lr=1e-2,
        critic_lr=1e-2,
        gamma=0.99
    )

    # 训练
    print("\n开始训练Actor-Critic...")
    rewards = train_actor_critic(
        wrapped_env,
        agent,
        n_episodes=500,
        print_interval=100
    )

    # 3. A2C解释
    print("\n" + "=" * 70)
    print("3. A2C (Advantage Actor-Critic)")
    explain_a2c()

    # 4. Actor-Critic vs REINFORCE
    print("\n" + "=" * 70)
    print("4. Actor-Critic vs REINFORCE 对比")
    print("-" * 70)

    comparison = """
    ┌────────────────┬─────────────────┬─────────────────┐
    │     特点       │   REINFORCE     │  Actor-Critic   │
    ├────────────────┼─────────────────┼─────────────────┤
    │ 更新时机       │ Episode结束     │ 每一步          │
    │ 方差           │ 高              │ 低              │
    │ 偏差           │ 无偏            │ 有偏（但可接受） │
    │ 需要Critic     │ 否              │ 是              │
    │ 样本效率       │ 低              │ 较高            │
    │ 适用场景       │ 短episode       │ 通用            │
    └────────────────┴─────────────────┴─────────────────┘
    """
    print(comparison)

    # 5. 与PPO的关系
    print("\n" + "=" * 70)
    print("5. 从Actor-Critic到PPO")
    print("-" * 70)

    transition = """
    Actor-Critic的问题：
    ════════════════════
    - Actor更新步长难以选择
    - 步长太大：策略变化剧烈，性能崩溃
    - 步长太小：学习太慢

    PPO的解决方案：
    ════════════════
    - 限制策略更新的幅度（Clipping）
    - 使用重要性采样比 r(θ) = π_new / π_old
    - 裁剪到 [1-ε, 1+ε] 范围内

    下一步学习：
    - gae.py: 如何更好地估计优势函数
    - 04_ppo/: PPO的完整实现
    """
    print(transition)

    print("\n" + "=" * 70)
    print("Actor-Critic演示完成！")
    print("下一步: 学习 gae.py 了解广义优势估计")
    print("=" * 70)
