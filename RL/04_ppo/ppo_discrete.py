"""
Module 04 - PPO离散动作版本

PPO (Proximal Policy Optimization) 是OpenAI在2017年提出的算法，
是目前应用最广泛的强化学习算法，也是RLHF的核心。

核心概念：
    1. 重要性采样：允许用旧策略的数据更新新策略
    2. 策略比 r(θ) = π_θ(a|s) / π_{θ_old}(a|s)
    3. Clipping：限制策略更新幅度，保证训练稳定

关键公式：

    PPO-Clip目标函数:
    L^CLIP(θ) = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]

    其中:
    - r(θ) = π_θ(a|s) / π_{θ_old}(a|s)  重要性采样比
    - A 是优势函数（用GAE计算）
    - ε = 0.2 是裁剪参数

    直觉理解:
    - 如果A > 0（好动作），我们想增加π(a|s)，即r > 1
      但clip限制r不超过1+ε，防止更新太大
    - 如果A < 0（差动作），我们想减少π(a|s)，即r < 1
      但clip限制r不低于1-ε，防止更新太大

为什么PPO这么成功？
    1. 简单：比TRPO实现简单得多
    2. 稳定：clipping防止策略崩溃
    3. 高效：可以多次复用数据（多个epoch）
    4. 通用：离散/连续动作空间都适用

与大模型的关联：
    - ChatGPT/Claude的RLHF就是用PPO训练
    - 语言模型生成token = 执行策略
    - 人类偏好奖励 → PPO优化策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
import sys
sys.path.append("../..")

from utils.networks import PolicyNetworkDiscrete, ValueNetwork
from utils.replay_buffer import RolloutBuffer
from RL.utils.training import clip_gradients


# ==================== PPO核心思想 ====================

def explain_ppo():
    """解释PPO的核心思想"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    PPO (Proximal Policy Optimization)            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  问题：普通策略梯度的更新步长难以选择                                 ║
    ║  ═════════════════════════════════════                           ║
    ║  - 步长太大：策略变化剧烈，性能可能崩溃                              ║
    ║  - 步长太小：学习太慢                                              ║
    ║                                                                  ║
    ║  TRPO的解决方案：                                                  ║
    ║  ════════════════                                                ║
    ║  - 添加KL散度约束：KL(π_old || π_new) < δ                         ║
    ║  - 保证新旧策略不会差太多                                          ║
    ║  - 问题：需要二阶优化，实现复杂                                      ║
    ║                                                                  ║
    ║  PPO的解决方案：                                                   ║
    ║  ══════════════                                                  ║
    ║  - 用Clipping代替KL约束                                           ║
    ║  - 简单且有效                                                     ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  PPO-Clip的工作原理：                                              ║
    ║  ═══════════════════                                             ║
    ║                                                                  ║
    ║  1. 定义策略比：r(θ) = π_new(a|s) / π_old(a|s)                    ║
    ║     - r = 1：新旧策略对这个动作的概率相同                            ║
    ║     - r > 1：新策略更倾向于这个动作                                 ║
    ║     - r < 1：新策略更不倾向于这个动作                                ║
    ║                                                                  ║
    ║  2. 目标函数（以A > 0为例）：                                       ║
    ║     L = min(r*A, clip(r, 1-ε, 1+ε)*A)                            ║
    ║                                                                  ║
    ║     当A > 0时，我们想最大化L，即增加r                                ║
    ║     但如果r > 1+ε，clip(r)*A = (1+ε)*A，梯度为0                   ║
    ║     → 防止r增加太多                                               ║
    ║                                                                  ║
    ║  3. 类似地，当A < 0时，clip防止r减少太多                            ║
    ║                                                                  ║
    ║  结果：策略更新被"proximal"（近端）约束                              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== PPO离散版实现 ====================

class PPODiscrete:
    """
    PPO离散动作版本

    适用于动作空间是有限集合的情况，如：
    - 游戏（上下左右）
    - 选择题（选项A/B/C/D）
    - 语言模型的token选择（RLHF中）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: PPO裁剪参数（关键超参数）
            value_coef: Critic损失系数
            entropy_coef: 熵奖励系数（鼓励探索）
            max_grad_norm: 梯度裁剪阈值
            n_epochs: 每次更新的epoch数
            batch_size: 小批次大小
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # 策略网络（Actor）
        self.policy = PolicyNetworkDiscrete(state_dim, action_dim, [hidden_dim, hidden_dim])

        # 价值网络（Critic）
        self.value = ValueNetwork(state_dim, [hidden_dim, hidden_dim])

        # 优化器（Actor和Critic共用一个优化器）
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        # Rollout缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=2048,
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            discrete_action=True
        )

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        根据当前策略选择动作

        Returns:
            action: 动作
            log_prob: log π(a|s)
            value: V(s)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor)
            value = self.value(state_tensor).item()

        return action, log_prob.item(), value

    def store_transition(self, state, action, reward, done, value, log_prob):
        """存储一个转移"""
        self.buffer.add(state, action, reward, done, value, log_prob)

    def update(self) -> Dict[str, float]:
        """
        PPO更新

        Returns:
            包含各种损失的字典
        """
        # 计算GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0)
            last_value = self.value(last_state).item()

        self.buffer.compute_returns_and_advantages(last_value)

        # 获取数据
        data = self.buffer.get()

        # 记录损失
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # 多个epoch
        for epoch in range(self.n_epochs):
            # 获取小批次
            batches = self.buffer.get_batches(self.batch_size)

            for batch in batches:
                # ===== 计算新策略的log概率和熵 =====
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch.states, batch.actions
                )

                # ===== 计算策略比 =====
                # r(θ) = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
                ratio = torch.exp(new_log_probs - batch.old_log_probs)

                # ===== PPO-Clip损失 =====
                # surr1 = r(θ) * A
                surr1 = ratio * batch.advantages

                # surr2 = clip(r(θ), 1-ε, 1+ε) * A
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch.advantages

                # 取min，这是PPO的核心
                policy_loss = -torch.min(surr1, surr2).mean()

                # ===== 价值函数损失 =====
                values = self.value(batch.states).squeeze()
                value_loss = F.mse_loss(values, batch.returns)

                # ===== 熵奖励（鼓励探索）=====
                entropy_loss = -entropy.mean()

                # ===== 总损失 =====
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # ===== 反向传播 =====
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # 记录
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 清空缓冲区
        self.buffer.reset()

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }


# ==================== 训练函数 ====================

def train_ppo(
    env,
    agent: PPODiscrete,
    n_steps: int = 100000,
    steps_per_update: int = 2048,
    print_interval: int = 10
):
    """
    训练PPO agent

    Args:
        env: 环境
        agent: PPO实例
        n_steps: 总训练步数
        steps_per_update: 每次更新收集的步数
        print_interval: 打印间隔（更新次数）
    """
    total_steps = 0
    episode_rewards = []
    current_episode_reward = 0
    n_updates = 0

    state = env.reset()

    while total_steps < n_steps:
        # 收集数据
        for _ in range(steps_per_update):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, done, value, log_prob)

            current_episode_reward += reward
            state = next_state
            total_steps += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state = env.reset()

        # PPO更新
        losses = agent.update()
        n_updates += 1

        # 打印进度
        if n_updates % print_interval == 0:
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Step {total_steps}: "
                      f"平均奖励 = {avg_reward:.2f}, "
                      f"策略损失 = {losses['policy_loss']:.4f}, "
                      f"价值损失 = {losses['value_loss']:.4f}")
            else:
                print(f"Step {total_steps}: 收集数据中...")

    return episode_rewards


# ==================== 演示 ====================

if __name__ == "__main__":
    from envs.gridworld import GridWorld

    print("=" * 70)
    print("Module 04: PPO离散动作版本")
    print("=" * 70)

    # 1. PPO原理
    print("\n" + "=" * 70)
    print("1. PPO核心思想")
    explain_ppo()

    # 2. 训练演示
    print("\n" + "=" * 70)
    print("2. 在GridWorld上训练PPO")
    print("-" * 70)

    env = GridWorld(size=4)

    # 包装环境
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

    # 创建PPO agent
    agent = PPODiscrete(
        state_dim=env.n_states,
        action_dim=env.n_actions,
        hidden_dim=64,
        lr=3e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=4,
        batch_size=32
    )

    print("\n开始训练PPO...")
    print("参数: clip_epsilon=0.2, n_epochs=4, batch_size=32")

    rewards = train_ppo(
        wrapped_env,
        agent,
        n_steps=20000,
        steps_per_update=512,
        print_interval=5
    )

    # 3. PPO的关键点
    print("\n" + "=" * 70)
    print("3. PPO的关键点")
    print("-" * 70)

    key_points = """
    PPO成功的关键因素：
    ═══════════════════

    1. Clipping (clip_epsilon=0.2)
       - 限制策略更新幅度
       - 防止性能崩溃
       - 0.2是常用默认值

    2. 多个Epoch (n_epochs=4-10)
       - 同一批数据训练多次
       - 提高样本效率
       - 但不能太多（策略会偏离太远）

    3. GAE (gae_lambda=0.95)
       - 平衡偏差和方差
       - 计算优势函数

    4. 熵奖励 (entropy_coef=0.01)
       - 鼓励探索
       - 防止策略过早收敛

    5. 梯度裁剪 (max_grad_norm=0.5)
       - 防止梯度爆炸
       - 保持训练稳定
    """
    print(key_points)

    # 4. PPO在RLHF中的应用
    print("\n" + "=" * 70)
    print("4. PPO在RLHF中的应用")
    print("-" * 70)

    rlhf_explanation = """
    RLHF中的PPO：
    ═══════════════

    状态 s: 对话历史 + 当前提示
    动作 a: 生成下一个token
    策略 π(a|s): 语言模型的输出分布

    奖励 r:
    - 中间步骤: 0（不对每个token单独评分）
    - 生成结束: reward_model(response) - β * KL(π || π_ref)
                 ↑                           ↑
             人类偏好分数                  防止偏离太远

    PPO优化:
    - 使生成的回复获得更高的奖励
    - 同时保持与原始模型的相似性（KL约束）

    这就是ChatGPT/Claude等大模型的训练方式！
    """
    print(rlhf_explanation)

    print("\n" + "=" * 70)
    print("PPO离散版演示完成！")
    print("下一步: 学习 ppo_continuous.py 了解连续动作PPO")
    print("=" * 70)
