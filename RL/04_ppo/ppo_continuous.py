"""
Module 04 - PPO连续动作版本

连续动作PPO用于动作空间是连续的场景，如：
- 机器人控制（关节力矩、速度）
- 自动驾驶（方向盘角度、油门）
- 游戏（鼠标位置、移动方向）

核心区别：
    离散动作：π(a|s) 是概率分布，如 [0.1, 0.3, 0.6]
    连续动作：π(a|s) 是高斯分布，输出 μ(s) 和 σ(s)

关键公式：

    连续策略:
    π(a|s) = N(μ(s), σ(s)²)
           = (1/√(2πσ²)) * exp(-(a-μ)²/(2σ²))

    log概率:
    log π(a|s) = -0.5 * [(a-μ)/σ]² - log(σ) - 0.5*log(2π)

    其余与离散PPO相同（ratio、clipping等）

与具身智能的关联：
    - 机器人的关节控制是连续的
    - SAC、TD3也使用类似的连续策略
    - 这是具身智能的核心技术
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple, Dict
import sys
sys.path.append("../..")


# ==================== 连续策略网络 ====================

class ContinuousPolicyNetwork(nn.Module):
    """
    连续动作策略网络

    输出高斯分布的参数: μ(s) 和 log σ(s)

    为什么用log σ而不是σ？
    - σ必须为正，但神经网络输出可以是任意值
    - log σ可以是任意值，然后用 exp 转换
    - 数值更稳定
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共享特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 均值头
        self.mean = nn.Linear(hidden_dim, action_dim)

        # 对数标准差（可学习参数，不依赖状态）
        # 也可以用网络输出，这里用简单版本
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            mean: 均值 μ(s)
            std: 标准差 σ
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        # tanh将均值限制在[-1, 1]
        mean = torch.tanh(mean)

        # 标准差
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        采样动作

        Args:
            state: 状态
            deterministic: 是否确定性（用于评估）

        Returns:
            action: 动作
            log_prob: log π(a|s)
        """
        mean, std = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            # 创建高斯分布
            dist = Normal(mean, std)
            # rsample: 重参数化采样（可以计算梯度）
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # 限制动作范围
        action = torch.clamp(action, -1.0, 1.0)

        return action, log_prob

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """
        评估给定的状态-动作对

        用于PPO更新时计算新策略下的log概率
        """
        mean, std = self.forward(states)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy


# ==================== 价值网络 ====================

class ContinuousValueNetwork(nn.Module):
    """价值网络 V(s)"""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== PPO连续版实现 ====================

class PPOContinuous:
    """
    PPO连续动作版本

    与离散版的主要区别：
    1. 策略输出高斯分布参数（μ, σ）
    2. 动作是连续值，而不是离散索引
    3. log概率计算方式不同

    其他部分（clipping、GAE等）完全相同
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
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim

        # 网络
        self.policy = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ContinuousValueNetwork(state_dim, hidden_dim)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        # 数据存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor, deterministic)
            value = self.value(state_tensor).item()

        return action.squeeze(0).numpy(), log_prob.item(), value

    def store_transition(self, state, action, reward, done, value, log_prob):
        """存储转移"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_gae(self, next_value: float):
        """计算GAE"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """PPO更新"""
        # 计算GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0)
            next_value = self.value(last_state).item()

        advantages, returns = self.compute_gae(next_value)

        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练
        n = len(self.states)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # 随机打乱
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # 新策略的log概率
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # 策略比
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO-Clip
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                values = self.value(batch_states).squeeze()
                value_loss = F.mse_loss(values, batch_returns)

                # 熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 清空存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }


# ==================== 训练函数 ====================

def train_ppo_continuous(
    env,
    agent: PPOContinuous,
    n_steps: int = 100000,
    steps_per_update: int = 2048,
    print_interval: int = 10
):
    """训练连续PPO"""
    total_steps = 0
    episode_rewards = []
    current_episode_reward = 0
    n_updates = 0

    state = env.reset()

    while total_steps < n_steps:
        for _ in range(steps_per_update):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, done, value, log_prob)

            current_episode_reward += reward
            state = next_state
            total_steps += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state = env.reset()

        losses = agent.update()
        n_updates += 1

        if n_updates % print_interval == 0 and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Step {total_steps}: 平均奖励 = {avg_reward:.2f}, "
                  f"策略损失 = {losses['policy_loss']:.4f}")

    return episode_rewards


# ==================== 演示 ====================

if __name__ == "__main__":
    from envs.simple_continuous import SimpleContinuousEnv

    print("=" * 70)
    print("Module 04: PPO连续动作版本")
    print("=" * 70)

    # 1. 连续策略解释
    print("\n" + "=" * 70)
    print("1. 连续动作策略")
    print("-" * 70)

    continuous_explanation = """
    连续动作空间 vs 离散动作空间:
    ═══════════════════════════════

    离散动作:
    - 动作是有限集合: {左, 右, 上, 下}
    - 策略输出概率分布: [0.1, 0.3, 0.4, 0.2]
    - 采样: 从Categorical分布采样

    连续动作:
    - 动作是连续值: 力 ∈ [-1, 1]
    - 策略输出高斯参数: μ(s), σ(s)
    - 采样: a = μ + σ * ε, 其中 ε ~ N(0,1)

    为什么用高斯分布？
    - 简单且有效
    - 天然支持探索（通过σ）
    - 梯度容易计算
    """
    print(continuous_explanation)

    # 2. 训练演示
    print("\n" + "=" * 70)
    print("2. 在SimpleContinuousEnv上训练")
    print("-" * 70)

    env = SimpleContinuousEnv()

    agent = PPOContinuous(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64
    )

    print(f"\n环境: SimpleContinuousEnv")
    print(f"状态维度: {env.state_dim}")
    print(f"动作维度: {env.action_dim}")
    print("\n开始训练...")

    rewards = train_ppo_continuous(
        env, agent,
        n_steps=30000,
        steps_per_update=1024,
        print_interval=5
    )

    # 3. 评估
    print("\n" + "=" * 70)
    print("3. 评估训练后的策略")
    print("-" * 70)

    state = env.reset()
    total_reward = 0

    for step in range(200):
        action, _, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            print(f"Step {step}: 距离目标 = {info['distance']:.2f}")

        if done:
            print(f"\n完成! 总奖励 = {total_reward:.2f}, 成功 = {info['success']}")
            break

        state = next_state

    # 4. 与具身智能的关联
    print("\n" + "=" * 70)
    print("4. 与具身智能的关联")
    print("-" * 70)

    embodied_explanation = """
    连续PPO在具身智能中的应用：
    ═════════════════════════════

    机器人控制:
    - 状态: 关节角度、角速度、传感器数据
    - 动作: 关节力矩或目标角度
    - 奖励: 任务完成度 - 能耗 - 安全惩罚

    PPO vs SAC (下一模块):
    - PPO: on-policy，适合仿真环境（采样便宜）
    - SAC: off-policy，样本效率更高
    - 两者都是机器人控制的主流方法

    实际应用:
    - 机械臂抓取
    - 双足/四足机器人行走
    - 自动驾驶
    - 无人机控制
    """
    print(embodied_explanation)

    print("\n" + "=" * 70)
    print("PPO连续版演示完成！")
    print("下一步: 学习 05_llm_alignment/ 了解RLHF和DPO")
    print("       或 06_continuous_control/ 了解SAC和TD3")
    print("=" * 70)
