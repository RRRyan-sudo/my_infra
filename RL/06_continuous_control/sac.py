"""
Module 06 - SAC (Soft Actor-Critic)

SAC是目前最流行的连续控制算法，广泛用于机器人控制。

核心概念：
    1. 最大熵强化学习：在最大化奖励的同时最大化策略熵
    2. 双Q网络：减少过估计
    3. 自动温度调节：自动平衡探索与利用

关键公式：

    最大熵目标:
    J(π) = E[Σ γ^t (r_t + α * H(π(·|s_t)))]
         = E[Σ γ^t (r_t - α * log π(a_t|s_t))]

    Soft Q函数:
    Q(s,a) = r + γ * E[Q(s',a') - α * log π(a'|s')]

    策略更新:
    π* = argmax_π E[Q(s,a) - α * log π(a|s)]

    温度自动调节:
    α* = argmin_α E[-α * (log π(a|s) + H_target)]

为什么SAC这么好？
    1. 最大熵：自动探索，不易陷入局部最优
    2. Off-policy：样本效率高
    3. 稳定：双Q网络和目标网络

应用：
    - 机械臂控制
    - 四足机器人行走
    - 自动驾驶
    - 无人机控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional
import copy


# ==================== SAC原理 ====================

def explain_sac():
    """解释SAC的原理"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              SAC (Soft Actor-Critic)                             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  传统RL的问题:                                                    ║
    ║  ═══════════════                                                 ║
    ║  - 策略可能过早收敛到次优解                                         ║
    ║  - 需要人工设计探索策略                                            ║
    ║  - 对超参数敏感                                                   ║
    ║                                                                  ║
    ║  最大熵强化学习的思想:                                              ║
    ║  ══════════════════════                                          ║
    ║  在最大化奖励的同时，最大化策略的熵                                   ║
    ║                                                                  ║
    ║  J = E[Σ r_t + α * H(π)]                                         ║
    ║        ↑         ↑                                               ║
    ║     奖励最大化  熵最大化（多样性）                                   ║
    ║                                                                  ║
    ║  α (温度系数) 控制两者的平衡:                                       ║
    ║  - α大：更注重探索                                                 ║
    ║  - α小：更注重利用                                                 ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  SAC的三个网络:                                                   ║
    ║  ═══════════════                                                 ║
    ║                                                                  ║
    ║  1. Actor (策略网络)                                              ║
    ║     输入: 状态 s                                                  ║
    ║     输出: 高斯分布参数 (μ, σ)                                      ║
    ║     目标: 最大化 Q(s,a) - α*log π(a|s)                            ║
    ║                                                                  ║
    ║  2. Critic (双Q网络)                                             ║
    ║     输入: 状态 s, 动作 a                                          ║
    ║     输出: Q值                                                     ║
    ║     用两个Q网络取min，减少过估计                                     ║
    ║                                                                  ║
    ║  3. 温度参数 α                                                    ║
    ║     自动调节探索程度                                               ║
    ║     目标: 使策略熵接近目标熵                                        ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  SAC vs PPO vs TD3:                                              ║
    ║  ═══════════════════                                             ║
    ║  ┌──────────┬────────┬────────┬────────┐                        ║
    ║  │   特点   │  SAC   │  PPO   │  TD3   │                        ║
    ║  ├──────────┼────────┼────────┼────────┤                        ║
    ║  │ 类型     │ Off-p  │ On-p   │ Off-p  │                        ║
    ║  │ 策略     │ 随机   │ 随机   │ 确定性 │                        ║
    ║  │ 探索     │ 熵驱动 │ 熵奖励 │ 噪声   │                        ║
    ║  │ 样本效率 │ 高     │ 中     │ 高     │                        ║
    ║  │ 稳定性   │ 高     │ 高     │ 高     │                        ║
    ║  │ 适用场景 │ 通用   │ 仿真   │ 确定性 │                        ║
    ║  └──────────┴────────┴────────┴────────┘                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 网络定义 ====================

class GaussianPolicy(nn.Module):
    """
    高斯策略网络

    输出高斯分布的均值和对数标准差
    使用重参数化技巧采样
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回均值和对数标准差"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并计算log概率

        使用重参数化技巧: a = tanh(μ + σ * ε), ε ~ N(0,1)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化采样
        normal = Normal(mean, std)
        x = normal.rsample()  # rsample允许梯度传播

        # Squashing (tanh)
        action = torch.tanh(x)

        # 计算log概率（考虑tanh变换）
        # log π(a|s) = log N(x|μ,σ) - log(1 - tanh²(x))
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== SAC实现 ====================

class SAC:
    """
    SAC (Soft Actor-Critic)

    特点:
    1. 最大熵RL：自动探索
    2. 双Q网络：减少过估计
    3. 自动温度调节：自适应探索程度
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 温度系数（初始值）
            auto_alpha: 是否自动调节温度
            target_entropy: 目标熵（默认-action_dim）
        """
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # 策略网络
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 双Q网络
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr
        )

        # 温度参数
        if auto_alpha:
            self.target_entropy = target_entropy if target_entropy else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state)

        return action.squeeze(0).numpy()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        SAC更新

        Returns:
            包含各种损失的字典
        """
        # ===== 更新Q网络 =====
        with torch.no_grad():
            # 采样下一个动作
            next_actions, next_log_probs = self.policy.sample(next_states)

            # 目标Q值（取两个Q网络的最小值）
            q1_target = self.q1_target(next_states, next_actions)
            q2_target = self.q2_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs

            # TD目标
            target = rewards + self.gamma * (1 - dones) * q_target

        # Q网络损失
        q1_loss = F.mse_loss(self.q1(states, actions), target)
        q2_loss = F.mse_loss(self.q2(states, actions), target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ===== 更新策略网络 =====
        new_actions, log_probs = self.policy.sample(states)

        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # 策略损失：最大化 Q - α*log_prob
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ===== 更新温度参数 =====
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ===== 软更新目标网络 =====
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'entropy': -log_probs.mean().item()
        }


# ==================== 经验回放 ====================

class ReplayBuffer:
    """简单的经验回放缓冲区"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx])
        )


# ==================== 训练函数 ====================

def train_sac(
    env,
    agent: SAC,
    n_steps: int = 100000,
    start_steps: int = 1000,
    batch_size: int = 256,
    update_every: int = 1,
    print_interval: int = 1000
):
    """训练SAC"""
    buffer = ReplayBuffer(100000, env.state_dim, env.action_dim)

    state = env.reset()
    episode_reward = 0
    episode_rewards = []
    n_episodes = 0

    for step in range(n_steps):
        # 初始随机探索
        if step < start_steps:
            action = np.random.uniform(-1, 1, env.action_dim)
        else:
            action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        episode_reward += reward
        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            n_episodes += 1
            state = env.reset()

        # 更新
        if step >= start_steps and step % update_every == 0:
            batch = buffer.sample(batch_size)
            agent.update(*batch)

        # 打印
        if (step + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Step {step+1}: Episodes={n_episodes}, Avg Reward={avg_reward:.2f}")

    return episode_rewards


# ==================== 演示 ====================

if __name__ == "__main__":
    import sys
    sys.path.append("../..")
    from envs.simple_continuous import SimpleContinuousEnv

    print("=" * 70)
    print("Module 06: SAC (Soft Actor-Critic)")
    print("=" * 70)

    # 1. SAC原理
    print("\n" + "=" * 70)
    print("1. SAC原理")
    explain_sac()

    # 2. 训练演示
    print("\n" + "=" * 70)
    print("2. SAC训练演示")
    print("-" * 70)

    env = SimpleContinuousEnv()
    print(f"\n环境: SimpleContinuousEnv")
    print(f"状态维度: {env.state_dim}")
    print(f"动作维度: {env.action_dim}")

    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True
    )

    print("\n开始训练SAC...")
    rewards = train_sac(
        env, agent,
        n_steps=20000,
        start_steps=500,
        print_interval=2000
    )

    # 3. 评估
    print("\n" + "=" * 70)
    print("3. 评估训练后的策略")
    print("-" * 70)

    state = env.reset()
    total_reward = 0

    for step in range(200):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"完成! 步数={step+1}, 总奖励={total_reward:.2f}, 成功={info['success']}")
            break
        state = next_state

    # 4. 与具身智能的关联
    print("\n" + "=" * 70)
    print("4. SAC在具身智能中的应用")
    print("-" * 70)

    embodied_ai = """
    SAC在机器人控制中的应用：
    ═══════════════════════════

    1. 机械臂控制
       - 状态: 关节角度、速度、末端位姿
       - 动作: 关节力矩或速度指令
       - 任务: 抓取、放置、装配

    2. 四足机器人
       - 状态: 身体姿态、关节状态、IMU数据
       - 动作: 12个关节的目标角度
       - 任务: 行走、奔跑、跨越障碍

    3. 自动驾驶
       - 状态: 激光雷达、摄像头、车辆状态
       - 动作: 方向盘、油门、刹车
       - 任务: 路径跟踪、避障

    SAC的优势：
    - 自动探索：不需要手动设计探索策略
    - 样本效率高：适合真实机器人（采样昂贵）
    - 稳定：适合连续控制任务

    Sim-to-Real:
    - 先在仿真中训练SAC
    - 用域随机化增加鲁棒性
    - 迁移到真实机器人
    """
    print(embodied_ai)

    print("\n" + "=" * 70)
    print("SAC演示完成！")
    print("恭喜完成连续控制模块！")
    print("=" * 70)
