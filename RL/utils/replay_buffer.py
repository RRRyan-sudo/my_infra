"""
经验回放缓冲区

本模块提供RL算法使用的数据存储结构。

核心概念：

1. 经验回放 (Experience Replay) - 用于off-policy算法（DQN、DDPG、TD3、SAC）
   - 存储历史经验 (s, a, r, s', done)
   - 随机采样打破数据相关性
   - 提高样本效率

2. Rollout缓冲区 - 用于on-policy算法（PPO、A2C）
   - 存储一个epoch的完整轨迹
   - 包含log_prob和value用于策略梯度计算
   - 用完即清空（on-policy不重用旧数据）

为什么需要经验回放？
    神经网络训练假设数据是独立同分布(i.i.d.)的，
    但RL中连续的状态高度相关（s_t和s_{t+1}很相似）。
    经验回放通过随机采样，打破这种时间相关性。
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


# ==================== 基础经验回放 ====================

class ReplayBuffer:
    """
    经验回放缓冲区

    用于off-policy算法（DDPG、TD3、SAC等）

    存储转移元组：(state, action, reward, next_state, done)

    工作流程：
        1. 与环境交互时，调用 add() 存储经验
        2. 训练时，调用 sample() 随机采样一批经验
        3. 缓冲区满时，覆盖最旧的经验（环形缓冲区）
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        discrete_action: bool = False
    ):
        """
        初始化缓冲区

        Args:
            capacity: 最大容量
            state_dim: 状态维度
            action_dim: 动作维度
            discrete_action: 是否为离散动作（影响存储方式）
        """
        self.capacity = capacity
        self.discrete_action = discrete_action

        # 预分配内存（比动态append更高效）
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        if discrete_action:
            self.actions = np.zeros((capacity, 1), dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)

        # 当前位置和大小
        self.ptr = 0  # 下一个写入位置
        self.size = 0  # 当前存储的经验数量

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        添加一条经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # 更新指针（环形缓冲区）
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        随机采样一批经验

        Args:
            batch_size: 批次大小

        Returns:
            包含states, actions, rewards, next_states, dones的字典
        """
        # 随机选择索引
        indices = np.random.randint(0, self.size, size=batch_size)

        # 转换为PyTorch张量
        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]) if self.discrete_action
            else torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }

        return batch

    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return self.size


# ==================== Rollout缓冲区 ====================

@dataclass
class RolloutData:
    """Rollout数据的数据类"""
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """
    Rollout缓冲区

    用于on-policy算法（PPO、A2C等）

    与ReplayBuffer的区别：
        1. 存储完整轨迹，不是随机转移
        2. 包含log_prob和value，用于策略梯度
        3. 训练后清空，不重用旧数据

    工作流程：
        1. 收集固定步数的轨迹
        2. 计算returns和advantages
        3. 用于多个epoch的训练
        4. 清空缓冲区，开始新一轮收集
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        discrete_action: bool = True
    ):
        """
        初始化缓冲区

        Args:
            buffer_size: 缓冲区大小（收集的步数）
            state_dim: 状态维度
            action_dim: 动作维度
            gamma: 折扣因子
            gae_lambda: GAE参数
            discrete_action: 是否为离散动作
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.discrete_action = discrete_action

        self.reset()

    def reset(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        self.advantages = None
        self.returns = None

        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """
        添加一步数据

        Args:
            state: 状态
            action: 动作
            reward: 奖励
            done: 是否终止
            value: V(s)估计值
            log_prob: log π(a|s)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float):
        """
        计算returns和advantages

        使用GAE (Generalized Advantage Estimation):
            δ_t = r_t + γ*V(s_{t+1}) - V(s_t)  (TD误差)
            A_t = Σ (γλ)^l * δ_{t+l}           (GAE)

        Args:
            last_value: 最后一个状态的价值估计（用于bootstrap）
        """
        # 转换为numpy数组
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])  # 多加一个用于计算

        # 计算GAE
        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            # TD误差
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]

            # GAE累积
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Returns = Advantages + Values
        returns = advantages + values[:-1]

        self.advantages = advantages
        self.returns = returns

    def get(self) -> RolloutData:
        """
        获取所有数据用于训练

        Returns:
            RolloutData对象
        """
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))

        if self.discrete_action:
            actions = torch.LongTensor(np.array(self.actions))
        else:
            actions = torch.FloatTensor(np.array(self.actions))

        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        advantages = torch.FloatTensor(self.advantages)
        returns = torch.FloatTensor(self.returns)
        values = torch.FloatTensor(np.array(self.values))

        # 标准化advantages（减少方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutData(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            values=values
        )

    def get_batches(self, batch_size: int) -> List[RolloutData]:
        """
        获取小批次数据（用于mini-batch训练）

        Args:
            batch_size: 批次大小

        Returns:
            RolloutData对象列表
        """
        data = self.get()
        n = len(self.states)

        # 随机打乱索引
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            batch = RolloutData(
                states=data.states[batch_indices],
                actions=data.actions[batch_indices],
                old_log_probs=data.old_log_probs[batch_indices],
                advantages=data.advantages[batch_indices],
                returns=data.returns[batch_indices],
                values=data.values[batch_indices]
            )
            batches.append(batch)

        return batches

    def __len__(self) -> int:
        return self.ptr


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("经验回放缓冲区演示")
    print("=" * 60)

    # 1. 测试ReplayBuffer
    print("\n1. ReplayBuffer (off-policy):")
    buffer = ReplayBuffer(
        capacity=1000,
        state_dim=4,
        action_dim=2,
        discrete_action=False
    )

    # 添加一些经验
    for i in range(100):
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(4).astype(np.float32)
        done = np.random.random() > 0.95

        buffer.add(state, action, reward, next_state, done)

    print(f"   缓冲区大小: {len(buffer)}")

    # 采样
    batch = buffer.sample(batch_size=32)
    print(f"   采样批次:")
    print(f"     - states: {batch['states'].shape}")
    print(f"     - actions: {batch['actions'].shape}")
    print(f"     - rewards: {batch['rewards'].shape}")

    # 2. 测试RolloutBuffer
    print("\n2. RolloutBuffer (on-policy):")
    rollout = RolloutBuffer(
        buffer_size=128,
        state_dim=4,
        action_dim=2,
        gamma=0.99,
        gae_lambda=0.95,
        discrete_action=True
    )

    # 收集一个rollout
    for i in range(128):
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randint(2)
        reward = np.random.randn()
        done = np.random.random() > 0.95
        value = np.random.randn()
        log_prob = np.random.randn()

        rollout.add(state, action, reward, done, value, log_prob)

    # 计算returns和advantages
    last_value = np.random.randn()
    rollout.compute_returns_and_advantages(last_value)

    print(f"   Rollout大小: {len(rollout)}")

    # 获取数据
    data = rollout.get()
    print(f"   完整数据:")
    print(f"     - states: {data.states.shape}")
    print(f"     - actions: {data.actions.shape}")
    print(f"     - advantages: {data.advantages.shape}")
    print(f"     - returns: {data.returns.shape}")

    # 获取小批次
    batches = rollout.get_batches(batch_size=32)
    print(f"   小批次数量: {len(batches)}")
    print(f"   每批次大小: {batches[0].states.shape[0]}")

    print("\n" + "=" * 60)
    print("缓冲区测试通过！")
