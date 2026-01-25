"""
神经网络架构

本模块提供RL算法常用的神经网络架构。

核心概念：
    1. 策略网络 (Policy Network): 输出动作的概率分布 π(a|s)
       - 离散动作：输出每个动作的概率（softmax）
       - 连续动作：输出高斯分布的均值和标准差

    2. 价值网络 (Value Network): 估计状态价值 V(s)
       - 输入：状态 s
       - 输出：标量价值

    3. Q网络 (Q Network): 估计动作价值 Q(s, a)
       - 离散动作：输入状态，输出每个动作的Q值
       - 连续动作：输入状态和动作，输出Q值

PyTorch基础说明：
    - nn.Module: PyTorch神经网络的基类
    - nn.Linear(in, out): 全连接层，执行 y = xW^T + b
    - F.relu(): ReLU激活函数，max(0, x)
    - forward(): 定义前向传播，PyTorch自动计算反向传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, List, Optional
import numpy as np


# ==================== 基础网络 ====================

class MLP(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron)

    这是最基础的神经网络结构：
        输入 -> [Linear -> ReLU] x N -> Linear -> 输出

    用途：作为策略网络和价值网络的主干网络
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: nn.Module = nn.ReLU
    ):
        """
        初始化MLP

        Args:
            input_dim: 输入维度（状态维度）
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表，如[64, 64]表示两个64维隐藏层
            activation: 激活函数类（默认ReLU）
        """
        super().__init__()

        # 构建层列表
        layers = []
        prev_dim = input_dim

        # 隐藏层：Linear + Activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        # 输出层：只有Linear，不加激活
        layers.append(nn.Linear(prev_dim, output_dim))

        # nn.Sequential: 按顺序执行各层
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, input_dim)

        Returns:
            输出张量，形状 (batch_size, output_dim)
        """
        return self.network(x)


# ==================== 策略网络 ====================

class PolicyNetworkDiscrete(nn.Module):
    """
    离散动作策略网络

    功能：给定状态，输出每个动作的概率分布 π(a|s)

    输出层使用softmax确保概率和为1：
        logits = MLP(s)
        π(a|s) = softmax(logits) = exp(logits_a) / Σ exp(logits_i)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64]
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        self.network = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算动作概率分布

        Args:
            state: 状态，形状 (batch_size, state_dim)

        Returns:
            动作概率，形状 (batch_size, action_dim)
        """
        logits = self.network(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        采样动作

        Args:
            state: 状态，形状 (state_dim,) 或 (1, state_dim)
            deterministic: 是否确定性选择（取概率最大的动作）

        Returns:
            action: 动作索引
            log_prob: log π(a|s)，用于策略梯度
        """
        # 确保state是2D
        if state.dim() == 1:
            state = state.unsqueeze(0)

        probs = self.forward(state)

        if deterministic:
            action = probs.argmax(dim=-1).item()
            log_prob = torch.log(probs[0, action] + 1e-8)
        else:
            # Categorical分布：根据概率采样
            dist = Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)

        return action, log_prob

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对

        用于PPO等算法计算新策略下的log概率

        Args:
            states: 状态批次，形状 (batch_size, state_dim)
            actions: 动作批次，形状 (batch_size,)

        Returns:
            log_probs: log π(a|s)，形状 (batch_size,)
            entropy: 策略熵，用于探索，形状 (batch_size,)
        """
        probs = self.forward(states)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class PolicyNetworkContinuous(nn.Module):
    """
    连续动作策略网络

    功能：给定状态，输出动作的高斯分布参数 π(a|s) = N(μ(s), σ(s))

    输出：
        - 均值 μ(s): 通过tanh限制在[-1, 1]
        - 标准差 σ(s): 通过softplus确保为正，或使用可学习的log_std

    数学原理：
        对于高斯分布 N(μ, σ²)，采样 a = μ + σ * ε，其中 ε ~ N(0, 1)
        log π(a|s) = -0.5 * ((a - μ) / σ)² - log(σ) - 0.5 * log(2π)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            log_std_min/max: log标准差的范围限制（防止数值问题）
        """
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共享的特征提取层
        self.shared = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1])

        # 均值头
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)

        # 标准差头（输出log_std）
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算高斯分布参数

        Args:
            state: 状态，形状 (batch_size, state_dim)

        Returns:
            mean: 均值，形状 (batch_size, action_dim)
            std: 标准差，形状 (batch_size, action_dim)
        """
        features = F.relu(self.shared(state))

        mean = torch.tanh(self.mean_head(features))  # 限制在[-1, 1]

        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        采样动作

        Args:
            state: 状态
            deterministic: 是否确定性（直接使用均值）

        Returns:
            action: 动作数组
            log_prob: log π(a|s)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mean, std = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            # 重参数化技巧：a = μ + σ * ε
            dist = Normal(mean, std)
            action = dist.rsample()  # rsample: 可微分的采样
            log_prob = dist.log_prob(action).sum(dim=-1)

        # 裁剪动作到有效范围
        action = torch.clamp(action, -1.0, 1.0)

        return action.squeeze(0).detach().numpy(), log_prob

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对

        Args:
            states: 状态批次
            actions: 动作批次

        Returns:
            log_probs: log π(a|s)
            entropy: 策略熵
        """
        mean, std = self.forward(states)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy


# ==================== 价值网络 ====================

class ValueNetwork(nn.Module):
    """
    状态价值网络 V(s)

    功能：估计给定状态的期望回报

    V(s) = E[Σ γ^t * r_t | s_0 = s]

    用途：
        - Actor-Critic中的Critic
        - 计算优势函数 A(s,a) = Q(s,a) - V(s)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [64, 64]
    ):
        """
        Args:
            state_dim: 状态维度
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        self.network = MLP(state_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算状态价值

        Args:
            state: 状态，形状 (batch_size, state_dim)

        Returns:
            价值，形状 (batch_size, 1)
        """
        return self.network(state)


# ==================== Q网络 ====================

class QNetwork(nn.Module):
    """
    动作价值网络 Q(s, a)

    功能：估计在给定状态下采取特定动作的期望回报

    Q(s, a) = E[Σ γ^t * r_t | s_0 = s, a_0 = a]

    两种模式：
        1. 离散动作：输入状态，输出所有动作的Q值
        2. 连续动作：输入状态和动作，输出单个Q值
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        continuous: bool = False
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            continuous: 是否为连续动作空间
        """
        super().__init__()
        self.continuous = continuous

        if continuous:
            # 连续动作：输入[state, action]，输出单个Q值
            self.network = MLP(state_dim + action_dim, 1, hidden_dims)
        else:
            # 离散动作：输入state，输出每个动作的Q值
            self.network = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Q值

        Args:
            state: 状态，形状 (batch_size, state_dim)
            action: 动作（连续空间时需要），形状 (batch_size, action_dim)

        Returns:
            Q值
            - 离散：形状 (batch_size, action_dim)
            - 连续：形状 (batch_size, 1)
        """
        if self.continuous:
            assert action is not None, "连续动作空间需要提供action参数"
            x = torch.cat([state, action], dim=-1)
            return self.network(x)
        else:
            return self.network(state)


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("神经网络架构演示")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)

    # 测试参数
    batch_size = 4
    state_dim = 8
    discrete_action_dim = 4
    continuous_action_dim = 2

    # 创建测试数据
    states = torch.randn(batch_size, state_dim)
    print(f"\n测试数据: states shape = {states.shape}")

    # 1. 测试离散策略网络
    print("\n1. 离散策略网络 PolicyNetworkDiscrete:")
    policy_discrete = PolicyNetworkDiscrete(state_dim, discrete_action_dim)
    probs = policy_discrete(states)
    print(f"   输出概率 shape: {probs.shape}")
    print(f"   概率示例: {probs[0].detach().numpy().round(3)}")
    print(f"   概率和: {probs[0].sum().item():.4f} (应该=1)")

    action, log_prob = policy_discrete.get_action(states[0])
    print(f"   采样动作: {action}, log_prob: {log_prob.item():.4f}")

    # 2. 测试连续策略网络
    print("\n2. 连续策略网络 PolicyNetworkContinuous:")
    policy_continuous = PolicyNetworkContinuous(state_dim, continuous_action_dim)
    mean, std = policy_continuous(states)
    print(f"   均值 shape: {mean.shape}")
    print(f"   标准差 shape: {std.shape}")
    print(f"   均值示例: {mean[0].detach().numpy().round(3)}")
    print(f"   标准差示例: {std[0].detach().numpy().round(3)}")

    action, log_prob = policy_continuous.get_action(states[0])
    print(f"   采样动作: {action.round(3)}")

    # 3. 测试价值网络
    print("\n3. 价值网络 ValueNetwork:")
    value_net = ValueNetwork(state_dim)
    values = value_net(states)
    print(f"   输出 shape: {values.shape}")
    print(f"   价值示例: {values.squeeze().detach().numpy().round(3)}")

    # 4. 测试Q网络（离散）
    print("\n4. Q网络（离散）:")
    q_discrete = QNetwork(state_dim, discrete_action_dim, continuous=False)
    q_values = q_discrete(states)
    print(f"   输出 shape: {q_values.shape}")
    print(f"   Q值示例: {q_values[0].detach().numpy().round(3)}")

    # 5. 测试Q网络（连续）
    print("\n5. Q网络（连续）:")
    q_continuous = QNetwork(state_dim, continuous_action_dim, continuous=True)
    actions = torch.randn(batch_size, continuous_action_dim)
    q_values = q_continuous(states, actions)
    print(f"   输出 shape: {q_values.shape}")
    print(f"   Q值示例: {q_values.squeeze().detach().numpy().round(3)}")

    print("\n" + "=" * 60)
    print("所有网络测试通过！")
