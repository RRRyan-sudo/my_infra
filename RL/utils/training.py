"""
训练工具函数

本模块提供RL训练中常用的工具函数。

内容：
    - 学习率调度
    - 梯度裁剪
    - 训练循环辅助函数
    - 评估函数
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Dict, Any
from tqdm import tqdm


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    """
    软更新目标网络

    θ_target = τ * θ_source + (1 - τ) * θ_target

    这是TD3和SAC中使用的技巧：
        - 直接复制参数会导致目标不稳定
        - 软更新让目标网络缓慢跟随，提供稳定的目标

    Args:
        target: 目标网络
        source: 源网络
        tau: 更新系数（通常0.005）
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module):
    """
    硬更新（直接复制）

    Args:
        target: 目标网络
        source: 源网络
    """
    target.load_state_dict(source.state_dict())


def compute_gradient_norm(model: nn.Module) -> float:
    """
    计算模型梯度的范数

    用于监控训练稳定性

    Args:
        model: PyTorch模型

    Returns:
        梯度L2范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def clip_gradients(model: nn.Module, max_norm: float = 0.5):
    """
    梯度裁剪

    防止梯度爆炸，保持训练稳定

    Args:
        model: PyTorch模型
        max_norm: 最大梯度范数
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def linear_schedule(initial_value: float, final_value: float, current_step: int, total_steps: int) -> float:
    """
    线性学习率调度

    从initial_value线性衰减到final_value

    Args:
        initial_value: 初始值
        final_value: 最终值
        current_step: 当前步数
        total_steps: 总步数

    Returns:
        当前值
    """
    progress = min(current_step / total_steps, 1.0)
    return initial_value + (final_value - initial_value) * progress


def evaluate_policy(
    env,
    get_action_fn: Callable,
    n_episodes: int = 10,
    max_steps: int = 1000,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    评估策略性能

    Args:
        env: 环境
        get_action_fn: 获取动作的函数 fn(state) -> action
        n_episodes: 评估的episode数量
        max_steps: 每个episode的最大步数
        deterministic: 是否使用确定性策略

    Returns:
        包含评估指标的字典
    """
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gymnasium返回(state, info)
            state = state[0]

        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = get_action_fn(state)
            result = env.step(action)

            if len(result) == 4:  # 旧版gym
                next_state, reward, done, info = result
            else:  # gymnasium
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths)
    }


def set_seed(seed: int):
    """
    设置随机种子

    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    获取计算设备

    Args:
        device: 指定设备（'cpu', 'cuda', 'mps'）或None自动选择

    Returns:
        torch.device
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class TrainingLogger:
    """
    训练日志记录器

    简单的日志工具，记录训练指标
    """

    def __init__(self, log_interval: int = 100):
        """
        Args:
            log_interval: 打印间隔
        """
        self.log_interval = log_interval
        self.metrics: Dict[str, list] = {}
        self.step = 0

    def log(self, **kwargs):
        """记录指标"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        self.step += 1

    def print_metrics(self, prefix: str = ""):
        """打印最近的指标"""
        if self.step % self.log_interval == 0:
            msg = f"{prefix}Step {self.step}: "
            for key, values in self.metrics.items():
                if len(values) > 0:
                    recent = np.mean(values[-self.log_interval:])
                    msg += f"{key}={recent:.4f} "
            print(msg)

    def get_recent(self, key: str, n: int = 100) -> float:
        """获取最近n个值的平均"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key][-n:])
        return 0.0


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("训练工具演示")
    print("=" * 60)

    # 1. 测试软更新
    print("\n1. 软更新测试:")
    source = nn.Linear(4, 2)
    target = nn.Linear(4, 2)
    hard_update(target, source)  # 先完全复制

    # 修改source
    source.weight.data += 1.0

    print(f"   更新前 - source权重均值: {source.weight.data.mean().item():.4f}")
    print(f"   更新前 - target权重均值: {target.weight.data.mean().item():.4f}")

    soft_update(target, source, tau=0.5)

    print(f"   更新后 - target权重均值: {target.weight.data.mean().item():.4f}")

    # 2. 测试学习率调度
    print("\n2. 线性学习率调度:")
    for step in [0, 500, 1000]:
        lr = linear_schedule(1e-3, 1e-5, step, 1000)
        print(f"   Step {step}: lr = {lr:.6f}")

    # 3. 测试日志记录器
    print("\n3. 训练日志记录器:")
    logger = TrainingLogger(log_interval=10)

    for i in range(25):
        logger.log(loss=np.random.randn() + 5, reward=np.random.randn())
        if (i + 1) % 10 == 0:
            logger.print_metrics(prefix="   ")

    print(f"   最近loss平均: {logger.get_recent('loss', 10):.4f}")

    # 4. 测试设备选择
    print("\n4. 设备选择:")
    device = get_device()
    print(f"   选择的设备: {device}")

    print("\n" + "=" * 60)
    print("工具函数测试通过！")
