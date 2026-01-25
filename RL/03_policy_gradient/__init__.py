"""
策略梯度模块

本模块介绍策略梯度方法，这是PPO和RLHF的理论基础。

内容：
    reinforce.py    - REINFORCE算法（最简单的策略梯度）
    actor_critic.py - Actor-Critic框架
    gae.py          - 广义优势估计（GAE）

学习顺序：reinforce → actor_critic → gae
"""

from .reinforce import REINFORCE
from .actor_critic import ActorCritic
from .gae import compute_gae

__all__ = ['REINFORCE', 'ActorCritic', 'compute_gae']
