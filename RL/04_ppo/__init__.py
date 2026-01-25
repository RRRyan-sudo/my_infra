"""
PPO模块 (Proximal Policy Optimization)

PPO是当前最流行的强化学习算法之一，是RLHF的核心。

内容：
    ppo_discrete.py   - 离散动作PPO
    ppo_continuous.py - 连续动作PPO
    PPO_EXPLAINED.md  - 详细原理讲解

学习顺序：先读PPO_EXPLAINED.md理解原理，再看代码实现
"""

from .ppo_discrete import PPODiscrete
from .ppo_continuous import PPOContinuous

__all__ = ['PPODiscrete', 'PPOContinuous']
