"""
连续控制模块 (Continuous Control)

本模块介绍用于连续动作空间的深度强化学习算法。

内容：
    sac.py  - Soft Actor-Critic（最重要）
    td3.py  - Twin Delayed DDPG

这些算法在具身智能和机器人控制中广泛应用。
"""

from .sac import SAC

__all__ = ['SAC']
