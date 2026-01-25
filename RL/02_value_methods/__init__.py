"""
价值方法模块

本模块介绍基于价值的强化学习方法。

内容：
    td_qlearning.py - TD学习、Q-learning、SARSA

这些是RL的经典算法，帮助理解"自举"的核心思想。
在现代深度RL中，价值方法的思想仍然重要（如Actor-Critic中的Critic）。
"""

from .td_qlearning import *
