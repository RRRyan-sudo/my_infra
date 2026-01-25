"""
大模型对齐模块 (LLM Alignment)

本模块介绍如何使用强化学习让大语言模型与人类偏好对齐。

内容：
    reward_model.py   - 奖励模型训练
    rlhf_pipeline.py  - RLHF完整流程
    dpo.py            - Direct Preference Optimization
    grpo.py           - Group Relative Policy Optimization
    ALIGNMENT_GUIDE.md - 对齐技术全景

这些是当前最热门的AI技术！
"""

from .reward_model import RewardModel
from .dpo import DPOTrainer
from .grpo import GRPOTrainer

__all__ = ['RewardModel', 'DPOTrainer', 'GRPOTrainer']
