"""
Module 05 - 奖励模型 (Reward Model)

奖励模型是RLHF的核心组件，用于预测人类对模型回复的偏好。

核心概念：
    1. Bradley-Terry模型：将偏好建模为概率
    2. 偏好数据：(prompt, response_win, response_lose)
    3. 奖励模型：r(prompt, response) → scalar

关键公式：

    Bradley-Terry偏好模型:
    P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

    其中 σ 是sigmoid函数:
    σ(z) = 1 / (1 + exp(-z))

    奖励模型损失:
    L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]

    直觉理解:
    - 如果r(y_w) >> r(y_l)，则P(y_w > y_l) ≈ 1，损失≈0
    - 如果r(y_w) << r(y_l)，则P(y_w > y_l) ≈ 0，损失很大

为什么需要奖励模型？
    - 直接让人类给每个回复打分成本太高
    - 训练奖励模型来模拟人类偏好
    - RLHF用奖励模型指导PPO优化

应用：
    - InstructGPT/ChatGPT的RLHF
    - Claude的Constitutional AI
    - Llama 2的RLHF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ==================== 偏好数据结构 ====================

@dataclass
class PreferenceData:
    """偏好数据"""
    prompt: str
    chosen: str      # 被选中的回复（y_win）
    rejected: str    # 被拒绝的回复（y_lose）


# ==================== 奖励模型 ====================

class RewardModel(nn.Module):
    """
    奖励模型

    在实际应用中，奖励模型通常基于预训练语言模型：
    - 输入：prompt + response
    - 输出：标量奖励值

    这里用简化版本演示核心思想。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128
    ):
        """
        简化的奖励模型

        Args:
            input_dim: 输入维度（实际是embedding维度）
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 简单的MLP奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量奖励
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算奖励

        Args:
            x: 输入特征，形状 (batch_size, input_dim)

        Returns:
            reward: 奖励值，形状 (batch_size, 1)
        """
        return self.reward_head(x)


def compute_preference_loss(
    reward_model: RewardModel,
    chosen_features: torch.Tensor,
    rejected_features: torch.Tensor
) -> torch.Tensor:
    """
    计算偏好损失

    Bradley-Terry模型:
    L = -log σ(r(chosen) - r(rejected))

    Args:
        reward_model: 奖励模型
        chosen_features: 被选中回复的特征
        rejected_features: 被拒绝回复的特征

    Returns:
        loss: 偏好损失
    """
    # 计算奖励
    r_chosen = reward_model(chosen_features)
    r_rejected = reward_model(rejected_features)

    # Bradley-Terry损失
    # P(chosen > rejected) = σ(r_chosen - r_rejected)
    # loss = -log(P) = -log σ(r_chosen - r_rejected)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()

    return loss


# ==================== 奖励模型训练 ====================

class RewardModelTrainer:
    """
    奖励模型训练器

    训练流程：
    1. 收集偏好数据 (prompt, chosen, rejected)
    2. 将文本转换为特征（实际用LLM embedding）
    3. 最小化Bradley-Terry损失
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-4
    ):
        self.reward_model = RewardModel(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=lr)

    def train_step(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor
    ) -> float:
        """
        单步训练

        Args:
            chosen_features: 被选中回复的特征
            rejected_features: 被拒绝回复的特征

        Returns:
            loss值
        """
        self.optimizer.zero_grad()

        loss = compute_preference_loss(
            self.reward_model,
            chosen_features,
            rejected_features
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_accuracy(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor
    ) -> float:
        """
        计算偏好预测准确率

        如果r(chosen) > r(rejected)，则预测正确
        """
        with torch.no_grad():
            r_chosen = self.reward_model(chosen_features)
            r_rejected = self.reward_model(rejected_features)

            correct = (r_chosen > r_rejected).float().mean().item()

        return correct


# ==================== 实际应用说明 ====================

def explain_reward_model_in_practice():
    """解释实际中的奖励模型"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                 实际中的奖励模型 (Reward Model)                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  架构：                                                           ║
    ║  ═════                                                           ║
    ║  ┌────────────────────────────────────────┐                      ║
    ║  │ Pretrained LLM (e.g., LLaMA, GPT)      │                      ║
    ║  │ [prompt + response] → hidden states    │                      ║
    ║  └──────────────┬─────────────────────────┘                      ║
    ║                 │                                                ║
    ║                 ▼                                                ║
    ║  ┌────────────────────────────────────────┐                      ║
    ║  │ Reward Head (Linear + Scalar output)   │                      ║
    ║  │ hidden states → reward (scalar)        │                      ║
    ║  └────────────────────────────────────────┘                      ║
    ║                                                                  ║
    ║  训练数据：                                                        ║
    ║  ═════════                                                       ║
    ║  - 人工标注的偏好对                                                ║
    ║  - 格式：(prompt, chosen_response, rejected_response)            ║
    ║  - 规模：通常数万到数十万对                                         ║
    ║                                                                  ║
    ║  训练技巧：                                                        ║
    ║  ═════════                                                       ║
    ║  1. 从SFT模型初始化（而不是基座模型）                               ║
    ║  2. 只训练reward head（冻结LLM）或低秩适配                         ║
    ║  3. 数据质量比数量更重要                                           ║
    ║                                                                  ║
    ║  InstructGPT的经验：                                              ║
    ║  ═══════════════════                                             ║
    ║  - 6B参数的奖励模型                                                ║
    ║  - 33K偏好对训练                                                  ║
    ║  - 标注一致性约73%（人类之间也有分歧）                              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Module 05: 奖励模型 (Reward Model)")
    print("=" * 70)

    # 1. Bradley-Terry模型解释
    print("\n" + "=" * 70)
    print("1. Bradley-Terry偏好模型")
    print("-" * 70)

    bt_explanation = """
    Bradley-Terry模型：

    假设每个回复有一个"潜在质量"r(x, y)
    那么人类选择y_w而不是y_l的概率为：

    P(y_w > y_l | x) = exp(r_w) / (exp(r_w) + exp(r_l))
                     = 1 / (1 + exp(-(r_w - r_l)))
                     = σ(r_w - r_l)

    其中σ是sigmoid函数。

    这个模型的优点：
    - 只需要相对偏好，不需要绝对评分
    - 数学形式简单，易于优化
    - 与人类决策过程相符
    """
    print(bt_explanation)

    # 2. 训练演示
    print("\n" + "=" * 70)
    print("2. 奖励模型训练演示")
    print("-" * 70)

    # 模拟训练数据
    input_dim = 64
    n_samples = 1000

    # 创建训练器
    trainer = RewardModelTrainer(input_dim=input_dim, hidden_dim=128, lr=1e-3)

    # 模拟偏好数据
    # 假设chosen_features比rejected_features"质量更高"
    print("\n模拟训练数据...")
    chosen_features = torch.randn(n_samples, input_dim) + 0.5  # 偏高
    rejected_features = torch.randn(n_samples, input_dim) - 0.5  # 偏低

    # 训练前准确率
    init_acc = trainer.compute_accuracy(chosen_features, rejected_features)
    print(f"训练前准确率: {init_acc:.2%}")

    # 训练
    print("\n开始训练...")
    batch_size = 64
    n_epochs = 10

    for epoch in range(n_epochs):
        # 随机打乱
        indices = np.random.permutation(n_samples)
        total_loss = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            loss = trainer.train_step(
                chosen_features[batch_idx],
                rejected_features[batch_idx]
            )
            total_loss += loss

        avg_loss = total_loss / (n_samples // batch_size)
        acc = trainer.compute_accuracy(chosen_features, rejected_features)
        print(f"Epoch {epoch+1}: 损失 = {avg_loss:.4f}, 准确率 = {acc:.2%}")

    # 3. 实际应用
    print("\n" + "=" * 70)
    print("3. 实际应用中的奖励模型")
    explain_reward_model_in_practice()

    # 4. 奖励模型的局限性
    print("\n" + "=" * 70)
    print("4. 奖励模型的局限性和改进")
    print("-" * 70)

    limitations = """
    局限性：
    ═════════

    1. 奖励黑客 (Reward Hacking)
       - 模型可能找到"欺骗"奖励模型的方式
       - 生成奖励模型认为好但实际不好的回复
       - 解决：KL约束、多奖励模型集成

    2. 分布偏移 (Distribution Shift)
       - 训练数据的分布与实际使用时不同
       - 奖励模型在新分布上可能不准确
       - 解决：持续收集数据、迭代训练

    3. 标注噪声
       - 人类标注者之间也有分歧
       - 某些偏好可能是个人化的
       - 解决：多标注者、置信度加权

    改进方向：
    ═══════════

    1. DPO (Direct Preference Optimization)
       - 完全绕过奖励模型
       - 直接从偏好数据学习策略

    2. GRPO (Group Relative Policy Optimization)
       - 不需要单独的奖励模型
       - 用组内相对排名作为信号

    3. Constitutional AI
       - 用AI自己来评判回复质量
       - 减少人工标注需求
    """
    print(limitations)

    print("\n" + "=" * 70)
    print("奖励模型演示完成！")
    print("下一步: 学习 rlhf_pipeline.py 了解完整的RLHF流程")
    print("       或 dpo.py 了解如何绕过奖励模型")
    print("=" * 70)
