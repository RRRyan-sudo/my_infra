"""
Module 05 - DPO (Direct Preference Optimization)

DPO是2023年提出的方法，直接从偏好数据学习策略，绕过显式的奖励模型。

核心概念：
    1. 不需要训练单独的奖励模型
    2. 直接用分类损失优化策略
    3. 数学上等价于RLHF，但实现更简单

关键公式：

    DPO损失:
    L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

    展开:
    L_DPO = -E[log σ(β * (log π(y_w|x) - log π_ref(y_w|x)
                        - log π(y_l|x) + log π_ref(y_l|x)))]

    直觉理解:
    - 增大 log π(y_w|x) 相对于 log π_ref(y_w|x)：更喜欢chosen
    - 减小 log π(y_l|x) 相对于 log π_ref(y_l|x)：更不喜欢rejected
    - β控制偏离参考模型的程度

为什么DPO更简单？
    RLHF需要:
    1. 训练奖励模型
    2. 采样生成
    3. PPO优化
    4. 多个模型（policy, ref, reward, value）

    DPO只需要:
    1. 一个损失函数
    2. 监督学习式的训练

应用：
    - Llama 2 (部分)
    - Zephyr
    - Mistral
    - 许多开源模型的微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional


# ==================== DPO原理 ====================

def explain_dpo():
    """解释DPO的原理"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              DPO (Direct Preference Optimization)                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  RLHF的问题:                                                     ║
    ║  ═══════════                                                     ║
    ║  1. 需要训练单独的奖励模型                                         ║
    ║  2. 需要在线采样生成                                               ║
    ║  3. PPO训练不稳定                                                 ║
    ║  4. 需要多个模型（资源消耗大）                                       ║
    ║                                                                  ║
    ║  DPO的关键洞察:                                                   ║
    ║  ═══════════════                                                 ║
    ║  RLHF的最优解有闭式形式！                                          ║
    ║                                                                  ║
    ║  RLHF目标: max E[r(x,y)] - β*KL(π || π_ref)                      ║
    ║  最优解:   π*(y|x) ∝ π_ref(y|x) * exp(r(x,y)/β)                  ║
    ║                                                                  ║
    ║  反解奖励模型:                                                     ║
    ║  r(x,y) = β * log(π*(y|x) / π_ref(y|x)) + const                  ║
    ║                                                                  ║
    ║  代入Bradley-Terry模型:                                           ║
    ║  P(y_w > y_l) = σ(r_w - r_l)                                     ║
    ║              = σ(β * (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))  ║
    ║                                                                  ║
    ║  这就是DPO损失的来源！                                             ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  DPO vs RLHF:                                                    ║
    ║  ═══════════                                                     ║
    ║  ┌──────────────┬─────────────────┬─────────────────┐           ║
    ║  │     特点     │      RLHF       │      DPO        │           ║
    ║  ├──────────────┼─────────────────┼─────────────────┤           ║
    ║  │ 奖励模型     │      需要        │     不需要      │           ║
    ║  │ 在线采样     │      需要        │     不需要      │           ║
    ║  │ 训练稳定性   │      较难        │     简单        │           ║
    ║  │ 理论基础     │      完整        │     等价        │           ║
    ║  │ 计算资源     │      高          │     低          │           ║
    ║  │ 代码复杂度   │      高          │     低          │           ║
    ║  └──────────────┴─────────────────┴─────────────────┘           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== DPO实现 ====================

class DPOTrainer:
    """
    DPO训练器

    简化版本，用于理解核心概念。
    实际应用中会用HuggingFace TRL库。
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        lr: float = 1e-5
    ):
        """
        Args:
            policy_model: 待训练的策略模型
            ref_model: 参考模型（冻结）
            beta: KL正则化系数
            lr: 学习率
        """
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta

        # 冻结参考模型
        for param in self.ref.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_dpo_loss(
        self,
        chosen_log_probs: torch.Tensor,     # log π(y_w|x)
        rejected_log_probs: torch.Tensor,   # log π(y_l|x)
        chosen_ref_log_probs: torch.Tensor, # log π_ref(y_w|x)
        rejected_ref_log_probs: torch.Tensor # log π_ref(y_l|x)
    ) -> torch.Tensor:
        """
        计算DPO损失

        L_DPO = -log σ(β * (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))
        """
        # 计算log比值
        chosen_log_ratio = chosen_log_probs - chosen_ref_log_probs
        rejected_log_ratio = rejected_log_probs - rejected_ref_log_probs

        # DPO损失
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        return loss

    def train_step(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        单步训练（简化版本）

        实际中需要计算真实的log概率，这里用简化版本演示
        """
        self.optimizer.zero_grad()

        # 计算策略模型的"log概率"（简化：用线性输出模拟）
        chosen_log_probs = self.policy(chosen_features).squeeze()
        rejected_log_probs = self.policy(rejected_features).squeeze()

        # 计算参考模型的"log概率"
        with torch.no_grad():
            chosen_ref_log_probs = self.ref(chosen_features).squeeze()
            rejected_ref_log_probs = self.ref(rejected_features).squeeze()

        # DPO损失
        loss = self.compute_dpo_loss(
            chosen_log_probs,
            rejected_log_probs,
            chosen_ref_log_probs,
            rejected_ref_log_probs
        )

        loss.backward()
        self.optimizer.step()

        # 计算一些统计量
        with torch.no_grad():
            chosen_reward = self.beta * (chosen_log_probs - chosen_ref_log_probs)
            rejected_reward = self.beta * (rejected_log_probs - rejected_ref_log_probs)
            accuracy = (chosen_reward > rejected_reward).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'chosen_reward': chosen_reward.mean().item(),
            'rejected_reward': rejected_reward.mean().item()
        }


# ==================== 简化的模型 ====================

class SimpleModel(nn.Module):
    """简化的模型，用于演示DPO"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ==================== DPO变体 ====================

def explain_dpo_variants():
    """解释DPO的变体"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                        DPO变体                                   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. IPO (Identity Preference Optimization)                       ║
    ║     ═══════════════════════════════════                          ║
    ║     - 解决DPO可能过拟合的问题                                      ║
    ║     - 使用更鲁棒的损失函数                                         ║
    ║     L_IPO = (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l) - 1/β)²  ║
    ║                                                                  ║
    ║  2. KTO (Knowledge Transfer Optimization)                        ║
    ║     ══════════════════════════════════════                       ║
    ║     - 不需要成对的偏好数据                                         ║
    ║     - 只需要"好"和"坏"的标签                                       ║
    ║     - 适合数据更容易获取的场景                                      ║
    ║                                                                  ║
    ║  3. cDPO (Conservative DPO)                                      ║
    ║     ═════════════════════════                                    ║
    ║     - 添加额外的正则化                                            ║
    ║     - 防止策略偏离太远                                            ║
    ║                                                                  ║
    ║  4. SimPO (Simple Preference Optimization)                       ║
    ║     ═══════════════════════════════════════                      ║
    ║     - 不需要参考模型                                              ║
    ║     - 使用长度归一化                                              ║
    ║     - 更简单的实现                                                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Module 05: DPO (Direct Preference Optimization)")
    print("=" * 70)

    # 1. DPO原理
    print("\n" + "=" * 70)
    print("1. DPO原理")
    explain_dpo()

    # 2. 训练演示
    print("\n" + "=" * 70)
    print("2. DPO训练演示")
    print("-" * 70)

    # 创建模型
    input_dim = 64
    policy_model = SimpleModel(input_dim)
    ref_model = SimpleModel(input_dim)

    # 复制参数（初始时policy = ref）
    ref_model.load_state_dict(policy_model.state_dict())

    # 创建DPO训练器
    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        beta=0.1,
        lr=1e-3
    )

    # 模拟偏好数据
    n_samples = 1000
    chosen_features = torch.randn(n_samples, input_dim) + 0.3
    rejected_features = torch.randn(n_samples, input_dim) - 0.3

    # 训练
    print("\n开始训练...")
    batch_size = 64
    n_epochs = 20

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        total_loss = 0
        total_acc = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            metrics = trainer.train_step(
                chosen_features[batch_idx],
                rejected_features[batch_idx]
            )
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: 损失 = {avg_loss:.4f}, 准确率 = {avg_acc:.2%}")

    # 3. DPO变体
    print("\n" + "=" * 70)
    print("3. DPO变体")
    explain_dpo_variants()

    # 4. DPO vs RLHF的选择
    print("\n" + "=" * 70)
    print("4. 什么时候用DPO？什么时候用RLHF？")
    print("-" * 70)

    comparison = """
    选择DPO的情况：
    ═══════════════
    - 有高质量的偏好数据
    - 计算资源有限
    - 需要快速迭代
    - 任务相对简单

    选择RLHF的情况：
    ════════════════
    - 需要在线采样和探索
    - 奖励函数复杂或多目标
    - 需要更细粒度的控制
    - 有足够的计算资源

    实践中的趋势：
    ═══════════════
    - 很多团队先用DPO快速迭代
    - 然后用RLHF做最后的精调
    - 或者完全只用DPO（如Zephyr）
    """
    print(comparison)

    print("\n" + "=" * 70)
    print("DPO演示完成！")
    print("下一步: 学习 grpo.py 了解DeepSeek使用的GRPO")
    print("=" * 70)
