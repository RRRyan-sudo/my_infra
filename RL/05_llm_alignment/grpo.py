"""
Module 05 - GRPO (Group Relative Policy Optimization)

GRPO是DeepSeek提出的方法，是PPO的简化版本。

核心概念：
    1. 不需要单独的Critic（价值网络）
    2. 用组内相对排名代替优势函数
    3. 更适合大模型训练

关键公式：

    传统PPO的优势函数:
    A(s,a) = Q(s,a) - V(s)  需要训练Critic

    GRPO的优势函数:
    A_i = (r_i - mean(r)) / std(r)
    对同一prompt采样多个响应，用组内相对奖励作为优势

    GRPO目标:
    L_GRPO = E[min(ratio * A, clip(ratio) * A)]
    与PPO相同，但A的计算方式不同

为什么GRPO更好？
    1. 不需要Critic：减少一半的参数和内存
    2. 更稳定：组内归一化减少方差
    3. 更简单：实现和调参更容易

应用：
    - DeepSeek-V2/V3的训练
    - 其他大规模语言模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple


# ==================== GRPO原理 ====================

def explain_grpo():
    """解释GRPO的原理"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              GRPO (Group Relative Policy Optimization)           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  PPO的问题（在大模型训练中）:                                       ║
    ║  ════════════════════════════                                    ║
    ║  1. 需要Critic网络 → 参数量翻倍                                   ║
    ║  2. Critic需要和Policy同步训练 → 不稳定                           ║
    ║  3. GAE计算复杂                                                  ║
    ║                                                                  ║
    ║  GRPO的核心思想:                                                  ║
    ║  ═══════════════                                                 ║
    ║  用"组内相对奖励"代替"优势函数"                                     ║
    ║                                                                  ║
    ║  对同一个prompt x，采样G个响应 {y_1, y_2, ..., y_G}                ║
    ║  计算每个响应的奖励 {r_1, r_2, ..., r_G}                           ║
    ║                                                                  ║
    ║  GRPO优势:                                                        ║
    ║  A_i = (r_i - mean(r)) / (std(r) + ε)                            ║
    ║                                                                  ║
    ║  这样做的好处:                                                     ║
    ║  - 不需要估计V(s)                                                 ║
    ║  - 组内比较减少奖励的绝对值影响                                      ║
    ║  - 自动归一化，更稳定                                              ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  GRPO vs PPO:                                                    ║
    ║  ═══════════                                                     ║
    ║  ┌──────────────┬─────────────────┬─────────────────┐           ║
    ║  │     特点     │      PPO        │     GRPO        │           ║
    ║  ├──────────────┼─────────────────┼─────────────────┤           ║
    ║  │ Critic网络   │      需要        │     不需要      │           ║
    ║  │ 内存占用     │      高          │     低          │           ║
    ║  │ 优势估计     │      GAE         │    组内归一化    │           ║
    ║  │ 采样要求     │      单次        │    多次/prompt  │           ║
    ║  │ 实现复杂度   │      中等        │     简单        │           ║
    ║  └──────────────┴─────────────────┴─────────────────┘           ║
    ║                                                                  ║
    ║  DeepSeek的GRPO变体:                                             ║
    ║  ═══════════════════                                             ║
    ║  还加入了一些技巧:                                                 ║
    ║  - KL正则化：防止偏离太远                                          ║
    ║  - 奖励裁剪：防止极端奖励                                          ║
    ║  - 分组策略：如何高效地分组采样                                      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== GRPO实现 ====================

class GRPOTrainer:
    """
    GRPO训练器

    简化版本，展示核心思想
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1,
        lr: float = 1e-5
    ):
        """
        Args:
            policy_model: 策略模型
            ref_model: 参考模型（用于KL计算）
            clip_epsilon: PPO裁剪参数
            kl_coef: KL正则化系数
            lr: 学习率
        """
        self.policy = policy_model
        self.ref = ref_model
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef

        # 冻结参考模型
        for param in self.ref.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算组内相对优势

        Args:
            rewards: 组内奖励，形状 (group_size,)

        Returns:
            advantages: 归一化的优势，形状 (group_size,)
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        advantages = (rewards - mean) / std
        return advantages

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,        # 新策略的log概率
        old_log_probs: torch.Tensor,    # 旧策略的log概率
        ref_log_probs: torch.Tensor,    # 参考模型的log概率
        advantages: torch.Tensor        # 组内优势
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算GRPO损失

        L = -min(ratio * A, clip(ratio) * A) + kl_coef * KL
        """
        # 策略比
        ratio = torch.exp(log_probs - old_log_probs)

        # PPO-Clip损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL正则化
        kl = (ref_log_probs - log_probs).mean()
        kl_loss = self.kl_coef * kl

        # 总损失
        total_loss = policy_loss + kl_loss

        metrics = {
            'policy_loss': policy_loss.item(),
            'kl': kl.item(),
            'ratio_mean': ratio.mean().item()
        }

        return total_loss, metrics

    def train_step(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int = 4
    ) -> Dict[str, float]:
        """
        单步训练

        Args:
            states: 状态特征，形状 (batch_size, feature_dim)
            rewards: 奖励，形状 (batch_size,)
            group_size: 每组的样本数

        Returns:
            训练指标
        """
        self.optimizer.zero_grad()

        batch_size = states.shape[0]
        n_groups = batch_size // group_size

        total_loss = 0
        total_policy_loss = 0
        total_kl = 0

        for i in range(n_groups):
            start = i * group_size
            end = start + group_size

            group_states = states[start:end]
            group_rewards = rewards[start:end]

            # 计算组内优势
            advantages = self.compute_group_advantages(group_rewards)

            # 计算log概率（简化版本）
            log_probs = self.policy(group_states).squeeze()
            with torch.no_grad():
                old_log_probs = log_probs.detach()
                ref_log_probs = self.ref(group_states).squeeze()

            # GRPO损失
            loss, metrics = self.compute_grpo_loss(
                log_probs, old_log_probs, ref_log_probs, advantages
            )

            total_loss += loss
            total_policy_loss += metrics['policy_loss']
            total_kl += metrics['kl']

        # 平均损失
        total_loss = total_loss / n_groups

        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'policy_loss': total_policy_loss / n_groups,
            'kl': total_kl / n_groups
        }


# ==================== 简化的模型 ====================

class SimplePolicy(nn.Module):
    """简化的策略模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== GRPO与其他方法的比较 ====================

def compare_methods():
    """比较不同的对齐方法"""
    comparison = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    对齐方法比较                                   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  ┌──────────┬────────┬────────┬────────┬────────┐               ║
    ║  │   方法   │  RLHF  │  DPO   │  GRPO  │  IPO   │               ║
    ║  ├──────────┼────────┼────────┼────────┼────────┤               ║
    ║  │ 奖励模型 │   需要  │  不需要 │  需要  │  不需要 │               ║
    ║  │ Critic  │   需要  │  不需要 │  不需要 │  不需要 │               ║
    ║  │ 在线采样 │   需要  │  不需要 │  需要  │  不需要 │               ║
    ║  │ 实现复杂 │   高    │   低   │   中   │   低   │               ║
    ║  │ 计算成本 │   高    │   低   │   中   │   低   │               ║
    ║  │ 训练稳定 │   中    │   高   │   高   │   高   │               ║
    ║  │ 探索能力 │   强    │   弱   │   中   │   弱   │               ║
    ║  └──────────┴────────┴────────┴────────┴────────┘               ║
    ║                                                                  ║
    ║  选择建议：                                                       ║
    ║  ═════════                                                       ║
    ║  - 资源有限 + 有偏好数据 → DPO                                    ║
    ║  - 需要在线探索 + 资源充足 → RLHF                                 ║
    ║  - 大规模训练 + 简单实现 → GRPO                                   ║
    ║  - 需要鲁棒性 → IPO                                              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(comparison)


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Module 05: GRPO (Group Relative Policy Optimization)")
    print("=" * 70)

    # 1. GRPO原理
    print("\n" + "=" * 70)
    print("1. GRPO原理")
    explain_grpo()

    # 2. 训练演示
    print("\n" + "=" * 70)
    print("2. GRPO训练演示")
    print("-" * 70)

    # 创建模型
    input_dim = 64
    policy = SimplePolicy(input_dim)
    ref = SimplePolicy(input_dim)
    ref.load_state_dict(policy.state_dict())

    # 创建训练器
    trainer = GRPOTrainer(
        policy_model=policy,
        ref_model=ref,
        clip_epsilon=0.2,
        kl_coef=0.1,
        lr=1e-3
    )

    # 模拟数据
    # 每组4个样本，共100组
    group_size = 4
    n_groups = 100
    n_samples = group_size * n_groups

    print(f"\n数据: {n_samples}个样本, {n_groups}组, 每组{group_size}个")

    # 模拟特征和奖励
    states = torch.randn(n_samples, input_dim)
    # 奖励有一定的结构（不是完全随机）
    rewards = torch.randn(n_samples) + states[:, 0]  # 与第一个特征相关

    # 训练
    print("\n开始训练...")
    n_epochs = 20

    for epoch in range(n_epochs):
        # 打乱数据（但保持组结构）
        group_indices = np.random.permutation(n_groups)

        total_loss = 0
        n_batches = 0

        # 每次处理多个组
        batch_groups = 10
        for i in range(0, n_groups, batch_groups):
            batch_group_idx = group_indices[i:i+batch_groups]

            # 收集这些组的数据
            batch_states = []
            batch_rewards = []
            for g in batch_group_idx:
                start = g * group_size
                end = start + group_size
                batch_states.append(states[start:end])
                batch_rewards.append(rewards[start:end])

            batch_states = torch.cat(batch_states)
            batch_rewards = torch.cat(batch_rewards)

            metrics = trainer.train_step(batch_states, batch_rewards, group_size)
            total_loss += metrics['loss']
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}: 损失 = {avg_loss:.4f}")

    # 3. 方法比较
    print("\n" + "=" * 70)
    print("3. 对齐方法比较")
    compare_methods()

    # 4. DeepSeek的经验
    print("\n" + "=" * 70)
    print("4. DeepSeek的GRPO经验")
    print("-" * 70)

    deepseek_experience = """
    DeepSeek-V2/V3使用GRPO的关键点：

    1. 采样策略
       - 每个prompt采样4-8个响应
       - 使用不同的温度参数
       - 保证多样性

    2. 奖励模型
       - 仍然需要训练奖励模型
       - 但不需要Critic网络
       - 奖励模型可以更大更强

    3. 训练技巧
       - 使用reward clipping防止极端值
       - 动态调整KL系数
       - 分阶段训练（先SFT，再GRPO）

    4. 效果
       - 训练更稳定
       - 内存效率更高
       - 效果不输RLHF

    这表明：不一定需要复杂的PPO + Critic
    简化的方法也能达到很好的效果！
    """
    print(deepseek_experience)

    print("\n" + "=" * 70)
    print("GRPO演示完成！")
    print("恭喜完成大模型对齐模块！")
    print("下一步: 学习 06_continuous_control/ 了解SAC和TD3")
    print("=" * 70)
