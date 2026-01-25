"""
Module 03 - 广义优势估计 (GAE)

GAE (Generalized Advantage Estimation) 是PPO中计算优势函数的核心方法。

核心概念：
    - 优势函数 A(s,a) = Q(s,a) - V(s) 衡量动作比平均好多少
    - GAE平衡了偏差（bias）和方差（variance）
    - 通过参数λ控制这个平衡

关键公式：

    TD误差（单步优势估计）:
    δ_t = r_t + γ*V(s_{t+1}) - V(s_t)

    GAE（多步加权平均）:
    A^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

    展开形式:
    A^GAE = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...

    特殊情况:
    - λ=0: A = δ_t （单步TD，高偏差低方差）
    - λ=1: A = G_t - V(s_t) （蒙特卡洛，低偏差高方差）

为什么重要？
    - PPO使用GAE计算优势函数
    - 是on-policy算法的标准做法
    - λ=0.95是常用默认值

与大模型的关联：
    - RLHF训练时用GAE估计每个token的优势
    - 指导模型哪些生成方式更好
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import sys
sys.path.append("../..")


# ==================== GAE解释 ====================

def explain_gae():
    """解释GAE的原理"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  GAE (Generalized Advantage Estimation)          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  问题：如何估计优势函数 A(s,a)？                                     ║
    ║  ═══════════════════════════════                                 ║
    ║                                                                  ║
    ║  方法1: 蒙特卡洛（MC）                                             ║
    ║         A_MC = G_t - V(s_t) = Σγ^k*r_{t+k} - V(s_t)              ║
    ║         优点：无偏                                                 ║
    ║         缺点：方差大（包含整个轨迹的随机性）                          ║
    ║                                                                  ║
    ║  方法2: TD(0)（单步自举）                                          ║
    ║         A_TD = r_t + γV(s_{t+1}) - V(s_t) = δ_t                  ║
    ║         优点：方差小                                               ║
    ║         缺点：有偏（V的估计误差会传播）                               ║
    ║                                                                  ║
    ║  GAE: 在两者之间取得平衡                                           ║
    ║       A^GAE = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...             ║
    ║               ↑                                                  ║
    ║            λ控制平衡                                              ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  λ的作用：                                                        ║
    ║  ═════════                                                       ║
    ║  λ = 0: A = δ_t                                                  ║
    ║         只看一步，高偏差，低方差                                     ║
    ║         （完全依赖Critic的估计）                                    ║
    ║                                                                  ║
    ║  λ = 1: A = δ_t + γδ_{t+1} + γ²δ_{t+2} + ...                     ║
    ║         = r_t + γr_{t+1} + ... - V(s_t)                          ║
    ║         = G_t - V(s_t) = MC估计                                  ║
    ║         低偏差，高方差                                             ║
    ║         （完全依赖采样的回报）                                       ║
    ║                                                                  ║
    ║  λ = 0.95（常用）: 平衡偏差和方差                                   ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== GAE实现 ====================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算GAE优势估计和回报

    这是PPO中最重要的辅助函数之一。

    Args:
        rewards: 奖励序列，形状 (T,)
        values: 价值估计序列，形状 (T,)，V(s_0), V(s_1), ..., V(s_{T-1})
        dones: 终止标志序列，形状 (T,)
        next_value: 最后一个状态的价值估计 V(s_T)
        gamma: 折扣因子
        gae_lambda: GAE参数λ

    Returns:
        advantages: GAE优势估计，形状 (T,)
        returns: 目标回报 = advantages + values，形状 (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T)

    # 从后往前计算（更高效）
    gae = 0
    for t in reversed(range(T)):
        # 下一步的价值
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # 如果done=1，则下一步价值为0（episode终止）
        next_val = next_val * (1 - dones[t])

        # TD误差
        delta = rewards[t] + gamma * next_val - values[t]

        # GAE累积
        # A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    # 回报 = 优势 + 价值
    returns = advantages + values

    return advantages, returns


def compute_gae_pytorch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch版本的GAE计算

    用于需要梯度的场景（如PPO实现中）
    """
    T = len(rewards)
    advantages = torch.zeros(T)

    gae = torch.tensor(0.0)
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_val = next_val * (1 - dones[t])

        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ==================== 可视化GAE ====================

def visualize_gae_effect():
    """可视化不同λ值对GAE的影响"""
    print("\n不同λ值对优势估计的影响:")
    print("-" * 60)

    # 模拟一个简单轨迹
    rewards = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    values = np.array([5.0, 4.5, 4.0, 3.0, 2.0])
    dones = np.array([0, 0, 0, 0, 1])
    next_value = 0.0
    gamma = 0.99

    print(f"奖励序列: {rewards}")
    print(f"价值估计: {values}")

    # 不同λ值
    for lam in [0.0, 0.5, 0.95, 1.0]:
        advantages, returns = compute_gae(rewards, values, dones, next_value, gamma, lam)
        print(f"\nλ = {lam}:")
        print(f"  优势: {advantages.round(2)}")
        print(f"  方差: {advantages.var():.4f}")


# ==================== GAE在PPO中的使用 ====================

def explain_gae_in_ppo():
    """解释GAE在PPO中如何使用"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    GAE在PPO中的使用                               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  PPO的训练流程：                                                   ║
    ║  ══════════════                                                  ║
    ║                                                                  ║
    ║  1. 收集数据                                                      ║
    ║     for t in range(T):                                           ║
    ║         action = policy(state)                                   ║
    ║         next_state, reward = env.step(action)                    ║
    ║         store(state, action, reward, value, log_prob)            ║
    ║                                                                  ║
    ║  2. 计算GAE  ← 这一步使用GAE                                       ║
    ║     advantages = compute_gae(rewards, values, dones, gamma, λ)   ║
    ║     returns = advantages + values                                ║
    ║                                                                  ║
    ║  3. 标准化优势（减少方差）                                          ║
    ║     advantages = (advantages - mean) / (std + ε)                 ║
    ║                                                                  ║
    ║  4. PPO更新（多个epoch）                                          ║
    ║     for epoch in range(K):                                       ║
    ║         for batch in mini_batches:                               ║
    ║             # Actor损失：用advantages指导更新                       ║
    ║             actor_loss = -min(ratio * adv, clip(ratio) * adv)    ║
    ║                                                                  ║
    ║             # Critic损失：用returns作为目标                         ║
    ║             critic_loss = (V(s) - returns)²                      ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  关键点：                                                         ║
    ║  ═══════                                                         ║
    ║  - GAE用于计算Actor更新时的"信号强度"                               ║
    ║  - advantages > 0: 这个动作好，增加概率                             ║
    ║  - advantages < 0: 这个动作差，减少概率                             ║
    ║  - returns用于训练Critic（V函数的目标值）                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== n-step returns对比 ====================

def explain_nstep_returns():
    """解释n-step returns与GAE的关系"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                n-step Returns vs GAE                             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  n-step回报（n步自举）：                                           ║
    ║  ═══════════════════════                                         ║
    ║  G_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})║
    ║                                                                  ║
    ║  n=1: G_t^(1) = r_t + γV(s_{t+1})  (TD(0))                       ║
    ║  n=2: G_t^(2) = r_t + γr_{t+1} + γ²V(s_{t+2})                    ║
    ║  n=∞: G_t^(∞) = r_t + γr_{t+1} + ...  (MC)                       ║
    ║                                                                  ║
    ║  GAE可以理解为n-step returns的加权平均：                            ║
    ║  ════════════════════════════════════                            ║
    ║  A^GAE = (1-λ)[A^(1) + λA^(2) + λ²A^(3) + ...]                   ║
    ║                                                                  ║
    ║  其中 A^(n) = G_t^(n) - V(s_t) 是n步优势估计                       ║
    ║                                                                  ║
    ║  这就是"广义"的含义：它是所有n-step估计的指数加权平均                  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Module 03: 广义优势估计 (GAE)")
    print("=" * 70)

    # 1. GAE原理
    print("\n" + "=" * 70)
    print("1. GAE原理")
    explain_gae()

    # 2. 不同λ的影响
    print("\n" + "=" * 70)
    print("2. 不同λ值的影响")
    visualize_gae_effect()

    # 3. n-step returns
    print("\n" + "=" * 70)
    print("3. n-step Returns与GAE的关系")
    explain_nstep_returns()

    # 4. GAE在PPO中的使用
    print("\n" + "=" * 70)
    print("4. GAE在PPO中的使用")
    explain_gae_in_ppo()

    # 5. 实际计算示例
    print("\n" + "=" * 70)
    print("5. 实际计算示例")
    print("-" * 70)

    # 创建一个简单的轨迹
    print("\n模拟一个10步轨迹:")

    T = 10
    rewards = np.random.randn(T) + 1  # 平均奖励为1
    values = np.linspace(10, 1, T)    # 递减的价值估计
    dones = np.zeros(T)
    dones[-1] = 1  # 最后一步终止
    next_value = 0.0

    print(f"奖励: {rewards.round(2)}")
    print(f"价值: {values.round(2)}")

    # 计算GAE
    advantages, returns = compute_gae(
        rewards, values, dones, next_value,
        gamma=0.99, gae_lambda=0.95
    )

    print(f"\nGAE优势: {advantages.round(2)}")
    print(f"目标回报: {returns.round(2)}")

    # 标准化优势
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    print(f"\n标准化优势: {advantages_normalized.round(2)}")
    print(f"标准化后均值: {advantages_normalized.mean():.4f} (应该接近0)")
    print(f"标准化后标准差: {advantages_normalized.std():.4f} (应该接近1)")

    # 6. GAE的重要性
    print("\n" + "=" * 70)
    print("6. GAE的重要性")
    print("-" * 70)

    importance = """
    为什么GAE很重要？
    ═════════════════

    1. 平衡偏差和方差
       - λ可调节，适应不同任务
       - 经验上λ=0.95是个好默认值

    2. 与V函数协同
       - Critic学习V(s)
       - GAE用V计算优势
       - 优势指导Actor更新
       - 形成正反馈循环

    3. 数值稳定
       - 标准化后优势均值为0
       - 避免策略更新过大或过小

    4. 广泛应用
       - PPO、A2C、TRPO都使用GAE
       - 是现代on-policy算法的标配
    """
    print(importance)

    print("\n" + "=" * 70)
    print("GAE演示完成！")
    print("下一步: 学习 04_ppo/ 了解PPO的完整实现")
    print("=" * 70)
