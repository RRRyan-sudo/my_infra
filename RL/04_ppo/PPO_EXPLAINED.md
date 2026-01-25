# PPO (Proximal Policy Optimization) 详解

## 概述

PPO是OpenAI在2017年提出的强化学习算法，因其简单、稳定、高效而成为应用最广泛的RL算法。
它是RLHF（人类反馈强化学习）的核心，用于训练ChatGPT、Claude等大模型。

## 为什么需要PPO？

### 问题：策略梯度的不稳定性

普通策略梯度（如REINFORCE）存在一个根本问题：

```
θ_new = θ_old + α * ∇J(θ)
```

- 如果学习率α太大：策略可能剧烈变化，导致性能崩溃
- 如果学习率α太小：学习效率低下

### TRPO的解决方案

TRPO（Trust Region Policy Optimization）提出用KL散度约束策略更新：

```
max J(θ)
s.t. KL(π_old || π_new) < δ
```

问题：需要二阶优化，实现复杂。

### PPO的解决方案

PPO用简单的Clipping代替KL约束，达到类似效果：

```
L^CLIP = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
```

## 核心公式详解

### 1. 策略比（Importance Sampling Ratio）

```
r(θ) = π_θ(a|s) / π_{θ_old}(a|s)
```

含义：
- r = 1：新旧策略对该动作的概率相同
- r > 1：新策略更倾向于该动作
- r < 1：新策略更不倾向于该动作

### 2. PPO-Clip目标函数

```
L^CLIP(θ) = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
```

其中ε通常取0.2，即r被限制在[0.8, 1.2]范围内。

### 3. 完整目标函数

```
L(θ) = L^CLIP(θ) - c1 * L^VF(θ) + c2 * H[π_θ]
```

- L^VF：Critic损失（价值函数拟合误差）
- H[π_θ]：策略熵（鼓励探索）
- c1 ≈ 0.5, c2 ≈ 0.01

## 直觉理解

想象你在优化一个策略，优势函数A告诉你某个动作好不好：

**情况1：A > 0（好动作）**
- 你想增加π(a|s)，即让r > 1
- 但clip限制r ≤ 1.2
- 防止一下子增加太多

**情况2：A < 0（差动作）**
- 你想减少π(a|s)，即让r < 1
- 但clip限制r ≥ 0.8
- 防止一下子减少太多

**结果**：策略更新被"近端"约束，不会偏离太远。

## 训练流程

```python
# PPO训练循环
for iteration in range(n_iterations):
    # 1. 收集数据
    for t in range(T):
        action = policy(state)
        next_state, reward = env.step(action)
        buffer.store(state, action, reward, log_prob, value)

    # 2. 计算GAE
    advantages = compute_gae(rewards, values, gamma, lambda)
    returns = advantages + values

    # 3. 多个epoch更新
    for epoch in range(K):
        for batch in buffer.get_batches():
            # 计算新策略的log概率
            new_log_probs = policy.log_prob(batch.states, batch.actions)

            # 策略比
            ratio = exp(new_log_probs - batch.old_log_probs)

            # PPO-Clip损失
            surr1 = ratio * batch.advantages
            surr2 = clip(ratio, 1-ε, 1+ε) * batch.advantages
            policy_loss = -min(surr1, surr2).mean()

            # Critic损失
            value_loss = MSE(critic(batch.states), batch.returns)

            # 更新
            loss = policy_loss + c1 * value_loss - c2 * entropy
            optimizer.step(loss)

    # 4. 清空buffer（on-policy）
    buffer.clear()
```

## 关键超参数

| 参数 | 典型值 | 作用 |
|------|--------|------|
| clip_epsilon | 0.2 | 控制策略更新幅度 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE参数 |
| n_epochs | 4-10 | 每批数据训练的次数 |
| batch_size | 64-256 | 小批次大小 |
| lr | 3e-4 | 学习率 |

## PPO在RLHF中的应用

```
RLHF三阶段：

阶段1: SFT（监督微调）
    基座模型 → 对话数据 → SFT模型

阶段2: 奖励模型训练
    收集人类偏好 (x, y_win, y_lose)
    训练奖励模型 r(x, y)

阶段3: PPO优化
    状态 s = 对话历史
    动作 a = 下一个token
    奖励 r = reward_model(response) - β*KL(π || π_ref)

    使用PPO优化策略
```

## 与其他算法的比较

| 算法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| REINFORCE | On-policy | 简单 | 高方差 |
| A2C | On-policy | 低方差 | 更新不稳定 |
| TRPO | On-policy | 理论保证 | 实现复杂 |
| **PPO** | On-policy | 简单+稳定 | 样本效率一般 |
| SAC | Off-policy | 样本效率高 | 仅适用于连续动作 |

## 总结

PPO成功的原因：
1. **简单**：Clipping比KL约束容易实现
2. **稳定**：限制策略更新幅度
3. **高效**：可以多次复用数据
4. **通用**：离散/连续动作都适用
5. **可扩展**：容易并行化

这就是为什么PPO成为RLHF的首选算法！
