# 强化学习系统学习指南

欢迎来到强化学习学习项目！本项目帮助你从零开始系统学习RL，直到掌握当前最热门的RLHF和具身智能算法。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行第一个演示
python 01_core_concepts/mdp_essentials.py
```

## 项目结构

```
RL/
├── 01_core_concepts/      # MDP核心概念（必学）
├── 02_value_methods/      # TD学习、Q-learning（了解）
├── 03_policy_gradient/    # 策略梯度、Actor-Critic（必学）
├── 04_ppo/                # PPO算法（核心）
├── 05_llm_alignment/      # RLHF、DPO、GRPO（大模型方向）
├── 06_continuous_control/ # SAC、TD3（具身智能方向）
├── envs/                  # 学习环境
├── utils/                 # 工具函数
└── START_HERE.md          # 本文件
```

## 学习路径

### 基础阶段（必学）

```
01_core_concepts → 03_policy_gradient → 04_ppo
      ↓                    ↓               ↓
    MDP基础            策略梯度          PPO算法
   （2-3小时）         （3-4小时）       （4-6小时）
```

### 进阶方向（选择其一或都学）

**方向A：AI大模型对齐**
```
04_ppo → 05_llm_alignment
           ├── reward_model.py  奖励模型
           ├── dpo.py           DPO（推荐）
           └── grpo.py          GRPO（DeepSeek）
```

**方向B：具身智能/机器人控制**
```
04_ppo → 06_continuous_control
           └── sac.py  SAC（最常用）
```

## 每个模块学什么

| 模块 | 核心内容 | 重要程度 | 预计时间 |
|------|----------|----------|----------|
| 01 | MDP、Bellman方程 | ⭐⭐⭐⭐⭐ | 2-3h |
| 02 | TD学习、Q-learning | ⭐⭐⭐ | 1-2h |
| 03 | REINFORCE、Actor-Critic、GAE | ⭐⭐⭐⭐⭐ | 3-4h |
| 04 | PPO-Clip、连续/离散PPO | ⭐⭐⭐⭐⭐ | 4-6h |
| 05 | RLHF、DPO、GRPO | ⭐⭐⭐⭐⭐ | 6-8h |
| 06 | SAC | ⭐⭐⭐⭐⭐ | 4-6h |

## 学习建议

1. **不要跳过基础**
   - 01和03模块是理解后续所有内容的基础
   - 即使你急着学RLHF，也要先过一遍

2. **运行每个演示**
   - 每个.py文件都可以独立运行
   - 看代码输出，理解算法行为

3. **关注公式和代码的对应**
   - 每个文件都有详细的公式注释
   - 理解公式如何转化为代码

4. **按自己的方向深入**
   - 做大模型→重点学05模块
   - 做机器人→重点学06模块

## 核心算法速查

### 大模型方向

| 算法 | 用途 | 代表应用 |
|------|------|----------|
| PPO | RLHF核心 | ChatGPT, Claude |
| DPO | 简化的偏好学习 | Llama 2, Zephyr |
| GRPO | 无Critic的PPO | DeepSeek |

### 具身智能方向

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| PPO | 通用、稳定 | 仿真训练 |
| SAC | 样本效率高 | 真实机器人 |

## 开始学习！

建议从第一个模块开始：

```bash
python 01_core_concepts/mdp_essentials.py
```

祝学习愉快！
