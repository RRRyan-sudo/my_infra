# 强化学习学习路径图

## 总览

```
                    ┌─────────────────────────────────────────┐
                    │             强化学习学习路径              │
                    └─────────────────────────────────────────┘

                              ┌──────────────┐
                              │ 01 MDP核心   │
                              │   概念       │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │ 02 价值方法   │ (可快速过)
                              │ TD/Q-learning│
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │ 03 策略梯度   │
                              │REINFORCE/A-C │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │   04 PPO     │ ← 核心！
                              │  (两方向交汇) │
                              └──────┬───────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
             ┌──────▼───────┐                 ┌──────▼───────┐
             │ 05 大模型对齐 │                 │ 06 连续控制   │
             │ RLHF/DPO/GRPO│                 │   SAC/TD3    │
             └──────────────┘                 └──────────────┘
                    │                                 │
                    ▼                                 ▼
              ChatGPT/Claude                    机器人控制
              大模型训练                          具身智能
```

## 详细学习计划

### 第一阶段：基础（必修）

**Day 1: MDP核心概念**
```
01_core_concepts/
├── mdp_essentials.py    学习MDP五元组、策略、回报
└── bellman_equations.py 学习Bellman方程、V和Q函数
```

**Day 2: 价值方法（选修）**
```
02_value_methods/
└── td_qlearning.py     了解TD学习、Q-learning
                        （可快速浏览，重点理解"自举"思想）
```

**Day 3-4: 策略梯度**
```
03_policy_gradient/
├── reinforce.py        学习最基础的策略梯度
├── actor_critic.py     学习Actor-Critic架构
└── gae.py              学习GAE（PPO的关键组件）
```

### 第二阶段：PPO（核心）

**Day 5-6: PPO**
```
04_ppo/
├── PPO_EXPLAINED.md    先读原理讲解
├── ppo_discrete.py     实现离散动作PPO
└── ppo_continuous.py   实现连续动作PPO
```

### 第三阶段：选择方向

**方向A: 大模型对齐 (Day 7-9)**
```
05_llm_alignment/
├── reward_model.py     学习奖励模型训练
├── dpo.py              学习DPO（重点！）
└── grpo.py             学习GRPO
```

**方向B: 具身智能 (Day 7-9)**
```
06_continuous_control/
└── sac.py              学习SAC（重点！）
```

## 核心算法关系图

```
                         强化学习算法演进
                         ═══════════════

基础：
    MDP → 动态规划 → 蒙特卡洛 → TD学习 → Q-learning

策略梯度：
    REINFORCE → Actor-Critic → A2C → TRPO → PPO
        │                               │
        │                               ▼
        │                          大模型对齐
        │                          ├── RLHF (PPO + RM)
        │                          ├── DPO (直接偏好)
        │                          └── GRPO (无Critic)
        │
        └─────────────────────────────────────┐
                                              │
连续控制：                                     │
    DDPG → TD3 → SAC ◄────────────────────────┘
              │
              ▼
          具身智能
          ├── 机械臂控制
          ├── 四足机器人
          └── 自动驾驶
```

## 知识点检查清单

### 基础概念
- [ ] 理解MDP五元组 (S, A, P, R, γ)
- [ ] 理解V(s)和Q(s,a)的含义
- [ ] 理解Bellman方程
- [ ] 理解策略梯度定理
- [ ] 理解Actor-Critic架构
- [ ] 理解GAE

### PPO
- [ ] 理解策略比 r(θ)
- [ ] 理解PPO-Clip的工作原理
- [ ] 能实现离散/连续PPO
- [ ] 理解PPO的关键超参数

### 大模型对齐
- [ ] 理解RLHF三阶段
- [ ] 理解奖励模型的训练
- [ ] 理解DPO的数学推导
- [ ] 理解GRPO与PPO的区别

### 连续控制
- [ ] 理解最大熵RL
- [ ] 理解SAC的三个网络
- [ ] 理解自动温度调节

## 推荐资源

### 论文
1. PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
2. SAC: "Soft Actor-Critic" (Haarnoja et al., 2018)
3. DPO: "Direct Preference Optimization" (Rafailov et al., 2023)
4. GRPO: "DeepSeekMath" (Shao et al., 2024)

### 书籍
1. "Reinforcement Learning: An Introduction" (Sutton & Barto) - RL圣经
2. "Spinning Up in Deep RL" (OpenAI) - 实践指南

### 代码库
1. CleanRL - 干净的单文件RL实现
2. Stable-Baselines3 - 工业级RL库
3. TRL - HuggingFace的RLHF库

祝学习顺利！
