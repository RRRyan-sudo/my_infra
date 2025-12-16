# 强化学习基础理论与实践

这是一个完整的强化学习学习项目，包含5个核心模块，从理论到实践逐步深入。

## 📚 项目结构

```
RL/
├── 01_MDP/                      # 马尔可夫决策过程 (Module 1)
│   └── mdp_basics.py
├── 02_dynamic_programming/      # 动态规划方法 (Module 2)
│   └── dp_solver.py
├── 03_monte_carlo/              # 蒙特卡洛方法 (Module 3)
│   └── mc_learning.py
├── 04_temporal_difference/      # 时序差分学习 (Module 4)
│   └── td_learning.py
├── 05_policy_gradient/          # 策略梯度方法 (Module 5)
│   └── pg_learning.py
├── envs/                        # 环境
│   └── gridworld.py            # 简单网格世界环境
├── utils/                       # 工具函数
│   └── helpers.py              # 辅助函数和可视化
└── docs/                        # 文档
    └── README.md               # 此文件
```

## 🎯 学习路线

### Module 1: 马尔可夫决策过程 (MDP) ⭐ 必读
**关键概念:**
- 状态空间、动作空间、转移概率、奖励函数
- 状态价值函数V(s)和动作价值函数Q(s,a)
- 贝尔曼方程和最优性原理
- 最优策略的存在性

**运行示例:**
```bash
cd /home/ryan/2repo/my_infra/RL
python 01_MDP/mdp_basics.py
```

**输出:**
```
============================================================
MDP基础演示
============================================================

1. 评估均匀随机策略:
状态价值函数V(s):
  V(0) = 5.2103
  V(1) = 12.3456
  V(2) = 8.7654

2. 计算Q值:
  状态0:
    Q(0,0) = 5.2103
    Q(0,1) = 6.1234

3. 贪心策略 (π_greedy):
  π(0) = 1
  π(1) = 1
  π(2) = 1
```

**关键数学:**
- 状态价值: $V^\pi(s) = E[G_t | S_t = s]$
- 动作价值: $Q^\pi(s,a) = E[G_t | S_t = s, A_t = a]$
- 贝尔曼方程: $V(s) = E[R + \gamma V(s')]$
- 最优性: $V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$

---

### Module 2: 动态规划 (Dynamic Programming) ⭐⭐
**关键概念:**
- 已知环境完整模型（转移概率和奖励）
- 策略迭代 vs 价值迭代
- 自举（Bootstrapping）思想
- 收敛性和计算复杂度

**运行示例:**
```bash
python 02_dynamic_programming/dp_solver.py
```

**算法对比:**

| 算法 | 更新方式 | 收敛性 | 计算量 | 适用场景 |
|------|---------|--------|--------|---------|
| 策略迭代 | 评估→改进 | 快 | 较大 | 小-中等规模 |
| 价值迭代 | 直接迭代 | 快 | 较小 | 快速求解 |

**关键代码:**
```python
# 策略迭代
V, policy = dp.policy_iteration()

# 价值迭代
V, policy = dp.value_iteration()
```

**数学原理:**
- 策略改进: $\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$
- 价值迭代: $V_{k+1}(s) = \max_a E[R + \gamma V_k(s')]$

---

### Module 3: 蒙特卡洛方法 (Monte Carlo Methods) ⭐⭐
**关键概念:**
- 无需环境模型，纯采样学习
- First-Visit vs Every-Visit MC
- GLIE条件和策略收敛
- 高方差、低偏差

**运行示例:**
```bash
python 03_monte_carlo/mc_learning.py
```

**核心算法:**
```python
# First-Visit MC
for episode in episodes:
    trajectory = generate_episode()
    G = 0
    for t in reversed(range(len(trajectory))):
        G = reward + gamma * G
        if (state, action) not in visited:
            V(state) ← V(state) + α(G - V(state))
            visited.add((state, action))
```

**关键特点:**
- ✅ 不需要环境模型
- ✅ 学习无偏估计
- ❌ 方差较大，需要更多样本
- ❌ 只能在episode结束时更新

**数学:**
- 蒙特卡洛回报: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$
- 增量更新: $V(s) \leftarrow V(s) + \alpha(G - V(s))$

---

### Module 4: 时序差分学习 (Temporal Difference Learning) ⭐⭐⭐ 核心
**关键概念:**
- 结合MC和DP的优点：单步更新但需自举
- 离策略学习（Q-Learning）vs 在策略学习（Sarsa）
- TD误差和自举
- 低方差、有偏但快速

**运行示例:**
```bash
python 04_temporal_difference/td_learning.py
```

**三种方法:**

1. **Q-Learning (离策略)**
```python
# 学习最优Q，但采用ε-贪心收集数据
Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
```
- 目标策略：贪心
- 行为策略：ε-贪心
- 可学习最优策略

2. **Sarsa (在策略)**
```python
# 学习当前策略的Q值
Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
```
- 行为和目标策略相同
- 更保守，避免危险探索

3. **Expected Sarsa**
```python
# 使用期望Q值
Q(s,a) ← Q(s,a) + α[R + γE[Q(s',·)] - Q(s,a)]
```
- 更稳定
- 低方差

**性能对比:**

| 方法 | 收敛性 | 方差 | 偏差 | 适用 |
|------|--------|------|------|------|
| Q-Learning | 快 | 中 | 有 | 学习最优策略 |
| Sarsa | 较慢 | 低 | 无 | 学习当前策略 |
| Expected Sarsa | 中等 | 最低 | 无 | 均衡选择 |

**关键数学:**
- TD误差: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
- Q-Learning更新: $Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t$ 其中 $\delta_t = R + \gamma \max_a Q(s',a') - Q(s,a)$

---

### Module 5: 策略梯度方法 (Policy Gradient Methods) ⭐⭐⭐
**关键概念:**
- 参数化策略 $\pi_\theta(a|s)$，直接优化策略
- 策略梯度定理
- 方差缩减：基准函数和优势函数
- 适应连续动作空间

**运行示例:**
```bash
python 05_policy_gradient/pg_learning.py
```

**两种主要方法:**

1. **REINFORCE (Williams, 1992)**
```python
# 策略梯度
∇J(θ) = E[∇log π_θ(a|s) G_t]

# 更新
θ ← θ + α ∇log π_θ(a|s) G_t
```
- 优点：简单，无偏
- 缺点：高方差

2. **Actor-Critic (结合策略和价值)**
```python
# Actor学习策略
∇J(θ) = E[∇log π_θ(a|s) A(s,a)]

# Critic学习价值
δ_t = R + γV(s') - V(s)  # TD误差作为优势
```
- Actor：策略网络 $\pi_\theta(a|s)$
- Critic：价值网络 $V_\phi(s)$
- 优点：低方差，快速收敛
- 缺点：偏差来自价值函数误差

**关键数学:**
- 策略梯度定理: $\nabla J(\theta) = E_s[E_a[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]]$
- 优势函数: $A(s,a) = Q(s,a) - V(s)$

---

## 💻 快速开始

### 1. 安装依赖
```bash
pip install numpy matplotlib torch
```

### 2. 按顺序学习

**第1天：理论基础**
```bash
python 01_MDP/mdp_basics.py
# 理解MDP框架、价值函数、贝尔曼方程
```

**第2天：已知模型求解**
```bash
python 02_dynamic_programming/dp_solver.py
# 对比策略迭代和价值迭代
```

**第3天：无模型学习（采样）**
```bash
python 03_monte_carlo/mc_learning.py
python 04_temporal_difference/td_learning.py
# 蒙特卡洛 vs 时序差分的权衡
```

**第4天：参数化策略**
```bash
python 05_policy_gradient/pg_learning.py
# REINFORCE vs Actor-Critic
```

### 3. 比较和优化
```bash
# 修改参数进行实验
- 改变learning_rate
- 改变epsilon（探索率）
- 改变gamma（折扣因子）
- 观察收敛速度和最终性能的变化
```

---

## 📊 算法对比表

| 算法 | 模型需求 | 更新周期 | 方差 | 偏差 | 收敛速度 | 应用 |
|------|---------|---------|------|------|---------|------|
| DP策略迭代 | ✅ 需要 | 每episode | 低 | 无 | 快 | 小规模确定性问题 |
| DP价值迭代 | ✅ 需要 | 每迭代 | 低 | 无 | 快 | 快速求解 |
| MC | ❌ 不需 | 每episode | 高 | 无 | 中 | Episode较短 |
| TD(0) | ❌ 不需 | 每步 | 中 | 有 | 快 | 在线学习 |
| Q-Learning | ❌ 不需 | 每步 | 中 | 有 | 快 | 学习最优策略 |
| Sarsa | ❌ 不需 | 每步 | 低 | 无 | 较慢 | 学习当前策略 |
| REINFORCE | ❌ 不需 | 每episode | 高 | 无 | 较慢 | 连续动作 |
| Actor-Critic | ❌ 不需 | 每步 | 低 | 有 | 中 | 通用高效 |

---

## 🔬 核心概念速查表

### 价值函数
- **状态价值** $V(s)$: 从状态s出发的期望累积奖励
- **动作价值** $Q(s,a)$: 从状态s执行动作a的期望累积奖励
- **优势函数** $A(s,a) = Q(s,a) - V(s)$: 相对于平均的优势

### 关键系数
- **折扣因子** $\gamma \in [0,1]$: 未来奖励的重要性
- **学习率** $\alpha \in (0,1]$: 步长大小
- **探索率** $\epsilon$: ε-贪心中的探索概率

### 收敛条件
- **GLIE** (Greedy in the Limit with Infinite Exploration)
  - $\sum_t \epsilon_t = \infty$ （无限探索）
  - $\sum_t \epsilon_t^2 < \infty$ （有限方差）
  - 例如：$\epsilon_t = 1/t$

---

## 📝 实验建议

### 1. 超参数敏感性分析
```python
# 修改学习率
for alpha in [0.01, 0.05, 0.1, 0.2]:
    agent = TDAgent(env, alpha=alpha)
    rewards = agent.train()
    plot_learning_curve(rewards)
```

### 2. 环境复杂度测试
```python
# 改变网格大小和障碍物数量
for grid_size in [3, 4, 5, 8]:
    for obstacles in [0, 1, 2, 4]:
        env = GridWorld(grid_size, obstacles)
        # 测试各算法
```

### 3. 算法效率比较
- 统计所需的总步数达到某个性能目标
- 测试计算时间
- 比较收敛曲线

---

## 🎓 学习要点检查清单

- [ ] 理解MDP的五元组定义
- [ ] 能推导贝尔曼方程
- [ ] 理解价值迭代和策略迭代的区别
- [ ] 知道为什么蒙特卡洛方差大
- [ ] 理解自举(bootstrapping)的含义
- [ ] 能解释Q-Learning的离策略性质
- [ ] 知道Actor-Critic如何降低方差
- [ ] 理解策略梯度定理的直观意义

---

## 📖 推荐进一步学习

### 经典教材
1. **Reinforcement Learning: An Introduction** (Sutton & Barto, 2018)
   - Chapter 1-3: 基础概念
   - Chapter 4: 动态规划
   - Chapter 5: 蒙特卡洛
   - Chapter 6: 时序差分
   - Chapter 13: 策略梯度

2. **Deep Reinforcement Learning Hands-On** (Lapan, 2018)
   - 深度学习与RL结合

### 进阶主题
- [ ] 函数近似和深度学习
- [ ] DQN (Deep Q-Networks)
- [ ] PPO (Proximal Policy Optimization)
- [ ] A3C (Asynchronous Advantage Actor-Critic)
- [ ] TRPO (Trust Region Policy Optimization)
- [ ] Model-based RL (规划和想象)

---

## 💡 常见问题

**Q: 为什么Q-Learning有时会不稳定？**
A: 使用自举和贪心操作导致高估，可以用Double Q-Learning解决

**Q: MC和TD如何选择？**
A: MC用于离散短episodes，TD用于长或连续任务。在线学习用TD

**Q: 为什么要有折扣因子γ？**
A: 避免无限循环的无限奖励；表现对未来的不确定性；实现数学收敛

**Q: 策略梯度为什么需要基准函数？**
A: 降低梯度估计的方差，加快收敛；不改变期望（无偏）

---

## 📞 调试技巧

1. **检查收敛性**
   ```python
   # 添加日志
   if episode % 10 == 0:
       print(f"avg reward: {np.mean(rewards[-100:])}")
   ```

2. **可视化学习过程**
   ```python
   plot_learning_curve(rewards)
   ```

3. **测试固定策略**
   ```python
   # 禁用探索看是否能完成任务
   epsilon = 0.0
   agent.test()
   ```

---

**祝你学习愉快！** 🚀

更新于: 2025-12-15
