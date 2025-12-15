# 🚀 强化学习完整学习项目

欢迎来到强化学习学习项目！这是一个从理论到实践的完整学习系统。

## ⚡ 快速开始（30秒）

```bash
cd /home/ryan/2repo/my_infra/RL
python 01_MDP/mdp_basics.py
```

或使用交互式菜单：

```bash
bash quickstart.sh
```

## 📚 项目包含内容

| 模块 | 文件 | 说明 |
|------|------|------|
| 1️⃣ MDP基础 | `01_MDP/mdp_basics.py` | 马尔可夫决策过程、价值函数 |
| 2️⃣ 动态规划 | `02_dynamic_programming/dp_solver.py` | 策略迭代、价值迭代 |
| 3️⃣ 蒙特卡洛 | `03_monte_carlo/mc_learning.py` | First-Visit、Every-Visit MC |
| 4️⃣ 时序差分 | `04_temporal_difference/td_learning.py` | Q-Learning、Sarsa、Expected Sarsa |
| 5️⃣ 策略梯度 | `05_policy_gradient/pg_learning.py` | REINFORCE、Actor-Critic |

## 📖 文档导览

- **[README.md](README.md)** - 完整的理论讲解和算法对比
- **[LEARNING_GUIDE.py](LEARNING_GUIDE.py)** - 11-15周学习计划
- **[QUICK_REFERENCE.py](QUICK_REFERENCE.py)** - 公式和代码速查手册
- **[PROJECT_OVERVIEW.py](PROJECT_OVERVIEW.py)** - 项目结构导览
- **[FINAL_CHECKLIST.py](FINAL_CHECKLIST.py)** - 完成度检查和后续指南

## 🎯 推荐学习路线

### 第一周：MDP基础 ⭐必修
```bash
python 01_MDP/mdp_basics.py        # 运行示例
```
关键概念：状态价值V(s)、动作价值Q(s,a)、贝尔曼方程

### 第二周：动态规划
```bash
python 02_dynamic_programming/dp_solver.py
```
关键概念：策略迭代 vs 价值迭代

### 第三周：蒙特卡洛与时序差分
```bash
python 03_monte_carlo/mc_learning.py
python 04_temporal_difference/td_learning.py
```
关键概念：采样学习、自举、离策略学习

### 第四周：策略梯度
```bash
pip install torch  # 如需要
python 05_policy_gradient/pg_learning.py
```
关键概念：参数化策略、策略梯度定理

## 🔧 系统要求

```bash
# 最小依赖
pip install numpy matplotlib

# 完整功能（包括策略梯度）
pip install numpy matplotlib torch
```

## 💻 常用命令

```bash
# 交互式菜单（推荐）
bash quickstart.sh

# 运行单个模块
python 01_MDP/mdp_basics.py
python 02_dynamic_programming/dp_solver.py
python 03_monte_carlo/mc_learning.py
python 04_temporal_difference/td_learning.py
python 05_policy_gradient/pg_learning.py

# 查看学习指南
python LEARNING_GUIDE.py

# 查看速查手册
python QUICK_REFERENCE.py

# 综合对比实验
python comparison_experiment.py
```

## 📊 核心算法对比

| 算法 | 模型需求 | 收敛速度 | 方差 | 推荐场景 |
|------|---------|---------|------|---------|
| 策略迭代 | ✅需要 | 快 | 低 | 已知模型的小问题 |
| 价值迭代 | ✅需要 | 快 | 低 | 快速求解最优解 |
| 蒙特卡洛 | ❌无需 | 中 | 高 | 离散短episode |
| Q-Learning | ❌无需 | 快 | 中 | **最常用** |
| Sarsa | ❌无需 | 中 | 低 | 保守的在线学习 |
| Actor-Critic | ❌无需 | 中 | 低 | **高效通用** |

## ✨ 项目特色

- ✅ **完整性** - 5个核心算法，理论+实践
- ✅ **易运行** - 最小依赖，包含脚本
- ✅ **教学友好** - 8份文档，大量注释
- ✅ **易扩展** - 清晰结构，支持修改
- ✅ **综合性** - 算法对比、性能分析

## 🎓 学习要点

完成本项目后，你将掌握：

✓ 理解MDP框架和贝尔曼方程  
✓ 实现基本的RL算法  
✓ 比较不同算法的优缺点  
✓ 在新环境上应用这些算法  
✓ 调参和优化性能  

## ❓ 常见问题

**Q: 需要多长时间完成？**  
A: 认真学习 2-3 个月。基础理论 2-3 周，深度理解需要反复实验。

**Q: 需要数学基础吗？**  
A: 需要基本的概率统计和线性代数。项目中的公式都有详细说明。

**Q: 能否跳过某些模块？**  
A: MDP 必须掌握。其他模块可按顺序学，相互补充。

**Q: 学完能做什么？**  
A: 能够理解DRL（深度强化学习）论文，在新问题上应用RL。

## 🚀 后续学习

掌握本项目后，可以继续学习：

- **深度强化学习** - DQN、PPO、A3C
- **多智能体RL** - 合作、竞争、通信
- **模型和规划** - World Models、MuZero
- **应用领域** - 游戏AI、机器人、推荐系统

## 📚 推荐资源

**必读教材：**
- Sutton & Barto: *Reinforcement Learning: An Introduction* (2018)

**推荐课程：**
- David Silver RL Course (UCL)
- 李宏毅 强化学习课程

## 📞 项目信息

- **创建日期** - 2025-12-15
- **位置** - `/home/ryan/2repo/my_infra/RL/`
- **代码行数** - 3500+
- **文档** - 8份详细文档

## 🎯 开始学习

现在就开始吧！

```bash
cd /home/ryan/2repo/my_infra/RL
bash quickstart.sh
```

或者直接查看完整文档：

```bash
python LEARNING_GUIDE.py
```

---

**祝你学习顺利！** 🎓

如有问题，查看 [README.md](README.md) 的FAQ部分。
