# SAC (Soft Actor-Critic) 模块

Soft Actor-Critic 是现代深度强化学习的代表算法，在样本效率和稳定性上表现优异。

## 🚀 快速开始

```bash
# 查看SAC关键概念（5分钟快速了解）
python sac_demo.py --concepts

# 运行SAC演示（15分钟看到学习效果）
python sac_demo.py --demo

# 交互式菜单（多种选择）
python sac_demo.py
```

## 📂 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| **sac_minimal.py** | 539行 | 核心实现：Actor、Critic、Agent |
| **sac_demo.py** | 389行 | 交互式演示：训练、可视化、评估 |
| **SAC_EXPLANATION.md** | 3000行 | 完整理论：12部分深度讲解 |
| **README.md** | 本文件 | 模块快速参考 |

## 🎯 5个关键概念

### 1. 最大熵目标
```
J(π) = E[r_t + α H(π)]
       ├─ 最大化奖励
       └─ 最大化策略熵（鼓励探索）
```

### 2. 重参数化采样
```
a = tanh(μ + σ·ε)    其中 ε ~ N(0,I)
├─ 使采样过程可微
├─ 梯度流向参数
└─ tanh压缩动作范围
```

### 3. 双Q网络
```
target = min(Q₁(s,a), Q₂(s,a))
├─ 减少Q值高估偏差
├─ 提高学习稳定性
└─ 两个独立网络取最小值
```

### 4. 自适应温度
```
如果 H(π) < 目标: α↑ 鼓励探索
如果 H(π) > 目标: α↓ 减少探索
└─ 自动调节，不需手动衰减
```

### 5. 三个网络更新
```
Actor:   最小化 α·log π - Q
Critic:  最小化 (Q - y)²
Alpha:   最小化 α·(log π + H_target)
```

## 📊 SAC vs 其他算法

| 特性 | PPO | **SAC** | DDPG | TD3 |
|------|-----|---------|------|-----|
| 样本效率 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 训练稳定 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 计算效率 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 超参敏感 | 中 | 低 | 低 | 中 |

**何时选择SAC？**
- ✅ 样本昂贵的真实环境（机器人、自驾车）
- ✅ 连续高维控制问题
- ✅ 需要鲁棒和稳定学习

## ⭐ 推荐超参数

```python
learning_rate = 3e-4    # 标准学习率
gamma = 0.99           # 折扣因子
tau = 0.005            # 软更新系数
alpha_init = 0.2       # 初始熵系数
batch_size = 256       # 批大小
hidden_dim = 256       # 网络宽度
```

## 🔑 常见问题

**Q: 为什么需要两个Q网络？**
A: 单个Q容易高估。两个Q的最小值更保守，显著提高稳定性。

**Q: 雅可比修正有多重要？**
A: 非常重要！`log π(a|s) = log p(z) - Σ log(1-a²)` 不能遗漏。

**Q: α怎么选？**
A: 初值0.2就很好。SAC自适应调节，不需要精细调整。

**Q: 学习不稳定怎么办？**
A: ↓学习率 → ↑批大小 → ↓α初值

## 📚 学习路径

**快速掌握（1小时）**
1. 运行：`python sac_demo.py --concepts`
2. 读：本README中的5个关键概念
3. 看：sac_minimal.py的核心部分

**深入学习（3小时）**
1. 运行：`python sac_demo.py --demo`
2. 读：SAC_EXPLANATION.md的前6部分
3. 对照：sac_minimal.py的实现细节

**精通掌握（6小时）**
1. 完整阅读：SAC_EXPLANATION.md
2. 深入研究：sac_minimal.py的所有代码
3. 实践修改：改变超参数，观察效果

## 🌟 SAC的核心优势

- **样本效率高**：比PPO高50%+
- **训练稳定**：双Q网络+目标网络
- **自动探索**：α自动调节，无需ε衰减
- **连续控制**：原生支持高维动作空间
- **鲁棒性好**：对环境变化容错能力强

## 🔗 外部资源

**论文（推荐）**
- Haarnoja et al. 2018: SAC原始论文 (ICML)
- Haarnoja et al. 2019: SAC完整版 (JMLR)

**实现参考**
- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

---

**下一步：**
```bash
python sac_demo.py --demo
```

看看SAC在GridWorld上的实际效果！
