# 算法工程师进阶知识手册

> 本文档覆盖大模型和具身智能领域的核心知识，目标是帮助从1年经验提升到顶级3年经验水平。
> 每个知识点包含：**概念解释 → 原理分析 → 面试常问**

---

## 目录

- [Part 1: 深度学习基础](#part-1-深度学习基础)
- [Part 2: 预训练技术](#part-2-预训练技术)
- [Part 3: 后训练/微调](#part-3-后训练微调)
- [Part 4: 推理优化与部署](#part-4-推理优化与部署)
- [Part 5: 多模态大模型](#part-5-多模态大模型)
- [Part 6: 具身智能](#part-6-具身智能)
- [Part 7: 前沿技术](#part-7-前沿技术)
- [Part 8: 面试高频问题清单](#part-8-面试高频问题清单)

---

# Part 1: 深度学习基础

## 1.1 激活函数

### ReLU 及其变体

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| **ReLU** | max(0, x) | 简单高效，但存在"死神经元"问题 |
| **Leaky ReLU** | max(0.01x, x) | 负区间有小梯度，缓解死神经元 |
| **PReLU** | max(αx, x)，α可学习 | 自适应的负斜率 |

### GELU（Gaussian Error Linear Unit）

**公式**：
$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)
$$

**特点**：
- GPT、BERT 等模型使用
- 平滑版的 ReLU，处处可微
- 概率解释：以概率 Φ(x) 保留输入

### SiLU / Swish

**公式**：
$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**特点**：
- 自门控机制
- 平滑、非单调
- 部分 LLM 使用

### SwiGLU（LLaMA 使用）

**公式**：
$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)
$$

**特点**：
- 门控线性单元的变体
- 在 FFN 中引入门控机制
- LLaMA、PaLM 等模型使用
- 比 GELU 效果更好，但参数量增加

**代码示例**：
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Swish(xW1) * xW2
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

### 面试常问

**Q: 为什么现代 LLM 更多使用 GELU 而非 ReLU？**

A: GELU 更平滑，梯度连续，在大模型中表现更稳定。ReLU 的硬截断会导致梯度不连续，可能影响训练。

---

## 1.2 优化器

### SGD 与 Momentum

```python
# SGD
θ = θ - lr * ∇L

# Momentum（动量）
v = β * v - lr * ∇L
θ = θ + v
```

**Momentum 作用**：加速收敛，减少震荡

### Adam

**公式**：
```python
m = β1 * m + (1 - β1) * ∇L      # 一阶矩估计（梯度均值）
v = β2 * v + (1 - β2) * (∇L)²   # 二阶矩估计（梯度方差）
m_hat = m / (1 - β1^t)           # 偏差修正
v_hat = v / (1 - β2^t)
θ = θ - lr * m_hat / (√v_hat + ε)
```

**超参数**：
- β1 = 0.9（一阶矩衰减）
- β2 = 0.999（二阶矩衰减）
- ε = 1e-8（数值稳定性）

### AdamW（权重衰减解耦）

**区别于 Adam + L2 正则化**：

```python
# Adam + L2（错误做法）
∇L = ∇L + λθ  # L2正则化项也会被自适应学习率缩放

# AdamW（正确做法）
θ = θ - lr * m_hat / (√v_hat + ε) - lr * λ * θ  # 权重衰减独立于自适应项
```

**为什么 AdamW 更好？**
- L2 正则化被自适应学习率影响，效果不一致
- AdamW 的权重衰减是真正的衰减，不受其他因素影响

### LAMB（大批量训练）

**核心思想**：层自适应学习率

$$
r = \frac{\|w\|}{\|Adam(w)\|}
$$

**用途**：大批量训练（如 batch_size=32K），保持训练稳定

### 学习率调度

**Warmup**：
```python
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
```

**Cosine Decay**：
$$
lr = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{t}{T}\pi))
$$

**典型配置**（LLM 预训练）：
- Warmup: 2000 步
- 之后 cosine decay 到最大步数

### 面试常问

**Q: Adam 和 AdamW 的区别？**

A: Adam 将 L2 正则化项加到梯度上，然后被自适应学习率缩放。AdamW 将权重衰减独立出来，直接在权重上衰减，效果更好。

**Q: 为什么需要 learning rate warmup？**

A: 训练初期模型参数随机，梯度不稳定。Warmup 让学习率从小到大增长，避免初期的大梯度更新破坏模型。

---

## 1.3 正则化

### Dropout

**原理**：训练时随机将部分神经元置零

```python
# 训练时
x = x * mask / (1 - p)  # mask中有p比例的0

# 推理时
x = x  # 不做任何操作
```

**注意**：训练时要除以 (1-p) 进行缩放，保证期望不变

### Layer Normalization vs Batch Normalization

| 特性 | Batch Norm | Layer Norm |
|------|------------|------------|
| 归一化维度 | 批次维度 (N) | 特征维度 (C) |
| 统计量 | 整个 batch | 单个样本 |
| 适用场景 | CNN、固定序列 | Transformer、变长序列 |
| 推理时 | 需要维护滑动均值 | 不需要 |

**为什么 Transformer 用 Layer Norm？**
- 序列长度可变
- 不依赖 batch size
- 推理时行为一致

### 梯度裁剪

```python
# 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**用途**：防止梯度爆炸，特别是在 RNN 和大 LLM 训练中

---

## 1.4 损失函数

### 交叉熵损失

**公式**：
$$
\text{CE} = -\sum_{i} y_i \log(\hat{y}_i)
$$

**语言模型中的应用**：
```python
# 预测下一个token的概率分布
loss = F.cross_entropy(logits, labels)  # logits: [B, seq_len, vocab_size]
```

### KL 散度

**公式**：
$$
D_{KL}(P || Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}
$$

**用途**：
- 知识蒸馏
- VAE
- RLHF 中约束策略变化

### 对比学习损失（InfoNCE）

**公式**：
$$
\mathcal{L} = -\log\frac{\exp(s(x, x^+)/\tau)}{\sum_j \exp(s(x, x_j)/\tau)}
$$

**用途**：CLIP、SimCLR 等对比学习方法

---

# Part 2: 预训练技术

## 2.1 语言模型预训练

### 自回归语言模型（GPT 系列）

**目标**：预测下一个 token

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1})
$$

**特点**：
- 单向注意力（因果掩码）
- 适合文本生成
- GPT、LLaMA、Qwen 使用

### 掩码语言模型（BERT）

**目标**：预测被 [MASK] 遮挡的 token

$$
\mathcal{L} = -\sum_{i \in M} \log P(x_i | x_{\backslash M})
$$

**训练策略**：
- 15% 的 token 被选中
- 其中 80% 替换为 [MASK]
- 10% 替换为随机 token
- 10% 保持不变

**特点**：
- 双向注意力
- 适合理解任务（分类、NER）
- 不适合生成

### Prefix LM vs Causal LM

| 类型 | 前缀部分 | 生成部分 | 例子 |
|------|----------|----------|------|
| Causal LM | 单向 | 单向 | GPT |
| Prefix LM | 双向 | 单向 | T5、GLM |

---

## 2.2 预训练数据

### 数据来源

| 来源 | 规模 | 质量 | 例子 |
|------|------|------|------|
| 网页爬取 | TB级 | 低 | Common Crawl |
| 书籍 | GB级 | 高 | Books3 |
| 代码 | TB级 | 高 | GitHub |
| 论文 | GB级 | 高 | arXiv |
| 百科 | GB级 | 高 | Wikipedia |

### 数据清洗

1. **去重**：MinHash、SimHash
2. **质量过滤**：困惑度过滤、分类器过滤
3. **有害内容过滤**：敏感词、有毒内容
4. **语言识别**：过滤非目标语言

### 数据配比

**典型配比**（LLaMA）：
- 英文网页：67%
- 代码：4.5%
- 维基百科：4.5%
- 书籍：4.5%
- 论文：2.5%
- ...

**配比原则**：
- 高质量数据多次重复
- 代码数据提升推理能力
- 多语言数据提升泛化性

---

## 2.3 训练技巧

### 混合精度训练

```python
# 使用 autocast
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)

# 使用 GradScaler 防止下溢
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP16 vs BF16**：
- FP16：更高精度，但容易溢出
- BF16：更大动态范围，训练更稳定（A100/H100 支持）

### 梯度累积

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**用途**：显存不足时模拟大 batch

### 分布式训练

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| **DDP** | 数据并行 | 多卡训练 |
| **FSDP** | 全切片数据并行 | 超大模型 |
| **DeepSpeed ZeRO** | 优化器/梯度/参数分片 | 超大模型 |
| **张量并行** | 切分矩阵计算 | 单层超大 |
| **流水线并行** | 层间切分 | 超深模型 |

### 面试常问

**Q: DeepSpeed ZeRO 的三个阶段分别优化什么？**

A:
- ZeRO-1：切分优化器状态（内存减少 4x）
- ZeRO-2：切分梯度（内存减少 8x）
- ZeRO-3：切分参数（内存减少线性于GPU数）

---

# Part 3: 后训练/微调

## 3.1 监督微调（SFT）

### 什么是 SFT？

在预训练模型基础上，使用**有标注的指令-回复对**进行微调。

```
指令: 请解释什么是机器学习
回复: 机器学习是人工智能的一个分支，它使计算机能够...
```

### 对话数据格式

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "请介绍一下自己"},
    {"role": "assistant", "content": "我是一个AI助手..."}
  ]
}
```

### 损失计算

只在 **assistant 回复部分** 计算损失：

```python
# labels 中，user 部分设为 -100（忽略）
labels[user_positions] = -100
loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

### 过拟合问题

**现象**：SFT 后模型变"傻"

**原因**：
- 训练数据量少
- 训练轮数过多
- 数据质量差

**解决**：
- 早停
- 降低学习率
- 数据增强
- 混入预训练数据

---

## 3.2 RLHF（基于人类反馈的强化学习）

### 三阶段流程

```
阶段1: SFT
预训练模型 → 指令微调 → SFT模型

阶段2: 训练奖励模型
收集人类偏好数据 (response1 > response2) → 训练 Reward Model

阶段3: 强化学习
使用 PPO 算法，最大化奖励同时保持与 SFT 模型接近
```

### 奖励模型

**训练目标**：

$$
\mathcal{L} = -\log\sigma(r(x, y_w) - r(x, y_l))
$$

其中 $y_w$ 是人类偏好的回复，$y_l$ 是不偏好的。

### PPO 目标

$$
\mathcal{L} = \mathbb{E}[r(x, y)] - \beta \cdot D_{KL}(\pi || \pi_{SFT})
$$

- 最大化奖励
- 同时限制与 SFT 策略的偏离（防止奖励 hacking）

### 面试常问

**Q: 为什么 SFT 之后还需要 RLHF？**

A: SFT 只学习"正确"的回复，不知道"错误"是什么。RLHF 通过奖励模型提供正负反馈信号，让模型学会避免不好的回复。

---

## 3.3 DPO（直接偏好优化）

### 核心思想

**绕过奖励模型**，直接用偏好数据优化策略。

**数学推导**：
- 从 RLHF 的最优策略出发
- 可以证明隐式的奖励函数为：

$$
r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}
$$

### DPO 损失函数

$$
\mathcal{L}_{DPO} = -\log\sigma\left(\beta \log\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

### DPO vs RLHF

| 特性 | RLHF | DPO |
|------|------|-----|
| 需要奖励模型 | 是 | 否 |
| 训练复杂度 | 高（需要PPO） | 低（类似SFT） |
| 内存需求 | 高（多个模型） | 较低 |
| 效果 | 略好 | 接近 |

### 代码示例

```python
def dpo_loss(policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    """
    policy_logps_w: 策略模型对偏好回复的log概率
    policy_logps_l: 策略模型对非偏好回复的log概率
    ref_logps_w/l: 参考模型（SFT模型）的log概率
    """
    logits = beta * (policy_logps_w - ref_logps_w) - beta * (policy_logps_l - ref_logps_l)
    loss = -F.logsigmoid(logits)
    return loss.mean()
```

---

## 3.4 参数高效微调（PEFT）

### LoRA（Low-Rank Adaptation）

**核心思想**：冻结原始权重，只训练低秩分解的增量

$$
W' = W + BA
$$

其中：
- W: 原始权重 [d, k]，冻结
- B: [d, r]，可训练
- A: [r, k]，可训练
- r << min(d, k)（如 r=8 或 16）

**参数量对比**：
- 原始：d × k = 4096 × 4096 = 16M
- LoRA (r=8)：d × r + r × k = 4096 × 8 + 8 × 4096 = 65K

**代码示例**：见 `lora_simple.py`

### QLoRA

**三大优化**：

1. **NF4 量化**：4-bit 量化，专为正态分布优化
2. **双重量化**：对量化常数再量化
3. **分页优化器**：OOM 时自动转移到 CPU

**显存节省**：
- 65B 模型：原需 130GB → QLoRA 只需 48GB

### 其他 PEFT 方法

| 方法 | 原理 | 参数量 |
|------|------|--------|
| **Adapter** | 在层间插入小型网络 | ~3% |
| **Prefix Tuning** | 在输入前加可学习前缀 | ~0.1% |
| **Prompt Tuning** | 只学习软提示嵌入 | ~0.01% |
| **LoRA** | 低秩分解 | ~0.1% |

### 面试常问

**Q: LoRA 的 rank 如何选择？**

A: 通常 r=4, 8, 16 已足够。实验表明 r=16 效果接近全参微调。r 过大会增加参数量和过拟合风险。

**Q: LoRA 一般应用在哪些层？**

A: 通常应用于 Attention 的 Q、V 投影层。扩展版本也可以应用于 K、O 和 FFN。

---

## 3.5 灾难性遗忘

### 现象

微调后模型丧失原有能力：
- 微调问答任务后，通用对话能力下降
- 微调特定语言后，其他语言能力退化

### 原因

- 新任务梯度覆盖旧知识
- 微调数据分布与预训练差异大
- 过度拟合新任务

### 解决方案

1. **混合训练**：微调数据混入预训练数据
2. **正则化**：EWC、L2-SP 等约束权重变化
3. **LoRA**：只训练少量参数，保留原始知识
4. **回炉训练**：定期用通用数据训练

---

# Part 4: 推理优化与部署

## 4.1 KV Cache

### 原理

自回归生成时，已生成 token 的 K、V 不会改变：

```
Step 1: 输入 [A]      → 计算 K1, V1 → 缓存
Step 2: 输入 [A, B]   → K1, V1 从缓存读取，只计算 K2, V2
Step 3: 输入 [A,B,C]  → K1,K2, V1,V2 从缓存读取，只计算 K3, V3
```

### 显存占用分析

```
KV Cache 大小 = 2 × num_layers × seq_len × num_heads × head_dim × batch_size × dtype_size
```

**示例**（LLaMA-7B，seq_len=2048，batch=1，FP16）：
- 2 × 32 × 2048 × 32 × 128 × 1 × 2 = 1GB

### KV Cache 量化

将 KV Cache 从 FP16 压缩到 INT8/INT4：
- INT8：节省 50% 显存
- INT4：节省 75% 显存

**代码示例**：见 `kv_cache_demo.py`

---

## 4.2 Flash Attention

### 问题背景

标准 Attention 的内存瓶颈：
- 需要存储完整的 [seq_len, seq_len] 注意力矩阵
- 频繁在 HBM（高带宽内存）和 SRAM 之间传输

### 核心技术

1. **Tiling（分块）**：
   - 将 Q, K, V 分成小块
   - 在 SRAM 中计算，避免 HBM 读写

2. **Recomputation（重计算）**：
   - 反向传播时不保存中间结果
   - 需要时重新计算

3. **Online Softmax**：
   - 增量计算 softmax，无需完整矩阵

### 性能提升

- 速度：2-4x
- 显存：5-20x 减少
- 精度：无损失

### 面试常问

**Q: Flash Attention 为什么能加速但不损失精度？**

A: Flash Attention 不改变计算结果，只改变计算顺序和内存访问模式。通过 tiling 减少 HBM 访问，利用更快的 SRAM 进行计算。

---

## 4.3 量化技术

### 量化基础

将高精度（FP32/FP16）转换为低精度（INT8/INT4）：

$$
x_{int} = \text{round}(x / s) + z
$$

其中 s 是缩放因子，z 是零点。

### INT8 量化

- **动态量化**：推理时计算缩放因子
- **静态量化**：预先校准缩放因子
- **精度损失**：<0.1%（短文本）

### INT4 量化（GPTQ、AWQ）

**GPTQ**：
- 基于 Hessian 的逐层量化
- 使用少量校准数据

**AWQ（Activation-aware Weight Quantization）**：
- 保护重要权重（激活值大的）
- 精度更高

### 量化方法对比

| 方法 | 精度 | 大小 | 速度 | 实现复杂度 |
|------|------|------|------|-----------|
| FP16 | 基准 | 1x | 1x | 低 |
| INT8 | -0.1% | 0.5x | 1.3x | 中 |
| INT4 (GPTQ) | -0.5% | 0.25x | 1.5x | 高 |
| INT4 (AWQ) | -0.3% | 0.25x | 1.5x | 高 |

---

## 4.4 推理框架

### vLLM

**核心技术：PagedAttention**

```
传统: KV Cache 连续存储 → 显存碎片化
vLLM: KV Cache 分页存储 → 显存利用率提升 24%+
```

**特点**：
- 高吞吐量
- 连续批处理（Continuous Batching）
- 支持多种模型

### TensorRT-LLM

**特点**：
- NVIDIA 官方优化
- 深度集成 CUDA
- 支持 INT8/FP8 量化
- 最高性能

### llama.cpp

**特点**：
- 纯 C/C++ 实现
- 支持 CPU 推理
- 极致的量化（2-8 bit）
- 适合边缘设备

### 框架选择建议

| 场景 | 推荐框架 |
|------|----------|
| 高吞吐服务 | vLLM |
| 最低延迟 | TensorRT-LLM |
| 边缘部署 | llama.cpp |
| 快速原型 | Hugging Face Transformers |

---

## 4.5 服务化部署

### 批处理策略

**静态批处理**：等待收集固定数量请求

**动态批处理（Continuous Batching）**：
- 请求到达立即加入批次
- 请求完成立即释放资源
- 最大化 GPU 利用率

### 流式输出

```python
async def generate_stream(prompt):
    for token in model.generate(prompt, stream=True):
        yield token
```

**优势**：
- 降低首 token 延迟
- 改善用户体验

---

# Part 5: 多模态大模型

## 5.1 视觉编码器

### ViT（Vision Transformer）

**核心思想**：将图像分割成 patches，当作 tokens 处理

```
图像 [H, W, C] → Patches [N, P×P×C] → Linear Projection → Transformer
```

**Patch 划分**：
- 通常 16×16 或 14×14
- 224×224 图像 → 196 个 patches

### CLIP

**双编码器架构**：
- Image Encoder（ViT 或 ResNet）
- Text Encoder（Transformer）

**对比学习训练**：
```
正样本：匹配的图文对
负样本：不匹配的图文对

损失：最大化正样本相似度，最小化负样本相似度
```

**能力**：
- 零样本图像分类
- 图文检索
- 作为 VLM 的视觉编码器

---

## 5.2 VLM 架构

### 典型架构

```
图像 → [视觉编码器] → [投影层] → 视觉 tokens
                            ↓
文本 → [Tokenizer] → [Embedding] → 文本 tokens → [LLM] → 输出
                            ↑
                      拼接视觉和文本 tokens
```

### LLaVA

**简洁设计**：
- 视觉编码器：CLIP ViT
- 投影层：**简单的 Linear 层**
- LLM：LLaMA/Vicuna

**训练阶段**：
1. 预训练：只训练投影层
2. 微调：训练投影层 + LLM

### Qwen-VL

**改进点**：
- 自训练的视觉编码器（非 CLIP）
- 动态分辨率输入
- 更强的 OCR 和定位能力

### 跨模态对齐方法

| 方法 | 原理 | 代表模型 |
|------|------|----------|
| **线性投影** | 直接线性变换 | LLaVA |
| **Q-Former** | Learnable Queries 提取特征 | BLIP-2 |
| **Perceiver** | 交叉注意力压缩 | Flamingo |

### 面试常问

**Q: LLaVA 用简单的线性层做投影，为什么效果好？**

A: 预训练的 CLIP 已经学到了良好的视觉-语义对齐。简单的线性层足以将视觉特征映射到 LLM 的嵌入空间。复杂的投影层反而可能破坏预训练的对齐。

---

## 5.3 多模态训练

### 预训练数据

| 数据类型 | 例子 | 规模 |
|----------|------|------|
| 图文对 | LAION-5B | 50亿 |
| 图文交错 | MMC4 | 1亿+ |
| OCR 数据 | 文档、截图 | 百万级 |
| 视频文本 | WebVid | 千万级 |

### 微调策略

1. **冻结视觉编码器**：只训练投影层和 LLM
2. **端到端微调**：全部参数可训练（需大量数据）
3. **LoRA 微调**：只在 LLM 上加 LoRA

---

# Part 6: 具身智能

## 6.1 强化学习基础

### MDP 框架

$$
(S, A, P, R, \gamma)
$$

- S：状态空间
- A：动作空间
- P：状态转移概率
- R：奖励函数
- γ：折扣因子

### 策略梯度

$$
\nabla J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]
$$

### PPO（Proximal Policy Optimization）

**核心思想**：限制策略更新幅度

$$
L^{CLIP} = \mathbb{E}[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]
$$

**优点**：
- 稳定性好
- 实现简单
- 广泛应用于机器人和 RLHF

### SAC（Soft Actor-Critic）

**核心思想**：最大化奖励 + 最大化熵

$$
J(\pi) = \mathbb{E}[\sum_t r_t + \alpha H(\pi(\cdot|s_t))]
$$

**优点**：
- 更好的探索
- 对超参数不敏感
- 适合连续控制任务

### DDPG（Deep Deterministic Policy Gradient）

**特点**：
- 确定性策略
- Actor-Critic 架构
- 适合连续动作空间

### 算法选择

| 场景 | 推荐算法 |
|------|----------|
| 离散动作 | PPO |
| 连续控制 | SAC、TD3 |
| 仿真训练 | PPO（并行化好） |
| 真机训练 | SAC（样本效率高） |

---

## 6.2 模仿学习

### 行为克隆（Behavior Cloning）

**方法**：将专家轨迹当作监督学习数据

$$
\mathcal{L} = \mathbb{E}_{(s,a) \sim D_{expert}}[\|a - \pi_\theta(s)\|^2]
$$

**问题**：分布偏移（Compounding Error）

### DAGGER

**解决分布偏移**：
1. 用当前策略收集轨迹
2. 让专家标注动作
3. 将新数据加入训练集

### GAIL（Generative Adversarial Imitation Learning）

**思想**：用 GAN 学习奖励函数

- Generator：策略网络
- Discriminator：区分专家和策略生成的轨迹

---

## 6.3 Sim2Real

### 为什么需要仿真？

- 真机数据采集成本高
- 真机可能损坏
- 仿真可以大规模并行

### 域随机化（Domain Randomization）

**思想**：在仿真中随机化各种参数，学习鲁棒策略

**随机化内容**：
- 物理参数（摩擦、质量）
- 视觉参数（颜色、纹理、光照）
- 传感器噪声

### 迁移方法

| 方法 | 原理 |
|------|------|
| **域随机化** | 仿真中随机化，真机是一种情况 |
| **域适应** | 减小仿真和真机的分布差异 |
| **渐进式迁移** | 先仿真后真机微调 |
| **Teacher-Student** | 仿真训练教师，真机蒸馏学生 |

---

## 6.4 VLA 模型

### 什么是 VLA？

**Vision-Language-Action**：将视觉、语言和动作统一建模

```
输入：图像 + 语言指令
输出：机器人动作
```

### RT-1 / RT-2

**RT-1**：
- 基于 Transformer
- 输入：图像序列 + 语言
- 输出：离散化的动作 token

**RT-2**：
- 基于 VLM（PaLI）
- 将动作表示为 token
- 端到端生成动作

### 当前研究方向

1. **IL + RL**：模仿学习 + 强化学习微调
2. **Sim2Real**：仿真预训练 + 真机适应
3. **多任务泛化**：一个模型处理多种任务

---

## 6.5 仿真平台

| 平台 | 特点 | 适用场景 |
|------|------|----------|
| **Isaac Gym/Sim** | NVIDIA 出品，GPU 并行，速度极快 | 大规模 RL 训练 |
| **MuJoCo** | 物理精确，接触建模好 | 精细操作 |
| **PyBullet** | 开源免费，易上手 | 快速原型 |
| **Gazebo** | ROS 集成好 | 系统集成 |

---

## 6.6 机器人学基础

### 运动学

**正运动学**：关节角度 → 末端位姿
**逆运动学**：末端位姿 → 关节角度

### 动力学

**牛顿-欧拉方程**：
$$
\tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q)
$$

### ROS（Robot Operating System）

**核心概念**：
- Node：功能模块
- Topic：消息通信
- Service：请求-响应
- Action：长时间任务

---

# Part 7: 前沿技术

## 7.1 MoE（Mixture of Experts）

### 核心思想

**稀疏激活**：每个 token 只激活部分专家

```
Input → Router → Top-K Experts → Weighted Sum → Output
```

### 路由机制

$$
G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))
$$

选择分数最高的 K 个专家（通常 K=2）

### 负载均衡

**问题**：某些专家被过度使用

**解决**：辅助损失鼓励均匀分配

$$
\mathcal{L}_{aux} = \alpha \cdot \sum_i f_i \cdot P_i
$$

### 代表模型

- Mixtral 8×7B：8 个专家，每次激活 2 个
- GPT-4（传闻）：MoE 架构

---

## 7.2 长上下文

### 挑战

- 注意力 O(n²) 复杂度
- 位置编码外推问题
- 显存限制

### 解决方案

| 方法 | 原理 | 代表 |
|------|------|------|
| **稀疏注意力** | 只计算部分位置 | Longformer |
| **位置插值** | 缩放位置编码 | PI、NTK |
| **窗口注意力** | 滑动窗口 | Mistral |
| **上下文压缩** | 压缩历史信息 | StreamingLLM |

### 位置插值（PI）

$$
f(x, m \cdot \frac{L}{L'}) \quad \text{其中 } L' > L
$$

将位置缩放到训练范围内

---

## 7.3 模型幻觉

### 定义

模型生成事实上不正确或与输入矛盾的内容。

### 原因

1. **训练数据问题**：数据本身有错误
2. **知识过时**：训练后的信息
3. **模式匹配**：统计关联而非真实理解
4. **解码策略**：高温度采样增加随机性

### 缓解方法

| 方法 | 原理 |
|------|------|
| **RAG** | 检索相关文档，基于事实生成 |
| **事实核查** | 生成后验证事实 |
| **引用来源** | 要求模型给出来源 |
| **低温度采样** | 减少随机性 |
| **CoT 推理** | 逐步推理减少跳跃 |

---

## 7.4 涌现能力

### 定义

小模型没有，大模型突然出现的能力。

### 典型能力

- In-context Learning
- Chain-of-Thought 推理
- 代码生成
- 指令遵循

### Scaling Law

$$
L(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_{\infty}
$$

- N：参数量
- D：数据量
- C：计算量

**Chinchilla 最优**：参数量和数据量应该同比例增长

---

# Part 8: 面试高频问题清单

## Transformer 基础

1. Transformer 的整体架构是什么？
2. 自注意力的计算公式和复杂度？
3. 为什么要除以 sqrt(d_k)？
4. 多头注意力的作用？
5. 位置编码有哪些方法？RoPE 的原理？
6. Pre-Norm vs Post-Norm？
7. 为什么大模型用 Decoder-only？

## 训练技术

8. Adam 和 AdamW 的区别？
9. 混合精度训练的原理？
10. DeepSpeed ZeRO 三个阶段分别优化什么？
11. 梯度累积的作用？

## 微调方法

12. SFT、RLHF、DPO 的区别？
13. LoRA 的原理和 rank 如何选择？
14. QLoRA 的三大优化？
15. 什么是灾难性遗忘？如何解决？

## 推理优化

16. KV Cache 的原理？显存占用如何计算？
17. Flash Attention 为什么能加速？
18. INT8 vs INT4 量化的权衡？
19. vLLM 的 PagedAttention 是什么？

## 多模态

20. CLIP 的训练方式？
21. VLM 如何做跨模态对齐？
22. LLaVA 的架构设计？

## 具身智能

23. PPO vs SAC 的区别和适用场景？
24. 什么是 Sim2Real？常用方法？
25. 域随机化的原理？

## 前沿技术

26. MoE 的稀疏激活原理？
27. 长上下文有哪些处理方法？
28. 模型幻觉的原因和解决方案？
29. 什么是涌现能力？
30. Scaling Law 的内容？

---

## 参考资源

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

### 博客
- [大模型微调基本概念指北](https://www.cnblogs.com/KubeExplorer/p/18674820)
- [Flash Attention 原理详解](https://zhuanlan.zhihu.com/p/676655352)
- [RoPE 旋转位置编码](https://blog.csdn.net/v_JULY_v/article/details/134085503)

### 代码
- 本目录下的 `attention_easy.py`、`positional_encoding.py`、`lora_simple.py`、`kv_cache_demo.py`
