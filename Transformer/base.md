# Transformer 架构基础知识

> 本文档整理 Transformer 相关的面试高频知识点，帮助深入理解架构原理。
> 配套代码参见 `attention_easy.py`。

---

## 目录

1. [Transformer 核心架构](#一transformer-核心架构)
2. [注意力机制](#二注意力机制)
3. [位置编码](#三位置编码)
4. [核心组件](#四核心组件)
5. [Mask 机制](#五mask-机制)
6. [架构对比](#六架构对比)
7. [Tokenizer](#七tokenizer)
8. [面试高频问题](#八面试高频问题)

---

## 一、Transformer 核心架构

### 1.1 整体架构

Transformer 由 **Encoder（编码器）** 和 **Decoder（解码器）** 两部分组成：

```
输入序列 → [Embedding + 位置编码] → Encoder × N → 编码表示
                                                      ↓
输出序列 → [Embedding + 位置编码] → Decoder × N → 线性层 → Softmax → 输出概率
```

**Encoder 结构**（每层包含）：
- Multi-Head Self-Attention（多头自注意力）
- Feed-Forward Network（前馈网络）
- 残差连接 + Layer Normalization

**Decoder 结构**（每层包含）：
- Masked Multi-Head Self-Attention（带掩码的自注意力）
- Multi-Head Cross-Attention（交叉注意力，Q来自Decoder，K/V来自Encoder）
- Feed-Forward Network
- 残差连接 + Layer Normalization

### 1.2 为什么 Transformer 取代 RNN/LSTM

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| **并行计算** | 顺序处理，无法并行 | 完全并行，训练速度快 |
| **长距离依赖** | 存在梯度消失，难以捕捉长距离依赖 | 自注意力直接建模任意距离的依赖 |
| **计算复杂度** | O(n) 但串行 | O(n²) 但高度并行 |
| **位置信息** | 隐式包含在循环结构中 | 需要显式的位置编码 |

### 1.3 Transformer 的计算复杂度

- **自注意力**：O(n² · d)，其中 n 是序列长度，d 是维度
- **FFN**：O(n · d²)

当序列很长时，自注意力的 O(n²) 成为瓶颈，这也是后续各种高效 Attention 方法的研究动机。

---

## 二、注意力机制

### 2.1 Scaled Dot-Product Attention

**核心公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**分步解释**：

1. **计算相似度**：$QK^T$
   - Q（Query）：当前位置"想找什么"
   - K（Key）：每个位置"是什么"
   - 结果是 [seq_len, seq_len] 的相似度矩阵

2. **缩放**：除以 $\sqrt{d_k}$
   - 防止点积值过大导致 softmax 饱和
   - 当 d_k 很大时，点积的方差为 d_k，缩放后方差恢复为 1

3. **Softmax**：转换为概率分布
   - 每行的和为 1
   - 表示当前位置对其他位置的注意力权重

4. **加权求和**：用注意力权重对 V（Value）加权
   - V 代表每个位置的"信息内容"
   - 输出是信息的加权平均

### 2.2 为什么要除以 sqrt(d_k)？

**问题**：当 d_k 较大时，点积的值会变得很大。

**数学解释**：
- 假设 q 和 k 的每个分量都是均值0、方差1的独立随机变量
- 点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$
- 点积的期望为 0，**方差为 d_k**

**后果**：
- 大的点积值会使 softmax 输出趋于 one-hot（极度尖锐）
- 梯度接近于 0，难以训练

**解决**：
- 除以 $\sqrt{d_k}$，使方差恢复为 1
- softmax 输出更平滑，梯度更稳定

### 2.3 多头注意力（Multi-Head Attention）

**为什么需要多头？**

单头注意力只能学习一种注意力模式，但语言理解需要同时关注多种关系：
- 语法关系（主谓宾）
- 语义关系（同义词、反义词）
- 位置关系（相邻词）
- 指代关系（代词指向）

**多头注意力公式**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**实现要点**：
- 将 d_model 分成 h 个头，每个头的维度 d_k = d_model / h
- 各头**并行计算**，最后拼接
- 通过 reshape 和 transpose 实现，无需创建多个模块

**代码参考**：见 `attention_easy.py` 中的 `MultiHeadAttention` 类

### 2.4 自注意力 vs 交叉注意力

| 类型 | Q 来源 | K/V 来源 | 用途 |
|------|--------|----------|------|
| **自注意力** | 同一序列 | 同一序列 | Encoder、Decoder的第一个注意力层 |
| **交叉注意力** | Decoder | Encoder输出 | Decoder的第二个注意力层，实现编解码器交互 |

---

## 三、位置编码

### 3.1 为什么需要位置编码？

Transformer 的自注意力是**置换不变**的：
- 打乱输入顺序，注意力权重不变
- 但语言是有顺序的，"我爱你"≠"你爱我"

**解决方案**：在输入嵌入中加入位置信息

### 3.2 Sinusoidal 位置编码（原始 Transformer）

**公式**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**特点**：
- 不需要学习，直接计算
- 不同维度对应不同频率的正弦波
- 相对位置可以通过线性变换表示：$PE_{pos+k}$ 可由 $PE_{pos}$ 线性变换得到

**代码实现**：见 `positional_encoding.py`

### 3.3 可学习位置编码（BERT）

```python
self.position_embeddings = nn.Embedding(max_position, d_model)
```

**特点**：
- 位置编码作为可学习参数
- 更灵活，但受最大长度限制
- BERT 使用此方式

### 3.4 RoPE 旋转位置编码（LLaMA、Qwen）

**核心思想**：
- 不直接加位置编码到嵌入
- 而是**旋转** Q 和 K 向量
- 使得 $q_m \cdot k_n$ 的结果只依赖于相对位置 $m - n$

**公式**（简化版）：

$$
q_m = R_m \cdot q, \quad k_n = R_n \cdot k
$$

$$
q_m^T k_n = q^T R_m^T R_n k = q^T R_{n-m} k
$$

其中 $R_\theta$ 是旋转矩阵。

**优点**：
- 相对位置编码，泛化性更好
- 具有外推性，可处理比训练时更长的序列
- 计算高效

**代码实现**：见 `positional_encoding.py`

### 3.5 ALiBi（Attention with Linear Biases）

**核心思想**：
- 不修改嵌入，而是在注意力分数上加偏置
- 偏置与相对距离成比例：距离越远，惩罚越大

$$
\text{softmax}(q_i^T k_j - m \cdot |i - j|)
$$

**优点**：
- 无需额外参数
- 外推性能优秀
- 本质是局部注意力的软版本

### 3.6 位置编码对比

| 方法 | 类型 | 外推性 | 额外参数 | 代表模型 |
|------|------|--------|----------|----------|
| Sinusoidal | 绝对 | 差 | 无 | 原始Transformer |
| Learned | 绝对 | 差 | 有 | BERT、GPT-2 |
| RoPE | 相对 | 较好 | 无 | LLaMA、Qwen |
| ALiBi | 相对 | 优秀 | 无 | BLOOM |

---

## 四、核心组件

### 4.1 Feed-Forward Network（FFN）

**结构**：

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),      # 升维：d_model → d_ff
    nn.GELU(),                      # 激活函数
    nn.Linear(d_ff, d_model),       # 降维：d_ff → d_model
    nn.Dropout(dropout)
)
```

**关键点**：
- d_ff 通常是 d_model 的 4 倍（如 512 → 2048）
- 这是一个**位置无关**的变换（每个位置独立计算）
- 相当于两层 MLP

**为什么需要 FFN？**
- 注意力层负责信息交换（位置间交互）
- FFN 负责特征变换（位置内变换）
- 两者配合，既有交互又有变换

### 4.2 残差连接（Residual Connection）

**公式**：

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

**作用**：
1. **缓解梯度消失**：梯度可以直接通过残差连接流回
2. **加速收敛**：网络可以学习恒等映射
3. **信息保留**：底层信息可以直接传递到高层

### 4.3 Layer Normalization

**公式**：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

其中 μ 和 σ 是在最后一个维度（特征维度）上计算的均值和标准差。

**Layer Norm vs Batch Norm**：

| 特性 | Batch Norm | Layer Norm |
|------|------------|------------|
| 归一化维度 | 批次维度 | 特征维度 |
| 依赖批次大小 | 是 | 否 |
| 适用场景 | CV | NLP/Transformer |
| 序列长度 | 需固定 | 可变 |

### 4.4 Pre-Norm vs Post-Norm

**Post-Norm**（原始 Transformer）：
```
x → Sublayer → Add → LayerNorm → output
```

**Pre-Norm**（现代 LLM 常用）：
```
x → LayerNorm → Sublayer → Add → output
```

**Pre-Norm 优点**：
- 训练更稳定
- 不需要精心设计的学习率 warmup
- 梯度流更顺畅

**现代 LLM 趋势**：使用 Pre-Norm + RMSNorm

### 4.5 RMSNorm（Root Mean Square Normalization）

**公式**：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}} \cdot \gamma
$$

**与 LayerNorm 区别**：
- 不计算均值，只归一化方差
- 计算更简单，速度更快
- LLaMA、Qwen 等模型使用

---

## 五、Mask 机制

### 5.1 Padding Mask

**目的**：屏蔽填充位置，避免其参与注意力计算

```python
# 输入序列（0表示padding）
# [我, 爱, 你, PAD, PAD]
# mask = [1, 1, 1, 0, 0]

# 应用于注意力分数
scores = scores.masked_fill(mask == 0, float('-inf'))
# softmax后，padding位置的注意力权重变为0
```

### 5.2 Causal Mask（因果掩码）

**目的**：在 Decoder 中，确保位置 i 只能看到位置 0 到 i-1

```python
def create_causal_mask(seq_len):
    """创建下三角掩码"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# 结果示例（seq_len=4）：
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

**为什么需要因果掩码？**
- 训练时：并行计算所有位置的损失，但每个位置不能"偷看"未来
- 推理时：自回归生成，逐个生成 token

### 5.3 代码示例

见 `attention_easy.py` 中的 `create_causal_mask()` 函数和 `demo_causal_mask()` 演示。

---

## 六、架构对比

### 6.1 三种主流架构

| 架构 | 注意力类型 | 代表模型 | 典型任务 |
|------|-----------|----------|----------|
| **Encoder-only** | 双向自注意力 | BERT、RoBERTa | 分类、NER、问答 |
| **Decoder-only** | 单向自注意力（因果） | GPT、LLaMA | 文本生成、对话 |
| **Encoder-Decoder** | 双向 + 单向 + 交叉 | T5、BART | 翻译、摘要 |

### 6.2 为什么大模型采用 Decoder-only？

**主要原因**：

1. **训练效率**
   - 自回归目标简单：预测下一个 token
   - 不需要设计复杂的预训练任务（如 BERT 的 MLM）

2. **工程简单**
   - 只有一个模块，代码更简洁
   - 推理时无需分离 encoder 和 decoder

3. **理论分析**（Dong et al., 2023）
   - Encoder 的双向注意力存在**低秩问题**
   - 注意力矩阵容易退化，影响表达能力
   - Decoder-only 的因果注意力矩阵更稳定

4. **涌现能力**
   - 规模扩大后，decoder-only 展现出更强的涌现能力
   - In-context Learning 等能力主要在 decoder-only 模型中观察到

### 6.3 Prefix LM vs Causal LM

| 类型 | 前缀部分 | 生成部分 | 代表模型 |
|------|----------|----------|----------|
| **Causal LM** | 全部单向 | 全部单向 | GPT、LLaMA |
| **Prefix LM** | 双向 | 单向 | T5、GLM |

**Prefix LM** 的优势：
- 前缀部分可以双向编码，理解更充分
- 适合条件生成任务

---

## 七、Tokenizer

### 7.1 为什么需要 Tokenizer？

模型只能处理数字，需要将文本转换为 token ID：
```
"Hello world" → [15496, 995] → 模型处理 → [32, 48, ...] → "你好世界"
```

### 7.2 BPE（Byte Pair Encoding）

**算法步骤**：
1. 将所有单词拆分成字符
2. 统计相邻字符对的频率
3. 将最频繁的字符对合并成新符号
4. 重复步骤 2-3，直到达到目标词表大小

**示例**：
```
初始词表: [a, b, c, d, ...]
文本: "aaabdaaabac"

第1轮: "aa"出现最多 → 合并为 "Z" → "ZabdZabac"
第2轮: "Za"出现最多 → 合并为 "Y" → "YbdYbac"
...
```

**优点**：
- 平衡词表大小和序列长度
- 可以处理未见过的词（拆分成子词）
- GPT 系列使用

### 7.3 WordPiece

与 BPE 类似，但选择合并对的标准不同：
- BPE：选择频率最高的
- WordPiece：选择使语言模型困惑度下降最多的

**BERT 使用 WordPiece**。

### 7.4 SentencePiece

- 直接在原始文本上训练（不需要预分词）
- 将空格也当作普通字符处理
- 支持 BPE 和 Unigram 两种算法
- LLaMA、T5 使用

---

## 八、面试高频问题

### Q1: Transformer 相比 RNN 的优势是什么？

**参考答案**：
1. **并行计算**：Transformer 可以并行处理整个序列，而 RNN 必须顺序处理
2. **长距离依赖**：自注意力直接建模任意位置间的关系，无梯度消失问题
3. **可扩展性**：更容易扩大模型规模

### Q2: 为什么注意力要除以 sqrt(d_k)？

**参考答案**：
当维度 d_k 较大时，点积的方差会变大（方差为 d_k）。大的点积值会使 softmax 输出趋于 one-hot，梯度接近 0。除以 sqrt(d_k) 使方差恢复为 1，让梯度更稳定。

### Q3: 多头注意力的作用是什么？

**参考答案**：
不同的头可以学习不同的注意力模式，如语法关系、语义关系、位置关系等。相当于在多个子空间并行计算注意力，提高模型的表达能力。

### Q4: Pre-Norm 和 Post-Norm 的区别？

**参考答案**：
- Post-Norm：先 Sublayer 再 Norm，原始 Transformer 使用
- Pre-Norm：先 Norm 再 Sublayer，现代 LLM 常用
- Pre-Norm 训练更稳定，不需要复杂的 warmup

### Q5: 为什么大模型都用 Decoder-only 架构？

**参考答案**：
1. 训练效率高：只需预测下一个 token，目标简单
2. 工程简单：只有一个模块
3. 理论优势：双向注意力存在低秩问题，影响表达能力
4. 涌现能力强：规模扩大后展现出更强的能力

### Q6: RoPE 相比 Sinusoidal 位置编码的优势？

**参考答案**：
1. RoPE 是相对位置编码，注意力分数只依赖相对位置
2. 具有更好的外推性，可处理比训练时更长的序列
3. 计算高效，不增加额外参数

### Q7: KV Cache 的原理是什么？

**参考答案**：
在自回归生成时，已生成 token 的 K、V 不会改变。KV Cache 将这些 K、V 缓存起来，避免重复计算。代价是增加显存占用，换取推理速度提升。

### Q8: BPE 算法的基本流程？

**参考答案**：
1. 初始化：将文本拆分成字符
2. 统计所有相邻字符对的频率
3. 将频率最高的字符对合并成新符号
4. 重复步骤 2-3 直到达到目标词表大小

### Q9: 如何理解 Transformer 中的残差连接？

**参考答案**：
1. 缓解梯度消失：梯度可以直接通过残差连接回传
2. 加速收敛：网络容易学习恒等映射
3. 信息保留：底层特征可以直接传递到高层

### Q10: Causal Mask 的作用是什么？

**参考答案**：
在 Decoder 中，确保位置 i 只能看到位置 0 到 i-1，防止"偷看"未来信息。实现方式是使用下三角掩码，将右上角的注意力分数设为负无穷。

---

## 推荐阅读

1. **原始论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. **RoPE 论文**：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
3. **ALiBi 论文**：[Train Short, Test Long](https://arxiv.org/abs/2108.12409)
4. **配套代码**：`attention_easy.py`、`positional_encoding.py`
