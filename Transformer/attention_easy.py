"""
===============================================================================
                    Attention 机制 - 简单易懂版本
===============================================================================

这个文件的目的是用最简单、最直观的方式帮助你理解 Transformer 中的 Attention 机制。
我们会从最基础的概念开始，一步步构建完整的 Multi-Head Attention。

目录：
    1. 什么是 Attention？（概念解释）
    2. scaled_dot_product_attention() - 最核心的函数
    3. SingleHeadAttention - 单头注意力
    4. MultiHeadAttention - 多头注意力
    5. 完整的可视化演示

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 第一部分：什么是 Attention？
# =============================================================================
"""
【直观理解 Attention】

想象你在读一句话："小猫追着球跑，它很开心"

当你读到"它"这个词时，你的大脑会自动"注意"到前面的"小猫"，
因为"它"指代的就是"小猫"。

这就是 Attention 的本质：
    - 在处理序列中的每个位置时，决定应该"关注"其他哪些位置
    - 不同位置之间的关联程度由"注意力分数"表示

【Q, K, V 的含义】

Attention 使用三个概念：Query（查询）、Key（键）、Value（值）

用图书馆的比喻来理解：
    - Query (Q): "我想找什么？" —— 代表当前位置的需求
    - Key (K):   "这本书是关于什么的？" —— 代表每个位置的标签/索引
    - Value (V): "这本书的内容是什么？" —— 代表每个位置的实际信息

工作流程：
    1. 用 Query 去和每个 Key 比较，得到相似度分数
    2. 把分数转换成概率（用 softmax）
    3. 用这些概率对 Value 进行加权求和

数学公式：
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

其中：
    - Q @ K^T: 计算 Query 和 Key 的点积相似度
    - sqrt(d_k): 缩放因子，防止点积值过大导致 softmax 饱和
    - softmax: 将分数转换为概率分布（和为1）
"""


# =============================================================================
# 第二部分：缩放点积注意力 - 最核心的函数
# =============================================================================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力 - Attention 的核心计算

    这是整个 Transformer 中最重要的函数，理解了它就理解了 Attention 的本质。

    参数说明：
        query: 查询张量，形状 [batch_size, seq_len, d_k]
               - batch_size: 批次大小（同时处理多少个样本）
               - seq_len: 序列长度（如句子有多少个词）
               - d_k: 每个向量的维度

        key:   键张量，形状 [batch_size, seq_len, d_k]

        value: 值张量，形状 [batch_size, seq_len, d_v]
               - d_v 可以和 d_k 不同，但通常相等

        mask:  可选的掩码张量，用于屏蔽某些位置
               - 在解码器中用于防止"偷看"未来的词
               - 形状 [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]

    返回：
        output: 注意力输出，形状 [batch_size, seq_len, d_v]
        attention_weights: 注意力权重，形状 [batch_size, seq_len, seq_len]
    """

    # -------------------------------------------------------------------------
    # 步骤 1：计算注意力分数（相似度）
    # -------------------------------------------------------------------------
    #
    # Q @ K^T 的含义：
    #   - Q 中的每个向量都和 K 中的所有向量做点积
    #   - 点积越大，表示两个向量越相似
    #
    # 形状变化：
    #   Q: [batch_size, seq_len_q, d_k]
    #   K^T: [batch_size, d_k, seq_len_k]  (转置最后两个维度)
    #   结果: [batch_size, seq_len_q, seq_len_k]
    #
    # 举例：如果序列长度为5，结果就是一个 5x5 的矩阵
    #       scores[i][j] 表示位置 i 对位置 j 的注意力分数

    d_k = query.size(-1)  # 获取维度 d_k

    # key.transpose(-2, -1) 将 key 的最后两个维度交换
    # 从 [batch, seq_len, d_k] 变为 [batch, d_k, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 此时 scores 的形状是 [batch_size, seq_len_q, seq_len_k]
    # 每一行代表一个查询位置对所有键位置的相似度分数

    # -------------------------------------------------------------------------
    # 步骤 2：缩放分数
    # -------------------------------------------------------------------------
    #
    # 为什么要除以 sqrt(d_k)？
    #
    # 问题：当 d_k 很大时，点积的方差也会很大
    #      - 假设 q 和 k 的每个分量都是均值0、方差1的随机变量
    #      - 它们的点积 = q1*k1 + q2*k2 + ... + qd*kd
    #      - 点积的方差 = d_k（因为每一项的方差是1，共有d_k项）
    #
    # 后果：大方差意味着有些值会特别大或特别小
    #      当值很大时，softmax 会变得非常"尖锐"
    #      几乎所有的权重都集中在一个位置上，梯度接近于0
    #
    # 解决：除以 sqrt(d_k)，使方差恢复到1
    #      这样 softmax 的输出分布更加平滑，梯度更稳定

    scores = scores / math.sqrt(d_k)

    # -------------------------------------------------------------------------
    # 步骤 3：应用掩码（如果有的话）
    # -------------------------------------------------------------------------
    #
    # 掩码的作用：让某些位置的注意力分数变成负无穷
    #            经过 softmax 后，这些位置的权重就会接近0
    #
    # 常见的掩码类型：
    #   1. Padding Mask: 屏蔽填充位置（因为填充的 token 没有意义）
    #   2. Causal Mask（因果掩码）: 在解码器中，位置 i 只能看到 i 之前的位置
    #      这是一个下三角矩阵，右上角都是 -inf
    #
    # 掩码示例（因果掩码）：
    #   [0, -inf, -inf, -inf]   位置0只能看自己
    #   [0,   0,  -inf, -inf]   位置1能看0和1
    #   [0,   0,    0,  -inf]   位置2能看0,1,2
    #   [0,   0,    0,    0 ]   位置3能看所有

    if mask is not None:
        # 将 mask 为 0 的位置填充为负无穷
        # masked_fill_ 是就地操作，这里用 masked_fill 返回新张量
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # -------------------------------------------------------------------------
    # 步骤 4：应用 Softmax 获取注意力权重
    # -------------------------------------------------------------------------
    #
    # Softmax 的作用：
    #   1. 将任意实数转换为正数
    #   2. 使所有值的和为1（形成概率分布）
    #
    # 公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
    #
    # 在最后一个维度上做 softmax (dim=-1)
    # 意味着：对于每个查询位置，它对所有键位置的注意力权重和为1

    attention_weights = F.softmax(scores, dim=-1)

    # 处理 NaN（当某行全是 -inf 时，softmax 会产生 NaN）
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # -------------------------------------------------------------------------
    # 步骤 5：加权求和得到输出
    # -------------------------------------------------------------------------
    #
    # 用注意力权重对 Value 进行加权求和
    #
    # 形状变化：
    #   attention_weights: [batch_size, seq_len_q, seq_len_k]
    #   V: [batch_size, seq_len_k, d_v]
    #   结果: [batch_size, seq_len_q, d_v]
    #
    # 直观理解：
    #   output[i] = sum(attention_weights[i][j] * V[j]) for all j
    #   即：输出的每个位置是所有 Value 的加权平均

    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# =============================================================================
# 第三部分：单头注意力 - 最简单的 Attention 模块
# =============================================================================

class SingleHeadAttention(nn.Module):
    """
    单头自注意力层

    这是最简单的 Attention 模块，用于帮助理解基本概念。
    后面的 Multi-Head Attention 就是多个这样的模块并行工作。

    "自注意力"(Self-Attention) 的含义：
        - Q, K, V 都来自同一个输入
        - 序列中的每个位置都会"关注"序列中的其他位置
        - 这让模型能够捕捉序列内部的依赖关系

    参数：
        d_model: 输入和输出的维度
        d_k: Query 和 Key 的维度（可以和 d_model 不同）
        d_v: Value 的维度（可以和 d_model 不同）
    """

    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()

        # 如果没有指定 d_k 和 d_v，默认使用 d_model
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.d_k = d_k
        self.d_v = d_v

        # =====================================================================
        # 线性投影层
        # =====================================================================
        #
        # 为什么需要这些投影层？
        #
        # 原始输入 x 的每个向量只有一种"身份"
        # 但在 Attention 中，同一个向量需要扮演三个角色：
        #   - 作为 Query：它想要什么信息？
        #   - 作为 Key：它代表什么信息？
        #   - 作为 Value：它包含什么信息？
        #
        # 这三个角色需要不同的表示，所以我们用三个不同的线性变换
        #
        # 线性变换 = 矩阵乘法：y = xW + b
        # nn.Linear(in_features, out_features) 做的就是这个操作

        self.W_q = nn.Linear(d_model, d_k)  # Query 投影：d_model -> d_k
        self.W_k = nn.Linear(d_model, d_k)  # Key 投影：d_model -> d_k
        self.W_v = nn.Linear(d_model, d_v)  # Value 投影：d_model -> d_v

        # 输出投影：将注意力输出映射回 d_model 维度
        self.W_o = nn.Linear(d_v, d_model)

    def forward(self, x, mask=None):
        """
        前向传播

        参数：
            x: 输入张量，形状 [batch_size, seq_len, d_model]
            mask: 可选的注意力掩码

        返回：
            output: 输出张量，形状 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重，形状 [batch_size, seq_len, seq_len]
        """

        # 步骤 1：生成 Q, K, V
        # 输入 x 通过三个不同的线性变换，得到不同的表示
        Q = self.W_q(x)  # [batch_size, seq_len, d_k]
        K = self.W_k(x)  # [batch_size, seq_len, d_k]
        V = self.W_v(x)  # [batch_size, seq_len, d_v]

        # 步骤 2：计算注意力
        # 这里调用我们前面定义的核心函数
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 步骤 3：输出投影
        # 将注意力输出映射回原始维度
        output = self.W_o(attn_output)  # [batch_size, seq_len, d_model]

        return output, attn_weights


# =============================================================================
# 第四部分：多头注意力 - Transformer 中实际使用的版本
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)

    =========================================================================
    为什么需要"多头"？
    =========================================================================

    单头注意力有个局限：每个位置只能学习一种注意力模式

    但在理解语言时，一个词可能需要同时关注多种不同的关系：
        - 语法关系：主语看谓语
        - 语义关系：代词看它指代的名词
        - 位置关系：临近的词可能更相关
        - 其他关系：......

    多头注意力的思想：
        - 让模型同时学习多种不同的注意力模式
        - 每个"头"专注于学习一种模式
        - 最后把所有头的输出拼接起来

    =========================================================================
    实现方式
    =========================================================================

    方法1（直观但低效）：
        - 创建 num_heads 个独立的 SingleHeadAttention
        - 分别计算，最后拼接

    方法2（高效实现，本文使用）：
        - 用一个大的线性层同时计算所有头的 Q/K/V
        - 通过 reshape 把它们分成多个头
        - 并行计算所有头的注意力
        - reshape 回来并投影

    关键技巧：
        - 如果 d_model = 512, num_heads = 8
        - 每个头的维度 d_k = 512 / 8 = 64
        - 投影后的总维度还是 512，只是被分成了 8 份

    参数：
        d_model: 模型维度
        num_heads: 注意力头的数量
        dropout: Dropout 概率
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # d_model 必须能被 num_heads 整除
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # =====================================================================
        # 线性投影层（为所有头共享一个大矩阵）
        # =====================================================================
        #
        # 虽然叫"多头"，但我们只用一个线性层来生成所有头的 Q/K/V
        # 输出维度还是 d_model，但会被 reshape 成 num_heads 份
        #
        # 这比创建 num_heads 个小矩阵更高效
        # 因为一次大的矩阵乘法比多次小的矩阵乘法更快（GPU并行优势）

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播

        参数：
            query: 查询，形状 [batch_size, seq_len, d_model]
            key:   键，形状 [batch_size, seq_len, d_model]
            value: 值，形状 [batch_size, seq_len, d_model]
            mask:  可选的掩码

        返回：
            output: 形状 [batch_size, seq_len, d_model]
            attention_weights: 形状 [batch_size, num_heads, seq_len, seq_len]

        注意：
            - 在自注意力中，query = key = value = x（同一个输入）
            - 在交叉注意力中，query 来自解码器，key/value 来自编码器
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # =====================================================================
        # 步骤 1：线性投影
        # =====================================================================
        # 将输入投影到 Q, K, V 空间

        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)    # [batch_size, seq_len, d_model]
        V = self.W_v(value)  # [batch_size, seq_len, d_model]

        # =====================================================================
        # 步骤 2：分割成多个头
        # =====================================================================
        #
        # 这是多头注意力的关键步骤！
        #
        # 原始形状: [batch_size, seq_len, d_model]
        # 目标形状: [batch_size, num_heads, seq_len, d_k]
        #
        # 变换过程：
        #   1. view: [batch, seq, d_model] -> [batch, seq, num_heads, d_k]
        #      把 d_model 拆分成 num_heads 个 d_k
        #   2. transpose: [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        #      把 num_heads 移到前面，方便并行计算

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 现在形状是 [batch_size, num_heads, seq_len, d_k]

        # =====================================================================
        # 步骤 3：计算注意力（所有头并行）
        # =====================================================================
        #
        # 由于 batch 维度现在是 [batch_size, num_heads, ...]
        # 矩阵乘法会自动在前两个维度上广播
        # 相当于同时计算了 batch_size * num_heads 个注意力

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, num_heads, seq_len, seq_len]

        # 应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配 [batch, num_heads, seq, seq]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        context = torch.matmul(attention_weights, V)
        # context: [batch_size, num_heads, seq_len, d_k]

        # =====================================================================
        # 步骤 4：合并多个头
        # =====================================================================
        #
        # 变换过程（与分割相反）：
        #   1. transpose: [batch, num_heads, seq, d_k] -> [batch, seq, num_heads, d_k]
        #   2. view: [batch, seq, num_heads, d_k] -> [batch, seq, d_model]
        #
        # contiguous(): 确保内存连续，因为 transpose 后可能不连续

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        # =====================================================================
        # 步骤 5：输出投影
        # =====================================================================
        # 最后再做一次线性变换，让不同头的信息进行融合

        output = self.W_o(context)

        return output, attention_weights


# =============================================================================
# 第五部分：辅助函数 - 创建因果掩码
# =============================================================================

def create_causal_mask(seq_len):
    """
    创建因果掩码（Causal Mask / Look-ahead Mask）

    用途：在解码器中，防止位置 i 看到位置 i+1, i+2, ... 的信息
          这是为了保证自回归生成时的正确性

    返回的掩码是一个下三角矩阵：
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    1 表示可以看到，0 表示看不到（会被 masked_fill 替换为 -inf）

    参数：
        seq_len: 序列长度

    返回：
        mask: 形状 [seq_len, seq_len] 的下三角掩码
    """
    # torch.tril 创建下三角矩阵
    # torch.ones 创建全1矩阵，然后 tril 把上三角变成0
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


# =============================================================================
# 第六部分：演示代码
# =============================================================================

def demo_attention_step_by_step():
    """
    逐步演示 Attention 的计算过程

    这个函数会打印每一步的中间结果，帮助你理解数据是如何流动的
    """
    print("\n" + "=" * 70)
    print("         Attention 计算过程 - 逐步演示")
    print("=" * 70)

    # 使用一个简单的例子
    batch_size = 1
    seq_len = 4
    d_model = 8

    # 创建一个简单的输入
    # 为了演示清晰，使用小的数值
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n【输入】")
    print(f"  形状: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]")
    print(f"  这代表 {batch_size} 个样本，每个样本有 {seq_len} 个位置，")
    print(f"  每个位置是一个 {d_model} 维的向量")

    # 创建简单的投影矩阵（这里用单位矩阵简化演示）
    W_q = torch.eye(d_model)
    W_k = torch.eye(d_model)
    W_v = torch.eye(d_model)

    # 生成 Q, K, V（这里 Q=K=V=x，因为使用单位矩阵）
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    print(f"\n【步骤 1: 生成 Q, K, V】")
    print(f"  Q = x @ W_q, 形状: {list(Q.shape)}")
    print(f"  K = x @ W_k, 形状: {list(K.shape)}")
    print(f"  V = x @ W_v, 形状: {list(V.shape)}")

    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"\n【步骤 2: 计算注意力分数】")
    print(f"  scores = Q @ K^T, 形状: {list(scores.shape)}")
    print(f"  这是一个 {seq_len}x{seq_len} 的矩阵")
    print(f"  scores[i][j] 表示位置 i 对位置 j 的原始注意力分数")
    print(f"\n  注意力分数矩阵（未缩放）:")
    print(scores[0].detach().numpy().round(2))

    # 缩放
    d_k = d_model
    scores_scaled = scores / math.sqrt(d_k)
    print(f"\n【步骤 3: 缩放】")
    print(f"  scores = scores / sqrt(d_k) = scores / sqrt({d_k})")
    print(f"  缩放后的分数矩阵:")
    print(scores_scaled[0].detach().numpy().round(2))

    # Softmax
    attention_weights = F.softmax(scores_scaled, dim=-1)
    print(f"\n【步骤 4: Softmax】")
    print(f"  attention_weights = softmax(scores, dim=-1)")
    print(f"  形状: {list(attention_weights.shape)}")
    print(f"\n  注意力权重矩阵（每行和为1）:")
    print(attention_weights[0].detach().numpy().round(3))
    print(f"\n  验证每行的和: {attention_weights[0].sum(dim=-1).detach().numpy().round(3)}")

    # 加权求和
    output = torch.matmul(attention_weights, V)
    print(f"\n【步骤 5: 加权求和】")
    print(f"  output = attention_weights @ V")
    print(f"  输出形状: {list(output.shape)}")
    print(f"\n  输出[0][0]（第一个位置的输出）:")
    print(f"  {output[0][0].detach().numpy().round(3)}")

    print("\n" + "=" * 70)


def demo_causal_mask():
    """
    演示因果掩码的效果
    """
    print("\n" + "=" * 70)
    print("         因果掩码 (Causal Mask) 演示")
    print("=" * 70)

    seq_len = 5

    print(f"\n序列长度: {seq_len}")
    print("\n【因果掩码矩阵】")
    mask = create_causal_mask(seq_len)
    print(mask.numpy().astype(int))
    print("\n解释：")
    print("  - 1 表示可以'看到'那个位置")
    print("  - 0 表示'看不到'那个位置（会被设为负无穷）")
    print("  - 每一行代表一个查询位置能看到哪些键位置")
    print("  - 第 i 行只有前 i+1 个位置是1，其余是0")

    # 演示掩码对注意力分数的影响
    print("\n【掩码前的注意力分数（假设）】")
    scores_before = torch.ones(seq_len, seq_len)
    print(scores_before.numpy().astype(int))

    print("\n【应用掩码后】")
    scores_after = scores_before.masked_fill(mask == 0, float('-inf'))
    # 为了显示方便，把 -inf 显示为 -999
    display = scores_after.clone()
    display[display == float('-inf')] = -999
    print(display.numpy().astype(int))
    print("  (这里 -999 代表负无穷)")

    print("\n【Softmax 后的注意力权重】")
    attn_weights = F.softmax(scores_after, dim=-1)
    print(attn_weights.numpy().round(3))
    print("\n解释：")
    print("  - 被掩码的位置（原来是负无穷）现在变成了 0")
    print("  - 每行的非零权重均匀分布（因为原始分数都是1）")
    print("  - 位置 i 只会'注意'位置 0 到 i")

    print("\n" + "=" * 70)


def demo_multi_head():
    """
    演示多头注意力
    """
    print("\n" + "=" * 70)
    print("         多头注意力 (Multi-Head Attention) 演示")
    print("=" * 70)

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4

    print(f"\n【参数设置】")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (每个头的维度): {d_model // num_heads}")

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n【输入】")
    print(f"  形状: {list(x.shape)}")

    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)

    # 前向传播（自注意力：Q=K=V=x）
    output, attn_weights = mha(x, x, x)

    print(f"\n【输出】")
    print(f"  output 形状: {list(output.shape)}")
    print(f"  attention_weights 形状: {list(attn_weights.shape)}")
    print(f"\n  attention_weights 的含义:")
    print(f"    - 维度 0 (batch_size={batch_size}): 批次中的样本")
    print(f"    - 维度 1 (num_heads={num_heads}): 不同的注意力头")
    print(f"    - 维度 2 (seq_len={seq_len}): 查询位置")
    print(f"    - 维度 3 (seq_len={seq_len}): 键位置")

    print(f"\n【注意力权重验证】")
    print(f"  每个查询位置的权重和（应该都是1.0）:")
    print(f"  {attn_weights[0, 0].sum(dim=-1).detach().numpy().round(3)}")

    # 演示不同头学习不同的模式
    print(f"\n【不同头的注意力模式】")
    print(f"  第一个样本、第一个查询位置对所有键位置的注意力:")
    for head in range(num_heads):
        weights = attn_weights[0, head, 0].detach().numpy().round(3)
        print(f"    头 {head}: {weights}")
    print(f"\n  可以看到不同的头学习了不同的注意力分布")

    print("\n" + "=" * 70)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("             Attention 机制学习演示")
    print("=" * 70)
    print("""
    这个脚本将演示 Attention 机制的核心概念：

    1. Attention 计算的逐步分解
    2. 因果掩码的作用
    3. 多头注意力的工作原理

    建议阅读顺序：
    1. 先阅读本文件开头的注释，理解 Q, K, V 的概念
    2. 阅读 scaled_dot_product_attention 函数的注释
    3. 运行下面的演示，观察数据形状的变化
    """)

    input("按 Enter 键开始第一个演示（逐步分解 Attention 计算）...")
    demo_attention_step_by_step()

    input("\n按 Enter 键开始第二个演示（因果掩码）...")
    demo_causal_mask()

    input("\n按 Enter 键开始第三个演示（多头注意力）...")
    demo_multi_head()

    print("\n" + "=" * 70)
    print("                    演示结束")
    print("=" * 70)
    print("""
    【学习建议】

    1. 反复阅读 scaled_dot_product_attention 函数
       这是 Attention 的核心，理解它就理解了大部分内容

    2. 用 print 打印中间结果
       在代码中添加 print 语句，观察张量形状的变化

    3. 修改参数
       尝试改变 d_model, num_heads, seq_len 等参数
       观察输出形状如何变化

    4. 画图理解
       在纸上画出 Q, K, V 的矩阵乘法过程
       这对理解形状变换很有帮助

    5. 阅读原论文
       "Attention Is All You Need" 是 Transformer 的原始论文
       论文中的图3清晰地展示了 Multi-Head Attention 的结构
    """)
