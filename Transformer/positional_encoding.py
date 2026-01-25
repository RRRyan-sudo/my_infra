"""
===============================================================================
                    位置编码（Positional Encoding）实现
===============================================================================

本文件实现了 Transformer 中常用的位置编码方法：
1. Sinusoidal 位置编码（原始 Transformer）
2. 可学习位置编码（BERT）
3. RoPE 旋转位置编码（LLaMA、Qwen）

每种方法都包含详细的中文注释，帮助理解其原理和实现。

===============================================================================
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# 第一部分：Sinusoidal 位置编码（原始 Transformer）
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦余弦位置编码 - 原始 Transformer 使用的方法

    原理：
        使用不同频率的正弦和余弦函数来编码位置信息。

        公式：
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        其中：
            - pos：token 在序列中的位置（0, 1, 2, ...）
            - i：维度索引（0, 1, 2, ..., d_model/2）
            - d_model：嵌入维度

    为什么用正弦余弦？
        1. 有界性：值域在 [-1, 1]，不会太大
        2. 周期性：不同维度有不同的周期
        3. 相对位置：PE(pos+k) 可以由 PE(pos) 线性变换得到
           这是因为 sin(a+b) = sin(a)cos(b) + cos(a)sin(b)

    参数：
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: dropout 概率
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 计算分母中的指数项
        # div_term = 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        # 这里使用 exp 和 log 是为了数值稳定性
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 填充位置编码
        # 偶数维度使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与训练，但会保存和加载）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 [batch_size, seq_len, d_model]

        返回：
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        # 取出对应长度的位置编码，加到输入上
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# =============================================================================
# 第二部分：可学习位置编码（BERT）
# =============================================================================

class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码 - BERT、GPT-2 使用的方法

    原理：
        将位置编码作为可学习的参数，通过训练学习最优的位置表示。
        实现上就是一个 Embedding 层，输入是位置索引，输出是位置嵌入。

    优点：
        - 更灵活，可以学习任务特定的位置模式
        - 实现简单

    缺点：
        - 受最大长度限制，无法外推
        - 增加了参数量

    参数：
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: dropout 概率
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置嵌入层
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # 位置索引 [0, 1, 2, ..., max_len-1]
        position_ids = torch.arange(max_len).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 [batch_size, seq_len, d_model]

        返回：
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(self.position_ids[:, :seq_len])

        # 加到输入上
        x = x + position_embeddings
        return self.dropout(x)


# =============================================================================
# 第三部分：RoPE 旋转位置编码（LLaMA、Qwen）
# =============================================================================

class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码（RoPE）- LLaMA、Qwen 等现代 LLM 使用

    核心思想：
        不是将位置编码加到嵌入上，而是通过旋转变换来编码位置。

        对于位置 m 的 query 向量 q 和位置 n 的 key 向量 k：
            q_m = R(m) * q
            k_n = R(n) * k

        其中 R(θ) 是旋转矩阵。

    关键性质：
        q_m · k_n = (R(m)q) · (R(n)k) = q · R(m-n) · k

        注意力分数只依赖于相对位置 (m-n)，这是相对位置编码的特性！

    实现细节：
        将向量的维度两两配对，每对构成一个 2D 平面。
        对每个平面应用旋转变换。

        例如 d_model=8 时：
            [x0, x1] → 旋转角度 θ0
            [x2, x3] → 旋转角度 θ1
            [x4, x5] → 旋转角度 θ2
            [x6, x7] → 旋转角度 θ3

    优点：
        1. 相对位置编码，泛化性好
        2. 具有外推性，可处理比训练时更长的序列
        3. 计算高效，不增加额外参数

    参数：
        d_model: 嵌入维度（必须是偶数）
        max_len: 最大序列长度
        base: 基础频率（默认 10000）
    """

    def __init__(self, d_model, max_len=2048, base=10000):
        super().__init__()
        assert d_model % 2 == 0, "d_model 必须是偶数"

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # 预计算频率
        # θ_i = 1 / (base^(2i/d_model))
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算位置的 cos 和 sin 值
        self._build_cache(max_len)

    def _build_cache(self, seq_len):
        """预计算 cos 和 sin 缓存"""
        # 位置索引 [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()

        # 计算 position * frequency
        # [seq_len, d_model/2]
        freqs = torch.outer(t, self.inv_freq)

        # 复制一份用于完整维度
        # [seq_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 缓存 cos 和 sin
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, q, k, seq_len=None):
        """
        对 Q 和 K 应用旋转位置编码

        参数：
            q: Query 张量 [batch_size, num_heads, seq_len, head_dim]
            k: Key 张量 [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度（如果为 None，从输入推断）

        返回：
            旋转后的 q, k
        """
        if seq_len is None:
            seq_len = q.size(2)

        # 确保缓存足够长
        if seq_len > self.max_len:
            self._build_cache(seq_len)

        # 获取当前长度的 cos 和 sin
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # 应用旋转
        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)

        return q_embed, k_embed

    def _apply_rotary(self, x, cos, sin):
        """
        应用旋转变换

        旋转公式（2D 旋转矩阵）：
            [cos θ, -sin θ] [x]   [x*cos - y*sin]
            [sin θ,  cos θ] [y] = [x*sin + y*cos]

        实现技巧：
            将 x 分成两半 [x1, x2]
            旋转后 = [x1*cos - x2*sin, x1*sin + x2*cos]

            或者用更简洁的形式：
            旋转后 = x * cos + rotate_half(x) * sin
            其中 rotate_half([x1, x2]) = [-x2, x1]
        """
        # x: [batch, num_heads, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim]

        # 扩展维度以便广播
        # [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # rotate_half: 将后半部分取负并与前半部分交换
        # [a, b, c, d] → [-c, -d, a, b]
        x_rotated = torch.cat((-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]), dim=-1)

        return x * cos + x_rotated * sin


# =============================================================================
# 第四部分：演示代码
# =============================================================================

def demo_sinusoidal():
    """演示 Sinusoidal 位置编码"""
    print("\n" + "=" * 60)
    print("Sinusoidal 位置编码演示")
    print("=" * 60)

    d_model = 512
    max_len = 100
    batch_size = 2
    seq_len = 20

    # 创建位置编码
    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)

    # 模拟输入（假设已经过词嵌入）
    x = torch.randn(batch_size, seq_len, d_model)

    # 应用位置编码
    output = pe(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"位置编码矩阵形状: {pe.pe.shape}")

    # 可视化位置编码的模式
    print("\n位置编码的前几个维度（位置 0-4）:")
    for pos in range(5):
        pe_values = pe.pe[0, pos, :8].numpy()
        print(f"  位置 {pos}: {pe_values.round(3)}")

    print("\n观察：")
    print("  - 不同位置有不同的编码")
    print("  - 编码值在 [-1, 1] 范围内")
    print("  - 低频维度变化慢，高频维度变化快")


def demo_learnable():
    """演示可学习位置编码"""
    print("\n" + "=" * 60)
    print("可学习位置编码演示")
    print("=" * 60)

    d_model = 512
    max_len = 512
    batch_size = 2
    seq_len = 20

    # 创建位置编码
    pe = LearnablePositionalEncoding(d_model, max_len, dropout=0.0)

    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 应用位置编码
    output = pe(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"可学习参数数量: {pe.position_embeddings.weight.numel():,}")
    print(f"  = max_len × d_model = {max_len} × {d_model} = {max_len * d_model:,}")

    print("\n特点：")
    print("  - 位置编码是可学习的参数")
    print("  - 需要更多显存存储参数")
    print("  - 无法处理超过 max_len 的序列")


def demo_rope():
    """演示 RoPE 旋转位置编码"""
    print("\n" + "=" * 60)
    print("RoPE 旋转位置编码演示")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    head_dim = d_model // num_heads
    batch_size = 2
    seq_len = 10

    # 创建 RoPE
    rope = RotaryPositionalEncoding(head_dim, max_len=2048)

    # 模拟 Q 和 K（已经过投影和分头）
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 应用 RoPE
    q_rotated, k_rotated = rope(q, k)

    print(f"\nQ 形状: {q.shape}")
    print(f"K 形状: {k.shape}")
    print(f"旋转后 Q 形状: {q_rotated.shape}")
    print(f"旋转后 K 形状: {k_rotated.shape}")

    # 验证相对位置性质
    print("\n验证相对位置性质:")
    print("  计算 q_m · k_n 对于不同的 (m, n) 组合...")

    # 计算注意力分数
    scores_before = torch.matmul(q, k.transpose(-2, -1))
    scores_after = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))

    print(f"\n  旋转前注意力分数（位置 0 对所有位置）:")
    print(f"    {scores_before[0, 0, 0, :5].detach().numpy().round(3)}")

    print(f"\n  旋转后注意力分数（位置 0 对所有位置）:")
    print(f"    {scores_after[0, 0, 0, :5].detach().numpy().round(3)}")

    print("\n特点：")
    print("  - RoPE 改变了 Q 和 K，使得注意力分数包含位置信息")
    print("  - 相对位置信息被编码进了点积结果中")
    print("  - 不增加额外参数，计算高效")


def demo_comparison():
    """对比三种位置编码"""
    print("\n" + "=" * 60)
    print("三种位置编码对比")
    print("=" * 60)

    comparison = """
    | 特性           | Sinusoidal        | Learnable         | RoPE              |
    |----------------|-------------------|-------------------|-------------------|
    | 类型           | 绝对位置          | 绝对位置          | 相对位置          |
    | 参数量         | 0                 | max_len × d_model | 0                 |
    | 外推性         | 理论上可以        | 不支持            | 支持              |
    | 实现复杂度     | 简单              | 最简单            | 中等              |
    | 代表模型       | 原始 Transformer  | BERT, GPT-2       | LLaMA, Qwen       |
    | 适用场景       | 中等长度序列      | 固定长度任务      | 长序列、LLM       |
    """
    print(comparison)

    print("\n选择建议：")
    print("  - 如果是固定长度的下游任务：可学习位置编码")
    print("  - 如果需要处理变长序列：Sinusoidal")
    print("  - 如果是 LLM 或需要外推：RoPE")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("           位置编码学习演示")
    print("=" * 60)
    print("""
    本脚本演示三种位置编码方法：
    1. Sinusoidal - 原始 Transformer
    2. Learnable - BERT
    3. RoPE - LLaMA/Qwen
    """)

    input("按 Enter 开始 Sinusoidal 位置编码演示...")
    demo_sinusoidal()

    input("\n按 Enter 开始可学习位置编码演示...")
    demo_learnable()

    input("\n按 Enter 开始 RoPE 演示...")
    demo_rope()

    input("\n按 Enter 查看三种方法对比...")
    demo_comparison()

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)
