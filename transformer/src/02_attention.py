"""
缩放点积注意力 (Scaled Dot-Product Attention)

注意力机制是Transformer的核心。它允许模型在处理序列中的每个元素时，
权衡地考虑序列中的所有其他元素。

注意力的直观理解：
- 想象你在阅读一篇文章时，当你读到某个词时，会自动关注相关的词
- 注意力机制就是让神经网络学会这种"关注"的能力

注意力的数学公式：
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V

其中：
- Q (Query): 查询向量 - "我想找什么"
- K (Key): 键向量 - "每个位置代表什么"
- V (Value): 值向量 - "每个位置的信息"
- d_k: 键向量的维度（用于缩放，避免梯度消失）
- softmax: 将注意力权重转换为概率分布

工作流程：
1. 计算查询和键的相似性：Q·K^T (形状: seq_len × seq_len)
2. 缩放相似性：除以√d_k (防止softmax饱和)
3. 可选：应用掩码（例如在解码器中隐藏未来信息）
4. 应用softmax转换为概率分布
5. 使用概率加权求和值向量：softmax(...)·V

这被称为"缩放点积注意力"因为：
- "点积" - 使用Q·K^T计算相似性
- "缩放" - 除以√d_k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    
    参数：
        d_k: 键向量的维度
        dropout: dropout概率
    """
    
    def __init__(self, d_k, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        
        参数：
            Q: 查询矩阵，形状 (batch_size, num_heads, seq_len, d_k)
            K: 键矩阵，形状 (batch_size, num_heads, seq_len, d_k)
            V: 值矩阵，形状 (batch_size, num_heads, seq_len, d_v)
            mask: 可选的掩码张量，用于隐藏某些位置
                  - 对于编码器：可以是padding掩码
                  - 对于解码器：通常是因果掩码（下三角矩阵）
        
        返回：
            output: 注意力输出，形状 (batch_size, num_heads, seq_len, d_v)
            attention_weights: 注意力权重，形状 (batch_size, num_heads, seq_len, seq_len)
        """
        
        # 步骤 1: 计算注意力分数 (相似性)
        # Q·K^T 计算查询和键之间的相似性
        # 形状: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 步骤 2: 缩放分数
        # 除以 √d_k，使得梯度更稳定
        # 直观理解：如果d_k很大，点积的值会很大，导致softmax后只有少数几个位置有显著权重
        # 缩放可以让多个位置都有合理的权重
        scores = scores / math.sqrt(self.d_k)
        
        # 步骤 3: 应用掩码（如果提供）
        if mask is not None:
            # 将掩码为0的位置设为极小值（负无穷），这样softmax后会接近0
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 步骤 4: 应用softmax获得注意力权重
        # 在最后一个维度（序列长度维度）应用softmax
        # 这样每个查询位置的注意力权重会求和为1
        attention_weights = F.softmax(scores, dim=-1)
        
        # 处理softmax后的inf值（由掩码产生的-inf经softmax会变成0或nan）
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # 步骤 5: 应用dropout（用于正则化）
        attention_weights = self.dropout(attention_weights)
        
        # 步骤 6: 使用注意力权重加权求和值向量
        # 形状: (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


# ============================================================================
# 多头注意力的前置知识：注意力机制的多个头
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    为什么使用多头注意力？
    - 单个注意力头可能不足以捕捉序列中不同方面的关系
    - 例如：在处理句子时，某个头可能学会关注语法关系，
      另一个头可能学会关注语义关系
    - 多头注意力通过并行运行多个注意力头，每个头学习不同的表示子空间
    
    工作流程：
    1. 将输入的Q, K, V线性投影到多个子空间
    2. 在每个子空间中并行执行缩放点积注意力
    3. 连接所有头的输出
    4. 应用最终的线性投影
    
    参数：
        d_model: 模型的维度
        num_heads: 注意力头的数量
        dropout: dropout概率
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        
        # 确保d_model能被num_heads整除
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性投影层：将d_model维的输入投影到d_model维的输出
        self.W_q = nn.Linear(d_model, d_model)  # 查询投影
        self.W_k = nn.Linear(d_model, d_model)  # 键投影
        self.W_v = nn.Linear(d_model, d_model)  # 值投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        
        参数：
            Q: 查询，形状 (batch_size, seq_len_q, d_model)
            K: 键，形状 (batch_size, seq_len_k, d_model)
            V: 值，形状 (batch_size, seq_len_v, d_model)
            mask: 可选掩码
        
        返回：
            output: 形状 (batch_size, seq_len_q, d_model)
            attention_weights: 形状 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q.size(0)
        
        # 步骤 1: 应用线性投影
        Q = self.W_q(Q)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(K)  # (batch_size, seq_len_k, d_model)
        V = self.W_v(V)  # (batch_size, seq_len_v, d_model)
        
        # 步骤 2: 重塑为多头
        # 从 (batch_size, seq_len, d_model) 
        # 到 (batch_size, seq_len, num_heads, d_k)
        # 然后转置为 (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 步骤 3: 应用缩放点积注意力
        context, attention_weights = self.attention(Q, K, V, mask)
        
        # 步骤 4: 连接多个头的输出
        # 从 (batch_size, num_heads, seq_len, d_k)
        # 到 (batch_size, seq_len, num_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        
        # 从 (batch_size, seq_len, num_heads, d_k)
        # 到 (batch_size, seq_len, d_model)
        context = context.view(batch_size, -1, self.d_model)
        
        # 步骤 5: 应用最终的线性投影
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("缩放点积注意力与多头注意力演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: {x.shape}")
    print(f"模型维度: {d_model}")
    print(f"序列长度: {seq_len}")
    print(f"注意力头数: {num_heads}")
    
    # 测试多头注意力
    mha = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = mha(x, x, x)
    
    print(f"\n多头注意力输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重总和（应为1）: {attention_weights[0, 0, 0].sum()}")
    
    print("\n✅ 注意力机制模块测试成功！")
