"""
前馈网络 (Feed Forward Network)

在Transformer的每个编码器和解码器层中，除了多头注意力外，
还有一个前馈网络（也称为位置级前馈网络）。

设计特点：
- 独立应用于序列中的每个位置
- 具有相同的权重，但在所有位置处独立应用
- 在时间维度上完全"卷积"（即提供相同的权重转换）

前馈网络的结构：
FFN(x) = max(0, x·W1 + b1)·W2 + b2

这是一个简单的两层全连接网络：
1. 第一层：d_model → d_ff (通常d_ff = 4 × d_model)
   - 使用ReLU激活函数扩展维度
   - 这允许模型学习非线性变换
2. 第二层：d_ff → d_model
   - 将维度投影回原始大小

为什么要扩展维度？
- 增加模型的表达能力
- 允许在中间层进行复杂的非线性组合
- 像"瓶颈"结构：先扩展再压缩，这种设计在许多深度网络中很有效

前馈网络对应原论文中的公式：
FFN(x) = max(0, xW1 + b1)W2 + b2
其中 d_ff = 2048（对于d_model = 512的情况）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    前馈网络模块
    
    参数：
        d_model: 模型的维度（输入和输出维度）
        d_ff: 中间层的维度（隐藏层维度）
        dropout: dropout概率
        activation: 激活函数名称（'relu' 或 'gelu'）
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(FeedForwardNetwork, self).__init__()
        
        # 如果没有指定d_ff，使用默认值：4倍的d_model
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 第一层：线性投影从d_model到d_ff
        # 这一层扩展维度，增加表达能力
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # 激活函数
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"未知的激活函数: {activation}")
        
        # Dropout层（用于正则化）
        self.dropout = nn.Dropout(p=dropout)
        
        # 第二层：线性投影从d_ff回到d_model
        # 这一层将维度投影回原始大小
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量，形状 (batch_size, seq_len, d_model)
        
        返回：
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        
        # 步骤 1: 第一层线性变换 + 激活函数
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 步骤 2: 第二层线性变换
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


# ============================================================================
# 实验：不同激活函数的效果
# ============================================================================

class ExperimentFFN(nn.Module):
    """
    实验FFN变体，用于演示不同的设计选择
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(ExperimentFFN, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # 变体1：使用ReLU
        self.ffn_relu = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout)
        )
        
        # 变体2：使用GELU（更现代的选择）
        self.ffn_gelu = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout)
        )
        
        # 变体3：两层Swish（SiLU）激活函数
        self.ffn_swish = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout)
        )


if __name__ == "__main__":
    print("=" * 60)
    print("前馈网络 (Feed Forward Network) 演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: {x.shape}")
    print(f"模型维度: {d_model}")
    print(f"中间层维度: {d_ff}")
    print(f"扩展比例: {d_ff / d_model}x")
    
    # 创建FFN
    ffn = FeedForwardNetwork(d_model, d_ff)
    
    # 前向传播
    output = ffn(x)
    
    print(f"\n前馈网络输出形状: {output.shape}")
    print(f"输入的第一个样本，第一个位置的前5个值:\n  {x[0, 0, :5]}")
    print(f"输出的第一个样本，第一个位置的前5个值:\n  {output[0, 0, :5]}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n前馈网络参数总数: {total_params:,}")
    print(f"  - fc1权重: {ffn.fc1.weight.numel():,} + 偏置: {ffn.fc1.bias.numel():,}")
    print(f"  - fc2权重: {ffn.fc2.weight.numel():,} + 偏置: {ffn.fc2.bias.numel():,}")
    
    # 测试不同的激活函数
    print("\n\n测试不同激活函数的效果:")
    print("-" * 40)
    
    ffn_relu = FeedForwardNetwork(d_model, d_ff, activation='relu')
    ffn_gelu = FeedForwardNetwork(d_model, d_ff, activation='gelu')
    
    out_relu = ffn_relu(x)
    out_gelu = ffn_gelu(x)
    
    print(f"ReLU输出范围: [{out_relu.min():.4f}, {out_relu.max():.4f}]")
    print(f"GELU输出范围: [{out_gelu.min():.4f}, {out_gelu.max():.4f}]")
    
    print("\n✅ 前馈网络模块测试成功！")
