"""
编码器层 (Encoder Layer)

编码器层是Transformer编码器的基本构件。
每个编码器层包含两个子层：
1. 多头自注意力（Self-Attention）
2. 前馈网络（Feed-Forward Network）

每个子层都遵循以下模式：
- 子层的计算
- 残差连接（Residual Connection）
- 层归一化（Layer Normalization）

这个设计被称为 Post-LN（归一化放在激活后）或 Pre-LN（归一化放在激活前）
本实现使用 Post-LN（标准Transformer）

编码器层的数据流：
1. 输入 x
2. 自注意力: x' = MultiHeadAttention(x, x, x)
3. 残差连接: x = x + Dropout(x')
4. 层归一化: x = LayerNorm(x)
5. 前馈网络: x' = FFN(x)
6. 残差连接: x = x + Dropout(x')
7. 层归一化: x = LayerNorm(x)
8. 输出 x

残差连接的作用：
- 允许梯度直接通过网络流动
- 缓解深网络中的梯度消失问题
- 使网络能够学习恒等映射

层归一化的作用：
- 稳定训练
- 独立于批次大小
- 在特征维度上进行归一化
"""

import torch
import torch.nn as nn
from typing import Optional


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    参数：
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout概率
        activation: 激活函数类型 ('relu' 或 'gelu')
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        
        # ============================================================
        # 第一个子层：多头自注意力
        # ============================================================
        from src.attention import MultiHeadAttention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # ============================================================
        # 第二个子层：前馈网络
        # ============================================================
        from src.feed_forward import FeedForwardNetwork
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数：
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 可选的掩码（用于padding位置）
        
        返回：
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        
        # ============================================================
        # 第一个子层：多头自注意力 + 残差连接 + 层归一化
        # ============================================================
        
        # 自注意力：三个输入都是x（自注意力）
        attention_output, _ = self.self_attention(x, x, x, mask)
        
        # 应用dropout
        attention_output = self.dropout1(attention_output)
        
        # 残差连接和层归一化（Post-LN）
        x = self.norm1(x + attention_output)
        
        # ============================================================
        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        # ============================================================
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 应用dropout
        ff_output = self.dropout2(ff_output)
        
        # 残差连接和层归一化
        x = self.norm2(x + ff_output)
        
        return x


# ============================================================================
# 改进版本：Pre-LN编码器层
# ============================================================================

class PreLNEncoderLayer(nn.Module):
    """
    使用Pre-LN（前置层归一化）的编码器层
    
    Pre-LN 相比 Post-LN 的优点：
    - 更容易训练深网络
    - 不需要 Learning Rate Warmup
    - 更稳定的训练动态
    
    数据流（Pre-LN）：
    1. 层归一化: x_norm = LayerNorm(x)
    2. 自注意力: x' = MultiHeadAttention(x_norm, x_norm, x_norm)
    3. 残差连接: x = x + Dropout(x')
    4. 层归一化: x_norm = LayerNorm(x)
    5. 前馈网络: x' = FFN(x_norm)
    6. 残差连接: x = x + Dropout(x')
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation='relu'):
        super(PreLNEncoderLayer, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        
        from src.attention import MultiHeadAttention
        from src.feed_forward import FeedForwardNetwork
        
        # 层归一化（放在前面）
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        """前向传播（Pre-LN变体）"""
        
        # 先归一化再注意力
        x_norm = self.norm1(x)
        attention_output, _ = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attention_output)
        
        # 先归一化再前馈
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_output)
        
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("编码器层 (Encoder Layer) 演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: {x.shape}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {num_heads}")
    print(f"中间层维度: {d_ff}")
    
    # 创建编码器层
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    
    # 前向传播
    output = encoder_layer(x)
    
    print(f"\n编码器层输出形状: {output.shape}")
    print(f"输入和输出形状相同: {x.shape == output.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"\n编码器层参数总数: {total_params:,}")
    
    # 测试Pre-LN变体
    print("\n\n测试Pre-LN编码器层:")
    print("-" * 40)
    
    preln_encoder_layer = PreLNEncoderLayer(d_model, num_heads, d_ff)
    output_preln = preln_encoder_layer(x)
    
    print(f"Pre-LN输出形状: {output_preln.shape}")
    preln_params = sum(p.numel() for p in preln_encoder_layer.parameters())
    print(f"Pre-LN参数总数: {preln_params:,}")
    print(f"参数数量相同: {total_params == preln_params}")
    
    print("\n✅ 编码器层模块测试成功！")
