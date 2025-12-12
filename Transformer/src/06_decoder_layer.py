"""
解码器层 (Decoder Layer)

解码器层是Transformer解码器的基本构件。
与编码器层不同，解码器层有三个子层：

1. 掩蔽多头自注意力（Masked Multi-Head Self-Attention）
   - 解码器在生成序列时，只能看到已生成的部分
   - 不能看到未来的信息
   - 使用因果掩码（Causal Mask）：下三角矩阵

2. 多头交叉注意力（Multi-Head Cross-Attention）
   - 查询来自解码器（自注意力的输出）
   - 键和值来自编码器（编码器的输出）
   - 允许解码器关注编码器输出中的相关信息

3. 前馈网络（Feed-Forward Network）
   - 与编码器层相同

数据流：
1. 输入 x (来自上一个解码器层或embedding)
2. 掩蔽自注意力 -> 残差 -> LayerNorm
3. 交叉注意力 (使用编码器输出作为K, V) -> 残差 -> LayerNorm
4. 前馈网络 -> 残差 -> LayerNorm
5. 输出

掩蔽（Causal Mask）的作用：
- 在推断时，模型一次生成一个词
- 在训练时，即使输入了整个序列，模型也不能看到未来的词
- 这确保了训练和推断的一致性

交叉注意力的作用：
- 连接编码器和解码器
- 允许模型根据源序列信息生成目标序列
- 在序列到序列任务中至关重要
"""

import torch
import torch.nn as nn
from typing import Optional


class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    参数：
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout概率
        activation: 激活函数类型
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        
        from src.attention import MultiHeadAttention
        from src.feed_forward import FeedForwardNetwork
        
        # ============================================================
        # 第一个子层：掩蔽多头自注意力
        # ============================================================
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # ============================================================
        # 第二个子层：多头交叉注意力
        # ============================================================
        # 交叉注意力使用来自解码器的Q，来自编码器的K和V
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # ============================================================
        # 第三个子层：前馈网络
        # ============================================================
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """
        前向传播
        
        参数：
            x: 解码器输入，形状 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出，形状 (batch_size, src_seq_len, d_model)
            self_attention_mask: 掩蔽自注意力的掩码（因果掩码）
            cross_attention_mask: 交叉注意力的掩码（通常用于padding）
        
        返回：
            输出张量，形状 (batch_size, tgt_seq_len, d_model)
        """
        
        # ============================================================
        # 第一个子层：掩蔽多头自注意力
        # ============================================================
        # 自注意力（所有输入都来自x）
        self_attn_output, _ = self.self_attention(x, x, x, self_attention_mask)
        self_attn_output = self.dropout1(self_attn_output)
        x = self.norm1(x + self_attn_output)
        
        # ============================================================
        # 第二个子层：多头交叉注意力
        # ============================================================
        # 交叉注意力：查询来自解码器，键和值来自编码器
        # Q: x (解码器当前层的输出)
        # K, V: encoder_output (编码器的输出)
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attention_mask)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = self.norm2(x + cross_attn_output)
        
        # ============================================================
        # 第三个子层：前馈网络
        # ============================================================
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.norm3(x + ff_output)
        
        return x


# ============================================================================
# Pre-LN解码器层
# ============================================================================

class PreLNDecoderLayer(nn.Module):
    """
    使用Pre-LN的解码器层
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation='relu'):
        super(PreLNDecoderLayer, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        
        from src.attention import MultiHeadAttention
        from src.feed_forward import FeedForwardNetwork
        
        # 掩蔽自注意力
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # 交叉注意力
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # 前馈网络
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """前向传播（Pre-LN变体）"""
        
        # 掩蔽自注意力（Pre-LN）
        x_norm = self.norm1(x)
        self_attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, self_attention_mask)
        x = x + self.dropout1(self_attn_output)
        
        # 交叉注意力（Pre-LN）
        x_norm = self.norm2(x)
        cross_attn_output, _ = self.cross_attention(x_norm, encoder_output, encoder_output, cross_attention_mask)
        x = x + self.dropout2(cross_attn_output)
        
        # 前馈网络（Pre-LN）
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout3(ff_output)
        
        return x


# ============================================================================
# 工具函数：创建掩码
# ============================================================================

def create_causal_mask(seq_len, device):
    """
    创建因果掩码（因果掩码）
    
    参数：
        seq_len: 序列长度
        device: 张量设备
    
    返回：
        因果掩码，形状 (1, 1, seq_len, seq_len)
        - 1表示可以参与注意力
        - 0表示被掩蔽（不能参与注意力）
    
    因果掩码是下三角矩阵：
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    
    这确保位置i只能参与位置0到i的注意力
    """
    # 创建下三角矩阵
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    # 添加batch和head维度
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask


def create_padding_mask(seq, pad_token_id=0):
    """
    创建padding掩码
    
    参数：
        seq: 序列张量，形状 (batch_size, seq_len)
        pad_token_id: padding token的id
    
    返回：
        padding掩码，形状 (batch_size, 1, 1, seq_len)
        - 1表示有效位置
        - 0表示padding位置
    """
    mask = (seq != pad_token_id).float()
    # 重塑为 (batch_size, 1, 1, seq_len)
    return mask.unsqueeze(1).unsqueeze(1)


if __name__ == "__main__":
    print("=" * 60)
    print("解码器层 (Decoder Layer) 演示")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    src_seq_len = 12
    tgt_seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # 创建输入
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    decoder_input = torch.randn(batch_size, tgt_seq_len, d_model)
    
    print(f"\n编码器输出形状: {encoder_output.shape}")
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {num_heads}")
    
    # 创建因果掩码
    causal_mask = create_causal_mask(tgt_seq_len, decoder_input.device)
    
    print(f"\n因果掩码形状: {causal_mask.shape}")
    print(f"因果掩码示例（3x3）:\n{causal_mask[0, 0, :3, :3]}")
    
    # 创建解码器层
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
    
    # 前向传播
    output = decoder_layer(decoder_input, encoder_output, causal_mask)
    
    print(f"\n解码器层输出形状: {output.shape}")
    print(f"输出形状与输入相同: {output.shape == decoder_input.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in decoder_layer.parameters())
    print(f"\n解码器层参数总数: {total_params:,}")
    
    print("\n✅ 解码器层模块测试成功！")
