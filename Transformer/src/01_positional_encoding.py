"""
位置编码 (Positional Encoding)

为什么需要位置编码？
- Transformer 使用自注意力机制，它本质上对输入序列中所有位置是对称的
- 这意味着模型无法天生理解序列中每个元素的位置
- 位置编码为序列中的每个位置添加一个独特的信号，使模型能够学习相对或绝对位置

位置编码的设计：
使用正弦和余弦函数的组合，理由如下：
1. 能够生成不同长度序列的编码
2. 模型可以学习相对位置关系（距离不变性）
3. 波长不同的正弦曲线可以编码不同的位置信息

位置编码公式：
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: 序列中的位置（0, 1, 2, ...）
- i: 维度索引（0, 1, 2, ..., d_model/2 - 1）
- d_model: 模型的维度
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    参数：
        d_model: 模型的维度（embedding维度）
        max_seq_length: 序列的最大长度（预先计算编码时使用）
        dropout: 应用的dropout比例
    """
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        # 形状: (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # 创建位置向量: [0, 1, 2, ..., max_seq_length-1]
        # 形状: (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 创建维度索引向量
        # 使用指数递减的分母来创建不同频率的波
        # dim_model = 2i, 2i+1 成对出现，所以分母基数为 0, 1, 2, ..., d_model/2-1
        # 形状: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        # 应用正弦和余弦编码
        # 偶数维度使用正弦
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 奇数维度使用余弦
        # 注意：当d_model为奇数时，最后一维没有对应的奇数维度，因此最后一个div_term被丢弃
        if d_model % 2 == 1:
            # 如果d_model是奇数，余弦维度会少一个
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        # 形状: (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer，这样在模型移动到GPU时会自动跟随
        # 但不会被当作可学习的参数
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量，形状为 (batch_size, seq_length, d_model)
        
        返回：
            x + 位置编码，形状为 (batch_size, seq_length, d_model)
        """
        # 获取序列长度
        seq_length = x.size(1)
        
        # 将位置编码与输入相加
        # self.pe[:, :seq_length, :] 形状: (1, seq_length, d_model)
        # 会被自动广播到 (batch_size, seq_length, d_model)
        x = x + self.pe[:, :seq_length, :].to(x.device)
        
        # 应用dropout
        return self.dropout(x)


# ============================================================================
# 辅助函数：用于可视化位置编码
# ============================================================================

def visualize_positional_encoding(d_model=512, max_seq_length=100):
    """
    可视化位置编码矩阵
    
    参数：
        d_model: 模型维度
        max_seq_length: 序列最大长度
    """
    # 创建位置编码
    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * 
        (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:-1]) if d_model % 2 == 1 else torch.cos(position * div_term)
    
    # 绘制
    plt.figure(figsize=(15, 5))
    plt.imshow(pe.numpy(), aspect='auto', cmap='RdBu')
    plt.title('位置编码热力图\n(横轴: 维度索引, 纵轴: 序列位置)')
    plt.xlabel('维度 (d_model)')
    plt.ylabel('序列位置')
    plt.colorbar()
    return pe


if __name__ == "__main__":
    # 简单测试
    print("=" * 60)
    print("位置编码 (Positional Encoding) 演示")
    print("=" * 60)
    
    # 参数设置
    d_model = 512
    batch_size = 2
    seq_length = 10
    
    # 创建位置编码层
    pos_encoding = PositionalEncoding(d_model=d_model, max_seq_length=100)
    
    # 创建虚拟输入 (通常这会是embedding层的输出)
    # 形状: (batch_size, seq_length, d_model)
    x = torch.randn(batch_size, seq_length, d_model)
    
    print(f"\n输入形状: {x.shape}")
    print(f"模型维度 (d_model): {d_model}")
    print(f"序列长度: {seq_length}")
    print(f"批次大小: {batch_size}")
    
    # 前向传播
    output = pos_encoding(x)
    
    print(f"\n输出形状: {output.shape}")
    print(f"输入的前几个值:\n{x[0, 0, :5]}")
    print(f"输出的前几个值:\n{output[0, 0, :5]}")
    
    # 显示位置编码的内部结构
    print("\n位置编码矩阵信息:")
    print(f"  形状: {pos_encoding.pe.shape}")
    print(f"  第一个位置的编码:\n  {pos_encoding.pe[0, 0, :10]}")
    print(f"  第十个位置的编码:\n  {pos_encoding.pe[0, 9, :10]}")
    
    print("\n✅ 位置编码模块测试成功！")
