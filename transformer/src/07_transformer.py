"""
完整的Transformer模型

这个文件包含完整的Transformer架构的实现，包括：
1. 编码器（Encoder）：多层编码器层的堆叠
2. 解码器（Decoder）：多层解码器层的堆叠
3. 完整模型（Transformer）：编码器和解码器的组合

Transformer模型的整体架构：
- 输入嵌入 + 位置编码
- 编码器：自注意力 + 前馈网络（N层）
- 解码器：自注意力 + 交叉注意力 + 前馈网络（N层）
- 输出线性层 + softmax（用于预测）

数据流：
1. 源序列 -> Embedding + Position Encoding -> 编码器 -> 编码器输出
2. 目标序列 -> Embedding + Position Encoding -> 解码器 -> 解码器输出
3. 解码器输出 -> 线性层 -> logits -> softmax -> 概率
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    参数：
        vocab_size: 词表大小
        d_model: 模型维度（embedding维度）
        num_heads: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        d_ff: 前馈网络隐藏层维度
        max_seq_length: 序列最大长度
        dropout: dropout概率
        activation: 激活函数类型
        use_pre_ln: 是否使用Pre-LN（True）或Post-LN（False）
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=None,
        max_seq_length=512,
        dropout=0.1,
        activation='relu',
        use_pre_ln=False
    ):
        super(Transformer, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # ============================================================
        # 共享的嵌入层
        # ============================================================
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 嵌入值缩放（Transformer论文中的技巧）
        # embedding乘以sqrt(d_model)可以让梯度流动更稳定
        self.embedding_scale = math.sqrt(d_model)
        
        from src.positional_encoding import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # ============================================================
        # 编码器
        # ============================================================
        if use_pre_ln:
            from src.encoder_layer import PreLNEncoderLayer as EncoderLayerClass
        else:
            from src.encoder_layer import EncoderLayer as EncoderLayerClass
        
        encoder_layer = EncoderLayerClass(d_model, num_heads, d_ff, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # ============================================================
        # 解码器
        # ============================================================
        if use_pre_ln:
            from src.decoder_layer import PreLNDecoderLayer as DecoderLayerClass
        else:
            from src.decoder_layer import DecoderLayer as DecoderLayerClass
        
        decoder_layer = DecoderLayerClass(d_model, num_heads, d_ff, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        # ============================================================
        # 输出层
        # ============================================================
        # 线性层将d_model维的输出投影到vocab_size维
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        # ============================================================
        # 权重初始化
        # ============================================================
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src_tokens,
        tgt_tokens,
        src_mask=None,
        tgt_mask=None,
        tgt_src_mask=None
    ):
        """
        前向传播
        
        参数：
            src_tokens: 源序列token IDs，形状 (batch_size, src_seq_len)
            tgt_tokens: 目标序列token IDs，形状 (batch_size, tgt_seq_len)
            src_mask: 源序列掩码（padding掩码）
            tgt_mask: 目标序列掩码（因果掩码）
            tgt_src_mask: 目标到源的交叉注意力掩码
        
        返回：
            logits: 预测的logits，形状 (batch_size, tgt_seq_len, vocab_size)
        """
        
        # ============================================================
        # 编码器处理源序列
        # ============================================================
        # 嵌入 + 位置编码
        src_embeddings = self.embedding(src_tokens) * self.embedding_scale
        src_embeddings = self.positional_encoding(src_embeddings)
        
        # 编码
        encoder_output = self.encoder(src_embeddings, src_mask)
        
        # ============================================================
        # 解码器处理目标序列
        # ============================================================
        # 嵌入 + 位置编码
        tgt_embeddings = self.embedding(tgt_tokens) * self.embedding_scale
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        
        # 解码
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask, tgt_src_mask)
        
        # ============================================================
        # 输出层
        # ============================================================
        logits = self.output_linear(decoder_output)
        
        return logits
    
    def encode(self, src_tokens, src_mask=None):
        """
        仅编码（推断时可用）
        
        参数：
            src_tokens: 源序列token IDs
            src_mask: 源序列掩码
        
        返回：
            编码器输出
        """
        src_embeddings = self.embedding(src_tokens) * self.embedding_scale
        src_embeddings = self.positional_encoding(src_embeddings)
        encoder_output = self.encoder(src_embeddings, src_mask)
        return encoder_output
    
    def decode(self, tgt_tokens, encoder_output, tgt_mask=None, tgt_src_mask=None):
        """
        仅解码（推断时可用）
        
        参数：
            tgt_tokens: 目标序列token IDs
            encoder_output: 编码器输出
            tgt_mask: 目标序列掩码
            tgt_src_mask: 交叉注意力掩码
        
        返回：
            解码器输出的logits
        """
        tgt_embeddings = self.embedding(tgt_tokens) * self.embedding_scale
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask, tgt_src_mask)
        logits = self.output_linear(decoder_output)
        return logits


# ============================================================
# 编码器和解码器堆栈
# ============================================================

class TransformerEncoder(nn.Module):
    """
    Transformer编码器：多层编码器层的堆叠
    
    参数：
        encoder_layer: 单个编码器层的实例
        num_layers: 编码器层的数量
        norm: 最终的层归一化
    """
    
    def __init__(self, encoder_layer, num_layers, norm):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            __import__('copy').deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.norm = norm
        self.num_layers = num_layers
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数：
            x: 输入张量
            mask: 掩码
        
        返回：
            编码器输出
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer解码器：多层解码器层的堆叠
    
    参数：
        decoder_layer: 单个解码器层的实例
        num_layers: 解码器层的数量
        norm: 最终的层归一化
    """
    
    def __init__(self, decoder_layer, num_layers, norm):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            __import__('copy').deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.norm = norm
        self.num_layers = num_layers
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """
        前向传播
        
        参数：
            x: 解码器输入
            encoder_output: 编码器输出
            self_attention_mask: 自注意力掩码
            cross_attention_mask: 交叉注意力掩码
        
        返回：
            解码器输出
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
        
        x = self.norm(x)
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("完整 Transformer 模型演示")
    print("=" * 60)
    
    # 参数设置
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 2048
    max_seq_length = 128
    
    # 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )
    
    print(f"\n模型参数:")
    print(f"  词表大小: {vocab_size}")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")
    print(f"  编码器层数: {num_encoder_layers}")
    print(f"  解码器层数: {num_decoder_layers}")
    print(f"  前馈网络维度: {d_ff}")
    print(f"  最大序列长度: {max_seq_length}")
    
    # 计算总参数数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数数: {total_params:,}")
    
    # 创建示例输入
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\n输入信息:")
    print(f"  批次大小: {batch_size}")
    print(f"  源序列长度: {src_seq_len}")
    print(f"  目标序列长度: {tgt_seq_len}")
    print(f"  源序列形状: {src_tokens.shape}")
    print(f"  目标序列形状: {tgt_tokens.shape}")
    
    # 前向传播
    logits = model(src_tokens, tgt_tokens)
    
    print(f"\n输出:")
    print(f"  Logits形状: {logits.shape}")
    print(f"  预期形状: ({batch_size}, {tgt_seq_len}, {vocab_size})")
    print(f"  形状正确: {logits.shape == (batch_size, tgt_seq_len, vocab_size)}")
    
    # 概率分布
    probabilities = torch.softmax(logits, dim=-1)
    print(f"\n概率分布:")
    print(f"  形状: {probabilities.shape}")
    print(f"  第一个样本，第一个位置的概率分布总和: {probabilities[0, 0].sum():.6f}")
    
    print("\n✅ Transformer 模型测试成功！")
