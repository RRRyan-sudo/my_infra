import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # 1. Attention 部分初始化
        # 使用合并的 Linear 层同时计算 Q, K, V，效率更高
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model) # Pre-Norm
        
        # 2. Feed-Forward 部分初始化
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # 现代 Transformer 常用 GELU 而非 ReLU
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model) # Pre-Norm
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: [Batch_Size, Seq_Len, d_model]
        B, T, C = x.shape
        
        # --- Block 1: Multi-Head Self-Attention (MHSA) ---
        # 1. Pre-Norm 这里的残差连接是 x + MHA(Norm(x))
        residual = x
        x = self.attn_norm(x)
        
        # 2. QKV 投影与重塑 (Split Heads)
        # qkv shape: [B, T, 3 * C] -> reshape -> permute -> [3, B, n_head, T, head_dim]
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. Scaled Dot-Product Attention
        # scores shape: [B, n_head, T, T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask 通常是 [B, 1, T, T] 或 [1, 1, T, T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 4. 加权求和与输出投影
        # out: [B, n_head, T, head_dim] -> transpose/reshape -> [B, T, C]
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.o_proj(out)
        x = self.dropout(x) + residual # 残差连接
        
        # --- Block 2: Feed-Forward Network (FFN) ---
        # Pre-Norm: x + FFN(Norm(x))
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual # 残差连接
        
        return x
