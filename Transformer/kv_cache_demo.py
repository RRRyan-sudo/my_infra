"""
===============================================================================
                    KV Cache 推理优化原理演示
===============================================================================

KV Cache 是大模型推理中最重要的优化技术之一。本文件通过简单的实现帮助理解其原理。

核心问题：
    自回归生成时，每生成一个新 token，都需要对整个序列计算 Attention。
    但是已生成 token 的 K 和 V 是不变的，重复计算是浪费。

解决方案：
    缓存已计算的 K 和 V，每次只计算新 token 的 K 和 V。

示例（生成 "Hello World"）：
    Step 1: 输入 "Hello"
        - 计算 K1, V1 → 存入缓存
        - 计算注意力 → 输出 " World"

    Step 2: 输入 "Hello World"（实际只输入 " World"）
        - K1, V1 从缓存读取
        - 只计算新的 K2, V2 → 存入缓存
        - 计算注意力（Q2 对 [K1, K2]）

显存占用计算：
    KV Cache 大小 = 2 × num_layers × seq_len × num_heads × head_dim × batch_size × dtype_size

    示例（LLaMA-7B，seq_len=2048，batch=1，FP16）：
    = 2 × 32 × 2048 × 32 × 128 × 1 × 2 bytes
    = 1,073,741,824 bytes
    ≈ 1 GB

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# =============================================================================
# 第一部分：不使用 KV Cache 的 Attention（低效）
# =============================================================================

class AttentionWithoutCache(nn.Module):
    """
    不使用 KV Cache 的 Attention（用于对比）

    问题：每次生成新 token 时，需要重新计算整个序列的 K 和 V
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        前向传播（每次计算完整的 K 和 V）

        参数：
            x: [batch_size, seq_len, d_model]
            mask: 因果掩码

        返回：
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V（每次都重新计算整个序列）
        Q = self.q_proj(x)
        K = self.k_proj(x)  # 这里每次都计算完整的 K
        V = self.v_proj(x)  # 这里每次都计算完整的 V

        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 合并头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(context)

        return output


# =============================================================================
# 第二部分：使用 KV Cache 的 Attention（高效）
# =============================================================================

class AttentionWithCache(nn.Module):
    """
    使用 KV Cache 的 Attention

    优化点：
        1. 缓存已计算的 K 和 V
        2. 增量生成时，只计算新 token 的 K 和 V
        3. 将新的 K, V 追加到缓存中
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=True):
        """
        带 KV Cache 的前向传播

        参数：
            x: 输入张量
               - 首次调用（prefill）: [batch_size, seq_len, d_model]
               - 增量生成（decode）: [batch_size, 1, d_model]
            kv_cache: 之前缓存的 (K, V)，形状为 (K_cache, V_cache)
                      K_cache: [batch_size, num_heads, cached_len, head_dim]
            use_cache: 是否返回更新后的缓存

        返回：
            output: [batch_size, seq_len/1, d_model]
            new_kv_cache: 更新后的 (K_cache, V_cache)（如果 use_cache=True）
        """
        batch_size, seq_len, _ = x.shape

        # 计算当前输入的 Q, K, V
        Q = self.q_proj(x)
        K_new = self.k_proj(x)  # 只计算新输入的 K
        V_new = self.v_proj(x)  # 只计算新输入的 V

        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_new = K_new.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_new = V_new.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 处理 KV Cache
        if kv_cache is not None:
            # 从缓存中取出之前的 K 和 V
            K_cache, V_cache = kv_cache

            # 将新的 K, V 追加到缓存的末尾
            K = torch.cat([K_cache, K_new], dim=2)  # [batch, heads, cached_len + new_len, head_dim]
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            # 首次调用，没有缓存
            K = K_new
            V = V_new

        # 计算注意力
        # Q: [batch, heads, new_len, head_dim]
        # K: [batch, heads, total_len, head_dim]
        # scores: [batch, heads, new_len, total_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果掩码：新 token 只能看到自己和之前的 token
        total_len = K.size(2)
        # 对于增量生成，Q 的长度是 1，K 的长度是 total_len
        # 新 token 可以看到所有之前的 token，所以不需要 mask
        # 但为了安全，这里还是应用因果掩码
        if seq_len > 1:  # prefill 阶段需要因果掩码
            causal_mask = torch.tril(torch.ones(seq_len, total_len, device=x.device))
            # 需要处理 cache 的情况
            if kv_cache is not None:
                cached_len = kv_cache[0].size(2)
                # 新 token 可以看到所有缓存的 token
                causal_mask = torch.cat([
                    torch.ones(seq_len, cached_len, device=x.device),
                    torch.tril(torch.ones(seq_len, seq_len, device=x.device))
                ], dim=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        context = torch.matmul(attn_weights, V)

        # 合并头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(context)

        if use_cache:
            # 返回更新后的缓存
            return output, (K, V)
        else:
            return output


# =============================================================================
# 第三部分：KV Cache 管理器
# =============================================================================

class KVCacheManager:
    """
    KV Cache 管理器

    用于管理多层 Transformer 的 KV Cache
    """

    def __init__(self, num_layers, batch_size, num_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # 预分配缓存空间（静态分配，避免动态分配的开销）
        self.k_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype
        )

        # 当前缓存的长度
        self.current_len = 0

    def update(self, layer_idx, new_k, new_v):
        """
        更新指定层的 KV Cache

        参数：
            layer_idx: 层索引
            new_k: 新的 K [batch, heads, new_len, head_dim]
            new_v: 新的 V [batch, heads, new_len, head_dim]
        """
        new_len = new_k.size(2)
        end_pos = self.current_len + new_len

        self.k_cache[layer_idx, :, :, self.current_len:end_pos, :] = new_k
        self.v_cache[layer_idx, :, :, self.current_len:end_pos, :] = new_v

    def get(self, layer_idx):
        """获取指定层的缓存"""
        if self.current_len == 0:
            return None
        return (
            self.k_cache[layer_idx, :, :, :self.current_len, :],
            self.v_cache[layer_idx, :, :, :self.current_len, :]
        )

    def advance(self, num_tokens):
        """前进缓存指针"""
        self.current_len += num_tokens

    def reset(self):
        """重置缓存"""
        self.current_len = 0

    def memory_usage(self):
        """计算内存占用（字节）"""
        element_size = 2 if self.dtype == torch.float16 else 4
        total_elements = 2 * self.num_layers * self.batch_size * self.num_heads * self.max_seq_len * self.head_dim
        return total_elements * element_size


# =============================================================================
# 第四部分：演示代码
# =============================================================================

def demo_kv_cache_concept():
    """演示 KV Cache 的基本概念"""
    print("\n" + "=" * 60)
    print("KV Cache 基本概念演示")
    print("=" * 60)

    print("""
    【问题】
    自回归生成时，每次都需要计算整个序列的注意力。

    例如生成 "Hello World!"：
    Step 1: 输入 "<s>"           计算 K1, V1
    Step 2: 输入 "<s> Hello"     计算 K1, K2, V1, V2  ← K1, V1 重复计算！
    Step 3: 输入 "<s> Hello World" 计算 K1, K2, K3...  ← 更多重复！

    【KV Cache 解决方案】
    缓存已计算的 K 和 V，只计算新 token 的。

    Step 1: 输入 "<s>"           计算 K1, V1 → 存入缓存
    Step 2: 输入 "Hello"（新）    K1,V1 从缓存读 + 计算 K2, V2
    Step 3: 输入 "World"（新）    K1,K2,V1,V2 从缓存读 + 计算 K3, V3
    """)

    # 可视化演示
    print("\n【可视化：无 Cache vs 有 Cache】")
    print("\n无 KV Cache（每步的计算量）：")
    for step in range(1, 6):
        calcs = "K" * step + ", " + "V" * step
        print(f"  Step {step}: 计算 {calcs}")

    print("\n有 KV Cache（每步的计算量）：")
    print(f"  Step 1: 计算 K1, V1（首次，无缓存）")
    for step in range(2, 6):
        print(f"  Step {step}: 计算 K{step}, V{step}（读取缓存 K1..{step-1}, V1..{step-1}）")


def demo_kv_cache_performance():
    """演示 KV Cache 的性能对比"""
    print("\n" + "=" * 60)
    print("KV Cache 性能对比演示")
    print("=" * 60)

    d_model = 256
    num_heads = 4
    batch_size = 1
    prompt_len = 10
    gen_len = 50  # 生成 50 个 token

    # 创建两种 Attention
    attn_no_cache = AttentionWithoutCache(d_model, num_heads)
    attn_with_cache = AttentionWithCache(d_model, num_heads)

    # 复制权重确保公平比较
    attn_with_cache.load_state_dict(attn_no_cache.state_dict())

    print(f"\n【配置】")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")
    print(f"  提示长度: {prompt_len}")
    print(f"  生成长度: {gen_len}")

    # 模拟生成过程（无 Cache）
    print(f"\n【无 KV Cache】")
    tokens = torch.randn(batch_size, prompt_len, d_model)

    start_time = time.time()
    for i in range(gen_len):
        seq_len = prompt_len + i + 1
        # 每次需要处理完整序列
        full_input = torch.randn(batch_size, seq_len, d_model)
        _ = attn_no_cache(full_input)
    no_cache_time = time.time() - start_time

    print(f"  总时间: {no_cache_time:.4f}s")

    # 模拟生成过程（有 Cache）
    print(f"\n【有 KV Cache】")

    # Prefill：处理提示
    prompt = torch.randn(batch_size, prompt_len, d_model)
    start_time = time.time()
    _, kv_cache = attn_with_cache(prompt, kv_cache=None, use_cache=True)

    # Decode：逐 token 生成
    for i in range(gen_len):
        new_token = torch.randn(batch_size, 1, d_model)
        _, kv_cache = attn_with_cache(new_token, kv_cache=kv_cache, use_cache=True)
    with_cache_time = time.time() - start_time

    print(f"  总时间: {with_cache_time:.4f}s")

    speedup = no_cache_time / with_cache_time
    print(f"\n【对比结果】")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  时间节省: {(1 - with_cache_time/no_cache_time) * 100:.1f}%")


def demo_memory_analysis():
    """演示 KV Cache 显存占用分析"""
    print("\n" + "=" * 60)
    print("KV Cache 显存占用分析")
    print("=" * 60)

    # 不同模型的配置
    models = {
        "LLaMA-7B": {"layers": 32, "heads": 32, "head_dim": 128},
        "LLaMA-13B": {"layers": 40, "heads": 40, "head_dim": 128},
        "LLaMA-70B": {"layers": 80, "heads": 64, "head_dim": 128},
    }

    batch_size = 1
    dtype_size = 2  # FP16

    print(f"\n【不同模型、不同序列长度的 KV Cache 显存占用】")
    print(f"\n{'模型':<15} {'序列长度':<12} {'KV Cache 大小':<15}")
    print("-" * 45)

    for model_name, config in models.items():
        for seq_len in [1024, 2048, 4096, 8192]:
            # 计算 KV Cache 大小
            # 2 (K和V) × layers × heads × seq_len × head_dim × batch × dtype
            cache_size = (2 * config["layers"] * config["heads"] *
                         seq_len * config["head_dim"] * batch_size * dtype_size)

            # 转换为 GB
            cache_gb = cache_size / (1024 ** 3)

            if seq_len == 1024:
                print(f"{model_name:<15} {seq_len:<12} {cache_gb:.2f} GB")
            else:
                print(f"{'':<15} {seq_len:<12} {cache_gb:.2f} GB")

    print(f"\n【公式】")
    print(f"  KV Cache = 2 × num_layers × num_heads × seq_len × head_dim × batch_size × dtype_size")

    print(f"\n【优化方向】")
    print(f"  1. KV Cache 量化：FP16 → INT8（减少 50%）或 INT4（减少 75%）")
    print(f"  2. 分页管理（PagedAttention）：减少显存碎片")
    print(f"  3. KV Cache 压缩：丢弃不重要的 token")


def demo_incremental_generation():
    """演示增量生成的完整流程"""
    print("\n" + "=" * 60)
    print("增量生成流程演示")
    print("=" * 60)

    d_model = 128
    num_heads = 4
    batch_size = 1

    # 创建 Attention 层
    attn = AttentionWithCache(d_model, num_heads)

    print(f"\n【模拟文本生成过程】")
    print(f"  假设我们要生成: 'Hello World!'")

    # 模拟 token 嵌入
    tokens = {
        "Hello": torch.randn(batch_size, 1, d_model),
        " ": torch.randn(batch_size, 1, d_model),
        "World": torch.randn(batch_size, 1, d_model),
        "!": torch.randn(batch_size, 1, d_model),
    }

    # Step 1: Prefill（处理提示，假设提示是 "<bos> Hello"）
    print(f"\n  Step 1: Prefill 阶段")
    prompt = torch.cat([tokens["Hello"], tokens[" "]], dim=1)  # 2 tokens
    print(f"    输入: ['<bos>', 'Hello'] (2 tokens)")
    output, kv_cache = attn(prompt, kv_cache=None, use_cache=True)
    print(f"    KV Cache 大小: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    print(f"    缓存了 {kv_cache[0].size(2)} 个 token 的 K/V")

    # Step 2-4: Decode（逐 token 生成）
    generated_tokens = ["World", "!"]
    for i, token_name in enumerate(generated_tokens, start=2):
        print(f"\n  Step {i}: 生成 '{token_name}'")
        new_token = tokens[token_name] if token_name in tokens else torch.randn(batch_size, 1, d_model)
        output, kv_cache = attn(new_token, kv_cache=kv_cache, use_cache=True)
        print(f"    输入: 1 个新 token")
        print(f"    KV Cache 大小: K={kv_cache[0].shape}")
        print(f"    总缓存 token 数: {kv_cache[0].size(2)}")

    print(f"\n  生成完成！")
    print(f"  最终 KV Cache 包含 {kv_cache[0].size(2)} 个 token 的历史信息")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("           KV Cache 推理优化学习演示")
    print("=" * 60)
    print("""
    KV Cache 是大模型推理中最重要的优化技术。

    核心思想：
    - 自回归生成时，已生成 token 的 K/V 不变
    - 缓存这些 K/V，避免重复计算
    - 显著加速推理，但增加显存占用
    """)

    input("按 Enter 开始基本概念演示...")
    demo_kv_cache_concept()

    input("\n按 Enter 开始性能对比演示...")
    demo_kv_cache_performance()

    input("\n按 Enter 开始显存分析演示...")
    demo_memory_analysis()

    input("\n按 Enter 开始增量生成流程演示...")
    demo_incremental_generation()

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)
    print("""
    【面试常见问题】

    Q1: KV Cache 的原理是什么？
    A: 缓存已生成 token 的 K 和 V，避免重复计算。
       每次只计算新 token 的 K/V，然后追加到缓存。

    Q2: KV Cache 的显存占用如何计算？
    A: 2 × num_layers × num_heads × seq_len × head_dim × batch × dtype_size

    Q3: 如何优化 KV Cache 的显存占用？
    A: 1. 量化（INT8/INT4）
       2. PagedAttention（分页管理）
       3. 稀疏化（只保留重要 token）

    Q4: vLLM 的 PagedAttention 是什么？
    A: 将 KV Cache 分成固定大小的页（blocks），
       通过指针管理，减少显存碎片，提高利用率。
    """)
