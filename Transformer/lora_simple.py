"""
===============================================================================
                    LoRA（Low-Rank Adaptation）原理演示
===============================================================================

LoRA 是目前最流行的参数高效微调方法。本文件通过简单的实现帮助理解其原理。

核心思想：
    - 冻结预训练模型的原始权重
    - 只训练低秩分解的增量矩阵
    - 推理时可以将增量合并到原始权重中，无额外开销

数学原理：
    原始权重更新: W' = W + ΔW
    LoRA 分解:    ΔW = B × A
    其中:
        W: [d_out, d_in]  原始权重，冻结
        A: [r, d_in]      低秩矩阵，可训练
        B: [d_out, r]     低秩矩阵，可训练
        r << min(d_out, d_in)  秩（通常 4, 8, 16）

参数量对比：
    原始: d_out × d_in = 4096 × 4096 = 16,777,216
    LoRA (r=8): d_out × r + r × d_in = 4096 × 8 + 8 × 4096 = 65,536
    减少比例: 99.6%

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 第一部分：LoRA 线性层实现
# =============================================================================

class LoRALinear(nn.Module):
    """
    带有 LoRA 适配器的线性层

    工作原理：
        原始: y = xW
        LoRA: y = xW + x(BA) × (alpha/r)

        其中：
        - W 是原始的冻结权重
        - B 和 A 是可训练的低秩矩阵
        - alpha 是缩放因子
        - r 是秩

    初始化策略：
        - A: 使用随机高斯初始化
        - B: 使用零初始化

        这样初始时 BA = 0，模型行为与原始相同

    参数：
        in_features: 输入维度
        out_features: 输出维度
        r: LoRA 的秩（rank），决定了增量矩阵的大小
        alpha: 缩放因子，通常设为 r 的倍数
        pretrained_weight: 预训练的权重（可选）
    """

    def __init__(self, in_features, out_features, r=8, alpha=16, pretrained_weight=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # 缩放因子：用于控制 LoRA 增量的影响程度
        # 当 alpha = r 时，缩放因子为 1
        self.scaling = alpha / r

        # =====================================================================
        # 原始线性层（冻结）
        # =====================================================================
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # 如果提供了预训练权重，加载它
        if pretrained_weight is not None:
            self.linear.weight.data = pretrained_weight.clone()

        # 冻结原始权重
        self.linear.weight.requires_grad = False

        # =====================================================================
        # LoRA 低秩矩阵（可训练）
        # =====================================================================
        # A: [r, in_features]  - "下投影"，将输入从 in_features 降到 r 维
        # B: [out_features, r] - "上投影"，将 r 维升回 out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化
        # A 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 使用零初始化，确保初始时 ΔW = BA = 0
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        前向传播

        计算: y = xW + x(BA) × scaling

        参数：
            x: 输入张量 [batch_size, ..., in_features]

        返回：
            输出张量 [batch_size, ..., out_features]
        """
        # 原始线性变换（使用冻结的预训练权重）
        original_output = self.linear(x)

        # LoRA 增量
        # x @ A^T: [batch, ..., in_features] @ [in_features, r] = [batch, ..., r]
        # (x @ A^T) @ B^T: [batch, ..., r] @ [r, out_features] = [batch, ..., out_features]
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)

        # 合并：原始输出 + 缩放后的 LoRA 输出
        return original_output + lora_output * self.scaling

    def merge_weights(self):
        """
        将 LoRA 权重合并到原始权重中

        推理时使用，合并后无额外计算开销

        数学原理：
            y = xW + x(BA) × scaling
              = x(W + BA × scaling)
              = xW'

        合并后 W' = W + BA × scaling
        """
        # 计算 ΔW = B @ A × scaling
        delta_w = (self.lora_B @ self.lora_A) * self.scaling

        # 合并到原始权重
        self.linear.weight.data += delta_w

        # 清零 LoRA 权重（可选，防止重复合并）
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

        print(f"权重已合并！")
        print(f"  原始权重形状: {self.linear.weight.shape}")
        print(f"  ΔW 形状: {delta_w.shape}")

    def get_trainable_params(self):
        """获取可训练参数数量"""
        return self.lora_A.numel() + self.lora_B.numel()

    def get_total_params(self):
        """获取总参数数量"""
        return self.linear.weight.numel() + self.lora_A.numel() + self.lora_B.numel()


# =============================================================================
# 第二部分：将 LoRA 应用到模型
# =============================================================================

def apply_lora_to_model(model, r=8, alpha=16, target_modules=None):
    """
    将 LoRA 应用到模型的指定层

    参数：
        model: PyTorch 模型
        r: LoRA 的秩
        alpha: 缩放因子
        target_modules: 要应用 LoRA 的模块名称列表
                       如 ['q_proj', 'v_proj'] 表示只对 Q 和 V 投影应用 LoRA
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    replaced_count = 0

    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 获取父模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # 创建 LoRA 层替换原始层
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    alpha=alpha,
                    pretrained_weight=module.weight.data
                )

                # 替换
                setattr(parent, child_name, lora_layer)
                replaced_count += 1

    print(f"已将 {replaced_count} 个层替换为 LoRA 层")
    return model


# =============================================================================
# 第三部分：简单的演示模型
# =============================================================================

class SimpleMLP(nn.Module):
    """用于演示的简单 MLP 模型"""

    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleTransformerBlock(nn.Module):
    """用于演示的简单 Transformer Block"""

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 注意力投影层（这些是通常应用 LoRA 的层）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # FFN
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 简化的注意力计算
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        x = attn @ v
        x = self.o_proj(x)
        x = x + residual

        # FFN
        residual = x
        x = self.norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


# =============================================================================
# 第四部分：演示代码
# =============================================================================

def demo_lora_basic():
    """演示 LoRA 的基本原理"""
    print("\n" + "=" * 60)
    print("LoRA 基本原理演示")
    print("=" * 60)

    in_features = 768
    out_features = 768
    r = 8
    alpha = 16

    # 创建 LoRA 层
    lora_layer = LoRALinear(in_features, out_features, r=r, alpha=alpha)

    print(f"\n【参数配置】")
    print(f"  输入维度: {in_features}")
    print(f"  输出维度: {out_features}")
    print(f"  LoRA 秩 (r): {r}")
    print(f"  缩放因子 (alpha): {alpha}")
    print(f"  实际缩放: {lora_layer.scaling}")

    print(f"\n【参数量对比】")
    original_params = in_features * out_features
    lora_params = lora_layer.get_trainable_params()
    print(f"  原始线性层参数: {original_params:,}")
    print(f"  LoRA 可训练参数: {lora_params:,}")
    print(f"  参数减少比例: {100 * (1 - lora_params / original_params):.2f}%")

    print(f"\n【LoRA 矩阵形状】")
    print(f"  A 矩阵: {lora_layer.lora_A.shape} (r × in_features)")
    print(f"  B 矩阵: {lora_layer.lora_B.shape} (out_features × r)")

    # 前向传播演示
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, in_features)
    y = lora_layer(x)

    print(f"\n【前向传播】")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")


def demo_lora_training():
    """演示 LoRA 微调过程"""
    print("\n" + "=" * 60)
    print("LoRA 微调过程演示")
    print("=" * 60)

    # 创建模型
    model = SimpleTransformerBlock(d_model=256, num_heads=4)

    # 统计原始参数
    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n【原始模型】")
    print(f"  总参数量: {total_params_before:,}")
    print(f"  可训练参数: {trainable_before:,}")

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换 Q 和 V 投影为 LoRA 层
    print(f"\n【应用 LoRA 到 Q 和 V 投影层】")

    # 手动替换 q_proj 和 v_proj
    model.q_proj = LoRALinear(256, 256, r=8, alpha=16, pretrained_weight=model.q_proj.weight.data)
    model.v_proj = LoRALinear(256, 256, r=8, alpha=16, pretrained_weight=model.v_proj.weight.data)

    # 统计 LoRA 后的参数
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n【LoRA 模型】")
    print(f"  总参数量: {total_params_after:,}")
    print(f"  可训练参数: {trainable_after:,}")
    print(f"  可训练参数占比: {100 * trainable_after / total_params_after:.2f}%")

    # 模拟训练
    print(f"\n【模拟训练一步】")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    x = torch.randn(4, 10, 256)
    target = torch.randn(4, 10, 256)

    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    print(f"  损失: {loss.item():.4f}")
    print(f"  只有 LoRA 参数被更新!")


def demo_weight_merging():
    """演示权重合并"""
    print("\n" + "=" * 60)
    print("LoRA 权重合并演示")
    print("=" * 60)

    in_features = 512
    out_features = 512

    # 创建 LoRA 层
    lora_layer = LoRALinear(in_features, out_features, r=8, alpha=16)

    # 模拟训练后的 LoRA 权重
    lora_layer.lora_A.data = torch.randn_like(lora_layer.lora_A) * 0.01
    lora_layer.lora_B.data = torch.randn_like(lora_layer.lora_B) * 0.01

    # 测试输入
    x = torch.randn(2, 10, in_features)

    # 合并前的输出
    with torch.no_grad():
        output_before = lora_layer(x)

    print(f"\n合并前:")
    print(f"  LoRA A 范数: {lora_layer.lora_A.norm().item():.4f}")
    print(f"  LoRA B 范数: {lora_layer.lora_B.norm().item():.4f}")

    # 合并权重
    print(f"\n执行权重合并...")
    lora_layer.merge_weights()

    # 合并后的输出
    with torch.no_grad():
        output_after = lora_layer(x)

    print(f"\n合并后:")
    print(f"  LoRA A 范数: {lora_layer.lora_A.norm().item():.4f}")
    print(f"  LoRA B 范数: {lora_layer.lora_B.norm().item():.4f}")

    # 验证输出一致性
    diff = (output_before - output_after).abs().max().item()
    print(f"\n验证: 合并前后输出最大差异: {diff:.2e}")
    print(f"  {'✓ 输出一致！' if diff < 1e-5 else '✗ 输出不一致'}")


def demo_rank_comparison():
    """演示不同秩的影响"""
    print("\n" + "=" * 60)
    print("不同秩 (rank) 的对比")
    print("=" * 60)

    in_features = 4096
    out_features = 4096

    ranks = [1, 4, 8, 16, 32, 64]

    print(f"\n原始线性层参数量: {in_features * out_features:,}")
    print(f"\n{'秩 (r)':<10} {'LoRA参数量':<15} {'占原始比例':<15} {'理论表达能力':<15}")
    print("-" * 55)

    for r in ranks:
        lora_params = r * in_features + out_features * r
        ratio = lora_params / (in_features * out_features) * 100
        capacity = f"rank={r}"
        print(f"{r:<10} {lora_params:<15,} {ratio:<15.2f}% {capacity:<15}")

    print(f"\n选择建议:")
    print(f"  r=4:  适合简单任务、资源受限场景")
    print(f"  r=8:  平衡选择，大多数任务适用")
    print(f"  r=16: 需要更强表达能力时使用")
    print(f"  r=32+: 接近全参微调，很少需要")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("           LoRA 原理学习演示")
    print("=" * 60)
    print("""
    LoRA (Low-Rank Adaptation) 是最流行的参数高效微调方法。

    核心优势：
    1. 大幅减少可训练参数（通常减少 99%+）
    2. 训练速度快，显存占用低
    3. 推理时可合并权重，无额外开销
    4. 可以为不同任务保存不同的 LoRA 适配器
    """)

    input("按 Enter 开始基本原理演示...")
    demo_lora_basic()

    input("\n按 Enter 开始微调过程演示...")
    demo_lora_training()

    input("\n按 Enter 开始权重合并演示...")
    demo_weight_merging()

    input("\n按 Enter 查看不同秩的对比...")
    demo_rank_comparison()

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)
    print("""
    【面试常见问题】

    Q1: LoRA 的 rank 如何选择？
    A: 通常 r=4,8,16 已足够。r=16 时效果接近全参微调。

    Q2: LoRA 应该应用在哪些层？
    A: 通常应用于 Attention 的 Q、V 投影层。
       扩展版本也可应用于 K、O 和 FFN。

    Q3: alpha 参数的作用？
    A: 控制 LoRA 增量的强度。通常设为 r 或 2r。

    Q4: LoRA vs 全参微调的效果差距？
    A: 在大多数任务上，LoRA 能达到全参微调 95%+ 的效果。
    """)
