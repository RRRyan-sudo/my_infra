#!/usr/bin/env python3
"""
Transformer 项目初始化和验证脚本

运行此脚本以验证项目设置并打印完整的使用说明。
"""

import os
import sys


def print_header():
    """打印项目头"""
    header = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        🎓 Transformer 架构从零实现 - 完整学习项目 🎓                       ║
║                                                                            ║
║                 从基础概念到完整模型实现的手把手教程                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)


def print_project_structure():
    """打印项目结构"""
    structure = """
📁 项目结构：
─────────────────────────────────────────────────────────────────────────────

transformer/
├── 📄 README.md                        项目简介
├── 📄 QUICKSTART.md                    快速开始指南（详细版）
├── 📄 requirements.txt                 Python 依赖
├── 🐍 LEARNING_GUIDE.py               完整学习路线
├── 🐍 demo.py                         交互式演示菜单
├── 🐍 setup.py                        此文件 - 项目初始化
│
├── 📁 src/                            核心实现代码
│   ├── __init__.py
│   ├── 01_positional_encoding.py      ⭐ 位置编码
│   ├── 02_attention.py                ⭐ 注意力机制
│   ├── 04_feed_forward.py             ⭐ 前馈网络
│   ├── 05_encoder_layer.py            ⭐ 编码器层
│   ├── 06_decoder_layer.py            ⭐ 解码器层
│   └── 07_transformer.py              ⭐ 完整模型
│
├── 📁 notebooks/                       交互式 Jupyter 笔记本
│   └── 01_transformer_tutorial.ipynb   完整教程
│
└── 📁 examples/                        实践示例
    └── train_example.py               机器翻译训练示例

⭐ = 每个文件都包含详细注释和演示代码
    """
    print(structure)


def print_installation_guide():
    """打印安装指南"""
    guide = """
🔧 安装步骤：
─────────────────────────────────────────────────────────────────────────────

【步骤 1】进入项目目录
  $ cd /home/ryan/2repo/my_infra/transformer

【步骤 2】安装依赖库
  $ pip install -r requirements.txt

  如果你已经安装了 PyTorch，可以跳过 torch 的安装。
  
  依赖库列表：
  ✓ torch==2.1.0            # PyTorch 深度学习框架
  ✓ numpy==1.24.3           # 数值计算
  ✓ matplotlib==3.7.2       # 数据可视化
  ✓ jupyter==1.0.0          # 交互式笔记本

【步骤 3】验证安装
  $ python -c "import torch; print(f'PyTorch {torch.__version__}')"

✅ 如果没有错误，说明安装成功！
    """
    print(guide)


def print_usage_guide():
    """打印使用指南"""
    guide = """
🚀 使用指南 - 三种学习方式：
─────────────────────────────────────────────────────────────────────────────

【方式 1】交互式菜单（推荐新手）⭐
  $ python demo.py
  
  优点：
  ✓ 菜单式交互，易于导航
  ✓ 逐个选择学习模块
  ✓ 包含详细的学习路线提示

【方式 2】命令行逐个运行（推荐初学者）
  按顺序运行每个模块：
  
  $ python src/01_positional_encoding.py    (5-10 分钟)
    └─ 学习位置编码的原理和实现
  
  $ python src/02_attention.py              (10-15 分钟)
    └─ 学习注意力机制和多头注意力
  
  $ python src/04_feed_forward.py           (5 分钟)
    └─ 学习前馈网络层
  
  $ python src/05_encoder_layer.py          (10 分钟)
    └─ 学习编码器层的组合
  
  $ python src/06_decoder_layer.py          (10 分钟)
    └─ 学习解码器层的设计
  
  $ python src/07_transformer.py            (10 分钟)
    └─ 学习完整 Transformer 模型
  
  总计：约 1 小时 (不包括深度思考的时间)

【方式 3】交互式 Jupyter Notebook（推荐探索者）
  $ jupyter notebook notebooks/01_transformer_tutorial.ipynb
  
  优点：
  ✓ 可以修改代码并即时看到结果
  ✓ 便于添加注释和笔记
  ✓ 支持数据可视化

【实践应用】运行完整示例
  $ python examples/train_example.py
  
  说明：
  - 演示完整的机器翻译任务
  - 包含数据准备、模型训练、评估、推断
  - 需要 5-10 分钟运行
    """
    print(guide)


def print_learning_objectives():
    """打印学习目标"""
    objectives = """
🎯 学习目标 - 完成后你将能够：
─────────────────────────────────────────────────────────────────────────────

基础理解 (✓ 必须掌握)
  ✓ 解释 Transformer 为什么比 RNN 更好
  ✓ 理解自注意力机制的工作原理
  ✓ 说明为什么需要位置编码
  ✓ 描述编码器和解码器的区别

数学公式 (✓ 理解不是死记)
  ✓ 推导缩放点积注意力公式
  ✓ 理解多头注意力如何并行工作
  ✓ 计算模型的参数数量
  ✓ 分析时间和空间复杂度

代码实现 (✓ 能够独立实现)
  ✓ 实现位置编码
  ✓ 实现缩放点积注意力
  ✓ 实现多头注意力
  ✓ 实现编码器层和解码器层
  ✓ 实现完整 Transformer 模型

实际应用 (✓ 了解流程)
  ✓ 准备序列数据
  ✓ 设置模型超参数
  ✓ 编写训练循环
  ✓ 评估模型性能
  ✓ 进行模型推断

扩展知识 (✓ 深入理解)
  ✓ 理解 Transformer 的变体（如 Pre-LN）
  ✓ 了解注意力机制的优化方法
  ✓ 学会调试和优化 Transformer 模型

🏆 完成检查清单
  [ ] 理解位置编码的作用
  [ ] 推导注意力公式
  [ ] 实现多头注意力
  [ ] 构建编码器层
  [ ] 构建解码器层
  [ ] 组装完整模型
  [ ] 在任务上成功训练
  [ ] 理解 Transformer 的限制和改进
    """
    print(objectives)


def print_key_concepts():
    """打印关键概念"""
    concepts = """
💡 关键概念速查 - 核心公式和思想：
─────────────────────────────────────────────────────────────────────────────

【位置编码 | Positional Encoding】
  why:  Transformer 并行处理，需要编码位置信息
  what: PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  how:  添加到 embedding，位置越远波长越长

【注意力机制 | Scaled Dot-Product Attention】
  why:  让模型学会关注相关的位置
  what: Attention(Q,K,V) = softmax(QK^T / √d_k) V
        其中 Q = Query, K = Key, V = Value, d_k = 维度
  how:  1. 计算相似度: QK^T
        2. 缩放: 除以√d_k
        3. softmax: 转换为概率
        4. 加权求和: 乘以 V

【多头注意力 | Multi-Head Attention】
  why:  不同的头学习不同的表示子空间
  what: 多个注意力头并行运行
  how:  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        最后: Concat(head_1, ..., head_h) W^O

【前馈网络 | Feed Forward Network】
  why:  在每个位置添加非线性变换
  what: FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
        d_ff 通常是 d_model 的 4 倍
  how:  两层全连接，在序列中每个位置独立应用

【编码器层 | Encoder Layer】
  why:  提取源序列的语义表示
  what: 多头自注意力 + 前馈网络
  how:  x' = Attention(x,x,x) → Residual + LayerNorm
        x' = FFN(x) → Residual + LayerNorm

【解码器层 | Decoder Layer】
  why:  根据源序列生成目标序列
  what: 掩蔽自注意力 + 交叉注意力 + 前馈网络
  how:  - 自注意力 (Q,K,V 都来自解码器) + 因果掩码
        - 交叉注意力 (Q 来自解码器, K,V 来自编码器)
        - 前馈网络

【完整模型 | Transformer】
  why:  编码器提取特征，解码器生成序列
  what: Encoder(N layers) + Decoder(N layers)
  how:  源序列 → Embedding + PE → 编码器
        目标序列 → Embedding + PE → 解码器 + 编码器输出

【残差连接 | Residual Connection】
  why:  梯度直接流动，缓解深网络训练困难
  what: y = x + f(x)
  how:  允许网络学习恒等映射

【层归一化 | Layer Normalization】
  why:  稳定训练，不依赖批次大小
  what: 在特征维度上进行归一化
  how:  LN(x) = (x - mean) / sqrt(variance + eps)

【因果掩码 | Causal Mask】
  why:  解码器不能看到未来信息
  what: 下三角矩阵，[1,0,0,0; 1,1,0,0; 1,1,1,0; 1,1,1,1]
  how:  在 softmax 前将未来位置设为 -inf
    """
    print(concepts)


def print_tips_and_tricks():
    """打印提示和技巧"""
    tips = """
💎 学习提示和技巧：
─────────────────────────────────────────────────────────────────────────────

【有效学习的 5 个步骤】
  1️⃣  读代码注释 - 理解设计思想
  2️⃣  运行演示代码 - 观察实际效果
  3️⃣  修改参数 - 观察输出变化
  4️⃣  添加 print() - 调试中间结果
  5️⃣  自己实现 - 从零写一遍

【调试技巧】
  • 检查张量形状：
    print(f"x.shape = {x.shape}")
  
  • 检查数值范围：
    print(f"x: min={x.min()}, max={x.max()}, mean={x.mean()}")
  
  • 检查梯度：
    print(f"gradient: {param.grad}")
  
  • 保存和加载模型：
    torch.save(model.state_dict(), 'model.pt')
    model.load_state_dict(torch.load('model.pt'))

【常见问题解决】
  Q: 为什么注意力权重全是 1/seq_len？
  A: 可能是因为初始化的问题。让梯度更新参数即可。

  Q: 为什么损失不下降？
  A: 检查学习率、批次大小、数据是否正确加载。

  Q: 为什么模型很慢？
  A: 可能是内存不足。减少批次大小或序列长度。

  Q: 如何调整模型大小？
  A: 修改 d_model 和 num_layers：
     - 小模型：d_model=256, num_layers=2
     - 中型模型：d_model=512, num_layers=6
     - 大型模型：d_model=768, num_layers=12

【进阶探索方向】
  • 尝试不同的激活函数 (ReLU, GELU, Swish)
  • 实现不同的掩码机制
  • 添加权重绑定 (Tied Embeddings)
  • 尝试 Pre-LN 而不是 Post-LN
  • 实现 Sparse Attention
  • 添加 Flash Attention 优化

【资源推荐】
  📊 可视化理解：
     https://jalammar.github.io/illustrated-transformer/
  
  🔬 原始论文：
     https://arxiv.org/abs/1706.03762
  
  🧠 深度理解：
     "Transformers from Scratch" - Peter Bloem
  
  💻 参考实现：
     PyTorch Transformer
     Hugging Face Transformers
    """
    print(tips)


def print_next_steps():
    """打印后续步骤"""
    steps = """
🎬 立即开始 - 推荐的后续步骤：
─────────────────────────────────────────────────────────────────────────────

【现在就开始】
  1. 打开终端，cd 到项目目录
  2. 运行: python demo.py
  3. 选择第一个模块学习

【预计时间投入】
  完整学习：2-3 小时
  - 快速浏览：30 分钟
  - 深度学习：2-3 小时
  - 实践应用：1-2 小时

【学习路线】
  Day 1: 基础概念 (位置编码、注意力)
  Day 2: 核心组件 (编码器、解码器)
  Day 3: 完整模型 + 实践

【成功指标】
  ✅ 能解释为什么需要位置编码
  ✅ 能实现缩放点积注意力
  ✅ 能构建完整 Transformer
  ✅ 能在任务上训练模型
  ✅ 理解论文和代码中的设计选择

【进阶方向】
  • 学习 Transformer 变体 (BERT, GPT, T5)
  • 研究效率优化 (Flash Attention, Linear Attention)
  • 应用到具体任务 (翻译、文本分类、问答)
  • 实现微调和迁移学习

【获取帮助】
  • 代码中有详细注释 - 逐行读
  • 运行示例观察效果 - 修改参数看变化
  • 查看参考资源 - 深入理解数学
  • 调试和实验 - 最好的学习方式
    """
    print(steps)


def main():
    """主函数"""
    print_header()
    print_project_structure()
    print_installation_guide()
    print_usage_guide()
    print_learning_objectives()
    print_key_concepts()
    print_tips_and_tricks()
    print_next_steps()
    
    footer = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                      🚀 准备好开始学习了吗？                              ║
║                                                                            ║
║                      运行这个命令开始交互式学习：                          ║
║                                                                            ║
║                          python demo.py                                    ║
║                                                                            ║
║                          祝你学习愉快！ 🎓                                ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(footer)


if __name__ == "__main__":
    main()
