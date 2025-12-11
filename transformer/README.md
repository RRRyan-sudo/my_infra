# Transformer 架构从零实现教程

本项目将带你从头开始理解和实现 Transformer 架构。

## 📚 学习路线

### 第一阶段：核心概念理解
1. **位置编码 (Positional Encoding)** - 如何在序列中编码位置信息
2. **缩放点积注意力 (Scaled Dot-Product Attention)** - 注意力机制的基础
3. **多头注意力 (Multi-Head Attention)** - 并行学习多个表示子空间

### 第二阶段：层级组件
4. **前馈网络 (Feed Forward Network)** - 每个位置独立的全连接网络
5. **编码器层 (Encoder Layer)** - 组合注意力和前馈网络
6. **解码器层 (Decoder Layer)** - 添加跨注意力机制

### 第三阶段：完整模型
7. **Transformer 模型** - 完整的编码器-解码器架构
8. **实践应用** - 训练和推断示例

## 📁 项目结构

```
transformer/
├── src/
│   ├── __init__.py
│   ├── 01_positional_encoding.py    # 位置编码
│   ├── 02_attention.py              # 注意力机制
│   ├── 03_multi_head_attention.py   # 多头注意力
│   ├── 04_feed_forward.py           # 前馈网络
│   ├── 05_encoder_layer.py          # 编码器层
│   ├── 06_decoder_layer.py          # 解码器层
│   └── 07_transformer.py            # 完整模型
├── notebooks/                        # Jupyter 笔记本教程
├── examples/                         # 实践示例
└── requirements.txt                  # 依赖
```

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```bash
python examples/train_example.py
```

## 💡 关键概念

### Transformer 的核心思想
- **自注意力机制**：允许模型关注序列中的任何位置
- **并行处理**：不像RNN那样顺序处理，可以并行处理所有位置
- **位置信息**：通过位置编码保留序列顺序信息
- **编码器-解码器**：编码器提取特征，解码器生成输出

### 数学基础
- 注意力：$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- 多头注意力：多个注意力头并行运行
- 位置编码：$PE(pos, 2i) = \sin(pos/10000^{2i/d})$，$PE(pos, 2i+1) = \cos(pos/10000^{2i/d})$

## 📖 参考资源

- 原论文：《Attention Is All You Need》(Vaswani et al., 2017)
- 相关链接：https://arxiv.org/abs/1706.03762

## 🎯 学习目标

通过完成本项目，你将能够：
- ✅ 理解 Transformer 每个组件的工作原理
- ✅ 从零开始实现完整的 Transformer 模型
- ✅ 理解注意力机制的数学原理
- ✅ 在实际任务中应用 Transformer
