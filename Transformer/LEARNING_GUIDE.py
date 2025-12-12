"""
Transformer 完整学习指南

本脚本提供了Transformer架构的完整实现和学习路径。

快速开始步骤：
1. 安装依赖: pip install -r requirements.txt
2. 运行各个模块进行学习:
   - python src/01_positional_encoding.py      # 学习位置编码
   - python src/02_attention.py                # 学习注意力机制
   - python src/04_feed_forward.py             # 学习前馈网络
   - python src/05_encoder_layer.py            # 学习编码器层
   - python src/06_decoder_layer.py            # 学习解码器层
   - python src/07_transformer.py              # 学习完整模型
3. 打开Jupyter notebooks进行交互式学习

学习路径指导：
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 70)
    print(" " * 15 + "欢迎来到 Transformer 架构学习之旅！")
    print("=" * 70)
    
    learning_path = """
📚 学习路径（建议顺序）：

【第一阶段】基础概念
  1️⃣  位置编码 (Positional Encoding)
      为什么需要位置编码？
      - Transformer是并行处理序列的，没有顺序信息
      - 位置编码将位置信息编码到向量中
      关键公式: PE(pos, 2i) = sin(pos/10000^(2i/d))
                PE(pos, 2i+1) = cos(pos/10000^(2i/d))

  2️⃣  注意力机制 (Attention Mechanism)
      什么是注意力？
      - 在处理每个位置时，关注相关的其他位置
      - Query (查询): "我想知道什么"
      - Key (键): "每个位置是什么"
      - Value (值): "每个位置的信息"
      关键公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V

【第二阶段】核心组件
  3️⃣  多头注意力 (Multi-Head Attention)
      为什么使用多个注意力头？
      - 不同的头可以学习不同的表示子空间
      - 并行运行多个注意力头，然后连接结果
      - 增强了模型的表达能力

  4️⃣  前馈网络 (Feed Forward Network)
      结构: Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)
      特点: 在每个位置独立应用相同的前馈网络
           通常 d_ff = 4 × d_model

  5️⃣  层构件（Residual & Layer Norm）
      残差连接: x = x + sublayer(x)
        - 允许梯度直接流动
        - 缓解深网络的训练困难
      
      层归一化: 在特征维度上进行归一化
        - 稳定训练
        - 独立于批次大小

【第三阶段】模型架构
  6️⃣  编码器层 (Encoder Layer)
      结构:
        x' = MultiHeadAttention(x, x, x)
        x = LayerNorm(x + Dropout(x'))
        x' = FeedForward(x)
        x = LayerNorm(x + Dropout(x'))

  7️⃣  解码器层 (Decoder Layer)
      结构:
        x' = MultiHeadAttention(x, x, x, causal_mask)  # 自注意力
        x = LayerNorm(x + Dropout(x'))
        x' = MultiHeadAttention(x, encoder_out, encoder_out)  # 交叉注意力
        x = LayerNorm(x + Dropout(x'))
        x' = FeedForward(x)
        x = LayerNorm(x + Dropout(x'))

  8️⃣  完整Transformer模型
      包含: Embedding + PositionalEncoding + Encoder + Decoder
      数据流: 源序列 → 编码器 → 解码器 → 目标序列预测

【第四阶段】实践应用
  9️⃣  数据预处理
      - 分词（Tokenization）
      - 构建词汇表（Vocabulary）
      - 填充和序列长度处理

  🔟 模型训练
      - 定义损失函数（通常使用交叉熵）
      - 选择优化器（Adam）
      - 实现训练循环
      - 评估模型性能

📊 关键数学公式速查：

1. 注意力: A(Q,K,V) = softmax(QK^T/√d_k)V

2. 多头注意力: MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O
              其中 headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)

3. 前馈: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

4. 位置编码: PE(pos,2i) = sin(pos/10000^(2i/d))
           PE(pos,2i+1) = cos(pos/10000^(2i/d))

5. 残差+归一化: y = LayerNorm(x + f(x))

💡 学习建议：
  - 先理解数学概念，再看代码实现
  - 运行每个模块的测试代码，观察输入输出形状
  - 使用print()和可视化工具理解数据流
  - 修改参数，观察对模型的影响
  - 实现一个简单的机器翻译任务来巩固学习

🚀 快速开始：
  
  # 1. 安装依赖
  pip install -r requirements.txt
  
  # 2. 逐个学习各个模块
  python src/01_positional_encoding.py
  python src/02_attention.py
  python src/04_feed_forward.py
  python src/05_encoder_layer.py
  python src/06_decoder_layer.py
  python src/07_transformer.py
  
  # 3. 进行交互式学习
  jupyter notebook notebooks/
  
  # 4. 运行实践示例
  python examples/train_example.py

📖 参考资源：
  - 论文: "Attention Is All You Need" (Vaswani et al., 2017)
  - 网址: https://arxiv.org/abs/1706.03762
  - 代码: https://github.com/pytorch/examples/blob/master/word_language_model/model.py

🎯 预期学习成果：
  ✓ 深入理解 Transformer 的每个组件
  ✓ 能够从零实现完整的 Transformer 模型
  ✓ 理解位置编码、注意力机制的数学原理
  ✓ 了解如何将 Transformer 应用于实际任务
  ✓ 能够调试和优化 Transformer 模型

有任何问题，请参考对应模块中的详细注释！
    """
    
    print(learning_path)
    print("=" * 70 + "\n")


def verify_installation():
    """验证所需的库是否已安装"""
    print("检查依赖库...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安装，请运行: pip install torch")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy 未安装，请运行: pip install numpy")
        return False
    
    print("\n所有依赖都已安装！\n")
    return True


def main():
    """主函数"""
    print_welcome()
    
    if not verify_installation():
        print("请先安装所需的库")
        return
    
    print("💡 现在你可以开始学习 Transformer 了！\n")
    print("建议的学习步骤：\n")
    
    steps = [
        ("1", "src/01_positional_encoding.py", "学习位置编码"),
        ("2", "src/02_attention.py", "学习注意力机制和多头注意力"),
        ("3", "src/04_feed_forward.py", "学习前馈网络"),
        ("4", "src/05_encoder_layer.py", "学习编码器层"),
        ("5", "src/06_decoder_layer.py", "学习解码器层"),
        ("6", "src/07_transformer.py", "学习完整Transformer模型"),
    ]
    
    for num, file_path, description in steps:
        print(f"  步骤 {num}: python {file_path}")
        print(f"           {description}\n")
    
    print("\n开始学习吧！祝你学习愉快！🚀\n")


if __name__ == "__main__":
    main()
