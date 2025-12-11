#!/usr/bin/env python3
"""
Transformer 交互式演示脚本

这个脚本提供了一个交互式菜单，让你可以逐个学习和测试 Transformer 的各个组件。
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "=" * 80)
    print(" " * 20 + "✨ Transformer 架构从零实现 - 交互式演示 ✨")
    print("=" * 80)


def print_menu():
    """打印主菜单"""
    menu = """
📚 主菜单 - 选择你想学习的模块：

基础概念：
  1. 位置编码 (Positional Encoding)
     └─ 学习如何编码序列中的位置信息

  2. 注意力机制 (Attention Mechanism)
     └─ 学习缩放点积注意力和多头注意力

中间组件：
  3. 前馈网络 (Feed Forward Network)
     └─ 学习位置级别的前馈网络

  4. 编码器层 (Encoder Layer)
     └─ 学习如何组合注意力和前馈网络

  5. 解码器层 (Decoder Layer)
     └─ 学习包含交叉注意力的解码器层

完整模型：
  6. 完整 Transformer 模型
     └─ 学习整个编码器-解码器架构

实践应用：
  7. 训练示例
     └─ 运行一个完整的机器翻译任务

其他：
  8. 显示学习路线图
  9. 查看快速开始指南
  0. 退出

请选择 (0-9):
    """
    print(menu)


def show_learning_path():
    """显示学习路线"""
    learning_path = """
📖 推荐学习路线：

【第一阶段】基础概念 (20-30 分钟)
  1. 运行：python src/01_positional_encoding.py
     学习内容：
     - 为什么需要位置编码？
     - 如何使用正弦/余弦函数编码位置
     - 位置编码的性质

  2. 运行：python src/02_attention.py  
     学习内容：
     - 缩放点积注意力的三个步骤
     - Query、Key、Value 的概念
     - 多头注意力的并行机制

【第二阶段】核心组件 (20-30 分钟)
  3. 运行：python src/04_feed_forward.py
     学习内容：
     - 前馈网络的两层结构
     - 为什么使用 4 倍的隐藏维度

  4. 运行：python src/05_encoder_layer.py
     学习内容：
     - 编码器层的完整结构
     - 残差连接的作用
     - 层归一化的作用

  5. 运行：python src/06_decoder_layer.py
     学习内容：
     - 解码器与编码器的区别
     - 因果掩码的重要性
     - 交叉注意力的原理

【第三阶段】完整模型 (15-20 分钟)
  6. 运行：python src/07_transformer.py
     学习内容：
     - 如何堆叠多个编码器层
     - 如何堆叠多个解码器层
     - 完整的前向传播流程

【第四阶段】实践应用 (30-60 分钟)
  7. 运行：python examples/train_example.py
     学习内容：
     - 数据集准备
     - 模型训练循环
     - 模型推断

💡 学习建议：
  • 每个模块都包含详细注释，务必阅读
  • 运行代码观察数据形状变化
  • 尝试修改参数（如 d_model, num_heads）
  • 使用 print() 调试中间结果
  • 不要跳过任何一个步骤，要确保理解
    """
    print(learning_path)


def show_quickstart():
    """显示快速开始指南"""
    quickstart = """
🚀 快速开始指南：

【第一步】安装依赖
  $ pip install -r requirements.txt

【第二步】选择学习方式

  方式 A：按模块顺序学习（推荐初学者）
    $ python src/01_positional_encoding.py
    $ python src/02_attention.py
    $ python src/04_feed_forward.py
    $ python src/05_encoder_layer.py
    $ python src/06_decoder_layer.py
    $ python src/07_transformer.py

  方式 B：交互式学习（本脚本）
    $ python demo.py
    选择相应的模块进行学习

  方式 C：Jupyter Notebook 学习
    $ jupyter notebook notebooks/

【第三步】运行训练示例
  $ python examples/train_example.py

【常见问题】

Q1: 运行时报错 "ModuleNotFoundError"？
A: 确保已安装依赖 (pip install -r requirements.txt)

Q2: 为什么要学习位置编码？
A: Transformer 是并行处理序列的，没有位置信息就无法理解词序。

Q3: 多头注意力真的有必要吗？
A: 是的！不同的头可以学习不同的表示子空间。

Q4: 我应该修改哪些参数？
A: 先理解代码，然后尝试修改：
   - d_model：模型维度（通常 256, 512, 768）
   - num_heads：注意力头数（通常 4, 8, 16）
   - num_layers：层数（通常 2-12）
   - d_ff：前馈网络维度（通常 4 × d_model）

【学习目标检查】

完成学习后，你应该能够：
  ✓ 解释为什么需要位置编码
  ✓ 推导并实现缩放点积注意力
  ✓ 说明多头注意力的优势
  ✓ 构建编码器和解码器层
  ✓ 实现完整的 Transformer 模型
  ✓ 在实际任务上训练和使用模型

【预计学习时间】
  总计：2-3 小时
  - 基础概念：30 分钟
  - 核心组件：30 分钟
  - 完整模型：20 分钟
  - 实践应用：30 分钟
  - 复习和巩固：30-60 分钟

祝学习愉快！ 🎓
    """
    print(quickstart)


def run_module(module_path, description):
    """运行指定的模块"""
    print("\n" + "=" * 80)
    print(f"📘 正在学习：{description}")
    print("=" * 80)
    print(f"\n运行命令：python {module_path}\n")
    
    # 这里在实际运行时会执行模块
    try:
        import subprocess
        full_path = os.path.join(project_root, module_path)
        result = subprocess.run(
            [sys.executable, full_path],
            capture_output=False,
            text=True
        )
    except Exception as e:
        print(f"❌ 运行失败：{e}")


def main():
    """主函数"""
    print_banner()
    
    # 模块映射
    modules = {
        '1': ('src/01_positional_encoding.py', '位置编码 (Positional Encoding)'),
        '2': ('src/02_attention.py', '注意力机制 (Attention Mechanism)'),
        '3': ('src/04_feed_forward.py', '前馈网络 (Feed Forward Network)'),
        '4': ('src/05_encoder_layer.py', '编码器层 (Encoder Layer)'),
        '5': ('src/06_decoder_layer.py', '解码器层 (Decoder Layer)'),
        '6': ('src/07_transformer.py', '完整 Transformer 模型'),
        '7': ('examples/train_example.py', '训练示例 (Training Example)'),
    }
    
    while True:
        print_menu()
        
        choice = input("请选择 (0-9): ").strip()
        
        if choice == '0':
            print("\n👋 感谢学习！祝你成为 Transformer 专家！\n")
            break
        
        elif choice in modules:
            module_path, description = modules[choice]
            run_module(module_path, description)
            
            input("\n按 Enter 键返回主菜单...")
        
        elif choice == '8':
            show_learning_path()
            input("\n按 Enter 键返回主菜单...")
        
        elif choice == '9':
            show_quickstart()
            input("\n按 Enter 键返回主菜单...")
        
        else:
            print("\n❌ 无效的选择，请输入 0-9 之间的数字。\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已退出演示。\n")
    except Exception as e:
        print(f"\n❌ 发生错误：{e}\n")
