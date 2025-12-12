"""
实践示例：简单的机器翻译任务

本示例演示如何使用实现的Transformer模型进行一个简单的机器翻译任务。
任务：英文 → 目标语言翻译（简化版本）

注：这是一个教学示例，使用简化的数据和配置以便快速运行。
在实际应用中，需要使用更大的数据集和调整超参数。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transformer import Transformer


# ============================================================================
# 数据集类
# ============================================================================

class SimpleTranslationDataset(Dataset):
    """
    简单的翻译数据集
    
    注：这是一个教学用的虚拟数据集，演示数据处理流程
    """
    
    def __init__(self, num_samples=100, src_vocab_size=100, tgt_vocab_size=100, 
                 src_seq_len=10, tgt_seq_len=8, pad_token_id=0):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.pad_token_id = pad_token_id
        
        # 生成虚拟数据（实际应用中应该加载真实数据）
        self.data = self._generate_dummy_data()
    
    def _generate_dummy_data(self):
        """生成虚拟数据用于演示"""
        data = []
        for _ in range(self.num_samples):
            # 随机生成源序列（词表ID）
            src = torch.randint(1, self.src_vocab_size, (self.src_seq_len,))
            # 随机生成目标序列
            tgt = torch.randint(1, self.tgt_vocab_size, (self.tgt_seq_len,))
            data.append((src, tgt))
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        # 在实际应用中，这里会返回实际的翻译对
        return src, tgt


def create_dummy_translation_dataset(num_samples=100):
    """创建虚拟翻译数据集"""
    return SimpleTranslationDataset(num_samples=num_samples)


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, log_interval=10):
    """
    训练一个epoch
    
    参数：
        model: Transformer模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备（CPU或GPU）
        epoch: 当前epoch数
        log_interval: 日志输出间隔
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 目标序列的输入和输出
        # 输入：删除最后一个token（teacher forcing）
        tgt_input = tgt[:, :-1]
        # 输出：删除第一个token（计算损失）
        tgt_output = tgt[:, 1:]
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(src, tgt_input)
        
        # 计算损失
        # logits 形状: (batch_size, tgt_seq_len-1, vocab_size)
        # tgt_output 形状: (batch_size, tgt_seq_len-1)
        # 需要reshape以适应CrossEntropyLoss
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Avg Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, val_loader, loss_fn, device):
    """
    评估模型
    
    参数：
        model: Transformer模型
        val_loader: 验证数据加载器
        loss_fn: 损失函数
        device: 设备
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("Transformer 机器翻译示例")
    print("=" * 70)
    
    # ================================================================
    # 配置参数
    # ================================================================
    print("\n配置参数...")
    
    # 数据参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    src_seq_len = 10
    tgt_seq_len = 8
    num_train_samples = 200
    num_val_samples = 50
    batch_size = 16
    
    # 模型参数
    d_model = 256
    num_heads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    max_seq_length = 128
    dropout = 0.1
    
    # 训练参数
    num_epochs = 5
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"设备: {device}")
    print(f"数据集大小: {num_train_samples} 训练 + {num_val_samples} 验证")
    print(f"模型配置: d_model={d_model}, num_heads={num_heads}, "
          f"layers={num_encoder_layers}")
    print(f"训练配置: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    # ================================================================
    # 创建数据集和数据加载器
    # ================================================================
    print("\n创建数据集...")
    
    train_dataset = SimpleTranslationDataset(
        num_samples=num_train_samples,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len
    )
    
    val_dataset = SimpleTranslationDataset(
        num_samples=num_val_samples,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # ================================================================
    # 创建模型
    # ================================================================
    print("\n创建模型...")
    
    model = Transformer(
        vocab_size=src_vocab_size,  # 注：简化版本中使用相同的词表
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    
    model = model.to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # ================================================================
    # 设置损失函数和优化器
    # ================================================================
    print("\n设置训练配置...")
    
    # 使用CrossEntropyLoss作为损失函数
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
    
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    print(f"损失函数: CrossEntropyLoss")
    print(f"优化器: Adam (lr={learning_rate})")
    
    # ================================================================
    # 训练循环
    # ================================================================
    print("\n开始训练...")
    print("-" * 70)
    
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(num_epochs):
            # 训练
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, device, epoch
            )
            train_losses.append(train_loss)
            
            # 验证
            val_loss = evaluate(model, val_loader, loss_fn, device)
            val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    except KeyboardInterrupt:
        print("\n\n训练被中断！")
    
    # ================================================================
    # 训练完成
    # ================================================================
    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")
    print("=" * 70)
    
    # ================================================================
    # 模型推断示例
    # ================================================================
    print("\n推断示例...")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        # 使用验证集的第一个样本
        src, _ = val_dataset[0]
        src = src.unsqueeze(0).to(device)  # 添加batch维度
        
        print(f"源序列: {src[0].tolist()}")
        
        # 编码
        encoder_output = model.encode(src)
        
        # 解码（自回归生成）
        max_gen_len = tgt_seq_len
        generated = []
        
        # 从起始token开始（这里简单地使用token 1）
        current_token = torch.tensor([[1]], device=device)
        
        for i in range(max_gen_len):
            # 预测下一个token
            logits = model.decode(current_token, encoder_output)
            next_token_logits = logits[:, -1, :]  # 取最后一个位置的预测
            next_token = next_token_logits.argmax(dim=-1)
            
            generated.append(next_token.item())
            
            # 添加到当前序列
            current_token = torch.cat([current_token, next_token.unsqueeze(0)], dim=1)
        
        print(f"生成序列: {generated}")
    
    print("\n✅ 示例完成！")
    print("\n提示：")
    print("- 这是一个教学示例，使用虚拟数据演示流程")
    print("- 在实际应用中，需要：")
    print("  1. 使用真实的翻译数据集")
    print("  2. 实现适当的分词（Tokenization）")
    print("  3. 调整超参数以适应具体任务")
    print("  4. 使用更大的模型和数据集")
    print("  5. 实现集束搜索（Beam Search）用于更好的生成质量")


if __name__ == "__main__":
    main()
