#!/usr/bin/env python3
"""
SAC Demo - GridWorld环境上的演示和测试
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sac_minimal import SACAgent

# 导入环境（GridWorld）
try:
    from envs.gridworld import GridWorld
except ImportError:
    print("警告：无法导入GridWorld环境")
    GridWorld = None


def demo_sac_training():
    """
    SAC在GridWorld环境上的训练演示
    """
    print("\n" + "="*80)
    print("SAC (Soft Actor-Critic) GridWorld 演示".center(80))
    print("="*80 + "\n")
    
    # 检查环境
    if GridWorld is None:
        print("❌ 错误：GridWorld环境不可用")
        print("请确保在项目根目录运行此脚本")
        return
    
    # 创建环境
    print("1️⃣  创建环境...")
    env = GridWorld(grid_size=4, max_steps=20)
    state_dim = 16  # 4x4 = 16种状态
    action_dim = 4  # 上下左右
    
    print(f"   ✓ 环境已创建: {state_dim}个状态, {action_dim}个动作")
    print(f"   ✓ 最大步数: {env.max_steps}\n")
    
    # 创建SAC Agent
    print("2️⃣  初始化SAC Agent...")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device='cpu'
    )
    print(f"   ✓ Agent已初始化")
    print(f"   ✓ Actor网络: 输入{state_dim} → 隐层128 → 均值/std×{action_dim}")
    print(f"   ✓ Critic网络: 输入{state_dim+action_dim} → 隐层128 → Q值\n")
    
    # 训练配置
    print("3️⃣  训练配置...")
    num_episodes = 100
    batch_size = 32
    update_freq = 4
    
    print(f"   ✓ 总回合数: {num_episodes}")
    print(f"   ✓ 批大小: {batch_size}")
    print(f"   ✓ 更新频率: 每{update_freq}步更新一次\n")
    
    # 训练循环
    print("4️⃣  开始训练...\n")
    
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 交互一个回合
        done = False
        while not done and episode_length < env.max_steps:
            # 选择动作（从连续分布采样）
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state]).to(agent.device)
                action_continuous, _ = agent.actor.sample(state_tensor)
                # 将连续动作[0,1]映射到离散动作
                action = int((action_continuous[0, 0].item() + 1) / 2 * action_dim) % action_dim
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # 存储经验
            agent.buffer.push(state, action, reward, next_state, done)
            
            # 更新
            if len(agent.buffer) > batch_size and episode_length % update_freq == 0:
                actor_loss, critic_loss = agent.update(batch_size)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 定期打印进度
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            print(f"   回合 {episode+1:3d}/{num_episodes} | "
                  f"奖励: {avg_reward:6.2f} | "
                  f"步数: {avg_length:5.1f}")
    
    print("\n   ✓ 训练完成！\n")
    
    # 打印统计信息
    print("5️⃣  训练结果统计:\n")
    print(f"   最终100回合平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   奖励范围: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    print(f"   平均回合长度: {np.mean(episode_lengths):.2f}")
    print(f"   总更新步数: {len(actor_losses)}\n")
    
    # 绘制学习曲线
    print("6️⃣  绘制学习曲线...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 奖励曲线
    ax = axes[0, 0]
    ax.plot(episode_rewards, label='Episode Reward')
    ax.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
            label='MA(10)', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('SAC Training: Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 回合长度
    ax = axes[0, 1]
    ax.plot(episode_lengths, alpha=0.5, label='Episode Length')
    ax.plot(np.convolve(episode_lengths, np.ones(10)/10, mode='valid'),
            label='MA(10)', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('SAC Training: Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Actor损失
    ax = axes[1, 0]
    if actor_losses:
        ax.plot(actor_losses, alpha=0.5, label='Actor Loss')
        ax.plot(np.convolve(actor_losses, np.ones(10)/10, mode='valid'),
                label='MA(10)', linewidth=2)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Critic损失
    ax = axes[1, 1]
    if critic_losses:
        ax.plot(critic_losses, alpha=0.5, label='Critic Loss')
        ax.plot(np.convolve(critic_losses, np.ones(10)/10, mode='valid'),
                label='MA(10)', linewidth=2)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sac_training_curves.png', dpi=100)
    print("   ✓ 曲线已保存: sac_training_curves.png\n")
    
    # 测试阶段
    print("7️⃣  评估模式（贪心策略）:\n")
    test_episodes = 10
    test_rewards = []
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state]).to(agent.device)
                # 使用均值（贪心）
                mean, _ = agent.actor.forward(state_tensor)
                action = int((mean[0, 0].item() + 1) / 2 * action_dim) % action_dim
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
        
        test_rewards.append(episode_reward)
        print(f"   测试回合 {episode+1:2d}: 奖励 = {episode_reward:6.2f}")
    
    print(f"\n   平均测试奖励: {np.mean(test_rewards):.2f}")
    print(f"   测试奖励范围: [{np.min(test_rewards):.2f}, {np.max(test_rewards):.2f}]\n")
    
    print("="*80)
    print("✅ SAC演示完成！".center(80))
    print("="*80 + "\n")
    
    return agent, env, episode_rewards


def explain_sac_key_concepts():
    """
    解释SAC的关键概念
    """
    print("\n" + "="*80)
    print("SAC 关键概念讲解".center(80))
    print("="*80 + "\n")
    
    concepts = {
        "1. 最大熵目标": """
目标函数：J(π) = E[r_t + α H(π)]
  - 不仅最大化奖励 r_t
  - 还最大化策略熵 H(π) = -E[log π(a|s)]
  - α权衡两者的重要性
  
好处：
  - 鼓励探索：不会过早收敛
  - 学习多模态策略：多条最优路径
  - 提高鲁棒性：对环境变化容错
""",
        
        "2. 重参数化技巧": """
采样流程：
  z ~ N(μ(s), σ²(s))     高斯采样
  a = tanh(z)            压缩到[-1,1]
  
为什么？
  - 直接采样 a~π 不可微
  - 重参数化使梯度能流向参数μ和σ
  - 关键：log π(a|s) = log p(z) - log|1-a²|
           └─────────┬─────────┘   └──────┬──────┘
                梯度项         雅可比修正（重要！）
""",
        
        "3. 双Q网络": """
设计：两个独立的Q网络 Q_1 和 Q_2
      目标值 = min(Q_1_target, Q_2_target)
  
为什么？
  - 单个Q容易系统性高估动作价值
  - 两个独立网络的最小值更保守
  - 大幅提高学习稳定性
  
权衡：
  - 计算量↑（两个网络）
  - 稳定性↑↑（最重要）
""",
        
        "4. 自适应温度": """
参数：α（熵系数），通过 log_α 学习

自动调节：
  目标熵 H_target = -action_dim
  
  如果 H(π) < H_target：
    → loss > 0, log_α↑, α↑
    → 更多奖励给熵 → 鼓励探索
  
  如果 H(π) > H_target：
    → loss < 0, log_α↓, α↓
    → 减少奖励给熵 → 专注高回报
  
好处：不需要手动调整探索程度！
""",
        
        "5. 三个网络更新": """
每步更新三个目标：

1️⃣  评论家(Critic)更新：
   最小化: L_Q = (Q(s,a) - y)²
   其中:   y = r + γ V(s') = r + γ(Q(s',a') - α log π(a'|s'))

2️⃣  演员(Actor)更新：
   最小化: L_π = α log π(a|s) - Q(s,a)
   效果:   增加高Q值动作概率 + 增加熵

3️⃣  温度(Alpha)更新：
   最小化: L_α = -α(log π(a|s) + H_target)
   效果:   自动调节探索程度
"""
    }
    
    for title, explanation in concepts.items():
        print(f"{title}")
        print("-" * 80)
        print(explanation)
        print()
    
    print("="*80 + "\n")


def main():
    """主程序"""
    print("\n" + "🚀 "*40)
    print("\n欢迎来到SAC (Soft Actor-Critic) 学习教程\n")
    print("🚀 "*40 + "\n")
    
    while True:
        print("\n请选择操作:")
        print("  1. 查看SAC关键概念讲解")
        print("  2. 运行SAC在GridWorld上的演示")
        print("  3. 查看SAC理论文档")
        print("  4. 查看SAC完整指南")
        print("  5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            explain_sac_key_concepts()
        
        elif choice == '2':
            try:
                demo_sac_training()
            except Exception as e:
                print(f"\n❌ 演示出错: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '3':
            print("\n📚 SAC理论文档位置: SAC_EXPLANATION.md")
            print("   包含内容:")
            print("   - 核心思想和为什么要最大化熵")
            print("   - 数学原理（目标函数、贝尔曼方程）")
            print("   - 重参数化技巧详解")
            print("   - 网络架构和算法流程")
            print("   - 数值稳定性技巧")
            print("   - 超参数调整指南")
            print("   - SAC vs其他算法对比")
            print("\n   在项目根目录打开: cat 06_soft_actor_critic/SAC_EXPLANATION.md")
        
        elif choice == '4':
            print("\n📖 SAC完整指南位置: sac_guide.py")
            print("   包含内容:")
            print("   - 10部分的详细讲解")
            print("   - 核心思想 → 数学原理 → 实现细节")
            print("   - 优缺点分析")
            print("   - 与其他算法对比")
            print("   - 常见问题解答")
            print("   - 高级话题（离线RL、分布式等）")
            print("\n   运行查看: python sac_guide.py")
        
        elif choice == '5':
            print("\n👋 谢谢使用！继续学习SAC吧！\n")
            break
        
        else:
            print("❌ 无效选择，请重试")


if __name__ == '__main__':
    # 如果有命令行参数，直接运行对应功能
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_sac_training()
        elif sys.argv[1] == '--concepts':
            explain_sac_key_concepts()
        elif sys.argv[1] == '--guide':
            import sac_guide
        elif sys.argv[1] == '--help':
            print("用法: python sac_demo.py [选项]")
            print("选项:")
            print("  --demo: 运行演示")
            print("  --concepts: 显示关键概念")
            print("  --guide: 显示完整指南")
            print("  --help: 显示此帮助")
            print("\n不带参数时进入交互菜单")
    else:
        main()
