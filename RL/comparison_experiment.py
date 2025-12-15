"""
综合对比实验脚本

展示5个模块在同一问题上的表现对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.gridworld import GridWorld
from utils.helpers import plot_learning_curve


def run_comparison():
    """运行综合对比实验"""
    
    print("\n" + "="*80)
    print(" "*20 + "强化学习5大模块综合对比")
    print("="*80)
    
    print("\n实验设置:")
    print("-" * 80)
    print("环境: 4x4 网格世界，1个障碍物")
    print("目标: 从随机位置到达目标位置")
    print("每个模块运行条件: 500个episode，最多100步/episode")
    print("-" * 80)
    
    # 创建测试环境
    env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    
    results = {}
    
    # 模块1: MDP基础 (仅展示理论，不做实验)
    print("\n【模块1】MDP基础 - 理论框架")
    print("  → 提供数学基础，不直接用于实验")
    print("  ✓ 理解价值函数和贝尔曼方程")
    results['MDP'] = "Theory"
    
    # 模块2: 动态规划
    print("\n【模块2】动态规划 - 已知模型求解")
    print("  需要: 完整的转移概率和奖励函数")
    print("  优点: 收敛快，解的质量高")
    print("  缺点: 计算复杂度高O(S²A)")
    try:
        from dynamic_programming.dp_solver import DP
        env_dp = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        dp = DP(env_dp, gamma=0.99)
        V_vi, policy_vi = dp.value_iteration(max_iterations=100)
        avg_reward_dp = np.sum(V_vi) / len(V_vi)
        results['DP'] = {
            'avg_value': avg_reward_dp,
            'method': 'Value Iteration',
            'time_complexity': 'O(S²A)',
            'model_required': 'Yes'
        }
        print(f"  ✓ 完成 (平均价值: {avg_reward_dp:.4f})")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['DP'] = None
    
    # 模块3: 蒙特卡洛
    print("\n【模块3】蒙特卡洛 - 完整采样学习")
    print("  需要: 环境交互，采集完整episode")
    print("  优点: 无偏，不需要模型")
    print("  缺点: 高方差，收敛慢")
    try:
        from monte_carlo.mc_learning import MonteCarloAgent
        env_mc = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        mc = MonteCarloAgent(env_mc, gamma=0.99, epsilon=0.2)
        rewards_mc = mc.train(num_episodes=500, max_steps=100, epsilon_decay=True)
        avg_reward_mc = np.mean(rewards_mc[-100:])
        results['MC'] = {
            'final_avg_reward': avg_reward_mc,
            'method': 'First-Visit MC',
            'convergence': f'{len(rewards_mc)} episodes',
            'model_required': 'No'
        }
        print(f"  ✓ 完成 (最后100个episode平均奖励: {avg_reward_mc:.4f})")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['MC'] = None
    
    # 模块4: 时序差分
    print("\n【模块4】时序差分 - 单步自举学习")
    print("  需要: 环境交互，每步更新")
    print("  优点: 快速收敛，低方差")
    print("  缺点: 有偏差")
    try:
        from temporal_difference.td_learning import TDAgent
        env_td = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        td = TDAgent(env_td, gamma=0.99, alpha=0.1, epsilon=0.1)
        
        # 比较三种TD方法
        rewards_ql, _ = td.q_learning(num_episodes=500, max_steps=100)
        avg_reward_td = np.mean(rewards_ql[-100:])
        
        results['TD'] = {
            'final_avg_reward': avg_reward_td,
            'method': 'Q-Learning',
            'convergence': f'{len(rewards_ql)} episodes',
            'model_required': 'No'
        }
        print(f"  ✓ 完成 Q-Learning (最后100个episode平均奖励: {avg_reward_td:.4f})")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['TD'] = None
    
    # 模块5: 策略梯度
    print("\n【模块5】策略梯度 - 参数化策略学习")
    print("  需要: 环境交互，反向传播")
    print("  优点: 支持连续动作，理论基础好")
    print("  缺点: 方差大（无基准函数时）")
    try:
        from policy_gradient.pg_learning import ActorCriticAgent
        env_pg = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        ac = ActorCriticAgent(env_pg, gamma=0.99, actor_lr=1e-2, critic_lr=1e-2)
        rewards_ac = ac.train(num_episodes=500, max_steps=100)
        avg_reward_ac = np.mean(rewards_ac[-100:])
        
        results['PG'] = {
            'final_avg_reward': avg_reward_ac,
            'method': 'Actor-Critic',
            'convergence': f'{len(rewards_ac)} episodes',
            'model_required': 'No'
        }
        print(f"  ✓ 完成 Actor-Critic (最后100个episode平均奖励: {avg_reward_ac:.4f})")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['PG'] = None
    
    # 打印对比总结
    print("\n" + "="*80)
    print("综合对比总结")
    print("="*80)
    
    print("\n1. 方法特征对比:")
    print("-" * 80)
    
    comparison_table = """
┌─────────────┬──────────────┬──────────┬────────┬──────┬──────────┐
│   算法      │ 完整性需求   │  收敛性  │ 方差   │ 偏差 │ 样本效率 │
├─────────────┼──────────────┼──────────┼────────┼──────┼──────────┤
│ DP策略迭代  │ 需要完整模型 │  快      │  低    │ 无   │   高     │
│ DP价值迭代  │ 需要完整模型 │  快      │  低    │ 无   │   高     │
│ 蒙特卡洛    │ 仅需交互     │  中等    │  高    │ 无   │   低     │
│ Q-Learning  │ 仅需交互     │  快      │  中    │ 有   │   中     │
│ Sarsa       │ 仅需交互     │  中等    │  低    │ 无   │   中     │
│ REINFORCE   │ 仅需交互     │  较慢    │  高    │ 无   │   低     │
│ Actor-Critic│ 仅需交互     │  中等    │  低    │ 有   │   中     │
└─────────────┴──────────────┴──────────┴────────┴──────┴──────────┘
"""
    print(comparison_table)
    
    print("\n2. 应用场景:")
    print("-" * 80)
    print("DP:       已知完整环境模型（如规划问题）")
    print("MC:       离散episode，需要精确回报（如游戏、仿真）")
    print("Q-Learn:  学习最优策略，离策略改进（推荐首选）")
    print("Sarsa:    学习当前策略，更保守的探索")
    print("REINFORCE:连续动作，易实现，但需要大量样本")
    print("A-C:      结合策略和价值，通用高效方案")
    
    print("\n3. 学习曲线特点:")
    print("-" * 80)
    print("DP:       立即获得最优解（无学习曲线）")
    print("MC:       缓慢上升，高方差")
    print("TD:       快速上升，平稳收敛")
    print("PG:       波动下降，长期改进")
    
    print("\n" + "="*80)
    print("学习建议")
    print("="*80)
    print("""
1. 理论基础:
   ✓ 从MDP开始，掌握贝尔曼方程
   ✓ 理解状态价值和动作价值的区别
   
2. 实践路径:
   ✓ DP了解最优解的样子
   ✓ MC理解采样学习的含义
   ✓ TD体验快速收敛的感受
   ✓ PG学习参数化策略的威力
   
3. 选择算法:
   ✓ 小规模、已知模型 → DP
   ✓ 中等规模、离散动作 → Q-Learning
   ✓ 连续动作空间 → Policy Gradient
   ✓ 需要鲁棒性 → Actor-Critic
   
4. 进阶方向:
   ✓ 函数近似 + DQN
   ✓ 加速收敛 + PPO
   ✓ 多智能体 + MARL
    """)
    
    print("="*80)
    print("✅ 实验完成！")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        run_comparison()
    except ImportError as e:
        print(f"\n⚠️  导入错误: {e}")
        print("确保在RL目录下运行此脚本")
        print("cd /home/ryan/2repo/my_infra/RL")
        print("python comparison_experiment.py")
