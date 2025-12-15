"""
强化学习5大模块快速实验脚本

运行此脚本进行快速演示，包含所有5个模块
"""

import sys
import os

# 添加路径
sys.path.insert(0, '/home/ryan/2repo/my_infra/RL')

def run_mdp_demo():
    """运行MDP演示"""
    print("\n" + "="*70)
    print("模块1: MDP基础")
    print("="*70)
    try:
        from RL.envs.gridworld import GridWorld
        from RL.mdp_basics import SimpleMDP, Policy, evaluate_policy_iteratively, compute_q_values
        
        mdp = SimpleMDP(gamma=0.9)
        policy = Policy(mdp, policy_type='uniform')
        
        V = evaluate_policy_iteratively(mdp, policy)
        print("\n✅ MDP演示成功！")
        return True
    except Exception as e:
        print(f"❌ MDP演示失败: {e}")
        return False


def run_dp_demo():
    """运行动态规划演示"""
    print("\n" + "="*70)
    print("模块2: 动态规划")
    print("="*70)
    try:
        from RL.envs.gridworld import GridWorld
        from RL.dp_solver import DP
        
        env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        dp = DP(env, gamma=0.99)
        V, policy = dp.value_iteration(max_iterations=100)
        
        print("✅ 动态规划演示成功！")
        return True
    except Exception as e:
        print(f"❌ 动态规划演示失败: {e}")
        return False


def run_mc_demo():
    """运行蒙特卡洛演示"""
    print("\n" + "="*70)
    print("模块3: 蒙特卡洛方法")
    print("="*70)
    try:
        from RL.envs.gridworld import GridWorld
        from RL.mc_learning import MonteCarloAgent
        
        env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        mc = MonteCarloAgent(env, gamma=0.99, epsilon=0.2)
        rewards = mc.train(num_episodes=100, max_steps=100)
        
        print("✅ 蒙特卡洛演示成功！")
        return True
    except Exception as e:
        print(f"❌ 蒙特卡洛演示失败: {e}")
        return False


def run_td_demo():
    """运行时序差分演示"""
    print("\n" + "="*70)
    print("模块4: 时序差分学习")
    print("="*70)
    try:
        from RL.envs.gridworld import GridWorld
        from RL.td_learning import TDAgent
        
        env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        td = TDAgent(env, gamma=0.99, alpha=0.1, epsilon=0.1)
        rewards, _ = td.q_learning(num_episodes=100, max_steps=100)
        
        print("✅ 时序差分演示成功！")
        return True
    except Exception as e:
        print(f"❌ 时序差分演示失败: {e}")
        return False


def run_pg_demo():
    """运行策略梯度演示"""
    print("\n" + "="*70)
    print("模块5: 策略梯度方法")
    print("="*70)
    try:
        from RL.envs.gridworld import GridWorld
        from RL.pg_learning import REINFORCEAgent
        
        env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
        agent = REINFORCEAgent(env, gamma=0.99, learning_rate=1e-2)
        rewards = agent.train(num_episodes=100, max_steps=100)
        
        print("✅ 策略梯度演示成功！")
        return True
    except Exception as e:
        print(f"❌ 策略梯度演示失败: {e}")
        return False


def main():
    """运行所有演示"""
    print("\n" + "🚀 "*35)
    print("强化学习5大模块快速实验")
    print("🚀 "*35)
    
    results = []
    
    # 依次运行各模块
    results.append(("MDP基础", run_mdp_demo()))
    results.append(("动态规划", run_dp_demo()))
    results.append(("蒙特卡洛", run_mc_demo()))
    results.append(("时序差分", run_td_demo()))
    results.append(("策略梯度", run_pg_demo()))
    
    # 打印总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name:20s} {status}")
    
    total = len(results)
    success = sum(1 for _, s in results if s)
    print(f"\n总体: {success}/{total} 模块成功")
    
    if success == total:
        print("\n🎉 所有模块演示完成！")
    else:
        print("\n⚠️  部分模块失败，请检查依赖")


if __name__ == '__main__':
    main()
