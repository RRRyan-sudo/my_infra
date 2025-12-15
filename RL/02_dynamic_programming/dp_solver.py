"""
模块2: 动态规划 (Dynamic Programming)

理论基础:
- 动态规划用于求解已知环境模型的问题
- 两种方式: 策略迭代和价值迭代

关键算法:
1. 策略迭代 (Policy Iteration)
   - 策略评估: 计算当前策略的价值
   - 策略改进: 根据价值更新策略
   - 重复直到收敛
   
2. 价值迭代 (Value Iteration)
   - 直接对最优价值函数迭代
   - V(s) = max_a E[R + γV(s')]
   - 收敛后提取最优策略

应用场景:
- 已知完整的环境动态模型
- 中等大小的问题 (状态空间不过大)
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os

# 添加路径以支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld


class DP:
    """
    动态规划求解器
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        """
        初始化DP求解器
        
        Args:
            env: 环境
            gamma: 折扣因子
        """
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # V(s) 状态价值函数
        self.V = np.zeros(self.num_states)
        
        # π(a|s) 策略，初始为均匀随机
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
    
    def _get_state_index(self, state: Tuple) -> int:
        """将状态转换为索引"""
        return state[0] * self.env.grid_size + state[1]
    
    def _get_state_from_index(self, index: int) -> Tuple:
        """将索引转换为状态"""
        return (index // self.env.grid_size, index % self.env.grid_size)
    
    def _build_transition_model(self):
        """
        构建转移模型
        P[s][a][s'] = 概率
        R[s][a][s'] = 奖励
        """
        self.P = {}
        self.R = {}
        
        for state_idx in range(self.num_states):
            self.P[state_idx] = {}
            self.R[state_idx] = {}
            
            state = self._get_state_from_index(state_idx)
            self.env.agent_pos = state
            
            for action in range(self.num_actions):
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self._get_state_index(next_state)
                
                # 记录转移和奖励
                self.P[state_idx][action] = {next_state_idx: 1.0}  # 确定性环境
                self.R[state_idx][action] = reward
                
                # 重置环境为当前状态（以继续探索其他动作）
                self.env.agent_pos = state
    
    def policy_evaluation(self, epsilon: float = 1e-6, max_iterations: int = 100) -> int:
        """
        策略评估: 计算当前策略的价值函数
        
        迭代计算: V(s) = E[R + γV(s')] 在当前策略下
        
        Args:
            epsilon: 收敛阈值
            max_iterations: 最大迭代次数
        
        Returns:
            实际迭代次数
        """
        for iteration in range(max_iterations):
            V_old = self.V.copy()
            
            for s in range(self.num_states):
                value = 0.0
                
                # 对所有可能的动作求期望
                for a in range(self.num_actions):
                    action_prob = self.policy[s, a]
                    
                    # 对所有可能的下一状态求期望
                    if s in self.P and a in self.P[s]:
                        for s_next, prob in self.P[s][a].items():
                            reward = self.R[s][a]
                            value += action_prob * prob * (reward + self.gamma * V_old[s_next])
                
                self.V[s] = value
            
            # 检查收敛
            delta = np.max(np.abs(self.V - V_old))
            if delta < epsilon:
                return iteration + 1
        
        return max_iterations
    
    def policy_improvement(self) -> bool:
        """
        策略改进: 根据价值函数更新策略
        
        π(s) = argmax_a Q(s,a)
        
        Returns:
            策略是否发生了改变
        """
        policy_stable = True
        
        for s in range(self.num_states):
            old_action = np.argmax(self.policy[s])
            
            # 计算Q值
            q_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                if s in self.P and a in self.P[s]:
                    for s_next, prob in self.P[s][a].items():
                        reward = self.R[s][a]
                        q_values[a] += prob * (reward + self.gamma * self.V[s_next])
            
            # 更新策略为贪心
            new_action = np.argmax(q_values)
            self.policy[s] = 0.0
            self.policy[s, new_action] = 1.0
            
            if old_action != new_action:
                policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略迭代算法
        
        1. 初始化策略为随机
        2. 循环:
           a. 评估当前策略
           b. 改进策略
        3. 直到策略收敛
        
        Args:
            max_iterations: 最大迭代次数
        
        Returns:
            最优价值函数和最优策略
        """
        print("构建转移模型...")
        self._build_transition_model()
        
        for iteration in range(max_iterations):
            print(f"\n策略迭代 #{iteration + 1}")
            
            # 策略评估
            eval_iters = self.policy_evaluation()
            print(f"  策略评估: {eval_iters} 次迭代收敛")
            
            # 策略改进
            policy_stable = self.policy_improvement()
            print(f"  策略改进: 策略{'稳定' if policy_stable else '有改进'}")
            
            if policy_stable:
                print(f"\n策略迭代收敛 (共{iteration + 1}次迭代)")
                break
        
        return self.V, self.policy
    
    def value_iteration(self, epsilon: float = 1e-6, 
                       max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        价值迭代算法
        
        直接迭代最优价值: V(s) = max_a E[R + γV(s')]
        
        Args:
            epsilon: 收敛阈值
            max_iterations: 最大迭代次数
        
        Returns:
            最优价值函数和最优策略
        """
        print("构建转移模型...")
        self._build_transition_model()
        
        print("\n执行价值迭代...")
        for iteration in range(max_iterations):
            V_old = self.V.copy()
            
            for s in range(self.num_states):
                if s not in self.P:
                    continue
                
                # 对每个动作计算价值
                q_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    if a in self.P[s]:
                        for s_next, prob in self.P[s][a].items():
                            reward = self.R[s][a]
                            q_values[a] += prob * (reward + self.gamma * V_old[s_next])
                
                # 取最大值
                self.V[s] = np.max(q_values)
            
            # 检查收敛
            delta = np.max(np.abs(self.V - V_old))
            if (iteration + 1) % 10 == 0 or delta < epsilon:
                print(f"  迭代{iteration + 1}: Δ = {delta:.2e}")
            
            if delta < epsilon:
                print(f"价值迭代收敛 (共{iteration + 1}次迭代)")
                break
        
        # 提取最优策略
        for s in range(self.num_states):
            if s not in self.P:
                continue
            
            q_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                if a in self.P[s]:
                    for s_next, prob in self.P[s][a].items():
                        reward = self.R[s][a]
                        q_values[a] += prob * (reward + self.gamma * self.V[s_next])
            
            best_action = np.argmax(q_values)
            self.policy[s] = 0.0
            self.policy[s, best_action] = 1.0
        
        return self.V, self.policy
    
    def extract_policy(self) -> Dict:
        """
        提取最优策略
        
        Returns:
            状态->动作的字典
        """
        policy_dict = {}
        action_names = ['↑', '↓', '←', '→']
        
        for s in range(self.num_states):
            action = np.argmax(self.policy[s])
            state = self._get_state_from_index(s)
            policy_dict[state] = action
        
        return policy_dict
    
    def test_policy(self, episodes: int = 5, max_steps: int = 100):
        """
        在环境中测试学到的策略
        
        Args:
            episodes: 测试轮数
            max_steps: 每轮最大步数
        """
        print(f"\n测试策略 ({episodes}轮):")
        
        total_reward = 0
        success_count = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = self._get_state_index(state)
                action = np.argmax(self.policy[state_idx])
                
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    success_count += 1
                    break
            
            total_reward += episode_reward
            print(f"  Episode {episode + 1}: 奖励 = {episode_reward:.4f}")
        
        print(f"平均奖励: {total_reward / episodes:.4f}")
        print(f"成功率: {success_count / episodes * 100:.1f}%")
        
        return total_reward / episodes


if __name__ == '__main__':
    # 创建环境
    env = GridWorld(grid_size=4, num_obstacles=2, seed=42)
    
    print("="*70)
    print("动态规划 - 策略迭代 vs 价值迭代")
    print("="*70)
    
    # 方法1: 策略迭代
    print("\n【方法1: 策略迭代】")
    print("-" * 70)
    dp_pi = DP(env.copy() if hasattr(env, 'copy') else env, gamma=0.99)
    V_pi, policy_pi = dp_pi.policy_iteration(max_iterations=10)
    dp_pi.test_policy()
    
    # 方法2: 价值迭代
    print("\n\n【方法2: 价值迭代】")
    print("-" * 70)
    dp_vi = DP(GridWorld(grid_size=4, num_obstacles=2, seed=42), gamma=0.99)
    V_vi, policy_vi = dp_vi.value_iteration(max_iterations=100)
    dp_vi.test_policy()
