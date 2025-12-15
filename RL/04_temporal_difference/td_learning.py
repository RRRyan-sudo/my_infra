"""
模块4: 时序差分学习 (Temporal Difference Learning)

理论基础:
- TD方法结合了蒙特卡洛和动态规划的优点
- 使用自举 (bootstrapping): V(s) ← V(s) + α[R + γV(s') - V(s)]
- 单步更新，比MC收敛快，比DP少需环境模型

关键算法:
1. TD(0) - 单步时序差分
   V(s) ← V(s) + α[R + γV(s') - V(s)]
   其中 δ = R + γV(s') - V(s) 是TD误差
   
2. Q-Learning - 离策略控制
   Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
   学习最优Q值，但采用ε-贪心策略收集数据
   
3. Sarsa - 在策略控制
   Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
   学习实际执行的策略的Q值
   
4. 预期Sarsa - 改进的在策略方法
   Q(s,a) ← Q(s,a) + α[R + γE[Q(s',·)] - Q(s,a)]

优势:
- 离线学习速度快
- 可处理长episode
- 适应动态环境
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# 添加路径以支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld
from utils.helpers import plot_learning_curve


class TDAgent:
    """
    时序差分学习智能体
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99, 
                 alpha: float = 0.1, epsilon: float = 0.1):
        """
        初始化TD智能体
        
        Args:
            env: 环境
            gamma: 折扣因子
            alpha: 学习率
            epsilon: ε-贪心探索率
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # 初始化Q值
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _state_to_index(self, state: Tuple) -> int:
        """将状态转换为索引"""
        return state[0] * self.env.grid_size + state[1]
    
    def epsilon_greedy_action(self, state_idx: int) -> int:
        """ε-贪心策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state_idx])
    
    def q_learning(self, num_episodes: int = 500, max_steps: int = 100,
                   alpha_decay: bool = True):
        """
        Q-Learning 离策略学习
        
        学习最优Q值，不管采用何种策略
        
        Args:
            num_episodes: 训练轮数
            max_steps: 每轮最大步数
            alpha_decay: 是否衰减学习率
        
        Returns:
            每轮的奖励和长度列表
        """
        print("Q-Learning 开始训练...")
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = self._state_to_index(state)
                
                # 选择动作（行为策略）
                action = self.epsilon_greedy_action(state_idx)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self._state_to_index(next_state)
                
                # Q-Learning 更新（目标策略是贪心）
                max_next_q = np.max(self.Q[next_state_idx])
                td_error = reward + self.gamma * max_next_q - self.Q[state_idx, action]
                
                self.Q[state_idx, action] += self.alpha * td_error
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            
            # 衰减学习率
            if alpha_decay:
                self.alpha = max(0.01, self.alpha * 0.9995)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}, α = {self.alpha:.4f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def sarsa(self, num_episodes: int = 500, max_steps: int = 100,
              alpha_decay: bool = True):
        """
        Sarsa 在策略学习
        
        学习当前策略的Q值（行为策略和目标策略相同）
        
        Args:
            num_episodes: 训练轮数
            max_steps: 每轮最大步数
            alpha_decay: 是否衰减学习率
        
        Returns:
            每轮的奖励和长度列表
        """
        print("Sarsa 开始训练...")
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state_idx = self._state_to_index(state)
            action = self.epsilon_greedy_action(state_idx)
            
            episode_reward = 0
            
            for step in range(max_steps):
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self._state_to_index(next_state)
                
                # 选择下一个动作
                next_action = self.epsilon_greedy_action(next_state_idx)
                
                # Sarsa 更新（目标策略也是ε-贪心）
                td_error = reward + self.gamma * self.Q[next_state_idx, next_action] \
                           - self.Q[state_idx, action]
                self.Q[state_idx, action] += self.alpha * td_error
                
                state_idx = next_state_idx
                action = next_action
                episode_reward += reward
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            
            if alpha_decay:
                self.alpha = max(0.01, self.alpha * 0.9995)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}, α = {self.alpha:.4f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def expected_sarsa(self, num_episodes: int = 500, max_steps: int = 100,
                      alpha_decay: bool = True):
        """
        期望Sarsa - 改进的在策略方法
        
        使用期望Q值而不是采样的动作
        
        Args:
            num_episodes: 训练轮数
            max_steps: 每轮最大步数
            alpha_decay: 是否衰减学习率
        
        Returns:
            每轮的奖励和长度列表
        """
        print("期望Sarsa 开始训练...")
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = self._state_to_index(state)
                action = self.epsilon_greedy_action(state_idx)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                next_state_idx = self._state_to_index(next_state)
                
                # 计算期望Q值
                q_values = self.Q[next_state_idx]
                best_action = np.argmax(q_values)
                expected_q = (1 - self.epsilon) * q_values[best_action] + \
                            self.epsilon * np.mean(q_values)
                
                # Expected Sarsa 更新
                td_error = reward + self.gamma * expected_q - self.Q[state_idx, action]
                self.Q[state_idx, action] += self.alpha * td_error
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            
            if alpha_decay:
                self.alpha = max(0.01, self.alpha * 0.9995)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}, α = {self.alpha:.4f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def test(self, num_episodes: int = 10, max_steps: int = 100):
        """测试策略"""
        print(f"\n测试策略 ({num_episodes}轮)...")
        
        total_reward = 0
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = self._state_to_index(state)
                action = np.argmax(self.Q[state_idx])  # 贪心
                
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                state = next_state
                if done:
                    success_count += 1
                    break
            
            total_reward += episode_reward
            print(f"  Episode {episode + 1}: 奖励 = {episode_reward:.4f}")
        
        avg_reward = total_reward / num_episodes
        success_rate = success_count / num_episodes * 100
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"成功率: {success_rate:.1f}%")
        
        return avg_reward


if __name__ == '__main__':
    # 创建环境
    env1 = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    env2 = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    env3 = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    
    print("="*70)
    print("时序差分学习 - 比较不同方法")
    print("="*70)
    
    # Q-Learning
    print("\n【Q-Learning (离策略)】")
    print("-" * 70)
    agent_ql = TDAgent(env1, gamma=0.99, alpha=0.1, epsilon=0.1)
    rewards_ql, _ = agent_ql.q_learning(num_episodes=500, max_steps=100)
    agent_ql.test()
    
    # Sarsa
    print("\n\n【Sarsa (在策略)】")
    print("-" * 70)
    agent_sarsa = TDAgent(env2, gamma=0.99, alpha=0.1, epsilon=0.1)
    rewards_sarsa, _ = agent_sarsa.sarsa(num_episodes=500, max_steps=100)
    agent_sarsa.test()
    
    # Expected Sarsa
    print("\n\n【期望Sarsa】")
    print("-" * 70)
    agent_esarsa = TDAgent(env3, gamma=0.99, alpha=0.1, epsilon=0.1)
    rewards_esarsa, _ = agent_esarsa.expected_sarsa(num_episodes=500, max_steps=100)
    agent_esarsa.test()
    
    # 比较学习曲线
    plot_learning_curve(rewards_ql, window_size=50, 
                       title='TD Learning Methods Comparison',
                       save_path='/tmp/td_comparison.png')
    print("\n学习曲线已保存到 /tmp/td_comparison.png")
