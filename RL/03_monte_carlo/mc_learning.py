"""
模块3: 蒙特卡洛方法 (Monte Carlo Methods)

理论基础:
- 蒙特卡洛方法通过采样完整的轨迹来学习
- 不需要知道环境的动态模型
- 只在episode结束时更新价值函数

关键算法:
1. 蒙特卡洛策略评估
   - 采样完整轨迹
   - 计算回报G_t
   - 更新V(s) = E[G]
   
2. 蒙特卡洛策略改进 (GLIE条件)
   - Greedy in the Limit with Infinite Exploration
   - 逐渐减少探索，最终收敛到贪心策略
   - ε-greedy或其他软性策略
   
3. 蒙特卡洛学习的变体:
   - First-visit MC: 只在首次访问时更新
   - Every-visit MC: 每次访问都更新
   - 带有importance sampling的off-policy学习

应用场景:
- 环境模型未知
- Episode相对较短
- 可以采样大量完整轨迹
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os

# 添加路径以支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld
from utils.helpers import plot_learning_curve


class MonteCarloAgent:
    """
    蒙特卡洛学习智能体
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99, 
                 epsilon: float = 0.1, learning_rate: float = 0.01):
        """
        初始化蒙特卡洛智能体
        
        Args:
            env: 环境
            gamma: 折扣因子
            epsilon: ε-贪心探索率
            learning_rate: 学习率（未使用，保留为接口）
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # Q(s,a) 动作价值函数
        self.Q = np.zeros((self.num_states, self.num_actions))
        
        # 访问计数
        self.N = defaultdict(lambda: np.zeros(self.num_actions))
        
        # 轨迹存储
        self.episode_rewards = []
    
    def _state_to_index(self, state: Tuple) -> int:
        """将状态转换为索引"""
        return state[0] * self.env.grid_size + state[1]
    
    def _index_to_state(self, index: int) -> Tuple:
        """将索引转换为状态"""
        return (index // self.env.grid_size, index % self.env.grid_size)
    
    def epsilon_greedy_action(self, state: int) -> int:
        """
        ε-贪心策略选择动作
        
        Args:
            state: 当前状态
        
        Returns:
            选择的动作
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def generate_episode(self, max_steps: int = 100) -> List[Tuple]:
        """
        生成一个完整的episode
        
        Returns:
            [(state, action, reward), ...] 的列表
        """
        trajectory = []
        state = self.env.reset()
        
        for step in range(max_steps):
            state_idx = self._state_to_index(state)
            action = self.epsilon_greedy_action(state_idx)
            
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append((state_idx, action, reward))
            
            state = next_state
            
            if done:
                break
        
        return trajectory
    
    def first_visit_mc_evaluation(self, trajectory: List[Tuple]):
        """
        First-Visit蒙特卡洛评估
        
        只在第一次访问状态时更新
        
        Args:
            trajectory: [(state, action, reward), ...] 的轨迹
        """
        visited = set()
        G = 0.0  # 累积回报
        
        # 从后向前处理轨迹
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = reward + self.gamma * G
            
            # 只在首次访问时更新
            if (state, action) not in visited:
                visited.add((state, action))
                self.N[state][action] += 1
                # 增量更新
                self.Q[state, action] += (G - self.Q[state, action]) / self.N[state][action]
    
    def every_visit_mc_evaluation(self, trajectory: List[Tuple]):
        """
        Every-Visit蒙特卡洛评估
        
        每次访问状态时都更新
        
        Args:
            trajectory: [(state, action, reward), ...] 的轨迹
        """
        G = 0.0
        
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = reward + self.gamma * G
            
            self.N[state][action] += 1
            # 增量更新
            self.Q[state, action] += (G - self.Q[state, action]) / self.N[state][action]
    
    def epsilon_decay(self, episode: int, initial_epsilon: float = 0.1, 
                     decay_rate: float = 0.995):
        """
        衰减探索率 (GLIE条件)
        
        ε_k = initial_epsilon * decay_rate^k
        
        Args:
            episode: 当前episode编号
            initial_epsilon: 初始探索率
            decay_rate: 衰减率
        """
        self.epsilon = max(0.01, initial_epsilon * (decay_rate ** episode))
    
    def train(self, num_episodes: int = 500, max_steps: int = 100, 
              use_every_visit: bool = False, epsilon_decay: bool = True):
        """
        训练蒙特卡洛智能体
        
        Args:
            num_episodes: 训练轮数
            max_steps: 每轮最大步数
            use_every_visit: 是否使用Every-Visit MC
            epsilon_decay: 是否衰减探索率
        
        Returns:
            每轮的奖励列表
        """
        self.episode_rewards = []
        
        print(f"开始训练 (使用{'Every-Visit' if use_every_visit else 'First-Visit'} MC)...")
        
        for episode in range(num_episodes):
            # 生成episode
            trajectory = self.generate_episode(max_steps)
            
            # 计算episode回报
            episode_reward = sum(r for _, _, r in trajectory)
            self.episode_rewards.append(episode_reward)
            
            # 更新价值函数
            if use_every_visit:
                self.every_visit_mc_evaluation(trajectory)
            else:
                self.first_visit_mc_evaluation(trajectory)
            
            # 衰减探索率
            if epsilon_decay:
                self.epsilon_decay(episode)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}, ε = {self.epsilon:.4f}")
        
        return self.episode_rewards
    
    def test(self, num_episodes: int = 10, max_steps: int = 100):
        """
        测试学到的策略
        
        Args:
            num_episodes: 测试轮数
            max_steps: 每轮最大步数
        """
        print(f"\n测试策略 ({num_episodes}轮)...")
        
        total_reward = 0
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = self._state_to_index(state)
                # 使用贪心策略
                action = np.argmax(self.Q[state_idx])
                
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
    env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    
    print("="*70)
    print("蒙特卡洛方法演示")
    print("="*70)
    
    # First-Visit MC
    print("\n【First-Visit Monte Carlo】")
    print("-" * 70)
    mc_fv = MonteCarloAgent(env, gamma=0.99, epsilon=0.2)
    rewards_fv = mc_fv.train(num_episodes=500, max_steps=100, 
                              use_every_visit=False, epsilon_decay=True)
    mc_fv.test(num_episodes=10)
    
    # 绘制学习曲线
    plot_learning_curve(rewards_fv, window_size=50, 
                       title='Monte Carlo Learning Curve (First-Visit)',
                       save_path='/tmp/mc_fv_learning_curve.png')
    print("\n学习曲线已保存到 /tmp/mc_fv_learning_curve.png")
