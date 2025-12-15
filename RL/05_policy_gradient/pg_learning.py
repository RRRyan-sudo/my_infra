"""
模块5: 策略梯度方法 (Policy Gradient Methods)

理论基础:
- 策略梯度方法直接学习策略，而不是价值函数
- 参数化策略：π_θ(a|s)，通过θ控制
- 优化目标：最大化期望回报 J(θ) = E[R]

关键概念:
1. 策略梯度定理 (Policy Gradient Theorem)
   ∇J(θ) ∝ E[∇log π_θ(a|s) Q^π(s,a)]
   
2. Reinforce 算法
   θ ← θ + α ∇log π_θ(a|s) G_t
   使用采样的回报G_t作为Q值的无偏估计
   
3. Actor-Critic 方法
   - Actor: 学习策略 π_θ(a|s)
   - Critic: 学习价值函数 V_φ(s)，估计Q值
   - 使用价值函数作为基准，降低方差
   
4. 优势函数 (Advantage Function)
   A(s,a) = Q(s,a) - V(s)
   表示相对于平均值的优势

优势:
- 可以学习连续动作
- 理论基础坚实
- 适应高维问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import sys
import os

# 添加路径以支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld
from utils.helpers import plot_learning_curve


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """输出动作概率分布"""
        return self.net(state)


class ValueNetwork(nn.Module):
    """价值网络（评论家）"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """输出状态价值"""
        return self.net(state)


class REINFORCEAgent:
    """
    Reinforce 算法（策略梯度方法）
    
    ∇J(θ) = E[∇log π_θ(a|s) G_t]
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99, 
                 learning_rate: float = 1e-2):
        """
        初始化REINFORCE智能体
        
        Args:
            env: 环境
            gamma: 折扣因子
            learning_rate: 学习率
        """
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # 创建策略网络
        self.policy_net = PolicyNetwork(1, self.num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=learning_rate)
        
        # 训练统计
        self.episode_rewards = []
    
    def _state_to_tensor(self, state_idx: int) -> torch.Tensor:
        """将状态转换为张量"""
        return torch.tensor([[float(state_idx)]], dtype=torch.float32)
    
    def select_action(self, state_idx: int) -> Tuple[int, float]:
        """
        根据策略选择动作
        
        Returns:
            (action, log_prob)
        """
        state = self._state_to_tensor(state_idx)
        with torch.no_grad():
            probs = self.policy_net(state)
        
        # 采样动作
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def train(self, num_episodes: int = 500, max_steps: int = 100):
        """
        训练REINFORCE智能体
        
        Args:
            num_episodes: 训练轮数
            max_steps: 每轮最大步数
        
        Returns:
            每轮的奖励列表
        """
        print("REINFORCE 开始训练...")
        self.episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            log_probs = []
            rewards = []
            
            # 采集轨迹
            for step in range(max_steps):
                state_idx = state[0] * self.env.grid_size + state[1]
                action, log_prob = self.select_action(state_idx)
                
                next_state, reward, done, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
                if done:
                    break
            
            # 计算回报
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns, dtype=torch.float32)
            # 归一化
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # 计算损失
            log_probs = torch.tensor(log_probs, dtype=torch.float32)
            loss = -(log_probs * returns).sum()
            
            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            episode_reward = sum(rewards)
            self.episode_rewards.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}")
        
        return self.episode_rewards
    
    def test(self, num_episodes: int = 10, max_steps: int = 100):
        """测试策略"""
        print(f"\n测试策略 ({num_episodes}轮)...")
        
        total_reward = 0
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = state[0] * self.env.grid_size + state[1]
                
                state_tensor = self._state_to_tensor(state_idx)
                with torch.no_grad():
                    probs = self.policy_net(state_tensor)
                
                # 贪心选择
                action = torch.argmax(probs).item()
                
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


class ActorCriticAgent:
    """
    Actor-Critic 方法
    
    Actor: 学习策略 π_θ(a|s)
    Critic: 学习价值函数 V_φ(s)
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99,
                 actor_lr: float = 1e-2, critic_lr: float = 1e-2):
        """
        初始化Actor-Critic智能体
        
        Args:
            env: 环境
            gamma: 折扣因子
            actor_lr: Actor学习率
            critic_lr: Critic学习率
        """
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # Actor和Critic网络
        self.actor = PolicyNetwork(1, self.num_actions)
        self.critic = ValueNetwork(1)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                          lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                           lr=critic_lr)
        
        self.episode_rewards = []
    
    def _state_to_tensor(self, state_idx: int) -> torch.Tensor:
        """将状态转换为张量"""
        return torch.tensor([[float(state_idx)]], dtype=torch.float32)
    
    def train(self, num_episodes: int = 500, max_steps: int = 100):
        """
        训练Actor-Critic智能体
        
        Args:
            num_episodes: 训练轮数
            max_steps: 每轮最大步数
        
        Returns:
            每轮的奖励列表
        """
        print("Actor-Critic 开始训练...")
        self.episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = state[0] * self.env.grid_size + state[1]
                state_tensor = self._state_to_tensor(state_idx)
                
                # Actor选择动作
                with torch.no_grad():
                    probs = self.actor(state_tensor)
                
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action.item())
                next_state_idx = next_state[0] * self.env.grid_size + next_state[1]
                next_state_tensor = self._state_to_tensor(next_state_idx)
                
                # Critic估计价值
                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)
                
                if done:
                    next_value = torch.tensor(0.0, dtype=torch.float32)
                
                # TD误差作为优势函数
                td_error = reward + self.gamma * next_value.detach() - value
                advantage = td_error.detach()
                
                # 更新Critic
                critic_loss = (td_error ** 2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # 更新Actor
                actor_loss = -(log_prob * advantage)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}")
        
        return self.episode_rewards
    
    def test(self, num_episodes: int = 10, max_steps: int = 100):
        """测试策略"""
        print(f"\n测试策略 ({num_episodes}轮)...")
        
        total_reward = 0
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                state_idx = state[0] * self.env.grid_size + state[1]
                state_tensor = self._state_to_tensor(state_idx)
                
                with torch.no_grad():
                    probs = self.actor(state_tensor)
                
                action = torch.argmax(probs).item()
                
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


if __name__ == '__main__':
    # 创建环境
    env1 = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    env2 = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    
    print("="*70)
    print("策略梯度方法演示")
    print("="*70)
    
    # REINFORCE
    print("\n【REINFORCE 算法】")
    print("-" * 70)
    agent_reinforce = REINFORCEAgent(env1, gamma=0.99, learning_rate=1e-2)
    rewards_reinforce = agent_reinforce.train(num_episodes=500, max_steps=100)
    agent_reinforce.test()
    
    # Actor-Critic
    print("\n\n【Actor-Critic 方法】")
    print("-" * 70)
    agent_ac = ActorCriticAgent(env2, gamma=0.99, actor_lr=1e-2, critic_lr=1e-2)
    rewards_ac = agent_ac.train(num_episodes=500, max_steps=100)
    agent_ac.test()
    
    # 绘制对比
    plot_learning_curve(rewards_reinforce, window_size=50,
                       title='Policy Gradient Methods',
                       save_path='/tmp/pg_learning_curve.png')
    print("\n学习曲线已保存到 /tmp/pg_learning_curve.png")
