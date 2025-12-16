"""
模块6: 软演员评论家 (Soft Actor-Critic, SAC)

理论基础:
- SAC是一种最大熵强化学习算法
- 结合Actor-Critic框架和最大熵原理
- 目标: 最大化期望奖励 + 策略熵

关键思想:
1. 最大熵RL目标
   J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
   其中 H(π) 是策略的熵，α 是温度参数
   
2. 为什么要最大化熵?
   - 鼓励探索
   - 学习多样化的策略
   - 避免陷入局部最优
   - 提高对环境变化的鲁棒性

3. 三个关键网络:
   - Actor π(a|s;θ) : 学习策略
   - Critic Q(s,a;φ) : 学习Q值
   - Target Critic Q'(s,a;φ') : 稳定学习（复制并延迟更新）

4. 核心更新规则:

   值函数更新:
   J_Q(φ) = E[(Q(s,a;φ) - (r + γ(1-d) * V(s';φ')))²]
   其中 V(s') = E_a'[Q(s',a';φ') - α log π(a'|s';θ)]
   
   策略更新:
   J_π(θ) = E[α log π(a|s;θ) - Q(s,a;φ)]
   
   熵系数自适应:
   J_α = -E[α(log π(a|s;θ) + H_target)]

核心特点:
✓ 支持连续和离散动作
✓ 样本高效
✓ 稳定性好
✓ 无需目标策略网络，用目标Q网络替代
✓ 自适应熵系数

应用场景:
- 机器人控制
- 自动驾驶
- 游戏AI
- 任何连续控制问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld
from utils.helpers import plot_learning_curve


class Actor(nn.Module):
    """
    演员网络：学习随机策略 π(a|s)
    
    输出策略参数（均值和标准差）
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        # 特征提取网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出: 均值
        self.mu = nn.Linear(hidden_dim, action_dim)
        
        # 输出: 对数标准差
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 动作范围限制
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mean: 策略均值
            log_std: 对数标准差
        """
        x = self.net(state)
        mean = self.mu(x)
        log_std = self.log_std(x)
        
        # 限制log_std范围以防止崩溃
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从策略采样动作
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 重参数化技巧: a = tanh(μ + σ*ε)
        eps = torch.randn_like(std)
        z = mean + std * eps
        action = torch.tanh(z)
        
        # 计算对数概率（考虑tanh变换的雅可比）
        log_prob = self._compute_log_prob(mean, log_std, z, action)
        
        return action, log_prob
    
    def _compute_log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, 
                         z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率
        
        log π(a|s) = log π(z|s) - Σ log(1 - a²)
        """
        # 高斯分布的对数概率
        normal_dist = torch.distributions.Normal(mean, torch.exp(log_std))
        log_prob_z = normal_dist.log_prob(z)
        
        # tanh变换的雅可比修正
        log_det_jacobian = torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob_z - log_det_jacobian.sum(dim=-1, keepdim=True)
        
        return log_prob


class Critic(nn.Module):
    """
    评论家网络：学习Q值函数 Q(s,a)
    
    双Q网络用于稳定性
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # 第一个Q网络
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 第二个Q网络（用于减少高估）
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算Q值
        
        Returns:
            q1: 第一个Q网络的输出
            q2: 第二个Q网络的输出
        """
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2
    
    def compute_min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """取两个Q值的最小值（减少高估）"""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样"""
        indices = np.random.randint(len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic 智能体
    
    最大熵强化学习算法实现
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, learning_rate: float = 3e-4):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            gamma: 折扣因子
            tau: 目标网络的软更新系数
            alpha: 熵系数（控制探索和利用的平衡）
            learning_rate: 学习率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # 目标网络初始化
        self._hard_update(self.critic_target, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 自适应熵系数
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -action_dim  # 目标熵 = -动作维度
        
        # 经验回放
        self.memory = ReplayBuffer()
        
        # 训练统计
        self.episode_rewards = []
    
    def _hard_update(self, target_net, source_net):
        """硬拷贝参数"""
        target_net.load_state_dict(source_net.state_dict())
    
    def _soft_update(self, target_net, source_net):
        """软更新目标网络参数"""
        for target_param, source_param in zip(target_net.parameters(), 
                                              source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> float:
        """
        选择动作
        
        Args:
            state: 当前状态
            eval_mode: 是否为评估模式（确定性动作）
        
        Returns:
            动作
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            
            if eval_mode:
                # 评估模式：使用均值
                action = torch.tanh(mean)
            else:
                # 训练模式：采样
                std = torch.exp(log_std)
                eps = torch.randn_like(std)
                z = mean + std * eps
                action = torch.tanh(z)
        
        return action.squeeze().item()
    
    def update(self, batch_size: int = 256):
        """
        更新神经网络
        
        Args:
            batch_size: 批大小
        
        Returns:
            损失字典
        """
        if len(self.memory) < batch_size:
            return {}
        
        # 采样一个批次
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1) if len(np.array(actions).shape) == 1 else torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)
        
        # ===== 更新评论家 =====
        with torch.no_grad():
            # 从下一状态采样动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 计算目标Q值
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # 目标值 = r + γ(1-d)(Q(s',a') - α log π(a'|s'))
            target_q = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * next_log_probs)
        
        # 计算评论家损失
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        
        # 更新评论家
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== 更新演员 =====
        # 从当前状态采样动作
        sampled_actions, sampled_log_probs = self.actor.sample(states)
        
        # 计算演员损失
        q1_a, q2_a = self.critic(states, sampled_actions)
        min_q_a = torch.min(q1_a, q2_a)
        
        # 演员损失 = E[α log π(a|s) - Q(s,a)]
        actor_loss = (self.alpha * sampled_log_probs - min_q_a).mean()
        
        # 更新演员
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== 更新熵系数 =====
        # 损失 = E[-α(log π(a|s) + H_target)]
        alpha_loss = -(self.log_alpha * (sampled_log_probs + self.target_entropy).detach()).mean()
        
        # 更新α
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 更新α值
        self.alpha = torch.exp(self.log_alpha).item()
        
        # ===== 软更新目标网络 =====
        self._soft_update(self.critic_target, self.critic)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha
        }
    
    def train_episode(self, env: GridWorld, max_steps: int = 100) -> float:
        """
        训练一个episode
        
        Returns:
            episode_reward: 该episode的总奖励
        """
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            if isinstance(state, tuple):
                state_idx = state[0] * env.grid_size + state[1]
            else:
                state_idx = int(state)
            
            action = self.select_action(np.array([state_idx], dtype=np.float32))
            action = int(np.clip(action, 0, env.num_actions - 1))
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            if isinstance(next_state, tuple):
                next_state_idx = next_state[0] * env.grid_size + next_state[1]
            else:
                next_state_idx = int(next_state)
            
            self.memory.add(
                state=np.array([state_idx], dtype=np.float32),
                action=action,
                reward=reward,
                next_state=np.array([next_state_idx], dtype=np.float32),
                done=done
            )
            
            # 更新网络
            self.update(batch_size=64)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        return episode_reward
    
    def train(self, env: GridWorld, num_episodes: int = 500, max_steps: int = 100):
        """
        训练SAC智能体
        
        Args:
            env: 环境
            num_episodes: 训练episode数
            max_steps: 每个episode的最大步数
        """
        print("开始SAC训练...")
        print(f"目标熵: {self.target_entropy:.4f}, 初始α: {self.alpha:.4f}")
        
        for episode in range(num_episodes):
            episode_reward = self.train_episode(env, max_steps)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}, α = {self.alpha:.4f}")
    
    def test(self, env: GridWorld, num_episodes: int = 10, max_steps: int = 100):
        """
        测试学到的策略
        
        Args:
            env: 环境
            num_episodes: 测试episode数
            max_steps: 每个episode的最大步数
        """
        print(f"\n测试策略 ({num_episodes}轮)...")
        
        total_reward = 0
        success_count = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                if isinstance(state, tuple):
                    state_idx = state[0] * env.grid_size + state[1]
                else:
                    state_idx = int(state)
                
                # 评估模式：使用均值
                action = self.select_action(np.array([state_idx], dtype=np.float32), eval_mode=True)
                action = int(np.clip(action, 0, env.num_actions - 1))
                
                next_state, reward, done, _ = env.step(action)
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
    print("="*70)
    print("Soft Actor-Critic (SAC) 算法演示")
    print("="*70)
    
    # 创建环境
    env = GridWorld(grid_size=4, num_obstacles=1, seed=42)
    
    # 创建SAC智能体
    agent = SACAgent(
        state_dim=1,
        action_dim=env.num_actions,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        learning_rate=3e-4
    )
    
    # 训练
    agent.train(env, num_episodes=500, max_steps=100)
    
    # 测试
    agent.test(env, num_episodes=10)
    
    # 绘制学习曲线
    plot_learning_curve(agent.episode_rewards, window_size=50,
                       title='SAC Learning Curve',
                       save_path='/tmp/sac_learning_curve.png')
    print("\n学习曲线已保存到 /tmp/sac_learning_curve.png")
