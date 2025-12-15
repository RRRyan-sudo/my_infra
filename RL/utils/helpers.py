"""
强化学习工具函数和辅助工具
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机抽样"""
        indices = np.random.randint(len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return np.array(states), np.array(actions), np.array(rewards), \
               np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)


def plot_learning_curve(rewards: List[float], window_size: int = 100, 
                       title: str = 'Learning Curve', save_path: str = None):
    """
    绘制学习曲线
    
    Args:
        rewards: 每个episode的奖励列表
        window_size: 移动平均窗口大小
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始奖励
    ax.plot(rewards, alpha=0.3, label='Raw', color='blue')
    
    # 绘制移动平均
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, 
                                 mode='valid')
        ax.plot(range(window_size-1, len(rewards)), moving_avg, 
               label=f'Moving Average ({window_size})', color='red', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.close()


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, 
                lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    
    Args:
        rewards: 奖励序列
        values: 值估计序列
        gamma: 折扣因子
        lam: GAE参数
    
    Returns:
        advantages: 优势估计
        returns: 回报估计
    """
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        td_error = rewards[t] + gamma * next_value - values[t]
        gae = td_error + gamma * lam * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def epsilon_greedy(q_values: np.ndarray, epsilon: float = 0.1) -> int:
    """
    ε-贪心策略
    
    Args:
        q_values: 动作价值
        epsilon: 探索概率
    
    Returns:
        选择的动作
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    else:
        return np.argmax(q_values)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax函数
    
    Args:
        x: 输入数组
        temperature: 温度参数
    
    Returns:
        概率分布
    """
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()
