"""
模块1: 马尔可夫决策过程 (Markov Decision Process)

理论基础:
- MDP是强化学习的数学框架
- 定义: (S, A, P, R, γ) 
  - S: 状态空间
  - A: 行动空间
  - P(s'|s,a): 转移概率
  - R(s,a): 奖励函数
  - γ: 折扣因子

关键概念:
1. 状态价值 V(s) = E[G_t | S_t = s]
   期望累积折扣奖励
   
2. 动作价值 Q(s,a) = E[G_t | S_t = s, A_t = a]
   在状态s采取动作a的期望回报
   
3. 贝尔曼方程
   V(s) = E[R + γV(s')]
   Q(s,a) = E[R + γE_{a'~π}[Q(s',a')]]
   
4. 最优价值函数
   V*(s) = max_π V^π(s)
   Q*(s,a) = max_π Q^π(s,a)
   
5. 最优策略
   π*(s) = argmax_a Q*(s,a)
"""

import numpy as np
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod


class MDP(ABC):
    """
    MDP的抽象基类
    """
    
    @abstractmethod
    def get_states(self) -> List:
        """获取所有状态"""
        pass
    
    @abstractmethod
    def get_actions(self, state) -> List:
        """获取给定状态下的可行动作"""
        pass
    
    @abstractmethod
    def get_transition_prob(self, state, action, next_state) -> float:
        """获取转移概率 P(s'|s,a)"""
        pass
    
    @abstractmethod
    def get_reward(self, state, action, next_state) -> float:
        """获取奖励 R(s,a,s')"""
        pass


class SimpleMDP(MDP):
    """
    简单的MDP实现示例
    
    状态: 0, 1, 2 (共3个状态)
    动作: 0 (左), 1 (右)
    
    转移:
    - 状态0: 动作0留在0，动作1到1
    - 状态1: 动作0到0，动作1到2
    - 状态2: 动作0到1，动作1留在2
    """
    
    def __init__(self, gamma: float = 0.9):
        """
        初始化MDP
        
        Args:
            gamma: 折扣因子
        """
        self.gamma = gamma
        self.states = [0, 1, 2]
        self.actions = [0, 1]  # 0=左, 1=右
        
        # 定义转移概率矩阵
        # P[s][a][s'] = 概率
        self.P = {
            0: {
                0: {0: 1.0},           # 状态0, 动作左, 留在0
                1: {1: 1.0}            # 状态0, 动作右, 到1
            },
            1: {
                0: {0: 1.0},           # 状态1, 动作左, 到0
                1: {2: 1.0}            # 状态1, 动作右, 到2
            },
            2: {
                0: {1: 1.0},           # 状态2, 动作左, 到1
                1: {2: 1.0}            # 状态2, 动作右, 留在2
            }
        }
        
        # 定义奖励函数
        # R[s][a][s'] = 奖励
        self.R = {
            0: {0: {0: 0}, 1: {1: 1}},
            1: {0: {0: 1}, 1: {2: 10}},
            2: {0: {1: -1}, 1: {2: 0}}
        }
    
    def get_states(self) -> List:
        return self.states
    
    def get_actions(self, state) -> List:
        return self.actions
    
    def get_transition_prob(self, state: int, action: int, 
                           next_state: int) -> float:
        """获取P(s'|s,a)"""
        if state not in self.P or action not in self.P[state]:
            return 0.0
        return self.P[state][action].get(next_state, 0.0)
    
    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """获取R(s,a,s')"""
        if state not in self.R or action not in self.R[state]:
            return 0.0
        return self.R[state][action].get(next_state, 0.0)
    
    def get_all_transitions(self, state: int, action: int) -> Dict[int, float]:
        """获取从(s,a)出发的所有可能转移"""
        if state in self.P and action in self.P[state]:
            return self.P[state][action].copy()
        return {}


class Policy:
    """
    策略类
    """
    
    def __init__(self, mdp: MDP, policy_type: str = 'uniform'):
        """
        初始化策略
        
        Args:
            mdp: MDP实例
            policy_type: 'uniform' (均匀随机) 或 'deterministic' (确定性)
        """
        self.mdp = mdp
        self.policy_type = policy_type
        self.states = mdp.get_states()
        
        # π(a|s) - 在状态s采取动作a的概率
        if policy_type == 'uniform':
            self.policy = self._create_uniform_policy()
        else:
            self.policy = {}  # 确定性策略: 状态->动作
    
    def _create_uniform_policy(self) -> Dict:
        """创建均匀随机策略"""
        policy = {}
        for state in self.states:
            actions = self.mdp.get_actions(state)
            num_actions = len(actions)
            policy[state] = {a: 1.0/num_actions for a in actions}
        return policy
    
    def get_action_prob(self, state: int, action: int) -> float:
        """获取π(a|s)"""
        return self.policy[state].get(action, 0.0)
    
    def sample_action(self, state: int) -> int:
        """根据策略采样动作"""
        if self.policy_type == 'uniform':
            actions = list(self.policy[state].keys())
            probs = [self.policy[state][a] for a in actions]
            return np.random.choice(actions, p=probs)
        else:
            return self.policy[state]
    
    def set_action(self, state: int, action: int):
        """设置确定性策略"""
        self.policy[state] = action
    
    def set_action_prob(self, state: int, action: int, prob: float):
        """设置动作概率"""
        if state not in self.policy:
            self.policy[state] = {}
        self.policy[state][action] = prob


def evaluate_policy_iteratively(mdp: MDP, policy: Policy, gamma: float = 0.9,
                               epsilon: float = 1e-6, max_iterations: int = 1000) -> Dict:
    """
    策略评估 - 迭代方法
    
    计算给定策略的价值函数
    
    Args:
        mdp: MDP实例
        policy: 策略
        gamma: 折扣因子
        epsilon: 收敛条件
        max_iterations: 最大迭代次数
    
    Returns:
        V: 状态价值函数字典
    """
    states = mdp.get_states()
    V = {s: 0.0 for s in states}
    
    for iteration in range(max_iterations):
        V_old = V.copy()
        
        for state in states:
            value = 0.0
            actions = mdp.get_actions(state)
            
            for action in actions:
                # π(a|s)
                prob_action = policy.get_action_prob(state, action)
                
                # E[R + γV(s')]
                expected_value = 0.0
                transitions = mdp.get_all_transitions(state, action)
                
                for next_state, prob_transition in transitions.items():
                    reward = mdp.get_reward(state, action, next_state)
                    expected_value += prob_transition * (reward + gamma * V_old[next_state])
                
                value += prob_action * expected_value
            
            V[state] = value
        
        # 检查收敛
        delta = max(abs(V[s] - V_old[s]) for s in states)
        if delta < epsilon:
            print(f"策略评估收敛 (迭代{iteration+1}次)")
            break
    
    return V


def compute_q_values(mdp: MDP, V: Dict, gamma: float = 0.9) -> Dict:
    """
    从价值函数计算Q值
    
    Q(s,a) = E[R + γV(s')]
    
    Args:
        mdp: MDP实例
        V: 状态价值函数
        gamma: 折扣因子
    
    Returns:
        Q: 动作价值函数嵌套字典 Q[s][a]
    """
    states = mdp.get_states()
    Q = {s: {} for s in states}
    
    for state in states:
        for action in mdp.get_actions(state):
            q_value = 0.0
            transitions = mdp.get_all_transitions(state, action)
            
            for next_state, prob in transitions.items():
                reward = mdp.get_reward(state, action, next_state)
                q_value += prob * (reward + gamma * V[next_state])
            
            Q[state][action] = q_value
    
    return Q


if __name__ == '__main__':
    # 示例: 评估简单MDP上的随机策略
    print("="*60)
    print("MDP基础演示")
    print("="*60)
    
    # 创建MDP
    mdp = SimpleMDP(gamma=0.9)
    
    # 创建均匀随机策略
    policy = Policy(mdp, policy_type='uniform')
    
    # 评估策略
    print("\n1. 评估均匀随机策略:")
    V_random = evaluate_policy_iteratively(mdp, policy, gamma=0.9)
    
    print("状态价值函数V(s):")
    for state in mdp.get_states():
        print(f"  V({state}) = {V_random[state]:.4f}")
    
    # 计算Q值
    print("\n2. 计算Q值:")
    Q_random = compute_q_values(mdp, V_random)
    for state in mdp.get_states():
        print(f"  状态{state}:")
        for action in mdp.get_actions(state):
            print(f"    Q({state},{action}) = {Q_random[state][action]:.4f}")
    
    # 提取贪心策略
    print("\n3. 贪心策略 (π_greedy):")
    greedy_policy = Policy(mdp, policy_type='deterministic')
    for state in mdp.get_states():
        best_action = max(mdp.get_actions(state), 
                         key=lambda a: Q_random[state][a])
        greedy_policy.set_action(state, best_action)
        print(f"  π({state}) = {best_action}")
