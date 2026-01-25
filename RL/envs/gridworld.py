"""
GridWorld - 网格世界环境

一个简单的离散状态/动作环境，用于学习RL基础概念。

环境描述：
    - 4x4的网格世界
    - Agent从起点(0,0)出发，目标是到达终点(3,3)
    - 每一步获得-1的奖励（鼓励尽快到达）
    - 到达终点获得+10的奖励

状态空间：
    - 离散：16个状态（4x4网格）
    - 状态编号：0-15，对应位置 (row, col) = (state // 4, state % 4)

动作空间：
    - 离散：4个动作
    - 0=上, 1=下, 2=左, 3=右

使用示例：
    >>> env = GridWorld()
    >>> state = env.reset()
    >>> next_state, reward, done, info = env.step(1)  # 向下移动
    >>> env.render()
"""

import numpy as np
from typing import Tuple, Dict, Optional


class GridWorld:
    """
    简单网格世界环境

    用于演示MDP的基本概念：状态、动作、奖励、转移。
    """

    # 动作定义
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # 动作名称（用于显示）
    ACTION_NAMES = ['↑', '↓', '←', '→']

    def __init__(self, size: int = 4, goal_reward: float = 10.0, step_reward: float = -1.0):
        """
        初始化网格世界

        Args:
            size: 网格大小（默认4x4）
            goal_reward: 到达终点的奖励
            step_reward: 每一步的奖励（通常为负，鼓励快速完成）
        """
        self.size = size
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        # 状态空间和动作空间大小
        self.n_states = size * size
        self.n_actions = 4

        # 起点和终点
        self.start_state = 0  # 左上角 (0, 0)
        self.goal_state = self.n_states - 1  # 右下角 (size-1, size-1)

        # 当前状态
        self.state = self.start_state

        # 构建转移概率矩阵（用于动态规划）
        self.P = self._build_transition_matrix()

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态编号转换为位置坐标"""
        return state // self.size, state % self.size

    def _pos_to_state(self, row: int, col: int) -> int:
        """位置坐标转换为状态编号"""
        return row * self.size + col

    def _build_transition_matrix(self) -> Dict:
        """
        构建转移概率矩阵 P[s][a] = [(prob, next_state, reward, done), ...]

        这是MDP的核心：给定状态s和动作a，返回所有可能的(下一状态, 奖励, 终止)

        在确定性环境中，每个(s,a)只有一个可能的结果，prob=1.0
        """
        P = {}

        for state in range(self.n_states):
            P[state] = {}
            row, col = self._state_to_pos(state)

            for action in range(self.n_actions):
                # 计算下一个位置
                if action == self.UP:
                    next_row, next_col = max(row - 1, 0), col
                elif action == self.DOWN:
                    next_row, next_col = min(row + 1, self.size - 1), col
                elif action == self.LEFT:
                    next_row, next_col = row, max(col - 1, 0)
                else:  # RIGHT
                    next_row, next_col = row, min(col + 1, self.size - 1)

                next_state = self._pos_to_state(next_row, next_col)

                # 判断是否到达终点
                done = (next_state == self.goal_state)
                reward = self.goal_reward if done else self.step_reward

                # 确定性环境：转移概率为1.0
                P[state][action] = [(1.0, next_state, reward, done)]

        return P

    def reset(self) -> int:
        """
        重置环境

        Returns:
            初始状态
        """
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作（0=上, 1=下, 2=左, 3=右）

        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否终止
            info: 额外信息
        """
        # 从转移矩阵获取结果
        transitions = self.P[self.state][action]
        prob, next_state, reward, done = transitions[0]  # 确定性环境只有一个结果

        self.state = next_state
        return next_state, reward, done, {}

    def render(self, policy: Optional[np.ndarray] = None, values: Optional[np.ndarray] = None):
        """
        渲染当前环境状态

        Args:
            policy: 可选，显示策略（每个状态的最优动作）
            values: 可选，显示价值函数
        """
        print("\n" + "=" * (self.size * 8 + 1))

        for row in range(self.size):
            # 上边框
            print("+" + "-------+" * self.size)

            # 内容行
            line = "|"
            for col in range(self.size):
                state = self._pos_to_state(row, col)
                cell = ""

                if state == self.state:
                    cell = "  A  "  # Agent位置
                elif state == self.goal_state:
                    cell = "  G  "  # 目标
                elif state == self.start_state:
                    cell = "  S  "  # 起点
                elif policy is not None:
                    cell = f"  {self.ACTION_NAMES[policy[state]]}  "
                elif values is not None:
                    cell = f"{values[state]:5.1f}"
                else:
                    cell = "     "

                line += f" {cell} |"

            print(line)

        # 下边框
        print("+" + "-------+" * self.size)
        print("=" * (self.size * 8 + 1))

        # 图例
        print("图例: A=Agent, S=起点, G=目标")
        if policy is not None:
            print("箭头表示策略（最优动作方向）")
        if values is not None:
            print("数字表示状态价值V(s)")

    def get_optimal_policy_and_value(self, gamma: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用价值迭代计算最优策略和价值函数（用于验证算法正确性）

        Args:
            gamma: 折扣因子

        Returns:
            policy: 最优策略
            values: 最优价值函数
        """
        values = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)

        # 价值迭代
        for _ in range(1000):
            new_values = np.zeros(self.n_states)
            for state in range(self.n_states):
                if state == self.goal_state:
                    new_values[state] = 0
                    continue

                q_values = []
                for action in range(self.n_actions):
                    prob, next_state, reward, done = self.P[state][action][0]
                    q = reward + gamma * values[next_state] * (1 - done)
                    q_values.append(q)

                new_values[state] = max(q_values)
                policy[state] = np.argmax(q_values)

            if np.max(np.abs(new_values - values)) < 1e-6:
                break
            values = new_values

        return policy, values


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("GridWorld 网格世界环境演示")
    print("=" * 60)

    # 创建环境
    env = GridWorld(size=4)
    print(f"\n环境信息：")
    print(f"  - 网格大小: {env.size}x{env.size}")
    print(f"  - 状态数量: {env.n_states}")
    print(f"  - 动作数量: {env.n_actions}")
    print(f"  - 起点: {env.start_state}")
    print(f"  - 终点: {env.goal_state}")

    # 显示初始状态
    print("\n1. 初始状态：")
    state = env.reset()
    env.render()

    # 执行几步随机动作
    print("\n2. 执行随机动作：")
    for i in range(5):
        action = np.random.randint(env.n_actions)
        next_state, reward, done, _ = env.step(action)
        print(f"   步骤{i+1}: 动作={env.ACTION_NAMES[action]}, "
              f"状态={state}->{next_state}, 奖励={reward:.1f}, 终止={done}")
        state = next_state
        if done:
            print("   到达目标！")
            break

    # 计算并显示最优策略
    print("\n3. 最优策略（使用价值迭代计算）：")
    env.reset()
    policy, values = env.get_optimal_policy_and_value()
    env.render(policy=policy)

    print("\n4. 最优价值函数：")
    env.render(values=values)

    # 使用最优策略走一遍
    print("\n5. 使用最优策略从起点到终点：")
    state = env.reset()
    trajectory = [state]
    total_reward = 0

    for step in range(20):
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        trajectory.append(next_state)

        if done:
            break
        state = next_state

    print(f"   轨迹: {' -> '.join(map(str, trajectory))}")
    print(f"   总奖励: {total_reward:.1f}")
    print(f"   步数: {len(trajectory) - 1}")
