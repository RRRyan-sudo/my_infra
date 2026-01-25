"""
SimpleContinuousEnv - 简单连续控制环境

一个简单的连续状态/动作环境，用于学习连续控制算法（PPO连续版、DDPG、TD3、SAC）。

环境描述：
    - 一个点质量在2D平面上移动
    - 目标是将点移动到原点(0, 0)
    - 状态是位置(x, y)和速度(vx, vy)
    - 动作是施加的力(fx, fy)

状态空间：
    - 连续：4维向量 [x, y, vx, vy]
    - x, y ∈ [-10, 10]
    - vx, vy ∈ [-5, 5]

动作空间：
    - 连续：2维向量 [fx, fy]
    - fx, fy ∈ [-1, 1]

奖励设计：
    - 距离惩罚: -distance_to_goal
    - 速度惩罚: -0.1 * speed（防止振荡）
    - 动作惩罚: -0.01 * action_magnitude（节约能量）

使用示例：
    >>> env = SimpleContinuousEnv()
    >>> state = env.reset()
    >>> action = np.array([0.5, -0.3])  # 连续动作
    >>> next_state, reward, done, info = env.step(action)
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SimpleContinuousEnv:
    """
    简单连续控制环境

    物理模型：点质量在2D平面上运动
    动力学方程：
        x_{t+1} = x_t + v_t * dt
        v_{t+1} = v_t + a_t * dt
        a_t = F_t / m  （质量m=1）

    这个环境帮助理解：
    1. 连续状态空间：状态是实数向量
    2. 连续动作空间：动作是实数向量
    3. 奖励塑形：如何设计奖励引导学习
    """

    def __init__(
        self,
        max_position: float = 10.0,
        max_velocity: float = 5.0,
        max_force: float = 1.0,
        dt: float = 0.1,
        max_steps: int = 200
    ):
        """
        初始化环境

        Args:
            max_position: 位置范围 [-max_position, max_position]
            max_velocity: 速度范围 [-max_velocity, max_velocity]
            max_force: 最大力 [-max_force, max_force]
            dt: 时间步长
            max_steps: 每个episode的最大步数
        """
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_force = max_force
        self.dt = dt
        self.max_steps = max_steps

        # 状态空间维度：[x, y, vx, vy]
        self.state_dim = 4
        # 动作空间维度：[fx, fy]
        self.action_dim = 2

        # 目标位置
        self.goal = np.array([0.0, 0.0])

        # 成功阈值（距离小于此值视为成功）
        self.success_threshold = 0.5

        # 当前状态和步数
        self.state = None
        self.steps = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置环境

        Args:
            seed: 随机种子（可选）

        Returns:
            初始状态 [x, y, vx, vy]
        """
        if seed is not None:
            np.random.seed(seed)

        # 随机初始位置（远离目标）
        position = np.random.uniform(-self.max_position * 0.8, self.max_position * 0.8, size=2)

        # 确保初始位置不在目标附近
        while np.linalg.norm(position - self.goal) < 2.0:
            position = np.random.uniform(-self.max_position * 0.8, self.max_position * 0.8, size=2)

        # 初始速度为0或小随机值
        velocity = np.random.uniform(-0.5, 0.5, size=2)

        self.state = np.concatenate([position, velocity])
        self.steps = 0

        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 连续动作 [fx, fy]，范围 [-1, 1]

        Returns:
            next_state: 下一个状态 [x, y, vx, vy]
            reward: 奖励
            done: 是否终止
            info: 额外信息
        """
        # 确保动作在有效范围内
        action = np.clip(action, -self.max_force, self.max_force)

        # 解析当前状态
        position = self.state[:2]
        velocity = self.state[2:]

        # 物理更新：简单欧拉积分
        # 加速度 = 力 / 质量（质量=1）
        acceleration = action

        # 更新速度
        new_velocity = velocity + acceleration * self.dt
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        # 更新位置
        new_position = position + new_velocity * self.dt
        new_position = np.clip(new_position, -self.max_position, self.max_position)

        # 如果撞墙，速度反向（简单的弹性碰撞）
        for i in range(2):
            if abs(new_position[i]) >= self.max_position:
                new_velocity[i] *= -0.5  # 碰撞损失一半能量

        # 更新状态
        self.state = np.concatenate([new_position, new_velocity])
        self.steps += 1

        # 计算奖励
        distance = np.linalg.norm(new_position - self.goal)
        speed = np.linalg.norm(new_velocity)
        action_mag = np.linalg.norm(action)

        # 奖励设计：
        # 1. 距离惩罚：越近越好
        # 2. 速度惩罚：防止振荡
        # 3. 动作惩罚：节约能量
        reward = -distance - 0.1 * speed - 0.01 * action_mag

        # 成功奖励
        success = distance < self.success_threshold
        if success:
            reward += 10.0

        # 终止条件
        done = success or (self.steps >= self.max_steps)

        # 额外信息
        info = {
            'distance': distance,
            'speed': speed,
            'success': success,
            'steps': self.steps
        }

        return self.state.copy(), reward, done, info

    def render(self, mode: str = 'text'):
        """
        渲染环境状态

        Args:
            mode: 渲染模式 ('text' 或 'array')
        """
        if self.state is None:
            print("环境未初始化，请先调用 reset()")
            return

        position = self.state[:2]
        velocity = self.state[2:]
        distance = np.linalg.norm(position - self.goal)

        if mode == 'text':
            print(f"位置: ({position[0]:6.2f}, {position[1]:6.2f})")
            print(f"速度: ({velocity[0]:6.2f}, {velocity[1]:6.2f})")
            print(f"距离目标: {distance:.2f}")
            print(f"步数: {self.steps}/{self.max_steps}")

        elif mode == 'array':
            # 简单的ASCII可视化
            grid_size = 21
            grid = np.full((grid_size, grid_size), '.')

            # 目标位置
            goal_x = grid_size // 2
            goal_y = grid_size // 2
            grid[goal_y, goal_x] = 'G'

            # Agent位置（缩放到grid）
            agent_x = int((position[0] / self.max_position + 1) * (grid_size - 1) / 2)
            agent_y = int((position[1] / self.max_position + 1) * (grid_size - 1) / 2)
            agent_x = np.clip(agent_x, 0, grid_size - 1)
            agent_y = np.clip(agent_y, 0, grid_size - 1)
            grid[grid_size - 1 - agent_y, agent_x] = 'A'  # 注意y轴反转

            # 打印grid
            print("+" + "-" * grid_size + "+")
            for row in grid:
                print("|" + "".join(row) + "|")
            print("+" + "-" * grid_size + "+")
            print(f"A=Agent, G=Goal, 距离={distance:.2f}")

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """状态空间形状"""
        return (self.state_dim,)

    @property
    def action_space_shape(self) -> Tuple[int]:
        """动作空间形状"""
        return (self.action_dim,)


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("SimpleContinuousEnv 连续控制环境演示")
    print("=" * 60)

    # 创建环境
    env = SimpleContinuousEnv()
    print(f"\n环境信息：")
    print(f"  - 状态维度: {env.state_dim} (x, y, vx, vy)")
    print(f"  - 动作维度: {env.action_dim} (fx, fy)")
    print(f"  - 位置范围: [-{env.max_position}, {env.max_position}]")
    print(f"  - 速度范围: [-{env.max_velocity}, {env.max_velocity}]")
    print(f"  - 力范围: [-{env.max_force}, {env.max_force}]")

    # 重置环境
    print("\n1. 初始状态：")
    state = env.reset(seed=42)
    env.render(mode='array')

    # 简单的比例控制策略（演示用）
    print("\n2. 使用简单比例控制（P控制）：")

    def simple_controller(state):
        """简单的P控制器：朝目标方向施力"""
        position = state[:2]
        velocity = state[2:]

        # P控制：力与位置误差成正比
        # D控制：加入速度阻尼
        kp = 0.3  # 位置增益
        kd = 0.5  # 速度阻尼

        force = -kp * position - kd * velocity
        return np.clip(force, -1, 1)

    # 运行一个episode
    state = env.reset(seed=42)
    total_reward = 0
    trajectory = [state[:2].copy()]

    for step in range(env.max_steps):
        action = simple_controller(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        trajectory.append(next_state[:2].copy())

        if step % 20 == 0:
            print(f"   步骤 {step:3d}: 距离={info['distance']:.2f}, 奖励={reward:.2f}")

        if done:
            print(f"\n   完成！成功={info['success']}, 总奖励={total_reward:.2f}, 步数={step+1}")
            break

        state = next_state

    # 显示最终状态
    print("\n3. 最终状态：")
    env.render(mode='array')

    # 对比随机策略
    print("\n4. 对比随机策略：")
    state = env.reset(seed=42)
    total_reward_random = 0

    for step in range(env.max_steps):
        action = np.random.uniform(-1, 1, size=2)  # 随机动作
        next_state, reward, done, info = env.step(action)
        total_reward_random += reward

        if done:
            break
        state = next_state

    print(f"   随机策略: 总奖励={total_reward_random:.2f}, 成功={info['success']}")
    print(f"   P控制器:  总奖励={total_reward:.2f}")
    print(f"   改进: {(total_reward - total_reward_random):.2f}")
