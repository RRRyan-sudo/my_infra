"""
简单的网格世界环境
用于强化学习算法的测试和学习
"""
import numpy as np
from typing import Tuple, Dict, Any


class GridWorld:
    """
    简单的网格世界环境
    
    - 代理可以上、下、左、右移动
    - 目标是到达目标位置
    - 可选择有障碍物
    """
    
    def __init__(self, grid_size: int = 4, num_obstacles: int = 0, seed: int = None):
        """
        初始化网格世界
        
        Args:
            grid_size: 网格大小 (grid_size x grid_size)
            num_obstacles: 障碍物数量
            seed: 随机数种子
        """
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.seed(seed)
        
        # 状态空间: 所有可能的位置
        self.state_space = [(i, j) for i in range(grid_size) 
                           for j in range(grid_size)]
        
        # 行动空间: 0=上, 1=下, 2=左, 3=右
        self.action_space = [0, 1, 2, 3]
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        
        # 初始化环境
        self.reset()
    
    def seed(self, seed=None):
        """设置随机数种子"""
        self.np_random = np.random.RandomState(seed)
    
    def reset(self):
        """重置环境"""
        # 随机选择起始位置
        self.agent_pos = tuple(self.np_random.randint(0, self.grid_size, 2))
        
        # 随机选择目标位置（与起始位置不同）
        while True:
            self.goal_pos = tuple(self.np_random.randint(0, self.grid_size, 2))
            if self.goal_pos != self.agent_pos:
                break
        
        # 生成障碍物
        self.obstacles = set()
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 1000:
            obs_pos = tuple(self.np_random.randint(0, self.grid_size, 2))
            if obs_pos != self.agent_pos and obs_pos != self.goal_pos:
                self.obstacles.add(obs_pos)
            attempts += 1
        
        return self._get_state()
    
    def _get_state(self) -> Tuple[int, int]:
        """获取当前状态"""
        return self.agent_pos
    
    def _state_to_index(self, state: Tuple[int, int]) -> int:
        """将状态转换为索引"""
        return state[0] * self.grid_size + state[1]
    
    def _index_to_state(self, index: int) -> Tuple[int, int]:
        """将索引转换为状态"""
        return (index // self.grid_size, index % self.grid_size)
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 行动 (0=上, 1=下, 2=左, 3=右)
        
        Returns:
            state: 新状态
            reward: 奖励
            done: 是否终止
            info: 额外信息
        """
        # 计算新位置
        row, col = self.agent_pos
        
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 下
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # 左
            col = max(0, col - 1)
        elif action == 3:  # 右
            col = min(self.grid_size - 1, col + 1)
        
        new_pos = (row, col)
        
        # 检查是否碰到障碍物
        if new_pos in self.obstacles:
            # 不移动
            new_pos = self.agent_pos
        
        self.agent_pos = new_pos
        
        # 计算奖励
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # 每步的成本
        
        return self._get_state(), reward, done, {}
    
    def render(self, mode: str = 'human') -> str:
        """
        渲染环境
        
        Args:
            mode: 渲染模式 ('human' 或 'rgb_array')
        
        Returns:
            表示环境的字符串
        """
        grid = [['.' for _ in range(self.grid_size)] 
                for _ in range(self.grid_size)]
        
        # 标记障碍物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = '#'
        
        # 标记目标
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # 标记代理
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        result = '\n'.join([''.join(row) for row in grid])
        
        if mode == 'human':
            print(result)
        
        return result
    
    @property
    def num_states(self) -> int:
        """状态空间大小"""
        return self.grid_size * self.grid_size
    
    @property
    def num_actions(self) -> int:
        """行动空间大小"""
        return len(self.action_space)
