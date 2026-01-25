"""
LeetCode 200. 岛屿数量 (Number of Islands)
难度: 中等
链接: https://leetcode.cn/problems/number-of-islands/

题目描述:
    给你一个由 '1'（陆地）和 '0'（水）组成的二维网格，请你计算网格中岛屿的数量。
    岛屿被水包围，通过水平或垂直方向相邻的陆地连接形成。

示例:
    输入: grid = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"],
      ["0","0","0","0","0"]
    ]
    输出: 1

思路分析:
    DFS/BFS + 标记访问:
        遍历每个格子:
            - 如果是陆地且未访问，岛屿数+1
            - 从该格子开始DFS/BFS，标记所有相连的陆地

    为什么这样做是对的?
        - 每个陆地只被访问一次
        - 每次从未访问的陆地开始，就是发现了一个新岛屿

    标记方法:
        - 可以用额外的visited数组
        - 也可以直接修改grid，将'1'改为'0'（节省空间）

复杂度分析:
    时间复杂度: O(m*n)
    空间复杂度: O(m*n) - DFS递归栈最坏情况

面试技巧:
    1. 图的遍历基础题
    2. DFS和BFS都要会
    3. 类似题目: 岛屿的最大面积、被围绕的区域
"""

from typing import List
from collections import deque


def numIslands(grid: List[List[str]]) -> int:
    """
    DFS解法

    遇到陆地就DFS标记所有相连的陆地
    """
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        # 边界检查和条件检查
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return

        # 标记为已访问（修改为'0'）
        grid[i][j] = '0'

        # 访问四个方向
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)

    return count


def numIslands_bfs(grid: List[List[str]]) -> int:
    """BFS解法"""
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    count = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def bfs(start_i, start_j):
        queue = deque([(start_i, start_j)])
        grid[start_i][start_j] = '0'

        while queue:
            i, j = queue.popleft()
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                    grid[ni][nj] = '0'
                    queue.append((ni, nj))

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                bfs(i, j)

    return count


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {
            "grid": [
                ["1", "1", "1", "1", "0"],
                ["1", "1", "0", "1", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "0", "0", "0"]
            ],
            "expected": 1
        },
        {
            "grid": [
                ["1", "1", "0", "0", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "1", "0", "0"],
                ["0", "0", "0", "1", "1"]
            ],
            "expected": 3
        },
    ]

    for i, tc in enumerate(test_cases):
        grid = [row.copy() for row in tc["grid"]]
        result = numIslands(grid)
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} 岛屿数量: {result}, 期望: {tc['expected']}")
