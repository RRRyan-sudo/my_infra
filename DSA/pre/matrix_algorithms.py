"""
矩阵类常见题型示例：
1. 顺时针螺旋遍历（遍历顺序技巧）。
2. 网格最短路（BFS），0 表示可走，1 表示障碍。
"""
from collections import deque
from typing import List, Tuple


def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    顺时针螺旋遍历矩阵，逐层收缩边界。
    """
    if not matrix or not matrix[0]:
        return []

    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    res: List[int] = []

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            res.append(matrix[top][c])
        top += 1

        for r in range(top, bottom + 1):
            res.append(matrix[r][right])
        right -= 1

        if top <= bottom:
            for c in range(right, left - 1, -1):
                res.append(matrix[bottom][c])
            bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                res.append(matrix[r][left])
            left += 1

    return res


def shortest_path_grid(grid: List[List[int]]) -> int:
    """
    在网格中从左上角到右下角的最短步数（4 方向移动）。
    0 表示可走，1 表示障碍；若不可达返回 -1。
    使用 BFS 层层扩展，每一层代表一步。
    """
    if not grid or not grid[0]:
        return -1
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[m - 1][n - 1] == 1:
        return -1

    queue: deque[Tuple[int, int]] = deque([(0, 0)])
    visited = {(0, 0)}
    steps = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            if (r, c) == (m - 1, n - 1):
                return steps
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < m
                    and 0 <= nc < n
                    and grid[nr][nc] == 0
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        steps += 1

    return -1


if __name__ == "__main__":
    mat = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    print("螺旋遍历:", spiral_order(mat))

    grid = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    ]
    print("网格最短步数:", shortest_path_grid(grid))
