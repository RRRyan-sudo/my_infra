"""
LeetCode 73. 矩阵置零 (Set Matrix Zeroes)
难度: 中等
链接: https://leetcode.cn/problems/set-matrix-zeroes/

题目描述:
    给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。
    请使用原地算法。

示例:
    输入: [[1,1,1],[1,0,1],[1,1,1]]
    输出: [[1,0,1],[0,0,0],[1,0,1]]

思路分析:
    方法1 - 额外空间 O(m+n):
        用两个数组记录哪些行和列需要置零

    方法2 - 原地算法 O(1):
        用矩阵的第一行和第一列来记录标记

        但是第一行和第一列本身需要特殊处理:
            - 先判断第一行/第一列是否本身有0
            - 用两个变量记录

        算法:
            1. 检查第一行/第一列是否有0
            2. 用第一行/第一列记录其他位置的0
            3. 根据标记置零（除了第一行/第一列）
            4. 最后处理第一行/第一列

复杂度分析:
    原地算法:
        时间复杂度: O(m*n)
        空间复杂度: O(1)

面试技巧:
    1. 原地算法的关键是找到"存储空间"
    2. 用矩阵本身的边界作为标记是常见技巧
    3. 注意处理顺序，避免覆盖标记
"""

from typing import List


def setZeroes(matrix: List[List[int]]) -> None:
    """
    原地算法: 用第一行/列作为标记

    空间复杂度 O(1)
    """
    if not matrix:
        return

    m, n = len(matrix), len(matrix[0])

    # 1. 检查第一行和第一列是否本身有0
    first_row_has_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_has_zero = any(matrix[i][0] == 0 for i in range(m))

    # 2. 用第一行/列记录其他位置的0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0  # 标记这一行
                matrix[0][j] = 0  # 标记这一列

    # 3. 根据标记置零（除了第一行/列）
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # 4. 处理第一行
    if first_row_has_zero:
        for j in range(n):
            matrix[0][j] = 0

    # 5. 处理第一列
    if first_col_has_zero:
        for i in range(m):
            matrix[i][0] = 0


def setZeroes_extra_space(matrix: List[List[int]]) -> None:
    """
    使用额外空间 O(m+n) - 更容易理解
    """
    m, n = len(matrix), len(matrix[0])
    rows = set()
    cols = set()

    # 记录哪些行和列需要置零
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                rows.add(i)
                cols.add(j)

    # 置零
    for i in range(m):
        for j in range(n):
            if i in rows or j in cols:
                matrix[i][j] = 0


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {
            "matrix": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            "expected": [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
        },
        {
            "matrix": [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]],
            "expected": [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
        },
    ]

    for i, tc in enumerate(test_cases):
        matrix = [row.copy() for row in tc["matrix"]]
        setZeroes(matrix)
        is_correct = matrix == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status}")
        print(f"       输出: {matrix}")
        print(f"       期望: {tc['expected']}")
