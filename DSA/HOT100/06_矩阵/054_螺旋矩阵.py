"""
LeetCode 54. 螺旋矩阵 (Spiral Matrix)
难度: 中等
链接: https://leetcode.cn/problems/spiral-matrix/

题目描述:
    给你一个 m 行 n 列的矩阵 matrix，请按照顺时针螺旋顺序，返回矩阵中的所有元素。

示例:
    输入: [[1,2,3],[4,5,6],[7,8,9]]
    输出: [1,2,3,6,9,8,7,4,5]

思路分析:
    模拟 + 边界收缩:
        定义四个边界: top, bottom, left, right

        按照 右→下→左→上 的顺序遍历:
            1. 从左到右遍历上边，然后 top++
            2. 从上到下遍历右边，然后 right--
            3. 从右到左遍历下边，然后 bottom--
            4. 从下到上遍历左边，然后 left++

        注意: 每次收缩后要检查边界是否还合法

复杂度分析:
    时间复杂度: O(m*n)
    空间复杂度: O(1) - 不算输出

面试技巧:
    1. 边界收缩法是最直观的解法
    2. 注意检查边界有效性，避免重复遍历
    3. 变体: 螺旋矩阵II（生成螺旋矩阵）
"""

from typing import List


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    """
    边界收缩法

    按 右→下→左→上 顺序，每次收缩一条边
    """
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # 从左到右遍历上边
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1

        # 从上到下遍历右边
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # 从右到左遍历下边（需要检查边界）
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1

        # 从下到上遍历左边（需要检查边界）
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "expected": [1, 2, 3, 6, 9, 8, 7, 4, 5]},
        {"matrix": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], "expected": [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]},
        {"matrix": [[1]], "expected": [1]},
        {"matrix": [[1, 2], [3, 4]], "expected": [1, 2, 4, 3]},
    ]

    for i, tc in enumerate(test_cases):
        result = spiralOrder(tc["matrix"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status}")
        print(f"       输出: {result}")
        print(f"       期望: {tc['expected']}")
