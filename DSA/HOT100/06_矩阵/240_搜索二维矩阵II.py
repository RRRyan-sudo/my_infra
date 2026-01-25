"""
LeetCode 240. 搜索二维矩阵 II (Search a 2D Matrix II)
难度: 中等
链接: https://leetcode.cn/problems/search-a-2d-matrix-ii/

题目描述:
    编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。
    该矩阵具有以下特性:
    - 每行的元素从左到右升序排列
    - 每列的元素从上到下升序排列

示例:
    输入: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
          target = 5
    输出: true

思路分析:
    方法1 - 暴力搜索: O(m*n)
        遍历所有元素

    方法2 - 每行二分: O(m * log n)
        对每一行进行二分查找

    方法3 - Z字形查找: O(m + n) (最优)
        从右上角开始搜索:
            - 如果当前值 == target，找到了
            - 如果当前值 > target，向左移动（排除当前列）
            - 如果当前值 < target，向下移动（排除当前行）

        为什么从右上角?
            - 右上角是当前行最大、当前列最小的值
            - 可以每次排除一行或一列
            - 同理也可以从左下角开始

复杂度分析:
    Z字形查找:
        时间复杂度: O(m + n)
        空间复杂度: O(1)

面试技巧:
    1. Z字形查找是这道题的精髓
    2. 理解为什么要从右上角或左下角开始
    3. 类似思想可用于其他有序矩阵问题
"""

from typing import List


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    """
    Z字形查找: 从右上角开始

    每次可以排除一行或一列
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    # 从右上角开始
    row, col = 0, n - 1

    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            # 当前值太大，向左移动
            col -= 1
        else:
            # 当前值太小，向下移动
            row += 1

    return False


def searchMatrix_binary(matrix: List[List[int]], target: int) -> bool:
    """
    每行二分查找 - O(m * log n)
    """
    import bisect

    for row in matrix:
        idx = bisect.bisect_left(row, target)
        if idx < len(row) and row[idx] == target:
            return True
    return False


# ==================== 测试代码 ====================
if __name__ == "__main__":
    matrix = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]

    test_cases = [
        {"target": 5, "expected": True},
        {"target": 20, "expected": False},
        {"target": 1, "expected": True},
        {"target": 30, "expected": True},
    ]

    for i, tc in enumerate(test_cases):
        result = searchMatrix(matrix, tc["target"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} target={tc['target']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
