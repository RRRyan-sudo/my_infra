"""
LeetCode 48. 旋转图像 (Rotate Image)
难度: 中等
链接: https://leetcode.cn/problems/rotate-image/

题目描述:
    给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
    你必须在原地旋转图像，直接修改输入的二维矩阵。

示例:
    输入: [[1,2,3],[4,5,6],[7,8,9]]
    输出: [[7,4,1],[8,5,2],[9,6,3]]

思路分析:
    方法1 - 转置 + 水平翻转:
        顺时针旋转90° = 先转置 + 再左右翻转

        转置: matrix[i][j] 和 matrix[j][i] 交换
        水平翻转: 每行左右对称交换

        例如:
        [1,2,3]    转置    [1,4,7]    翻转    [7,4,1]
        [4,5,6]    →      [2,5,8]    →      [8,5,2]
        [7,8,9]           [3,6,9]           [9,6,3]

    方法2 - 四点旋转:
        直接旋转，每次处理4个对应位置
        (i,j) → (j,n-1-i) → (n-1-i,n-1-j) → (n-1-j,i) → (i,j)

复杂度分析:
    时间复杂度: O(n²)
    空间复杂度: O(1)

面试技巧:
    1. 转置+翻转方法最好记忆和实现
    2. 逆时针旋转 = 转置 + 上下翻转
    3. 旋转180° = 上下翻转 + 左右翻转
"""

from typing import List


def rotate(matrix: List[List[int]]) -> None:
    """
    转置 + 水平翻转

    顺时针90° = 转置 + 左右翻转
    """
    n = len(matrix)

    # 1. 转置: matrix[i][j] 和 matrix[j][i] 交换
    for i in range(n):
        for j in range(i + 1, n):  # 只遍历上三角
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # 2. 水平翻转: 每行左右对称交换
    for i in range(n):
        for j in range(n // 2):
            matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]


def rotate_four_way(matrix: List[List[int]]) -> None:
    """
    四点旋转法

    每次处理4个对应位置的元素
    """
    n = len(matrix)

    # 只需要遍历1/4的元素
    for i in range(n // 2):
        for j in range(i, n - 1 - i):
            # 保存左上角
            temp = matrix[i][j]
            # 左下 → 左上
            matrix[i][j] = matrix[n-1-j][i]
            # 右下 → 左下
            matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
            # 右上 → 右下
            matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
            # 左上 → 右上
            matrix[j][n-1-i] = temp


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "expected": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]},
        {"matrix": [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]],
         "expected": [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]},
    ]

    for i, tc in enumerate(test_cases):
        matrix = [row.copy() for row in tc["matrix"]]
        rotate(matrix)
        is_correct = matrix == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status}")
        print(f"       输出: {matrix}")
        print(f"       期望: {tc['expected']}")
