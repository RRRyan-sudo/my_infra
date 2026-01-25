"""
LeetCode 11. 盛最多水的容器 (Container With Most Water)
难度: 中等
链接: https://leetcode.cn/problems/container-with-most-water/

题目描述:
    给定 n 个非负整数 a1,a2,...,an，每个数代表坐标中的一个点 (i, ai)。
    找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

示例:
    输入: height = [1,8,6,2,5,4,8,3,7]
    输出: 49
    解释: 选择 height[1]=8 和 height[8]=7，宽度为 8-1=7
         面积 = min(8,7) * 7 = 49

思路分析:
    暴力法: O(n²)
        枚举所有 (i,j) 组合，计算面积取最大值

    双指针法: O(n)
        核心思想: 从两端向中间收缩

        为什么可以用双指针?
        - 容器面积 = min(height[left], height[right]) * (right - left)
        - 宽度随着指针移动只会减小
        - 要想面积可能增大，必须移动较短的那条边

        为什么移动较短的边?
        - 假设 height[left] < height[right]
        - 如果移动 right (较长的边)，宽度减小，高度最多不变 (还是受限于 left)
          所以面积一定减小
        - 如果移动 left (较短的边)，宽度减小，但高度可能增加
          面积可能增大

        这就是贪心策略的正确性证明!

复杂度分析:
    时间复杂度: O(n) - 双指针最多移动 n 次
    空间复杂度: O(1)

面试技巧:
    1. 一定要能解释清楚为什么移动短边
    2. 这是双指针的经典应用: 缩减搜索空间
    3. 类似题目: 三数之和、接雨水
"""

from typing import List


def maxArea(height: List[int]) -> int:
    """
    双指针法

    策略: 每次移动较短的那条边
    """
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        # 计算当前容器面积
        # 宽度 = right - left
        # 高度 = min(两边高度)
        width = right - left
        h = min(height[left], height[right])
        area = width * h
        max_area = max(max_area, area)

        # 移动较短的那条边
        # 因为移动较长的边不可能得到更大面积
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area


def maxArea_bruteforce(height: List[int]) -> int:
    """暴力法 - O(n²)，用于对比理解"""
    n = len(height)
    max_area = 0
    for i in range(n):
        for j in range(i + 1, n):
            area = (j - i) * min(height[i], height[j])
            max_area = max(max_area, area)
    return max_area


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"height": [1, 8, 6, 2, 5, 4, 8, 3, 7], "expected": 49},
        {"height": [1, 1], "expected": 1},
        {"height": [4, 3, 2, 1, 4], "expected": 16},
        {"height": [1, 2, 1], "expected": 2},
    ]

    for i, tc in enumerate(test_cases):
        result = maxArea(tc["height"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} height={tc['height']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
