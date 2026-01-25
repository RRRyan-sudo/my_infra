"""
LeetCode 42. 接雨水 (Trapping Rain Water)
难度: 困难
链接: https://leetcode.cn/problems/trapping-rain-water/

题目描述:
    给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，
    计算按此排列的柱子，下雨之后能接多少雨水。

示例:
    输入: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    输出: 6
    解释: 上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，
         在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）

思路分析:
    核心思想: 每个位置能接多少水，取决于它左右两边最高柱子的较小值

    对于位置 i:
        water[i] = max(0, min(左边最高, 右边最高) - height[i])

    方法1 - 暴力法: O(n²)
        对每个位置，分别向左右扫描找最大值

    方法2 - 预处理: O(n)
        预先计算每个位置的左边最大值和右边最大值

    方法3 - 双指针: O(n)，空间O(1)
        核心洞察:
        - 我们不需要知道确切的左右最大值
        - 只需要知道"较小的那一边"的最大值就够了

        维护: left, right 指针，left_max, right_max

        如果 left_max < right_max:
            当前位置水量由 left_max 决定（木桶效应）
            处理左边，left++
        否则:
            处理右边，right--

    方法4 - 单调栈: O(n)
        横向计算积水，使用单调递减栈

复杂度分析:
    双指针法:
        时间复杂度: O(n)
        空间复杂度: O(1)

面试技巧:
    1. 这是经典难题，多种解法都要会
    2. 双指针法最优雅，但需要理解"木桶效应"
    3. 单调栈解法更通用，可以解决类似问题
"""

from typing import List


def trap(height: List[int]) -> int:
    """
    双指针法 (最优解)

    核心: 哪边低，哪边就决定了当前位置的水量
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0  # 左右两边遇到的最大高度
    water = 0

    while left < right:
        # 更新左右最大值
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        # 哪边低，就处理哪边
        if left_max < right_max:
            # 左边低，当前位置水量由 left_max 决定
            water += left_max - height[left]
            left += 1
        else:
            # 右边低（或相等），当前位置水量由 right_max 决定
            water += right_max - height[right]
            right -= 1

    return water


def trap_precompute(height: List[int]) -> int:
    """
    预处理法 - 更容易理解

    预先计算每个位置的 left_max 和 right_max
    """
    if not height:
        return 0

    n = len(height)

    # left_max[i] = max(height[0..i])
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])

    # right_max[i] = max(height[i..n-1])
    right_max = [0] * n
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])

    # 计算每个位置的积水
    water = 0
    for i in range(n):
        water += min(left_max[i], right_max[i]) - height[i]

    return water


def trap_stack(height: List[int]) -> int:
    """
    单调栈法 - 横向计算积水

    栈中存储索引，保持高度单调递减
    遇到更高的柱子时，计算凹槽积水
    """
    stack = []  # 单调递减栈，存索引
    water = 0

    for i, h in enumerate(height):
        # 当前高度大于栈顶，形成凹槽
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()  # 凹槽底部
            if not stack:
                break
            # 计算凹槽积水
            left = stack[-1]
            width = i - left - 1
            bounded_height = min(height[left], h) - height[bottom]
            water += width * bounded_height

        stack.append(i)

    return water


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"height": [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], "expected": 6},
        {"height": [4, 2, 0, 3, 2, 5], "expected": 9},
        {"height": [1, 2], "expected": 0},
        {"height": [], "expected": 0},
    ]

    for i, tc in enumerate(test_cases):
        result = trap(tc["height"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} height={tc['height']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
