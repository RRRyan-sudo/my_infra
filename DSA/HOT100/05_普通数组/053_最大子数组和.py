"""
LeetCode 53. 最大子数组和 (Maximum Subarray)
难度: 中等
链接: https://leetcode.cn/problems/maximum-subarray/

题目描述:
    给你一个整数数组 nums，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），
    返回其最大和。

示例:
    输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
    输出: 6
    解释: 连续子数组 [4,-1,2,1] 的和最大，为 6

思路分析:
    方法1 - Kadane算法 (动态规划):
        定义: dp[i] = 以 nums[i] 结尾的最大子数组和

        状态转移:
            dp[i] = max(dp[i-1] + nums[i], nums[i])
            解释: 要么把 nums[i] 加到前面的子数组，要么从 nums[i] 重新开始

        空间优化: 只需要一个变量记录前一个状态

    方法2 - 分治法:
        将数组分成左右两半，最大子数组要么在左边，要么在右边，要么跨越中点
        时间复杂度 O(n log n)

    为什么 Kadane 算法是对的?
        - 如果前面的和是负数，加上它只会让结果更小
        - 所以当前面的和为负时，不如从当前元素重新开始

复杂度分析:
    Kadane算法:
        时间复杂度: O(n)
        空间复杂度: O(1)

面试技巧:
    1. Kadane算法是必会的经典算法
    2. 理解"以i结尾"这个定义方式
    3. 可以扩展到"最大子矩阵和"等问题
"""

from typing import List


def maxSubArray(nums: List[int]) -> int:
    """
    Kadane算法

    核心: 以当前元素结尾的最大和 = max(前一个最大和+当前, 当前)
    """
    # current_sum: 以当前元素结尾的最大子数组和
    current_sum = nums[0]
    max_sum = nums[0]

    for i in range(1, len(nums)):
        # 如果 current_sum > 0，加上当前元素
        # 否则，从当前元素重新开始
        current_sum = max(current_sum + nums[i], nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum


def maxSubArray_dp(nums: List[int]) -> int:
    """
    标准DP写法 (便于理解)
    """
    n = len(nums)
    # dp[i] = 以 nums[i] 结尾的最大子数组和
    dp = [0] * n
    dp[0] = nums[0]

    for i in range(1, n):
        # 要么延续前面的子数组，要么从自己开始
        dp[i] = max(dp[i-1] + nums[i], nums[i])

    return max(dp)


def maxSubArray_divide_conquer(nums: List[int]) -> int:
    """
    分治法 - O(n log n)
    用于理解另一种思路
    """
    def helper(left, right):
        if left == right:
            return nums[left]

        mid = (left + right) // 2

        # 左半部分最大子数组和
        left_max = helper(left, mid)
        # 右半部分最大子数组和
        right_max = helper(mid + 1, right)

        # 跨越中点的最大子数组和
        # 从中点向左延伸
        left_sum = float('-inf')
        total = 0
        for i in range(mid, left - 1, -1):
            total += nums[i]
            left_sum = max(left_sum, total)

        # 从中点向右延伸
        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, right + 1):
            total += nums[i]
            right_sum = max(right_sum, total)

        cross_max = left_sum + right_sum

        return max(left_max, right_max, cross_max)

    return helper(0, len(nums) - 1)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [-2, 1, -3, 4, -1, 2, 1, -5, 4], "expected": 6},
        {"nums": [1], "expected": 1},
        {"nums": [5, 4, -1, 7, 8], "expected": 23},
        {"nums": [-1], "expected": -1},
        {"nums": [-2, -1], "expected": -1},
    ]

    for i, tc in enumerate(test_cases):
        result = maxSubArray(tc["nums"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
