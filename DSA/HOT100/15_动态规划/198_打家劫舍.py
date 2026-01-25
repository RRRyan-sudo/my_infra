"""
LeetCode 198. 打家劫舍 (House Robber)
难度: 中等
链接: https://leetcode.cn/problems/house-robber/

题目描述:
    你是一个专业的小偷，沿街房屋排列成一排。每个房间存放了一定数量的现金。
    相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
    给定一个代表每个房屋存放金额的非负整数数组，计算在不触动警报的情况下，一夜之内能够偷窃到的最高金额。

示例:
    输入: [1,2,3,1]
    输出: 4
    解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)，总金额 = 1 + 3 = 4

思路分析:
    动态规划:
        设 dp[i] = 偷到第 i 个房子能获得的最大金额

        状态转移:
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
            解释:
                - 不偷第i个房子: dp[i-1]
                - 偷第i个房子: dp[i-2] + nums[i]（不能偷相邻的i-1）

    空间优化:
        只需要保存前两个状态

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1)

面试技巧:
    1. 经典的"选或不选"型DP
    2. 状态转移方程是关键
    3. 变体: 打家劫舍II（房子成环）
"""

from typing import List


def rob(nums: List[int]) -> int:
    """
    动态规划 (空间优化)

    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    # prev2 = dp[i-2], prev1 = dp[i-1]
    prev2, prev1 = 0, nums[0]

    for i in range(1, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current

    return prev1


def rob_dp_array(nums: List[int]) -> int:
    """标准DP数组写法"""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[n-1]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 2, 3, 1], "expected": 4},
        {"nums": [2, 7, 9, 3, 1], "expected": 12},
        {"nums": [2, 1, 1, 2], "expected": 4},
    ]

    for i, tc in enumerate(test_cases):
        result = rob(tc["nums"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
