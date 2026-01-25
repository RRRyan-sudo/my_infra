"""
LeetCode 70. 爬楼梯 (Climbing Stairs)
难度: 简单
链接: https://leetcode.cn/problems/climbing-stairs/

题目描述:
    假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶？

示例:
    输入: n = 3
    输出: 3
    解释: 有三种方法: 1+1+1, 1+2, 2+1

思路分析:
    动态规划:
        设 dp[i] = 爬到第 i 阶的方法数

        状态转移:
            dp[i] = dp[i-1] + dp[i-2]
            解释: 到第i阶可以从第i-1阶爬1步，或从第i-2阶爬2步

        本质上就是斐波那契数列!

    空间优化:
        只需要保存前两个状态

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1)

面试技巧:
    1. 动态规划入门题
    2. 理解状态定义和状态转移
    3. 空间优化是加分项
"""


def climbStairs(n: int) -> int:
    """
    动态规划 (空间优化)

    dp[i] = dp[i-1] + dp[i-2]
    """
    if n <= 2:
        return n

    # 只保存前两个状态
    prev2, prev1 = 1, 2

    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1


def climbStairs_dp_array(n: int) -> int:
    """标准DP数组写法"""
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"n": 2, "expected": 2},
        {"n": 3, "expected": 3},
        {"n": 4, "expected": 5},
        {"n": 5, "expected": 8},
    ]

    for i, tc in enumerate(test_cases):
        result = climbStairs(tc["n"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} n={tc['n']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
