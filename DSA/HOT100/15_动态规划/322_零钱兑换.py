"""
LeetCode 322. 零钱兑换 (Coin Change)
难度: 中等
链接: https://leetcode.cn/problems/coin-change/

题目描述:
    给你一个整数数组 coins 表示不同面额的硬币，以及一个整数 amount 表示总金额。
    计算凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
    每种硬币的数量是无限的。

示例:
    输入: coins = [1, 2, 5], amount = 11
    输出: 3
    解释: 11 = 5 + 5 + 1

思路分析:
    完全背包问题:
        设 dp[i] = 凑成金额 i 所需的最少硬币数

        状态转移:
            dp[i] = min(dp[i - coin] + 1) for coin in coins
            解释: 尝试每种硬币，取最小值

        初始化:
            dp[0] = 0  (金额0需要0枚硬币)
            dp[i] = inf  (初始化为无穷大)

复杂度分析:
    时间复杂度: O(amount * n)
    空间复杂度: O(amount)

面试技巧:
    1. 完全背包的经典题目
    2. 理解"完全背包"和"0-1背包"的区别
    3. 注意处理无法凑成的情况
"""

from typing import List


def coinChange(coins: List[int], amount: int) -> int:
    """
    完全背包DP

    dp[i] = 凑成金额i所需的最少硬币数
    """
    # 初始化: 0金额需要0枚硬币，其他为无穷大
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    # 遍历每个金额
    for i in range(1, amount + 1):
        # 尝试每种硬币
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"coins": [1, 2, 5], "amount": 11, "expected": 3},
        {"coins": [2], "amount": 3, "expected": -1},
        {"coins": [1], "amount": 0, "expected": 0},
    ]

    for i, tc in enumerate(test_cases):
        result = coinChange(tc["coins"], tc["amount"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} coins={tc['coins']}, amount={tc['amount']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
