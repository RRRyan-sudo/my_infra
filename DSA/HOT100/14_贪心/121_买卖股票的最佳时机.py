"""
LeetCode 121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)
难度: 简单
链接: https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/

题目描述:
    给定一个数组 prices，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
    你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。
    计算你所能获取的最大利润。如果不能获取利润，返回 0。

示例:
    输入: [7,1,5,3,6,4]
    输出: 5
    解释: 在第 2 天（价格 = 1）买入，在第 5 天（价格 = 6）卖出，利润 = 6-1 = 5

思路分析:
    贪心:
        维护到目前为止的最低价格
        对于每一天，计算在这天卖出的利润（当天价格 - 历史最低价）
        取所有利润的最大值

    为什么贪心是对的?
        要获得最大利润，需要在最低点买入，在之后的最高点卖出
        我们只能在过去买入，所以维护历史最低价格就足够了

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1)

面试技巧:
    1. 这是贪心的经典入门题
    2. 理解"维护历史最低价"的思想
    3. 变体: 买卖股票II（可以多次交易）
"""

from typing import List


def maxProfit(prices: List[int]) -> int:
    """
    贪心: 维护历史最低价格

    每天计算"今天卖出的利润"，取最大值
    """
    if not prices:
        return 0

    min_price = float('inf')
    max_profit = 0

    for price in prices:
        # 更新历史最低价格
        min_price = min(min_price, price)
        # 计算今天卖出的利润
        profit = price - min_price
        # 更新最大利润
        max_profit = max(max_profit, profit)

    return max_profit


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"prices": [7, 1, 5, 3, 6, 4], "expected": 5},
        {"prices": [7, 6, 4, 3, 1], "expected": 0},  # 只跌不涨
        {"prices": [1, 2], "expected": 1},
    ]

    for i, tc in enumerate(test_cases):
        result = maxProfit(tc["prices"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} prices={tc['prices']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
