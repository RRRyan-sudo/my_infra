"""
动态规划（Dynamic Programming）常见入门示例。
核心套路：
1. 明确状态：子问题的定义是什么。
2. 明确转移：当前状态如何由更小的子问题推导。
3. 处理好边界与初始化。
下面用两个经典问题示范：斐波那契数列（递归记忆化 + 自底向上）和硬币凑金额（最少硬币数）。
"""
from functools import lru_cache
from typing import Dict, List


# 递归 + 记忆化：避免重复计算
@lru_cache(maxsize=None)
def fib_top_down(n: int) -> int:
    """
    斐波那契数列的自顶向下写法。
    状态：f(n) 表示第 n 项的值。
    转移：f(n) = f(n-1) + f(n-2)。
    """
    if n <= 1:
        return n
    return fib_top_down(n - 1) + fib_top_down(n - 2)


def fib_bottom_up(n: int) -> int:
    """
    斐波那契数列的自底向上写法。
    用数组从小到大迭代，避免递归栈。
    """
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def coin_change_min_coins(coins: List[int], amount: int) -> int:
    """
    给定硬币面额 coins，求凑成 amount 的最少硬币数，凑不出返回 -1。
    状态：dp[x] 表示凑成金额 x 的最少硬币数。
    转移：dp[x] = min(dp[x - coin] + 1) for coin in coins if x - coin >= 0。
    初始化：dp[0] = 0，其余置为无穷大表示暂不可达。
    """
    INF = amount + 1  # 一个足够大的不可达标记
    dp = [INF] * (amount + 1)
    dp[0] = 0

    for x in range(1, amount + 1):
        for coin in coins:
            if x - coin >= 0:
                dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != INF else -1


if __name__ == "__main__":
    n = 10
    print(f"fib_top_down({n}) =", fib_top_down(n))
    print(f"fib_bottom_up({n}) =", fib_bottom_up(n))

    coins = [1, 2, 5]
    amount = 11
    print(
        f"面额 {coins} 凑出 {amount} 的最少硬币数:",
        coin_change_min_coins(coins, amount),
    )
