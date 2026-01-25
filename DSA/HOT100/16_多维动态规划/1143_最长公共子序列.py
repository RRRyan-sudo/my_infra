"""
LeetCode 1143. 最长公共子序列 (Longest Common Subsequence)
难度: 中等
链接: https://leetcode.cn/problems/longest-common-subsequence/

题目描述:
    给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
    子序列是指可以通过删除某些字符（也可以不删除）但不改变剩余字符相对顺序得到的字符串。

示例:
    输入: text1 = "abcde", text2 = "ace"
    输出: 3
    解释: 最长公共子序列是 "ace"

思路分析:
    二维动态规划:
        设 dp[i][j] = text1[0:i] 和 text2[0:j] 的最长公共子序列长度

        状态转移:
            如果 text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1  # 公共字符
            否则:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # 取较大值

        初始化:
            dp[0][j] = dp[i][0] = 0

复杂度分析:
    时间复杂度: O(m*n)
    空间复杂度: O(m*n)，可优化到O(min(m,n))

面试技巧:
    1. 这是二维DP的经典题目
    2. 理解"子序列"和"子串"的区别
    3. 可以用来求编辑距离等问题
"""


def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    二维DP

    dp[i][j] = text1[0:i] 和 text2[0:j] 的LCS长度
    """
    m, n = len(text1), len(text2)

    # dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的LCS长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                # 字符相同，LCS长度+1
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # 字符不同，取较大值
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def longestCommonSubsequence_optimized(text1: str, text2: str) -> int:
    """
    空间优化版本 - O(n)空间
    """
    m, n = len(text1), len(text2)
    if m < n:
        text1, text2 = text2, text1
        m, n = n, m

    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i-1] == text2[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp

    return dp[n]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"text1": "abcde", "text2": "ace", "expected": 3},
        {"text1": "abc", "text2": "abc", "expected": 3},
        {"text1": "abc", "text2": "def", "expected": 0},
    ]

    for i, tc in enumerate(test_cases):
        result = longestCommonSubsequence(tc["text1"], tc["text2"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} text1=\"{tc['text1']}\", text2=\"{tc['text2']}\"")
        print(f"       输出: {result}, 期望: {tc['expected']}")
