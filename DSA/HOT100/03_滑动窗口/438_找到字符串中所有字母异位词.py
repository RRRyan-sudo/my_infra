"""
LeetCode 438. 找到字符串中所有字母异位词 (Find All Anagrams in a String)
难度: 中等
链接: https://leetcode.cn/problems/find-all-anagrams-in-a-string/

题目描述:
    给定两个字符串 s 和 p，找到 s 中所有 p 的异位词的子串，返回这些子串的起始索引。

示例:
    输入: s = "cbaebabacd", p = "abc"
    输出: [0, 6]
    解释:
        起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词
        起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词

思路分析:
    固定大小的滑动窗口:
        - 窗口大小固定为 len(p)
        - 维护窗口内的字符计数
        - 当窗口字符计数等于 p 的字符计数时，找到一个异位词

    优化: 使用 diff 变量
        - 不需要每次比较整个计数数组
        - 维护"还差多少字符"的计数
        - 当 diff == 0 时，说明窗口是异位词

复杂度分析:
    时间复杂度: O(n) - n是s的长度
    空间复杂度: O(1) - 只需要26个字母的计数

面试技巧:
    1. 固定窗口大小的滑动窗口模板
    2. 用字符计数判断异位词
    3. diff优化是加分项
"""

from typing import List
from collections import Counter


def findAnagrams(s: str, p: str) -> List[int]:
    """
    固定窗口滑动 + 字符计数
    """
    if len(s) < len(p):
        return []

    result = []
    p_count = Counter(p)
    window_count = Counter()
    p_len = len(p)

    for i in range(len(s)):
        # 加入新字符
        window_count[s[i]] += 1

        # 移除窗口左边的字符 (当窗口大小超过 p_len)
        if i >= p_len:
            left_char = s[i - p_len]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]

        # 当窗口大小等于 p_len 时，检查是否是异位词
        if window_count == p_count:
            result.append(i - p_len + 1)

    return result


def findAnagrams_optimized(s: str, p: str) -> List[int]:
    """
    优化版: 使用 diff 计数

    diff 表示窗口与目标还差多少字符
    当 diff == 0 时，窗口就是异位词
    """
    if len(s) < len(p):
        return []

    result = []
    p_len = len(p)

    # 初始化: 负数表示需要这个字符，正数表示多余
    count = [0] * 26
    for c in p:
        count[ord(c) - ord('a')] -= 1

    # diff: 不匹配的字符种类数
    diff = sum(1 for c in count if c != 0)

    for i, c in enumerate(s):
        # 加入新字符
        idx = ord(c) - ord('a')
        if count[idx] == 0:
            diff += 1  # 从平衡变成不平衡
        count[idx] += 1
        if count[idx] == 0:
            diff -= 1  # 变成平衡

        # 移除窗口左边的字符
        if i >= p_len:
            left_idx = ord(s[i - p_len]) - ord('a')
            if count[left_idx] == 0:
                diff += 1
            count[left_idx] -= 1
            if count[left_idx] == 0:
                diff -= 1

        # 检查是否是异位词
        if diff == 0:
            result.append(i - p_len + 1)

    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"s": "cbaebabacd", "p": "abc", "expected": [0, 6]},
        {"s": "abab", "p": "ab", "expected": [0, 1, 2]},
        {"s": "a", "p": "ab", "expected": []},
    ]

    for i, tc in enumerate(test_cases):
        result = findAnagrams(tc["s"], tc["p"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} s=\"{tc['s']}\", p=\"{tc['p']}\"")
        print(f"       输出: {result}, 期望: {tc['expected']}")
