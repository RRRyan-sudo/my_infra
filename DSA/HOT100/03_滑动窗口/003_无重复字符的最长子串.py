"""
LeetCode 3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)
难度: 中等
链接: https://leetcode.cn/problems/longest-substring-without-repeating-characters/

题目描述:
    给定一个字符串 s，请你找出其中不含有重复字符的最长子串的长度。

示例:
    输入: s = "abcabcbb"
    输出: 3
    解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3

    输入: s = "bbbbb"
    输出: 1

思路分析:
    滑动窗口 + 哈希集合:
        维护一个窗口 [left, right]，窗口内没有重复字符

        1. right 向右扩展，将字符加入窗口
        2. 如果新字符导致重复，left 向右收缩，直到没有重复
        3. 更新最大长度

    为什么用滑动窗口?
        - 暴力法需要 O(n²) 枚举所有子串
        - 滑动窗口利用了一个性质: 如果 [i,j] 有重复，[i,j+1] 也有重复
        - 所以 left 只需要单向移动，不需要回退

    优化: 用哈希表记录字符位置
        - 当发现重复时，直接跳到重复字符的下一个位置
        - 避免 left 一步步移动

复杂度分析:
    时间复杂度: O(n) - 每个字符最多被访问两次
    空间复杂度: O(min(n, m)) - m是字符集大小

面试技巧:
    1. 滑动窗口模板题，必须掌握
    2. 注意边界条件: 空字符串
    3. 优化版本在面试中加分
"""


def lengthOfLongestSubstring(s: str) -> int:
    """
    滑动窗口 + 哈希集合
    """
    if not s:
        return 0

    char_set = set()  # 窗口内的字符
    left = 0
    max_length = 0

    for right in range(len(s)):
        # 如果新字符已在窗口内，收缩左边界
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        # 将新字符加入窗口
        char_set.add(s[right])

        # 更新最大长度
        max_length = max(max_length, right - left + 1)

    return max_length


def lengthOfLongestSubstring_optimized(s: str) -> int:
    """
    优化版: 哈希表记录字符位置，直接跳转

    当发现重复字符时，直接把 left 移到重复位置的下一个
    """
    if not s:
        return 0

    char_index = {}  # 字符 -> 最后出现的索引
    left = 0
    max_length = 0

    for right, char in enumerate(s):
        # 如果字符已存在且在窗口内
        if char in char_index and char_index[char] >= left:
            # 直接跳到重复字符的下一个位置
            left = char_index[char] + 1

        # 更新字符位置
        char_index[char] = right

        # 更新最大长度
        max_length = max(max_length, right - left + 1)

    return max_length


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"s": "abcabcbb", "expected": 3},
        {"s": "bbbbb", "expected": 1},
        {"s": "pwwkew", "expected": 3},
        {"s": "", "expected": 0},
        {"s": " ", "expected": 1},
        {"s": "abba", "expected": 2},  # 边界: a在前面出现过但已不在窗口内
    ]

    for i, tc in enumerate(test_cases):
        result = lengthOfLongestSubstring(tc["s"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} s=\"{tc['s']}\"")
        print(f"       输出: {result}, 期望: {tc['expected']}")
