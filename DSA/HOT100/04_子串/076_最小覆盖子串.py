"""
LeetCode 76. 最小覆盖子串 (Minimum Window Substring)
难度: 困难
链接: https://leetcode.cn/problems/minimum-window-substring/

题目描述:
    给你一个字符串 s、一个字符串 t。返回 s 中涵盖 t 所有字符的最小子串。
    如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 ""。

    注意:
    - 如果 s 中存在这样的子串，我们保证它是唯一的答案
    - t 中可能包含重复字母

示例:
    输入: s = "ADOBECODEBANC", t = "ABC"
    输出: "BANC"
    解释: 最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'

思路分析:
    滑动窗口 + 字符计数:
        1. 先扩展右边界，直到窗口包含t的所有字符
        2. 再收缩左边界，直到刚好不满足条件
        3. 记录满足条件时的最小窗口
        4. 重复以上步骤

    关键变量:
        - need: t中每个字符需要的数量
        - window: 窗口中每个字符的数量
        - valid: 已经满足条件的字符种类数

    什么时候窗口满足条件?
        当 valid == len(need)，即所有需要的字符都达到了要求数量

复杂度分析:
    时间复杂度: O(n + m) - n是s的长度，m是t的长度
    空间复杂度: O(k) - k是字符集大小

面试技巧:
    1. 这是滑动窗口的模板题，难度较高
    2. 先写出框架，再填充细节
    3. 理解"满足条件"的判断是关键
"""

from collections import Counter, defaultdict


def minWindow(s: str, t: str) -> str:
    """
    滑动窗口

    维护窗口满足条件 = 包含t的所有字符（含重复）
    """
    if not s or not t:
        return ""

    # t中每个字符需要的数量
    need = Counter(t)
    # 窗口中的字符计数
    window = defaultdict(int)

    # valid: 已经满足条件的字符种类数
    valid = 0
    # 需要满足的字符种类数
    required = len(need)

    # 记录最小窗口
    min_len = float('inf')
    min_start = 0

    left = 0
    for right in range(len(s)):
        # 扩展右边界
        c = s[right]
        if c in need:
            window[c] += 1
            # 当某个字符数量刚好满足时，valid++
            if window[c] == need[c]:
                valid += 1

        # 当窗口满足条件时，尝试收缩左边界
        while valid == required:
            # 更新最小窗口
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left

            # 收缩左边界
            d = s[left]
            if d in need:
                # 当某个字符数量不再满足时，valid--
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
            left += 1

    return "" if min_len == float('inf') else s[min_start:min_start + min_len]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"s": "ADOBECODEBANC", "t": "ABC", "expected": "BANC"},
        {"s": "a", "t": "a", "expected": "a"},
        {"s": "a", "t": "aa", "expected": ""},
        {"s": "aa", "t": "aa", "expected": "aa"},
    ]

    for i, tc in enumerate(test_cases):
        result = minWindow(tc["s"], tc["t"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} s=\"{tc['s']}\", t=\"{tc['t']}\"")
        print(f"       输出: \"{result}\", 期望: \"{tc['expected']}\"")
