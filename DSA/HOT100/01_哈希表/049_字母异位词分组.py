"""
LeetCode 49. 字母异位词分组 (Group Anagrams)
难度: 中等
链接: https://leetcode.cn/problems/group-anagrams/

题目描述:
    给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。
    字母异位词: 由重新排列源单词的所有字母得到的一个新单词。

示例:
    输入: strs = ["eat","tea","tan","ate","nat","bat"]
    输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

思路分析:
    核心问题: 如何判断两个字符串是字母异位词?

    方法1 - 排序作为key:
        - 字母异位词排序后结果相同
        - "eat" -> "aet", "tea" -> "aet", "ate" -> "aet"
        - 用排序后的字符串作为哈希表的key

    方法2 - 字符计数作为key:
        - 统计每个字符出现的次数
        - 将计数结果转换为不可变的元组作为key
        - 例如: "eat" -> (1,0,0,0,1,0,...,1,...) 表示a=1,e=1,t=1

    选择方法1的原因:
        - 代码更简洁
        - 对于短字符串，排序的常数因子更小
        - 面试时更容易写对

复杂度分析:
    时间复杂度: O(n * k * log k) - n是字符串数量，k是最长字符串长度
    空间复杂度: O(n * k) - 存储所有字符串

面试技巧:
    1. 问清楚: 字符串是否只包含小写字母?
    2. 两种方法都要会，根据实际情况选择
    3. 字符计数法在字符串很长时更优
"""

from typing import List
from collections import defaultdict


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    """
    方法1: 排序作为key

    思路: 异位词排序后相同，用排序结果做分组依据
    """
    # 哈希表: 排序后的字符串 -> 原始字符串列表
    groups = defaultdict(list)

    for s in strs:
        # 排序后作为key
        # sorted返回列表，需要转成字符串或元组
        key = tuple(sorted(s))
        groups[key].append(s)

    return list(groups.values())


def groupAnagrams_count(strs: List[str]) -> List[List[str]]:
    """
    方法2: 字符计数作为key

    思路: 用26个字母的计数元组作为key
    优势: 当字符串很长时，计数是O(k)，比排序O(k*logk)更快
    """
    groups = defaultdict(list)

    for s in strs:
        # 统计每个字符出现次数
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1

        # 将计数转为元组作为key (列表不能作为字典的key)
        key = tuple(count)
        groups[key].append(s)

    return list(groups.values())


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {
            "input": ["eat", "tea", "tan", "ate", "nat", "bat"],
            "expected_groups": 3,  # 应该分成3组
        },
        {
            "input": [""],
            "expected_groups": 1,
        },
        {
            "input": ["a"],
            "expected_groups": 1,
        },
    ]

    for i, tc in enumerate(test_cases):
        result = groupAnagrams(tc["input"])
        is_correct = len(result) == tc["expected_groups"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} 输入: {tc['input']}")
        print(f"       分组结果: {result}")
        print(f"       组数: {len(result)}, 期望: {tc['expected_groups']}")
        print()
