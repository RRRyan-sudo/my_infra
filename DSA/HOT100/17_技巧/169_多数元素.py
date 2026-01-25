"""
LeetCode 169. 多数元素 (Majority Element)
难度: 简单
链接: https://leetcode.cn/problems/majority-element/

题目描述:
    给定一个大小为 n 的数组 nums，返回其中的多数元素。
    多数元素是指在数组中出现次数大于 n/2 的元素。
    你可以假设数组是非空的，并且给定的数组总是存在多数元素。

示例:
    输入: nums = [3,2,3]
    输出: 3

思路分析:
    方法1 - 哈希表: O(n)时间, O(n)空间
        统计每个元素出现次数

    方法2 - 排序: O(n log n)
        排序后中间元素一定是多数元素

    方法3 - Boyer-Moore投票算法: O(n)时间, O(1)空间 (最优)
        核心思想:
            - 维护一个候选人和计数器
            - 遇到相同的数，计数+1
            - 遇到不同的数，计数-1
            - 计数为0时，更换候选人
            - 最后剩下的候选人就是多数元素

        为什么这样做是对的?
            多数元素出现次数 > n/2，其他所有元素加起来 < n/2
            所以多数元素的"票数"不可能被完全抵消

复杂度分析:
    Boyer-Moore:
        时间复杂度: O(n)
        空间复杂度: O(1)

面试技巧:
    1. Boyer-Moore投票是经典算法
    2. 理解"抵消"的思想
    3. 如果不保证存在多数元素，需要二次验证
"""

from typing import List


def majorityElement(nums: List[int]) -> int:
    """
    Boyer-Moore投票算法

    候选人和不同的数字"抵消"，多数元素一定剩下
    """
    candidate = None
    count = 0

    for num in nums:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1

    return candidate


def majorityElement_hash(nums: List[int]) -> int:
    """哈希表计数"""
    from collections import Counter
    count = Counter(nums)
    return max(count.keys(), key=count.get)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [3, 2, 3], "expected": 3},
        {"nums": [2, 2, 1, 1, 1, 2, 2], "expected": 2},
    ]

    for i, tc in enumerate(test_cases):
        result = majorityElement(tc["nums"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
