"""
LeetCode 128. 最长连续序列 (Longest Consecutive Sequence)
难度: 中等
链接: https://leetcode.cn/problems/longest-consecutive-sequence/

题目描述:
    给定一个未排序的整数数组 nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例:
    输入: nums = [100, 4, 200, 1, 3, 2]
    输出: 4
    解释: 最长数字连续序列是 [1, 2, 3, 4]，长度为 4

思路分析:
    关键约束: 时间复杂度必须是O(n)，不能排序!

    暴力思路 (超时):
        对于每个数，不断查找 num+1, num+2, ... 是否存在
        问题: 每个数都可能作为起点，导致重复计算

    优化思路 - 只从序列起点开始:
        核心洞察: 只有当 num-1 不存在时，num 才是某个序列的起点
        例如: [1,2,3,4] 中，只有1是起点，2,3,4都不需要作为起点

    算法步骤:
        1. 将所有数放入哈希集合 (去重 + O(1)查找)
        2. 遍历集合中的每个数
        3. 如果 num-1 不在集合中，说明 num 是序列起点
        4. 从起点开始，不断查找 num+1, num+2, ... 计算序列长度
        5. 更新最长长度

复杂度分析:
    时间复杂度: O(n)
        - 每个数最多被访问两次: 一次遍历，一次在序列延伸中
        - 因为只从起点开始延伸，不会重复计算
    空间复杂度: O(n) - 哈希集合

面试技巧:
    1. 这道题的关键是"只从起点开始"这个优化
    2. 面试时先说暴力解法，再说优化思路
    3. 解释清楚为什么时间复杂度是O(n)而不是O(n²)
"""

from typing import List


def longestConsecutive(nums: List[int]) -> int:
    """
    哈希集合 + 只从序列起点开始计数
    """
    if not nums:
        return 0

    # 转换为集合，实现O(1)查找 + 去重
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # 关键优化: 只有当num-1不存在时，num才是序列起点
        # 这保证了每个序列只被计算一次
        if num - 1 not in num_set:
            # num是某个序列的起点，开始延伸
            current_num = num
            current_length = 1

            # 不断查找下一个数
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            max_length = max(max_length, current_length)

    return max_length


def longestConsecutive_sorting(nums: List[int]) -> int:
    """
    排序解法 - O(n log n)
    用于对比理解，不满足题目O(n)的要求
    """
    if not nums:
        return 0

    nums = sorted(set(nums))  # 去重并排序
    max_length = 1
    current_length = 1

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1

    return max_length


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [100, 4, 200, 1, 3, 2], "expected": 4},
        {"nums": [0, 3, 7, 2, 5, 8, 4, 6, 0, 1], "expected": 9},
        {"nums": [], "expected": 0},
        {"nums": [1, 2, 0, 1], "expected": 3},  # 有重复元素
    ]

    for i, tc in enumerate(test_cases):
        result = longestConsecutive(tc["nums"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
