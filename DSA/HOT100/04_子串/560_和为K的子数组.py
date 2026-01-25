"""
LeetCode 560. 和为 K 的子数组 (Subarray Sum Equals K)
难度: 中等
链接: https://leetcode.cn/problems/subarray-sum-equals-k/

题目描述:
    给你一个整数数组 nums 和一个整数 k，请你统计并返回该数组中和为 k 的子数组的个数。
    子数组是数组中元素的连续非空序列。

示例:
    输入: nums = [1,1,1], k = 2
    输出: 2
    解释: [1,1] 有两个

    输入: nums = [1,2,3], k = 3
    输出: 2
    解释: [1,2] 和 [3]

思路分析:
    暴力法: O(n²)
        枚举所有子数组，计算和

    前缀和 + 哈希表: O(n)
        核心公式:
            子数组 [i,j] 的和 = prefix[j] - prefix[i-1]
            如果这个和等于 k，则 prefix[i-1] = prefix[j] - k

        算法:
            1. 计算前缀和 prefix[j]
            2. 查找之前有多少个前缀和等于 prefix[j] - k
            3. 用哈希表存储前缀和出现的次数

    为什么不能用滑动窗口?
        - 滑动窗口适用于元素非负的情况
        - 本题元素可能为负，窗口扩大/缩小和增减不确定

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(n)

面试技巧:
    1. 前缀和是处理子数组和问题的利器
    2. 哈希表优化查找是关键
    3. 注意初始化: 空前缀和为0出现1次
"""

from typing import List
from collections import defaultdict


def subarraySum(nums: List[int], k: int) -> int:
    """
    前缀和 + 哈希表

    核心: 找有多少个 prefix[i] 使得 prefix[j] - prefix[i] = k
    """
    # 哈希表: 前缀和 -> 出现次数
    prefix_count = defaultdict(int)
    # 初始化: 空前缀和为0，出现1次
    # 这样当 prefix_sum == k 时，能找到一个有效子数组
    prefix_count[0] = 1

    prefix_sum = 0
    count = 0

    for num in nums:
        # 计算当前前缀和
        prefix_sum += num

        # 查找有多少个前缀和等于 prefix_sum - k
        # 如果有，说明从那个位置到当前位置的子数组和为k
        count += prefix_count[prefix_sum - k]

        # 记录当前前缀和
        prefix_count[prefix_sum] += 1

    return count


def subarraySum_bruteforce(nums: List[int], k: int) -> int:
    """暴力法 - O(n²)，用于对比理解"""
    n = len(nums)
    count = 0
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += nums[j]
            if total == k:
                count += 1
    return count


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 1, 1], "k": 2, "expected": 2},
        {"nums": [1, 2, 3], "k": 3, "expected": 2},
        {"nums": [1, -1, 0], "k": 0, "expected": 3},  # 含负数和0
        {"nums": [-1, -1, 1], "k": 0, "expected": 1},
    ]

    for i, tc in enumerate(test_cases):
        result = subarraySum(tc["nums"], tc["k"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}, k={tc['k']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
