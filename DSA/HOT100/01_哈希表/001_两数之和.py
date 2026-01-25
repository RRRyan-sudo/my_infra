"""
LeetCode 1. 两数之和 (Two Sum)
难度: 简单
链接: https://leetcode.cn/problems/two-sum/

题目描述:
    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出
    和为目标值 target 的那两个整数，并返回它们的数组下标。

    你可以假设每种输入只会对应一个答案，并且你不能使用两次同一个元素。
    你可以按任意顺序返回答案。

示例:
    输入: nums = [2,7,11,15], target = 9
    输出: [0,1]
    解释: 因为 nums[0] + nums[1] == 9 ，返回 [0, 1]

思路分析:
    方法1 - 暴力枚举: O(n²)
        两层循环，对于每个元素，遍历剩余元素寻找target-nums[i]

    方法2 - 哈希表 (最优): O(n)
        核心思想: 用哈希表存储"我需要什么"
        - 遍历数组时，计算 need = target - nums[i]
        - 检查哈希表中是否已有 need
        - 如果有，说明之前遍历过的某个数正好满足条件
        - 如果没有，将当前数及其索引存入哈希表

    为什么哈希表更好?
        - 哈希表查找是O(1)，总时间复杂度O(n)
        - 空间换时间，用O(n)空间换取O(n)时间

复杂度分析:
    时间复杂度: O(n) - 一次遍历
    空间复杂度: O(n) - 哈希表存储

面试技巧:
    1. 先问清楚: 数组是否有序? 是否有重复元素? 是否一定有解?
    2. 从暴力解法说起，然后优化到哈希表解法
    3. 注意: 不能使用同一个元素两次
"""

from typing import List


def twoSum(nums: List[int], target: int) -> List[int]:
    """
    哈希表一次遍历解法

    关键点:
    1. 哈希表存储 {值: 索引}
    2. 边遍历边检查，避免使用同一元素两次
    """
    # 哈希表: 值 -> 索引
    num_to_index = {}

    for i, num in enumerate(nums):
        # 计算配对数字
        need = target - num

        # 检查配对数字是否已在哈希表中
        if need in num_to_index:
            return [num_to_index[need], i]

        # 将当前数字存入哈希表
        num_to_index[num] = i

    # 题目保证有解，这里不会执行
    return []


def twoSum_bruteforce(nums: List[int], target: int) -> List[int]:
    """暴力解法 - 用于对比理解"""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):  # j > i 避免重复
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {"nums": [2, 7, 11, 15], "target": 9, "expected": [0, 1]},
        {"nums": [3, 2, 4], "target": 6, "expected": [1, 2]},
        {"nums": [3, 3], "target": 6, "expected": [0, 1]},
    ]

    for i, tc in enumerate(test_cases):
        result = twoSum(tc["nums"], tc["target"])
        # 验证结果正确性 (两种顺序都可以)
        is_correct = (sorted(result) == sorted(tc["expected"]))
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}, target={tc['target']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
