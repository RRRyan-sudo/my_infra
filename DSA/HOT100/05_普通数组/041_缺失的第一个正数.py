"""
LeetCode 41. 缺失的第一个正数 (First Missing Positive)
难度: 困难
链接: https://leetcode.cn/problems/first-missing-positive/

题目描述:
    给你一个未排序的整数数组 nums，请你找出其中没有出现的最小的正整数。
    请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

示例:
    输入: nums = [3,4,-1,1]
    输出: 2
    解释: 1 在数组中，2 不在数组中

    输入: nums = [7,8,9,11,12]
    输出: 1
    解释: 1 不在数组中

思路分析:
    关键观察:
        - 答案一定在 [1, n+1] 范围内
        - 如果 1~n 都存在，答案是 n+1
        - 否则答案是 1~n 中第一个缺失的

    原地哈希:
        - 把每个数放到它应该在的位置
        - nums[i] 应该放在 nums[nums[i]-1] 的位置
        - 例如: 数字 3 应该放在索引 2 的位置

    算法:
        1. 遍历数组，把每个在 [1,n] 范围内的数放到正确位置
        2. 再次遍历，找第一个 nums[i] != i+1 的位置
        3. 如果都正确，答案是 n+1

    为什么时间复杂度是 O(n)?
        - 虽然有嵌套循环，但每个数最多被交换一次
        - 总交换次数 <= n

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1)

面试技巧:
    1. 原地哈希是解决这类问题的关键技巧
    2. 理解"答案范围"的观察很重要
    3. 类似题目: 找重复数、找消失的数字
"""

from typing import List


def firstMissingPositive(nums: List[int]) -> int:
    """
    原地哈希

    把数字 x 放到索引 x-1 的位置
    """
    n = len(nums)

    # 把每个数放到正确的位置
    for i in range(n):
        # 当 nums[i] 在 [1,n] 范围内，且不在正确位置时，交换
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # 把 nums[i] 放到 nums[nums[i]-1]
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]

    # 找第一个不在正确位置的数
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # 如果 1~n 都在正确位置，答案是 n+1
    return n + 1


def firstMissingPositive_marking(nums: List[int]) -> int:
    """
    另一种原地哈希: 标记法

    用负数标记某个数是否存在
    """
    n = len(nums)

    # 1. 把所有非正数变成 n+1（一个不影响结果的正数）
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1

    # 2. 对于每个在 [1,n] 范围内的数，把对应位置标记为负数
    for i in range(n):
        num = abs(nums[i])
        if 1 <= num <= n:
            idx = num - 1
            if nums[idx] > 0:
                nums[idx] = -nums[idx]

    # 3. 找第一个正数的位置
    for i in range(n):
        if nums[i] > 0:
            return i + 1

    return n + 1


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 2, 0], "expected": 3},
        {"nums": [3, 4, -1, 1], "expected": 2},
        {"nums": [7, 8, 9, 11, 12], "expected": 1},
        {"nums": [1], "expected": 2},
        {"nums": [1, 2, 3], "expected": 4},
    ]

    for i, tc in enumerate(test_cases):
        nums = tc["nums"].copy()
        result = firstMissingPositive(nums)
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
