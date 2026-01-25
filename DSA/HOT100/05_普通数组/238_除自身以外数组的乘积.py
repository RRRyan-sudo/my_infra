"""
LeetCode 238. 除自身以外数组的乘积 (Product of Array Except Self)
难度: 中等
链接: https://leetcode.cn/problems/product-of-array-except-self/

题目描述:
    给你一个整数数组 nums，返回数组 answer，其中 answer[i] 等于 nums 中除 nums[i] 之外
    其余各元素的乘积。

    题目数据保证数组 nums 之中任意元素的全部前缀元素和后缀的乘积都在 32 位整数范围内。
    请不要使用除法，且在 O(n) 时间复杂度内完成此题。

示例:
    输入: nums = [1,2,3,4]
    输出: [24,12,8,6]
    解释: answer[0] = 2*3*4 = 24, answer[1] = 1*3*4 = 12, ...

思路分析:
    为什么不能用除法?
        - 有可能存在0，除法会出错
        - 题目明确要求不使用除法

    前缀积 + 后缀积:
        answer[i] = (nums[0] * ... * nums[i-1]) * (nums[i+1] * ... * nums[n-1])
                  = 左边所有元素的乘积 * 右边所有元素的乘积

        算法:
            1. 计算前缀积: prefix[i] = nums[0] * ... * nums[i-1]
            2. 计算后缀积: suffix[i] = nums[i+1] * ... * nums[n-1]
            3. answer[i] = prefix[i] * suffix[i]

    空间优化:
        - 先用 answer 数组存前缀积
        - 再从右到左遍历，乘以后缀积（只用一个变量维护）

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1) - 不算输出数组

面试技巧:
    1. 前缀/后缀的思想很重要
    2. 空间优化是加分项
    3. 类似思想: 接雨水、前缀和
"""

from typing import List


def productExceptSelf(nums: List[int]) -> List[int]:
    """
    前缀积 + 后缀积 (空间优化版)

    用 answer 先存前缀积，再乘后缀积
    """
    n = len(nums)
    answer = [1] * n

    # 第一遍: 计算前缀积
    # answer[i] = nums[0] * nums[1] * ... * nums[i-1]
    prefix = 1
    for i in range(n):
        answer[i] = prefix
        prefix *= nums[i]

    # 第二遍: 乘以后缀积
    # answer[i] *= nums[i+1] * nums[i+2] * ... * nums[n-1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix
        suffix *= nums[i]

    return answer


def productExceptSelf_two_arrays(nums: List[int]) -> List[int]:
    """
    使用两个数组 - 更容易理解
    """
    n = len(nums)

    # 前缀积
    prefix = [1] * n
    for i in range(1, n):
        prefix[i] = prefix[i-1] * nums[i-1]

    # 后缀积
    suffix = [1] * n
    for i in range(n-2, -1, -1):
        suffix[i] = suffix[i+1] * nums[i+1]

    # 结果
    return [prefix[i] * suffix[i] for i in range(n)]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 2, 3, 4], "expected": [24, 12, 8, 6]},
        {"nums": [-1, 1, 0, -3, 3], "expected": [0, 0, 9, 0, 0]},
        {"nums": [0, 0], "expected": [0, 0]},
    ]

    for i, tc in enumerate(test_cases):
        result = productExceptSelf(tc["nums"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
