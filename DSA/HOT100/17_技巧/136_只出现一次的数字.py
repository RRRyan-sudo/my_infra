"""
LeetCode 136. 只出现一次的数字 (Single Number)
难度: 简单
链接: https://leetcode.cn/problems/single-number/

题目描述:
    给你一个非空整数数组 nums，除了某个元素只出现一次以外，其余每个元素均出现两次。
    找出那个只出现了一次的元素。
    要求: 线性时间复杂度，不使用额外空间

示例:
    输入: nums = [2,2,1]
    输出: 1

思路分析:
    异或运算的性质:
        1. a ^ a = 0  (相同的数异或为0)
        2. a ^ 0 = a  (任何数和0异或等于自己)
        3. a ^ b ^ a = b  (异或满足交换律和结合律)

    算法:
        将所有数字异或起来，成对的数字抵消为0，剩下的就是只出现一次的数字

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1)

面试技巧:
    1. 异或是解决这类问题的利器
    2. 理解异或的性质很重要
    3. 变体: 只出现一次的数字II（其他数出现3次）
"""

from typing import List


def singleNumber(nums: List[int]) -> int:
    """
    异或运算

    a ^ a = 0, a ^ 0 = a
    所有数异或，成对的抵消，剩下单独的
    """
    result = 0
    for num in nums:
        result ^= num
    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [2, 2, 1], "expected": 1},
        {"nums": [4, 1, 2, 1, 2], "expected": 4},
        {"nums": [1], "expected": 1},
    ]

    for i, tc in enumerate(test_cases):
        result = singleNumber(tc["nums"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
