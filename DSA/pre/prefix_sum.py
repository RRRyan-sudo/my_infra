"""
前缀和（Prefix Sum）用于快速查询区间和。
核心思路：prefix[i] 记录下标 < i 的元素和（含 0，但不含 i），
那么区间 [l, r] 的和为 prefix[r+1] - prefix[l]，查询复杂度 O(1)。
"""
from typing import List


def build_prefix(nums: List[int]) -> List[int]:
    """
    构建前缀和数组，长度为 len(nums) + 1。
    prefix[0] = 0，prefix[i] 是前 i 个元素的和。
    """
    prefix = [0] * (len(nums) + 1)
    for i, num in enumerate(nums, start=1):
        prefix[i] = prefix[i - 1] + num
    return prefix


def range_sum(prefix: List[int], left: int, right: int) -> int:
    """
    使用前缀和数组求闭区间 [left, right] 的和。
    """
    if left < 0 or right >= len(prefix) - 1 or left > right:
        raise ValueError("查询区间非法")
    return prefix[right + 1] - prefix[left]


if __name__ == "__main__":
    nums = [3, 1, 4, 1, 5, 9]
    prefix = build_prefix(nums)
    print("原数组:", nums)
    print("前缀和数组:", prefix)
    print("区间 [1, 3] 之和:", range_sum(prefix, 1, 3))
    print("区间 [0, 5] 之和:", range_sum(prefix, 0, 5))
