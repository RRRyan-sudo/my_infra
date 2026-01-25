"""
LeetCode 189. 轮转数组 (Rotate Array)
难度: 中等
链接: https://leetcode.cn/problems/rotate-array/

题目描述:
    给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

示例:
    输入: nums = [1,2,3,4,5,6,7], k = 3
    输出: [5,6,7,1,2,3,4]
    解释:
        向右轮转 1 步: [7,1,2,3,4,5,6]
        向右轮转 2 步: [6,7,1,2,3,4,5]
        向右轮转 3 步: [5,6,7,1,2,3,4]

思路分析:
    方法1 - 使用额外数组: O(n)空间
        新位置 = (原位置 + k) % n

    方法2 - 三次翻转: O(1)空间 (最优)
        观察: [1,2,3,4,5,6,7] 向右转3步 -> [5,6,7,1,2,3,4]

        技巧:
            1. 整体翻转:     [7,6,5,4,3,2,1]
            2. 翻转前k个:    [5,6,7,4,3,2,1]
            3. 翻转后n-k个:  [5,6,7,1,2,3,4]

    方法3 - 环状替换: O(1)空间
        从位置0开始，不断将元素放到正确位置
        处理完所有元素需要gcd(n,k)轮

    为什么三次翻转是对的?
        右转k步 = 把后k个元素移到前面
        翻转相当于"镜像"操作，三次镜像恰好达到目标

复杂度分析:
    三次翻转:
        时间复杂度: O(n)
        空间复杂度: O(1)

面试技巧:
    1. 三次翻转是最优雅的解法
    2. 注意 k 可能大于 n，需要取模
    3. 翻转是原地操作，非常高效
"""

from typing import List


def rotate(nums: List[int], k: int) -> None:
    """
    三次翻转法

    1. 整体翻转
    2. 翻转前k个
    3. 翻转后n-k个
    """
    n = len(nums)
    k = k % n  # 处理 k > n 的情况

    def reverse(start: int, end: int):
        """翻转 nums[start:end+1]"""
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

    # 三次翻转
    reverse(0, n - 1)      # 整体翻转
    reverse(0, k - 1)      # 翻转前k个
    reverse(k, n - 1)      # 翻转后n-k个


def rotate_extra_array(nums: List[int], k: int) -> None:
    """
    使用额外数组 - 更直观
    """
    n = len(nums)
    k = k % n
    temp = nums.copy()
    for i in range(n):
        nums[(i + k) % n] = temp[i]


def rotate_cyclic(nums: List[int], k: int) -> None:
    """
    环状替换 - 另一种O(1)空间解法
    """
    from math import gcd

    n = len(nums)
    k = k % n
    count = gcd(n, k)  # 需要的轮数

    for start in range(count):
        current = start
        prev = nums[start]
        while True:
            next_idx = (current + k) % n
            nums[next_idx], prev = prev, nums[next_idx]
            current = next_idx
            if current == start:
                break


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 2, 3, 4, 5, 6, 7], "k": 3, "expected": [5, 6, 7, 1, 2, 3, 4]},
        {"nums": [-1, -100, 3, 99], "k": 2, "expected": [3, 99, -1, -100]},
        {"nums": [1, 2], "k": 3, "expected": [2, 1]},  # k > n
    ]

    for i, tc in enumerate(test_cases):
        nums = tc["nums"].copy()
        rotate(nums, tc["k"])
        is_correct = nums == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} 原数组={tc['nums']}, k={tc['k']}")
        print(f"       输出: {nums}, 期望: {tc['expected']}")
