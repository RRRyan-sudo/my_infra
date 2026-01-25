"""
LeetCode 78. 子集 (Subsets)
难度: 中等
链接: https://leetcode.cn/problems/subsets/

题目描述:
    给你一个整数数组 nums，数组中的元素互不相同。返回该数组所有可能的子集（幂集）。
    解集不能包含重复的子集。

示例:
    输入: nums = [1,2,3]
    输出: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

思路分析:
    方法1 - 回溯:
        和全排列类似，但不需要用完所有元素
        每个位置可以选择"加入"或"不加入"

    方法2 - 迭代:
        从空集开始，每次添加一个新元素
        新子集 = 旧子集 + 旧子集每个元素加上新元素

    方法3 - 位运算:
        n个元素有2^n个子集
        用0到2^n-1的二进制表示每个子集

复杂度分析:
    时间复杂度: O(n * 2^n)
    空间复杂度: O(n)

面试技巧:
    1. 回溯和迭代方法都要会
    2. 位运算方法很巧妙
    3. 子集II有重复元素，需要排序+剪枝
"""

from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    """
    回溯法

    每个元素选择"加入"或"不加入"
    """
    result = []
    path = []

    def backtrack(start):
        # 每个路径都是一个有效子集
        result.append(path.copy())

        # 从start开始选择，避免重复
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)  # 从i+1开始，不能重复选
            path.pop()

    backtrack(0)
    return result


def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    迭代法

    每次加入一个新元素，生成新的子集
    """
    result = [[]]  # 从空集开始

    for num in nums:
        # 新子集 = 旧子集每个元素加上num
        result += [subset + [num] for subset in result]

    return result


def subsets_bitmask(nums: List[int]) -> List[List[int]]:
    """
    位运算法

    用0到2^n-1的二进制表示每个子集
    """
    n = len(nums)
    result = []

    for mask in range(1 << n):  # 0 到 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # 检查第i位是否为1
                subset.append(nums[i])
        result.append(subset)

    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 2, 3], "expected_count": 8},
        {"nums": [0], "expected_count": 2},
    ]

    for i, tc in enumerate(test_cases):
        result = subsets(tc["nums"])
        status = "✓" if len(result) == tc["expected_count"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       子集数: {len(result)}, 期望: {tc['expected_count']}")
        print(f"       结果: {result}")
