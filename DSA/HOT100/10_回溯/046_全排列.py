"""
LeetCode 46. 全排列 (Permutations)
难度: 中等
链接: https://leetcode.cn/problems/permutations/

题目描述:
    给定一个不含重复数字的数组 nums，返回其所有可能的全排列。

示例:
    输入: nums = [1,2,3]
    输出: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

思路分析:
    回溯法:
        回溯的核心框架:
            1. 路径: 已经做出的选择
            2. 选择列表: 当前可以做的选择
            3. 结束条件: 到达决策树底层，无法再做选择

        模板:
            def backtrack(路径, 选择列表):
                if 满足结束条件:
                    result.append(路径)
                    return
                for 选择 in 选择列表:
                    做选择
                    backtrack(新路径, 新选择列表)
                    撤销选择

    去重方法:
        - 使用 used 数组标记已使用的元素
        - 或者使用交换的方法（但不太直观）

复杂度分析:
    时间复杂度: O(n * n!)
    空间复杂度: O(n)

面试技巧:
    1. 回溯模板要熟练
    2. 理解"选择-递归-撤销"的过程
    3. 全排列II有重复元素，需要额外去重
"""

from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    """
    回溯法

    用 used 数组标记已使用的元素
    """
    result = []
    path = []
    used = [False] * len(nums)

    def backtrack():
        # 结束条件: 路径长度等于数组长度
        if len(path) == len(nums):
            result.append(path.copy())  # 注意要copy
            return

        # 遍历选择列表
        for i in range(len(nums)):
            # 跳过已使用的元素
            if used[i]:
                continue

            # 做选择
            path.append(nums[i])
            used[i] = True

            # 递归
            backtrack()

            # 撤销选择
            path.pop()
            used[i] = False

    backtrack()
    return result


def permute_swap(nums: List[int]) -> List[List[int]]:
    """
    交换法

    通过交换元素位置生成排列
    """
    result = []

    def backtrack(first):
        if first == len(nums):
            result.append(nums.copy())
            return

        for i in range(first, len(nums)):
            # 交换
            nums[first], nums[i] = nums[i], nums[first]
            # 递归
            backtrack(first + 1)
            # 撤销交换
            nums[first], nums[i] = nums[i], nums[first]

    backtrack(0)
    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 2, 3], "expected_count": 6},
        {"nums": [0, 1], "expected_count": 2},
        {"nums": [1], "expected_count": 1},
    ]

    for i, tc in enumerate(test_cases):
        result = permute(tc["nums"])
        status = "✓" if len(result) == tc["expected_count"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       排列数: {len(result)}, 期望: {tc['expected_count']}")
        print(f"       结果: {result}")
