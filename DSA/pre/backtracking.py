"""
回溯算法（Backtracking）用于在解空间中试探 + 撤销，常见于排列、组合、棋盘问题。
模板思路：
1. 选择：当前可以尝试哪些选项。
2. 约束：哪些选择是合法的。
3. 结束：满足条件时记录答案。
下面给出两个典型示例：全排列与组合求和（每个数字可重复使用）。
"""
from typing import List


def permutations(nums: List[int]) -> List[List[int]]:
    """
    返回 nums 的所有全排列。
    用 visited 数组记录已使用的元素，避免重复放入路径。
    """
    res: List[List[int]] = []
    path: List[int] = []
    used = [False] * len(nums)

    def dfs() -> None:
        if len(path) == len(nums):
            res.append(path.copy())
            return
        for i, num in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(num)
            dfs()  # 递归进入下一层
            path.pop()  # 撤销选择
            used[i] = False

    dfs()
    return res


def combination_sum(nums: List[int], target: int) -> List[List[int]]:
    """
    组合求和：nums 中的数字可重复使用，找出和为 target 的所有组合。
    使用 start 控制下一个搜索起点，确保组合不重复（避免排列）。
    """
    res: List[List[int]] = []
    path: List[int] = []
    nums = sorted(nums)  # 排序便于剪枝

    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path.copy())
            return
        for i in range(start, len(nums)):
            num = nums[i]
            if num > remain:
                break  # 剪枝：后续数字更大，无需继续
            path.append(num)
            dfs(i, remain - num)  # i 而不是 i+1，允许重复使用当前数字
            path.pop()

    dfs(0, target)
    return res


if __name__ == "__main__":
    nums = [1, 2, 3]
    print("全排列:")
    for p in permutations(nums):
        print(p)

    print("\n组合求和 (nums=[2,3,6,7], target=7):")
    for c in combination_sum([2, 3, 6, 7], 7):
        print(c)
