"""
LeetCode 33. 搜索旋转排序数组 (Search in Rotated Sorted Array)
难度: 中等
链接: https://leetcode.cn/problems/search-in-rotated-sorted-array/

题目描述:
    整数数组 nums 按升序排列，数组中的值互不相同。
    在传递给函数之前，nums 在预先未知的某个下标 k 上进行了旋转，
    使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]

    给你旋转后的数组 nums 和一个整数 target，如果 nums 中存在这个目标值 target，
    则返回它的下标，否则返回 -1。

    要求: O(log n) 时间复杂度

示例:
    输入: nums = [4,5,6,7,0,1,2], target = 0
    输出: 4

思路分析:
    二分查找变体:
        关键观察: 将数组一分为二，其中一半一定是有序的

        判断哪半有序:
            - 如果 nums[left] <= nums[mid]，左半部分有序
            - 否则，右半部分有序

        确定target在哪半:
            - 如果在有序的那半，正常二分
            - 否则，在另一半继续找

复杂度分析:
    时间复杂度: O(log n)
    空间复杂度: O(1)

面试技巧:
    1. 关键是判断哪半有序
    2. 注意边界条件的等号
    3. 类似题目: 寻找旋转数组的最小值
"""

from typing import List


def search(nums: List[int], target: int) -> int:
    """
    二分查找

    核心: 每次一半是有序的，判断target在哪半
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # 判断哪半有序
        if nums[left] <= nums[mid]:
            # 左半有序
            if nums[left] <= target < nums[mid]:
                # target在左半
                right = mid - 1
            else:
                # target在右半
                left = mid + 1
        else:
            # 右半有序
            if nums[mid] < target <= nums[right]:
                # target在右半
                left = mid + 1
            else:
                # target在左半
                right = mid - 1

    return -1


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [4, 5, 6, 7, 0, 1, 2], "target": 0, "expected": 4},
        {"nums": [4, 5, 6, 7, 0, 1, 2], "target": 3, "expected": -1},
        {"nums": [1], "target": 0, "expected": -1},
        {"nums": [1, 3], "target": 3, "expected": 1},
    ]

    for i, tc in enumerate(test_cases):
        result = search(tc["nums"], tc["target"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}, target={tc['target']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
