"""
LeetCode 15. 三数之和 (3Sum)
难度: 中等
链接: https://leetcode.cn/problems/3sum/

题目描述:
    给你一个整数数组 nums，判断是否存在三元组 [nums[i], nums[j], nums[k]]
    满足 i != j、i != k 且 j != k，同时还满足 nums[i] + nums[j] + nums[k] == 0。
    返回所有和为 0 且不重复的三元组。

    注意: 答案中不可以包含重复的三元组。

示例:
    输入: nums = [-1,0,1,2,-1,-4]
    输出: [[-1,-1,2],[-1,0,1]]

思路分析:
    暴力法: O(n³)
        三重循环枚举所有组合，用集合去重

    排序 + 双指针: O(n²)
        1. 先排序数组
        2. 固定第一个数 nums[i]
        3. 在 [i+1, n-1] 范围内用双指针找两个数，使三数之和为0

    去重技巧 (难点):
        1. 第一个数去重: 如果 nums[i] == nums[i-1]，跳过
        2. 找到解后，跳过重复的 left 和 right

    为什么要排序?
        - 排序后可以用双指针
        - 排序后相同元素相邻，方便去重

复杂度分析:
    时间复杂度: O(n²) - 排序 O(n log n) + 双指针遍历 O(n²)
    空间复杂度: O(1) - 不计算返回值的空间

面试技巧:
    1. 去重是这道题的难点，一定要处理好
    2. 可以先做"两数之和"热身
    3. 扩展: 四数之和用同样的思路，固定两个数用双指针
"""

from typing import List


def threeSum(nums: List[int]) -> List[List[int]]:
    """
    排序 + 双指针

    核心: 固定一个数，剩下两个数用双指针
    """
    n = len(nums)
    if n < 3:
        return []

    nums.sort()  # 排序是关键
    result = []

    for i in range(n - 2):
        # 剪枝: 最小的数都大于0，不可能有解
        if nums[i] > 0:
            break

        # 去重: 跳过重复的第一个数
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 双指针找剩下两个数
        left, right = i + 1, n - 1
        target = -nums[i]  # 需要找的两数之和

        while left < right:
            total = nums[left] + nums[right]

            if total == target:
                result.append([nums[i], nums[left], nums[right]])

                # 去重: 跳过重复的 left 和 right
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1

            elif total < target:
                left += 1  # 和太小，left右移
            else:
                right -= 1  # 和太大，right左移

    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {
            "nums": [-1, 0, 1, 2, -1, -4],
            "expected": [[-1, -1, 2], [-1, 0, 1]]
        },
        {
            "nums": [0, 1, 1],
            "expected": []
        },
        {
            "nums": [0, 0, 0],
            "expected": [[0, 0, 0]]
        },
    ]

    for i, tc in enumerate(test_cases):
        result = threeSum(tc["nums"])
        # 排序后比较，因为顺序可能不同
        result_sorted = [sorted(x) for x in result]
        expected_sorted = [sorted(x) for x in tc["expected"]]
        is_correct = sorted(result_sorted) == sorted(expected_sorted)
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}")
        print(f"       输出: {result}")
        print(f"       期望: {tc['expected']}")
