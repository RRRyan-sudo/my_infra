"""
LeetCode 215. 数组中的第K个最大元素 (Kth Largest Element in an Array)
难度: 中等
链接: https://leetcode.cn/problems/kth-largest-element-in-an-array/

题目描述:
    给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
    注意是排序后的第 k 个最大，而不是第 k 个不同的元素。

示例:
    输入: [3,2,1,5,6,4], k = 2
    输出: 5

思路分析:
    方法1 - 排序: O(n log n)
        排序后返回第k大

    方法2 - 堆: O(n log k)
        维护一个大小为k的最小堆
        遍历完成后堆顶就是第k大

    方法3 - 快速选择: O(n) 平均
        基于快速排序的partition
        只需要在一侧递归

复杂度分析:
    快速选择:
        时间复杂度: O(n) 平均, O(n²) 最坏
        空间复杂度: O(1)
    堆:
        时间复杂度: O(n log k)
        空间复杂度: O(k)

面试技巧:
    1. 快速选择是最优解，但实现稍复杂
    2. 堆解法更容易写对，实际工作中常用
    3. 注意第k大对应的索引是 n-k
"""

from typing import List
import heapq
import random


def findKthLargest(nums: List[int], k: int) -> int:
    """
    快速选择算法

    第k大 = 第(n-k)小
    """
    def partition(left, right, pivot_idx):
        pivot = nums[pivot_idx]
        # 1. 把pivot移到最右边
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]

        # 2. 把小于pivot的都移到左边
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1

        # 3. 把pivot放到正确位置
        nums[store_idx], nums[right] = nums[right], nums[store_idx]
        return store_idx

    def quick_select(left, right, k_smallest):
        if left == right:
            return nums[left]

        # 随机选择pivot，避免最坏情况
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)

        if pivot_idx == k_smallest:
            return nums[pivot_idx]
        elif pivot_idx < k_smallest:
            return quick_select(pivot_idx + 1, right, k_smallest)
        else:
            return quick_select(left, pivot_idx - 1, k_smallest)

    # 第k大 = 第(n-k)小
    return quick_select(0, len(nums) - 1, len(nums) - k)


def findKthLargest_heap(nums: List[int], k: int) -> int:
    """
    堆解法

    维护大小为k的最小堆，堆顶是第k大
    """
    # 方法1: 使用nlargest
    # return heapq.nlargest(k, nums)[-1]

    # 方法2: 维护最小堆
    heap = nums[:k]
    heapq.heapify(heap)

    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)

    return heap[0]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [3, 2, 1, 5, 6, 4], "k": 2, "expected": 5},
        {"nums": [3, 2, 3, 1, 2, 4, 5, 5, 6], "k": 4, "expected": 4},
        {"nums": [1], "k": 1, "expected": 1},
    ]

    for i, tc in enumerate(test_cases):
        nums = tc["nums"].copy()
        result = findKthLargest(nums, tc["k"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}, k={tc['k']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
