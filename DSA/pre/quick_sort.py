"""
快速排序（Quick Sort）示例。
核心思路：
1. 选择一个枢轴值（这里用中间位置元素，便于避免最坏情况）。
2. 用两个指针从两端向中间移动，把比枢轴小的放左边，大的放右边。
3. 完成一轮分区后，递归排序左右两段。
平均时间复杂度 O(n log n)，最坏 O(n^2)（当每次分区极度不平衡时）。
"""
from typing import List


def quick_sort(nums: List[int]) -> List[int]:
    """
    对输入列表进行快速排序，返回新列表，不修改原列表。
    """
    arr = list(nums)  # 复制一份，保护调用方数据

    def _quick_sort(lo: int, hi: int) -> None:
        # 递归终止：区间为空或只有一个元素
        if lo >= hi:
            return

        pivot = arr[(lo + hi) // 2]  # 选择中间值作为枢轴
        i, j = lo, hi

        # 分区：把小于 pivot 的放左边，大于 pivot 的放右边
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]  # 交换到正确侧
                i += 1
                j -= 1

        # 此时 j 在左半部分末尾，i 在右半部分开头
        _quick_sort(lo, j)
        _quick_sort(i, hi)

    _quick_sort(0, len(arr) - 1)
    return arr


if __name__ == "__main__":
    demo = [6, 1, 2, 7, 9, 3, 4, 5, 10, 8]
    print("原始数组:", demo)
    print("排序结果:", quick_sort(demo))
