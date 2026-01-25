"""
LeetCode 56. 合并区间 (Merge Intervals)
难度: 中等
链接: https://leetcode.cn/problems/merge-intervals/

题目描述:
    以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [start_i, end_i]。
    请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

示例:
    输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
    输出: [[1,6],[8,10],[15,18]]
    解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6]

思路分析:
    排序 + 贪心合并:
        1. 按区间起点排序
        2. 遍历区间，如果当前区间的起点 <= 上一个区间的终点，则合并
        3. 否则，当前区间是新的独立区间

    为什么排序后就能贪心?
        排序后，如果区间A能和区间B合并，那么它们一定相邻
        不可能出现A不能和B合并，但能和C合并的情况

复杂度分析:
    时间复杂度: O(n log n) - 排序
    空间复杂度: O(n) - 存储结果（或O(log n)排序栈空间）

面试技巧:
    1. 区间问题首先考虑排序
    2. 理解"重叠"的判断条件
    3. 相关题目: 插入区间、无重叠区间
"""

from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    排序 + 贪心合并
    """
    if not intervals:
        return []

    # 按区间起点排序
    intervals.sort(key=lambda x: x[0])

    result = [intervals[0]]

    for i in range(1, len(intervals)):
        # 获取结果中最后一个区间
        last = result[-1]
        current = intervals[i]

        # 判断是否重叠: 当前起点 <= 上一个终点
        if current[0] <= last[1]:
            # 合并: 更新终点为两者的最大值
            last[1] = max(last[1], current[1])
        else:
            # 不重叠: 添加新区间
            result.append(current)

    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"intervals": [[1, 3], [2, 6], [8, 10], [15, 18]], "expected": [[1, 6], [8, 10], [15, 18]]},
        {"intervals": [[1, 4], [4, 5]], "expected": [[1, 5]]},
        {"intervals": [[1, 4], [0, 4]], "expected": [[0, 4]]},  # 未排序
        {"intervals": [[1, 4], [2, 3]], "expected": [[1, 4]]},  # 完全包含
    ]

    for i, tc in enumerate(test_cases):
        result = merge([interval.copy() for interval in tc["intervals"]])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} intervals={tc['intervals']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
