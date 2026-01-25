"""
LeetCode 239. 滑动窗口最大值 (Sliding Window Maximum)
难度: 困难
链接: https://leetcode.cn/problems/sliding-window-maximum/

题目描述:
    给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
    你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
    返回滑动窗口中的最大值。

示例:
    输入: nums = [1,3,-1,-3,5,3,6,7], k = 3
    输出: [3,3,5,5,6,7]
    解释:
        滑动窗口位置                最大值
        ---------------           -----
        [1  3  -1] -3  5  3  6  7   3
         1 [3  -1  -3] 5  3  6  7   3
         1  3 [-1  -3  5] 3  6  7   5
         1  3  -1 [-3  5  3] 6  7   5
         1  3  -1  -3 [5  3  6] 7   6
         1  3  -1  -3  5 [3  6  7]  7

思路分析:
    暴力法: O(nk)
        每个窗口遍历k个元素找最大值

    单调队列: O(n)
        维护一个单调递减的双端队列，队首是当前窗口的最大值

        核心思想:
            - 如果一个元素比它前面的元素大，那前面的元素永远不可能成为最大值
            - 所以可以把前面较小的元素从队尾移除

        算法:
            1. 遍历数组，对于每个元素:
               a) 移除队尾所有小于当前元素的索引（它们不可能成为最大值）
               b) 将当前索引加入队尾
               c) 移除队首已经不在窗口内的索引
               d) 如果窗口已形成，队首就是当前窗口最大值

复杂度分析:
    时间复杂度: O(n) - 每个元素最多入队出队各一次
    空间复杂度: O(k) - 队列最多存k个元素

面试技巧:
    1. 单调队列是解决滑动窗口极值问题的利器
    2. 队列存储索引而不是值，方便判断是否出界
    3. 理解"单调"的含义: 队列中元素对应的值单调递减
"""

from typing import List
from collections import deque


def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    """
    单调递减队列

    队列存储索引，保持对应值单调递减
    队首始终是当前窗口的最大值索引
    """
    if not nums:
        return []

    result = []
    q = deque()  # 存储索引，对应值单调递减

    for i in range(len(nums)):
        # 1. 移除队尾所有小于当前元素的索引
        # 因为它们永远不可能成为最大值
        while q and nums[q[-1]] < nums[i]:
            q.pop()

        # 2. 将当前索引加入队尾
        q.append(i)

        # 3. 移除队首已经不在窗口内的索引
        # 窗口范围是 [i-k+1, i]
        if q[0] < i - k + 1:
            q.popleft()

        # 4. 当窗口形成后，记录最大值
        if i >= k - 1:
            result.append(nums[q[0]])

    return result


def maxSlidingWindow_bruteforce(nums: List[int], k: int) -> List[int]:
    """暴力法 - O(nk)，用于对比理解"""
    if not nums:
        return []
    return [max(nums[i:i+k]) for i in range(len(nums) - k + 1)]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"nums": [1, 3, -1, -3, 5, 3, 6, 7], "k": 3, "expected": [3, 3, 5, 5, 6, 7]},
        {"nums": [1], "k": 1, "expected": [1]},
        {"nums": [1, -1], "k": 1, "expected": [1, -1]},
        {"nums": [9, 11], "k": 2, "expected": [11]},
    ]

    for i, tc in enumerate(test_cases):
        result = maxSlidingWindow(tc["nums"], tc["k"])
        is_correct = result == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} nums={tc['nums']}, k={tc['k']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
