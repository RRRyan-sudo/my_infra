"""
LeetCode 283. 移动零 (Move Zeroes)
难度: 简单
链接: https://leetcode.cn/problems/move-zeroes/

题目描述:
    给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，
    同时保持非零元素的相对顺序。

    注意: 必须在不复制数组的情况下原地对数组进行操作。

示例:
    输入: nums = [0,1,0,3,12]
    输出: [1,3,12,0,0]

思路分析:
    双指针法 (快慢指针):
        - slow: 指向下一个非零元素应该放置的位置
        - fast: 遍历整个数组，寻找非零元素

    算法过程:
        1. slow 从 0 开始，fast 从 0 开始
        2. fast 向前遍历，遇到非零元素:
           - 将该元素放到 slow 位置
           - slow 向前移动一位
        3. 遍历结束后，slow 之后的位置全部填0

    为什么这样做是对的?
        - fast 遇到非零元素时，slow 一定 <= fast
        - 所有非零元素按原顺序被移到了数组前部
        - slow 之后的位置就是需要填0的位置

    优化: 交换法
        - 不需要最后填0，直接交换 nums[slow] 和 nums[fast]
        - 这样更简洁

复杂度分析:
    时间复杂度: O(n) - 一次遍历
    空间复杂度: O(1) - 原地操作

面试技巧:
    1. 这是快慢指针的经典应用
    2. 类似的题目: 删除排序数组中的重复项、移除元素
    3. 关键是理解 slow 的含义: "已处理好的区域的边界"
"""

from typing import List


def moveZeroes(nums: List[int]) -> None:
    """
    双指针交换法 (最优解)

    思路:
    - slow 指向下一个非零元素应放的位置
    - fast 遍历数组，遇到非零元素就和 slow 位置交换
    """
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] != 0:
            # 交换 (当 slow == fast 时交换也没问题)
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


def moveZeroes_twoPass(nums: List[int]) -> None:
    """
    两次遍历法 (更容易理解)

    第一次: 把所有非零元素移到前面
    第二次: 剩余位置填0
    """
    slow = 0

    # 第一次遍历: 移动非零元素
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1

    # 第二次遍历: 填充0
    while slow < len(nums):
        nums[slow] = 0
        slow += 1


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"input": [0, 1, 0, 3, 12], "expected": [1, 3, 12, 0, 0]},
        {"input": [0], "expected": [0]},
        {"input": [1, 2, 3], "expected": [1, 2, 3]},
        {"input": [0, 0, 1], "expected": [1, 0, 0]},
    ]

    for i, tc in enumerate(test_cases):
        nums = tc["input"].copy()  # 复制，因为是原地修改
        moveZeroes(nums)
        is_correct = nums == tc["expected"]
        status = "✓" if is_correct else "✗"
        print(f"测试{i+1}: {status} 输入: {tc['input']}")
        print(f"       输出: {nums}, 期望: {tc['expected']}")
