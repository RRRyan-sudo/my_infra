"""
LeetCode 21. 合并两个有序链表 (Merge Two Sorted Lists)
难度: 简单
链接: https://leetcode.cn/problems/merge-two-sorted-lists/

题目描述:
    将两个升序链表合并为一个新的升序链表并返回。
    新链表是通过拼接给定的两个链表的所有节点组成的。

示例:
    输入: l1 = [1,2,4], l2 = [1,3,4]
    输出: [1,1,2,3,4,4]

思路分析:
    方法1 - 迭代 (推荐):
        使用虚拟头节点，每次比较两个链表的当前节点，选择较小的接到结果链表

    方法2 - 递归:
        递归比较两个头节点，较小的接上递归结果

    虚拟头节点的好处:
        - 不用特殊处理头节点
        - 代码更简洁
        - 最后返回 dummy.next

复杂度分析:
    时间复杂度: O(m+n)
    空间复杂度: 迭代O(1), 递归O(m+n)

面试技巧:
    1. 虚拟头节点是链表题的常用技巧
    2. 归并排序链表会用到这个子问题
"""

import sys
sys.path.append('..')
from _list_node import ListNode, create_linked_list, linked_list_to_list
from typing import Optional


def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    迭代法 + 虚拟头节点
    """
    # 虚拟头节点
    dummy = ListNode(-1)
    current = dummy

    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    # 接上剩余部分
    current.next = list1 if list1 else list2

    return dummy.next


def mergeTwoLists_recursive(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """递归法"""
    if not list1:
        return list2
    if not list2:
        return list1

    if list1.val <= list2.val:
        list1.next = mergeTwoLists_recursive(list1.next, list2)
        return list1
    else:
        list2.next = mergeTwoLists_recursive(list1, list2.next)
        return list2


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"l1": [1, 2, 4], "l2": [1, 3, 4], "expected": [1, 1, 2, 3, 4, 4]},
        {"l1": [], "l2": [], "expected": []},
        {"l1": [], "l2": [0], "expected": [0]},
    ]

    for i, tc in enumerate(test_cases):
        l1 = create_linked_list(tc["l1"])
        l2 = create_linked_list(tc["l2"])
        result = mergeTwoLists(l1, l2)
        output = linked_list_to_list(result)
        status = "✓" if output == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} l1={tc['l1']}, l2={tc['l2']}")
        print(f"       输出: {output}, 期望: {tc['expected']}")
