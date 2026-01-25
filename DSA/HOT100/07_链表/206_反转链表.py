"""
LeetCode 206. 反转链表 (Reverse Linked List)
难度: 简单
链接: https://leetcode.cn/problems/reverse-linked-list/

题目描述:
    给你单链表的头节点 head，请你反转链表，并返回反转后的链表。

示例:
    输入: head = [1,2,3,4,5]
    输出: [5,4,3,2,1]

思路分析:
    方法1 - 迭代:
        用三个指针: prev, curr, next
        每次把 curr.next 指向 prev，然后三个指针都前进一步

    方法2 - 递归:
        递归到链表末尾，然后回溯时反转指针
        reverseList(head) 返回反转后的头节点

        递归思路:
            1. 递归到最后一个节点，它就是新的头
            2. 回溯时，让 head.next.next = head (让下一个节点指向自己)
            3. head.next = None (断开原来的连接)

复杂度分析:
    迭代: 时间O(n), 空间O(1)
    递归: 时间O(n), 空间O(n)（递归栈）

面试技巧:
    1. 反转链表是最基础的链表操作，必须熟练
    2. 两种方法都要会
    3. 是很多其他链表题的子问题
"""

import sys
sys.path.append('..')
from _list_node import ListNode, create_linked_list, linked_list_to_list
from typing import Optional


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    迭代法

    用prev和curr两个指针，逐个反转
    """
    prev = None
    curr = head

    while curr:
        # 保存下一个节点
        next_temp = curr.next
        # 反转指针
        curr.next = prev
        # 移动指针
        prev = curr
        curr = next_temp

    return prev


def reverseList_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    递归法

    递归到末尾，回溯时反转
    """
    # 基本情况
    if not head or not head.next:
        return head

    # 递归反转剩余部分
    new_head = reverseList_recursive(head.next)

    # 反转当前节点
    head.next.next = head  # 让下一个节点指向自己
    head.next = None       # 断开原来的连接

    return new_head


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        [1, 2, 3, 4, 5],
        [1, 2],
        [1],
        [],
    ]

    for values in test_cases:
        head = create_linked_list(values)
        result = reverseList(head)
        output = linked_list_to_list(result)
        expected = values[::-1]
        status = "✓" if output == expected else "✗"
        print(f"{status} 输入: {values}, 输出: {output}, 期望: {expected}")
