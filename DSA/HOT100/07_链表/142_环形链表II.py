"""
LeetCode 142. 环形链表 II (Linked List Cycle II)
难度: 中等
链接: https://leetcode.cn/problems/linked-list-cycle-ii/

题目描述:
    给定一个链表的头节点 head，返回链表开始入环的第一个节点。
    如果链表无环，则返回 null。

示例:
    输入: head = [3,2,0,-4], pos = 1
    输出: 返回索引为 1 的链表节点

思路分析:
    Floyd算法找环入口:
        设:
            a = 头节点到环入口的距离
            b = 环入口到相遇点的距离
            c = 环的长度

        相遇时:
            慢指针走了: a + b
            快指针走了: a + b + n*c (n是快指针多绕的圈数)

        因为快指针速度是慢指针的2倍:
            2(a + b) = a + b + n*c
            a + b = n*c
            a = n*c - b = (n-1)*c + (c-b)

        结论:
            从头节点到环入口的距离 = 从相遇点到环入口的距离
            (可能多绕几圈)

        所以:
            相遇后，让一个指针从头开始，一个从相遇点开始
            两者同速前进，再次相遇的点就是环入口

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(1)

面试技巧:
    1. 数学推导是难点，一定要理解
    2. 分两步: 先找相遇点，再找入口
"""

import sys
sys.path.append('..')
from _list_node import ListNode, create_linked_list
from typing import Optional


def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Floyd算法找环入口

    1. 快慢指针找相遇点
    2. 从头和相遇点同速出发，再次相遇就是入口
    """
    if not head or not head.next:
        return None

    # 第一步: 快慢指针找相遇点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        # 无环
        return None

    # 第二步: 从头和相遇点同速出发
    ptr1 = head
    ptr2 = slow
    while ptr1 != ptr2:
        ptr1 = ptr1.next
        ptr2 = ptr2.next

    return ptr1


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建有环链表: 3 -> 2 -> 0 -> -4 -> 2(环)
    head = create_linked_list([3, 2, 0, -4])
    node1 = head.next  # 值为2的节点（环入口）
    tail = head
    while tail.next:
        tail = tail.next
    tail.next = node1  # 创建环

    result = detectCycle(head)
    print(f"环入口节点值: {result.val if result else None}")
    print(f"期望: 2")
