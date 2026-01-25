"""
LeetCode 141. 环形链表 (Linked List Cycle)
难度: 简单
链接: https://leetcode.cn/problems/linked-list-cycle/

题目描述:
    给你一个链表的头节点 head，判断链表中是否有环。

示例:
    输入: head = [3,2,0,-4], pos = 1 (尾节点连接到索引1的节点)
    输出: true

思路分析:
    方法1 - 哈希集合: O(n)时间, O(n)空间
        遍历链表，如果遇到访问过的节点，说明有环

    方法2 - 快慢指针 (Floyd判圈): O(n)时间, O(1)空间
        - 快指针每次走2步，慢指针每次走1步
        - 如果有环，快慢指针一定会相遇
        - 如果无环，快指针会先到达null

        为什么一定会相遇?
            当慢指针进入环后，快指针每次比慢指针多走1步
            所以它们的距离每次减少1，最终会相遇

复杂度分析:
    快慢指针:
        时间复杂度: O(n)
        空间复杂度: O(1)

面试技巧:
    1. Floyd判圈算法是经典算法
    2. 证明快慢指针一定相遇很重要
    3. 进阶: 找环的入口 (LeetCode 142)
"""

import sys
sys.path.append('..')
from _list_node import ListNode, create_linked_list
from typing import Optional


def hasCycle(head: Optional[ListNode]) -> bool:
    """
    快慢指针 (Floyd判圈)

    快指针走2步，慢指针走1步
    如果有环，一定会相遇
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False


def hasCycle_hash(head: Optional[ListNode]) -> bool:
    """哈希集合法"""
    visited = set()
    while head:
        if head in visited:
            return True
        visited.add(head)
        head = head.next
    return False


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试1: 有环
    head1 = create_linked_list([3, 2, 0, -4])
    # 创建环: 尾节点指向索引1的节点
    tail = head1
    node1 = head1.next
    while tail.next:
        tail = tail.next
    tail.next = node1

    print(f"测试1 (有环): {hasCycle(head1)}, 期望: True")

    # 测试2: 无环
    head2 = create_linked_list([1, 2, 3])
    print(f"测试2 (无环): {hasCycle(head2)}, 期望: False")

    # 测试3: 单节点无环
    head3 = create_linked_list([1])
    print(f"测试3 (单节点): {hasCycle(head3)}, 期望: False")
