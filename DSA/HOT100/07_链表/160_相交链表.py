"""
LeetCode 160. 相交链表 (Intersection of Two Linked Lists)
难度: 简单
链接: https://leetcode.cn/problems/intersection-of-two-linked-lists/

题目描述:
    给你两个单链表的头节点 headA 和 headB，请你找出并返回两个单链表相交的起始节点。
    如果两个链表不存在相交节点，返回 null。

示例:
    输入: A: 4->1->8->4->5, B: 5->6->1->8->4->5 (在节点8处相交)
    输出: 8

思路分析:
    方法1 - 哈希集合: O(m+n)时间, O(m)空间
        遍历A，存所有节点；遍历B，找第一个在集合中的节点

    方法2 - 双指针: O(m+n)时间, O(1)空间 (最优)
        让两个指针分别从A和B出发:
            - pA 走完A后走B
            - pB 走完B后走A
            - 相遇点就是交点（或都为null）

        为什么?
            假设A长度为a，B长度为b，公共部分长度为c
            pA走的路径: a + (b-c) = a + b - c
            pB走的路径: b + (a-c) = a + b - c
            两者相等！所以一定会同时到达交点

复杂度分析:
    双指针法:
        时间复杂度: O(m+n)
        空间复杂度: O(1)

面试技巧:
    1. 双指针的数学证明要能说清楚
    2. 这是链表双指针的经典应用
"""

import sys
sys.path.append('..')
from _list_node import ListNode, create_linked_list
from typing import Optional


def getIntersectionNode(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """
    双指针法

    A走完走B，B走完走A，相遇点就是交点
    """
    if not headA or not headB:
        return None

    pA, pB = headA, headB

    # 当pA和pB相遇时停止
    # 如果没有交点，最终都会变成None，也会相等
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA

    return pA


def getIntersectionNode_hash(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """哈希集合法"""
    nodes = set()
    while headA:
        nodes.add(headA)
        headA = headA.next

    while headB:
        if headB in nodes:
            return headB
        headB = headB.next

    return None


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建相交链表
    # A: 4 -> 1 -> 8 -> 4 -> 5
    # B: 5 -> 6 -> 1 -> 8 -> 4 -> 5
    #                  ↑ 相交点

    common = create_linked_list([8, 4, 5])
    headA = ListNode(4, ListNode(1, common))
    headB = ListNode(5, ListNode(6, ListNode(1, common)))

    result = getIntersectionNode(headA, headB)
    print(f"相交节点值: {result.val if result else None}")
    print(f"期望: 8")
