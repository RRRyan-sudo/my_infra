"""
链表通用定义和辅助函数

这个文件包含了链表题目中常用的节点定义和辅助函数
"""

from typing import Optional, List


class ListNode:
    """单链表节点"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """从列表创建链表"""
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """将链表转换为列表"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def print_linked_list(head: Optional[ListNode]) -> None:
    """打印链表"""
    values = linked_list_to_list(head)
    print(" -> ".join(map(str, values)) if values else "Empty")
