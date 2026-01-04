"""
单链表与环形链表（检测是否有环）。
常用操作：插入、遍历、检测环（Floyd 快慢指针）。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ListNode:
    val: int
    next: Optional["ListNode"] = None


class LinkedList:
    def __init__(self) -> None:
        self.head: Optional[ListNode] = None

    def append(self, val: int) -> None:
        """在尾部插入新节点。"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return

        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = new_node

    def to_list(self, limit: int = 20) -> list[int]:
        """
        以 Python 列表形式返回链表值。
        limit 防止意外遇到有环链表时无限循环，仅用于演示。
        """
        res = []
        cur = self.head
        steps = 0
        while cur and steps < limit:
            res.append(cur.val)
            cur = cur.next
            steps += 1
        return res


def detect_cycle(head: Optional[ListNode]) -> bool:
    """
    Floyd 快慢指针判环：
    - fast 每次走两步，slow 每次走一步。
    - 若存在环，fast 终会追上 slow；否则 fast 先到达 None。
    """
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


if __name__ == "__main__":
    ll = LinkedList()
    for num in [1, 2, 3, 4, 5]:
        ll.append(num)

    print("链表内容:", ll.to_list())
    print("是否有环:", detect_cycle(ll.head))

    # 制造一个简单的环：让尾节点指向第二个节点
    tail = ll.head
    while tail and tail.next:
        tail = tail.next
    if tail and ll.head and ll.head.next:
        tail.next = ll.head.next
    print("人为制造环后是否有环:", detect_cycle(ll.head))
