"""
LeetCode 146. LRU 缓存 (LRU Cache)
难度: 中等
链接: https://leetcode.cn/problems/lru-cache/

题目描述:
    设计和实现一个 LRU (最近最少使用) 缓存机制。
    实现 LRUCache 类:
    - LRUCache(capacity): 以正整数作为容量初始化 LRU 缓存
    - get(key): 如果key存在，返回value；否则返回-1
    - put(key, value): 如果key存在，更新value；否则插入。超过容量时删除最久未使用的

    get 和 put 必须以 O(1) 的平均时间复杂度运行。

思路分析:
    数据结构选择:
        需要满足:
            1. O(1) 查找 -> 哈希表
            2. O(1) 更新访问顺序 -> 双向链表

        哈希表 + 双向链表:
            - 哈希表: key -> 链表节点
            - 双向链表: 按访问时间排序，最近访问的在头部，最久未访问的在尾部

    操作:
        get:
            1. 哈希表查找节点
            2. 将节点移到链表头部

        put:
            1. 如果key存在，更新值并移到头部
            2. 如果key不存在:
               - 创建新节点，插入头部
               - 如果超过容量，删除尾部节点

复杂度分析:
    时间复杂度: O(1) - 所有操作
    空间复杂度: O(capacity)

面试技巧:
    1. 这是设计题的经典题目
    2. 双向链表的操作要熟练
    3. 使用dummy head和dummy tail简化边界处理
"""


class DLinkedNode:
    """双向链表节点"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    LRU缓存: 哈希表 + 双向链表

    - 链表头部是最近使用的
    - 链表尾部是最久未使用的
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> DLinkedNode

        # 使用dummy head和dummy tail简化操作
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: DLinkedNode):
        """将节点添加到头部"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode):
        """删除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: DLinkedNode):
        """将节点移到头部"""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> DLinkedNode:
        """删除尾部节点（最久未使用）"""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # 移到头部（最近使用）
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 更新值并移到头部
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # 创建新节点
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)

            # 超过容量，删除尾部
            if len(self.cache) > self.capacity:
                removed = self._remove_tail()
                del self.cache[removed.key]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    cache = LRUCache(2)

    cache.put(1, 1)
    cache.put(2, 2)
    print(f"get(1): {cache.get(1)}")  # 返回 1

    cache.put(3, 3)  # 淘汰 key=2
    print(f"get(2): {cache.get(2)}")  # 返回 -1 (未找到)

    cache.put(4, 4)  # 淘汰 key=1
    print(f"get(1): {cache.get(1)}")  # 返回 -1 (未找到)
    print(f"get(3): {cache.get(3)}")  # 返回 3
    print(f"get(4): {cache.get(4)}")  # 返回 4
