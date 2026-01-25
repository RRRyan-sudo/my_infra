"""
二叉树通用定义和辅助函数
"""

from typing import Optional, List
from collections import deque


class TreeNode:
    """二叉树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


def create_tree(values: List[Optional[int]]) -> Optional[TreeNode]:
    """
    从列表创建二叉树 (层序表示)
    values = [1, 2, 3, None, 4] 表示:
          1
         / \
        2   3
         \
          4
    """
    if not values:
        return None

    root = TreeNode(values[0])
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        # 左子节点
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        # 右子节点
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """将二叉树转换为列表 (层序)"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)

    # 去除尾部的None
    while result and result[-1] is None:
        result.pop()

    return result
