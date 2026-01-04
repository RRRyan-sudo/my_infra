"""
二叉树基础：节点结构、构建、遍历。
示例包括：
- 从层序数组构建树（None 表示空子节点）。
- 中序遍历（递归）。
- 层序遍历（BFS）。
"""
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def build_tree_level_order(values: Iterable[Optional[int]]) -> Optional[TreeNode]:
    """
    按层序（BFS）顺序的数组构建二叉树，None 代表该位置为空。
    例如 [1, 2, 3, None, 4] 表示：
          1
         / \
        2   3
         \
          4
    """
    values = list(values)
    if not values:
        return None

    iter_vals = iter(values)
    root_val = next(iter_vals)
    if root_val is None:
        return None
    root = TreeNode(root_val)
    queue: deque[TreeNode] = deque([root])

    for val_left, val_right in zip(iter_vals, iter_vals):
        parent = queue.popleft()
        if val_left is not None:
            parent.left = TreeNode(val_left)
            queue.append(parent.left)
        if val_right is not None:
            parent.right = TreeNode(val_right)
            queue.append(parent.right)

    return root


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """中序遍历（左-根-右）。递归实现易理解。"""
    if not root:
        return []
    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right)


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """层序遍历（BFS），逐层返回节点值。"""
    if not root:
        return []
    res: List[List[int]] = []
    queue: deque[TreeNode] = deque([root])

    while queue:
        level_size = len(queue)
        level_vals: List[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level_vals.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level_vals)
    return res


if __name__ == "__main__":
    # 用数组构建一棵简单的树
    data = [1, 2, 3, None, 4, 5, 6]
    root = build_tree_level_order(data)

    print("中序遍历结果:", inorder_traversal(root))
    print("层序遍历结果:", level_order(root))
