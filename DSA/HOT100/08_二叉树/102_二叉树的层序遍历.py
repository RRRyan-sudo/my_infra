"""
LeetCode 102. 二叉树的层序遍历 (Binary Tree Level Order Traversal)
难度: 中等
链接: https://leetcode.cn/problems/binary-tree-level-order-traversal/

题目描述:
    给你二叉树的根节点 root，返回其节点值的层序遍历。
    （即逐层地，从左到右访问所有节点）

示例:
    输入: root = [3,9,20,null,null,15,7]
    输出: [[3],[9,20],[15,7]]

思路分析:
    BFS (广度优先搜索):
        使用队列实现层序遍历
        关键: 每次处理一整层，记录当前层的节点数

    也可以用DFS，但BFS更直观

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(n)

面试技巧:
    1. 层序遍历是BFS的基础应用
    2. 关键是"每次处理一层"的技巧
    3. 变体: 之字形遍历、右视图等
"""

import sys
sys.path.append('..')
from _tree_node import TreeNode, create_tree
from typing import Optional, List
from collections import deque


def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS层序遍历

    每次处理一整层
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        level_size = len(queue)  # 当前层的节点数

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


def levelOrder_dfs(root: Optional[TreeNode]) -> List[List[int]]:
    """
    DFS实现层序遍历

    传递depth参数，将节点放入对应层
    """
    result = []

    def dfs(node, depth):
        if not node:
            return

        # 如果是新的一层，添加空列表
        if depth == len(result):
            result.append([])

        result[depth].append(node.val)
        dfs(node.left, depth + 1)
        dfs(node.right, depth + 1)

    dfs(root, 0)
    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"values": [3, 9, 20, None, None, 15, 7], "expected": [[3], [9, 20], [15, 7]]},
        {"values": [1], "expected": [[1]]},
        {"values": [], "expected": []},
    ]

    for i, tc in enumerate(test_cases):
        root = create_tree(tc["values"])
        result = levelOrder(root)
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} 输入: {tc['values']}")
        print(f"       输出: {result}")
        print(f"       期望: {tc['expected']}")
