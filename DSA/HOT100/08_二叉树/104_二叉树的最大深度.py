"""
LeetCode 104. 二叉树的最大深度 (Maximum Depth of Binary Tree)
难度: 简单
链接: https://leetcode.cn/problems/maximum-depth-of-binary-tree/

题目描述:
    给定一个二叉树 root，返回其最大深度。
    最大深度是指从根节点到最远叶子节点的最长路径上的节点数。

示例:
    输入: root = [3,9,20,null,null,15,7]
    输出: 3

思路分析:
    方法1 - DFS递归:
        maxDepth(root) = 1 + max(maxDepth(left), maxDepth(right))

    方法2 - BFS:
        层序遍历，统计层数

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(n) - 最坏情况（链状）

面试技巧:
    1. DFS递归是最简洁的写法
    2. 这是很多树题的基础
"""

import sys
sys.path.append('..')
from _tree_node import TreeNode, create_tree
from typing import Optional
from collections import deque


def maxDepth(root: Optional[TreeNode]) -> int:
    """
    DFS递归

    深度 = 1 + max(左子树深度, 右子树深度)
    """
    if not root:
        return 0

    return 1 + max(maxDepth(root.left), maxDepth(root.right))


def maxDepth_bfs(root: Optional[TreeNode]) -> int:
    """BFS层序遍历"""
    if not root:
        return 0

    depth = 0
    queue = deque([root])

    while queue:
        depth += 1
        # 遍历当前层
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return depth


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"values": [3, 9, 20, None, None, 15, 7], "expected": 3},
        {"values": [1, None, 2], "expected": 2},
        {"values": [], "expected": 0},
    ]

    for i, tc in enumerate(test_cases):
        root = create_tree(tc["values"])
        result = maxDepth(root)
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} 输入: {tc['values']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
