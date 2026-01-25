"""
LeetCode 94. 二叉树的中序遍历 (Binary Tree Inorder Traversal)
难度: 简单
链接: https://leetcode.cn/problems/binary-tree-inorder-traversal/

题目描述:
    给定一个二叉树的根节点 root，返回它的中序遍历。
    中序遍历: 左 -> 根 -> 右

示例:
    输入: root = [1,null,2,3]
    输出: [1,3,2]

思路分析:
    三种遍历方式:
        - 前序: 根 -> 左 -> 右
        - 中序: 左 -> 根 -> 右
        - 后序: 左 -> 右 -> 根

    方法1 - 递归:
        最直观，代码简洁

    方法2 - 迭代 (使用栈):
        显式使用栈模拟递归过程
        - 不断将左子节点入栈
        - 弹出时访问节点，然后转向右子树

    方法3 - Morris遍历:
        O(1)空间复杂度，利用叶子节点的空指针

复杂度分析:
    递归/迭代:
        时间复杂度: O(n)
        空间复杂度: O(n)
    Morris:
        时间复杂度: O(n)
        空间复杂度: O(1)

面试技巧:
    1. 递归和迭代两种方法都要会
    2. 中序遍历BST会得到有序序列
    3. Morris遍历是进阶技巧
"""

import sys
sys.path.append('..')
from _tree_node import TreeNode, create_tree
from typing import Optional, List


def inorderTraversal(root: Optional[TreeNode]) -> List[int]:
    """
    递归法
    """
    result = []

    def inorder(node):
        if not node:
            return
        inorder(node.left)      # 左
        result.append(node.val)  # 根
        inorder(node.right)     # 右

    inorder(root)
    return result


def inorderTraversal_iterative(root: Optional[TreeNode]) -> List[int]:
    """
    迭代法 (使用栈)

    核心: 一直往左走入栈，弹出时访问，然后转向右子树
    """
    result = []
    stack = []
    current = root

    while current or stack:
        # 一直往左走，将节点入栈
        while current:
            stack.append(current)
            current = current.left

        # 弹出节点，访问
        current = stack.pop()
        result.append(current.val)

        # 转向右子树
        current = current.right

    return result


def inorderTraversal_morris(root: Optional[TreeNode]) -> List[int]:
    """
    Morris遍历 - O(1)空间

    利用叶子节点的空指针指向后继节点
    """
    result = []
    current = root

    while current:
        if not current.left:
            # 没有左子树，访问当前节点，转向右子树
            result.append(current.val)
            current = current.right
        else:
            # 找到左子树的最右节点（前驱节点）
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            if not predecessor.right:
                # 建立连接
                predecessor.right = current
                current = current.left
            else:
                # 已访问过左子树，断开连接
                predecessor.right = None
                result.append(current.val)
                current = current.right

    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"values": [1, None, 2, 3], "expected": [1, 3, 2]},
        {"values": [], "expected": []},
        {"values": [1], "expected": [1]},
        {"values": [1, 2, 3, 4, 5], "expected": [4, 2, 5, 1, 3]},
    ]

    for i, tc in enumerate(test_cases):
        root = create_tree(tc["values"])
        result = inorderTraversal(root)
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} 输入: {tc['values']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
