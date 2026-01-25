"""
LeetCode 236. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree)
难度: 中等
链接: https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/

题目描述:
    给定一个二叉树, 找到该树中两个指定节点的最近公共祖先(LCA)。
    最近公共祖先: 对于有根树T的两个节点p、q，最近公共祖先表示为一个节点x，
    满足x是p、q的祖先且x的深度尽可能大。

示例:
    输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
    输出: 3

思路分析:
    递归后序遍历:
        对于每个节点:
            - 如果是null，返回null
            - 如果是p或q，返回自己
            - 递归查找左右子树

        返回值的含义:
            - 返回null: 该子树不包含p和q
            - 返回节点: 该子树包含p或q（或两者的LCA）

        结果判断:
            - 如果左右都不为null，当前节点就是LCA
            - 如果只有一边不为null，返回那一边的结果

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(n) - 递归栈

面试技巧:
    1. 理解递归返回值的含义是关键
    2. 后序遍历保证先处理子树
    3. 变体: BST的LCA可以利用BST性质优化
"""

import sys
sys.path.append('..')
from _tree_node import TreeNode, create_tree


def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    递归后序遍历

    返回值:
        - null: 该子树不包含p和q
        - 节点: 该子树包含p或q（或两者的LCA）
    """
    # 基本情况
    if not root or root == p or root == q:
        return root

    # 递归查找左右子树
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    # 判断结果
    if left and right:
        # p和q分别在左右子树，当前节点就是LCA
        return root
    elif left:
        # p和q都在左子树
        return left
    else:
        # p和q都在右子树（或都不存在）
        return right


# ==================== 测试代码 ====================
if __name__ == "__main__":
    #       3
    #      / \
    #     5   1
    #    / \ / \
    #   6  2 0  8
    #     / \
    #    7   4
    root = create_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p = root.left  # 节点5
    q = root.right  # 节点1

    lca = lowestCommonAncestor(root, p, q)
    print(f"p={p.val}, q={q.val} 的LCA: {lca.val}")
    print(f"期望: 3")

    # 测试2: p=5, q=4 (4是5的后代)
    q2 = root.left.right.right  # 节点4
    lca2 = lowestCommonAncestor(root, p, q2)
    print(f"p={p.val}, q={q2.val} 的LCA: {lca2.val}")
    print(f"期望: 5")
