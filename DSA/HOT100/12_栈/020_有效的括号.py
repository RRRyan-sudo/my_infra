"""
LeetCode 20. 有效的括号 (Valid Parentheses)
难度: 简单
链接: https://leetcode.cn/problems/valid-parentheses/

题目描述:
    给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s，判断字符串是否有效。
    有效字符串需满足:
    1. 左括号必须用相同类型的右括号闭合
    2. 左括号必须以正确的顺序闭合
    3. 每个右括号都有一个对应的相同类型的左括号

示例:
    输入: s = "()[]{}"
    输出: true

    输入: s = "(]"
    输出: false

思路分析:
    栈:
        - 遇到左括号，入栈
        - 遇到右括号，检查栈顶是否匹配
            - 匹配则出栈
            - 不匹配则返回false
        - 最后栈应该为空

复杂度分析:
    时间复杂度: O(n)
    空间复杂度: O(n)

面试技巧:
    1. 栈的经典应用
    2. 注意边界: 空字符串、只有右括号
"""


def isValid(s: str) -> bool:
    """
    栈匹配

    左括号入栈，右括号检查栈顶
    """
    stack = []
    # 括号映射: 右括号 -> 左括号
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # 右括号: 检查栈顶
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # 左括号: 入栈
            stack.append(char)

    # 栈应该为空
    return len(stack) == 0


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"s": "()", "expected": True},
        {"s": "()[]{}", "expected": True},
        {"s": "(]", "expected": False},
        {"s": "([)]", "expected": False},
        {"s": "{[]}", "expected": True},
        {"s": "]", "expected": False},
    ]

    for i, tc in enumerate(test_cases):
        result = isValid(tc["s"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} s=\"{tc['s']}\"")
        print(f"       输出: {result}, 期望: {tc['expected']}")
