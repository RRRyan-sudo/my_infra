"""
LeetCode 207. 课程表 (Course Schedule)
难度: 中等
链接: https://leetcode.cn/problems/course-schedule/

题目描述:
    你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses-1。
    在选修某些课程之前需要一些先修课程。先修课程按数组 prerequisites 给出，
    其中 prerequisites[i] = [ai, bi]，表示如果要学习课程 ai 则必须先学习课程 bi。

    请你判断是否可能完成所有课程的学习？

示例:
    输入: numCourses = 2, prerequisites = [[1,0]]
    输出: true
    解释: 总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。所以可能。

    输入: numCourses = 2, prerequisites = [[1,0],[0,1]]
    输出: false
    解释: 形成环，不可能完成

思路分析:
    本质是检测有向图是否有环，可以用拓扑排序

    方法1 - BFS (Kahn算法):
        1. 计算每个节点的入度
        2. 将入度为0的节点入队
        3. 每次取出一个节点，将其邻居的入度-1
        4. 如果邻居入度变为0，入队
        5. 最后检查是否所有节点都被处理

    方法2 - DFS:
        使用三种状态: 未访问、访问中、已完成
        如果访问到"访问中"的节点，说明有环

复杂度分析:
    时间复杂度: O(V + E) - 节点数 + 边数
    空间复杂度: O(V + E)

面试技巧:
    1. 拓扑排序是图论的经典算法
    2. 理解入度的概念
    3. 进阶: 课程表II返回一个可行的学习顺序
"""

from typing import List
from collections import deque, defaultdict


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    BFS拓扑排序 (Kahn算法)

    核心: 不断移除入度为0的节点
    """
    # 构建邻接表和入度数组
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # 将入度为0的节点入队
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0  # 已处理的节点数

    while queue:
        node = queue.popleft()
        count += 1

        # 将邻居的入度-1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 如果所有节点都被处理，说明无环
    return count == numCourses


def canFinish_dfs(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    DFS检测环

    三种状态: 0=未访问, 1=访问中, 2=已完成
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 0: 未访问, 1: 访问中, 2: 已完成
    state = [0] * numCourses

    def has_cycle(node):
        if state[node] == 1:
            return True  # 访问到正在访问的节点，有环
        if state[node] == 2:
            return False  # 已完成，无需再访问

        state[node] = 1  # 标记为访问中

        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True

        state[node] = 2  # 标记为已完成
        return False

    for i in range(numCourses):
        if has_cycle(i):
            return False

    return True


# ==================== 测试代码 ====================
if __name__ == "__main__":
    test_cases = [
        {"numCourses": 2, "prerequisites": [[1, 0]], "expected": True},
        {"numCourses": 2, "prerequisites": [[1, 0], [0, 1]], "expected": False},
        {"numCourses": 3, "prerequisites": [[1, 0], [2, 1]], "expected": True},
    ]

    for i, tc in enumerate(test_cases):
        result = canFinish(tc["numCourses"], tc["prerequisites"])
        status = "✓" if result == tc["expected"] else "✗"
        print(f"测试{i+1}: {status} numCourses={tc['numCourses']}, prerequisites={tc['prerequisites']}")
        print(f"       输出: {result}, 期望: {tc['expected']}")
