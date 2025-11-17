# Task

## Code Base Generations

- Graph definitions
- BFS & DFS simple algorithms

## Task1: Shortest Path and Connected Components

### Task1.1 Finding the shortest path

- 基本的 BFS 过程
- 需要维护 parent 字典，方便后续构建 BFS 树

### Task1.2 Connected Components

Running full-BFS (or DFS)

## Task2: Topological Sort

### Task2.1 Running Topological Sort

#### Method1: Using BFS

- 找到入度为 0 的节点
- 跑 BFS 并对应的减少后继节点的入度
- 更新新的入度为 0 的节点
- 环的判断：如果 queue 为空的时候（循环中断），但是此时仍有节点未被访问

#### Method2: Using DFS

- 构建依赖图（每一个图指向的节点是其**依赖项**）
- 跑递归 DFS 当所有的依赖项已经入栈之后当前节点入栈，保证拓扑性。
- Running Full DFS, 保证所有节点全部被访问