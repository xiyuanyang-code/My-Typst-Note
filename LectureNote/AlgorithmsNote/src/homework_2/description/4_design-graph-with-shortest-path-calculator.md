# Design Graph With Shortest Path Calculator（设计可以求最短路径的图类）
There is a **directed weighted** graph that consists of `n` nodes numbered from `0` to `n - 1`. The edges of the graph are initially represented by the given array `edges` where `edges[i] = [fromi, toi, edgeCosti]` meaning that there is an edge from `fromi` to `toi` with the cost `edgeCosti`.

Implement the `Graph` class:

*   `Graph(int n, int[][] edges)` initializes the object with `n` nodes and the given edges.
*   `addEdge(int[] edge)` adds an edge to the list of edges where `edge = [from, to, edgeCost]`. It is guaranteed that there is no edge between the two nodes before adding this one.
*   `int shortestPath(int node1, int node2)` returns the **minimum** cost of a path from `node1` to `node2`. If no path exists, return `-1`. The cost of a path is the sum of the costs of the edges in the path.

**Example 1:**

![](https://assets.leetcode.com/uploads/2023/01/11/graph3drawio-2.png)

**Input**

\["Graph", "shortestPath", "shortestPath", "addEdge", "shortestPath"\]
\[\[4, \[\[0, 2, 5\], \[0, 1, 2\], \[1, 2, 1\], \[3, 0, 3\]\]\], \[3, 2\], \[0, 3\], \[\[1, 3, 4\]\], \[0, 3\]\]

**Output**

\[null, 6, -1, null, 6\]

**Explanation**

Graph g = new Graph(4, \[\[0, 2, 5\], \[0, 1, 2\], \[1, 2, 1\], \[3, 0, 3\]\]);

g.shortestPath(3, 2); // return 6. The shortest path from 3 to 2 in the first diagram above is 3 -> 0 -> 1 -> 2 with a total cost of 3 + 2 + 1 = 6.

g.shortestPath(0, 3); // return -1. There is no path from 0 to 3.

g.addEdge(\[1, 3, 4\]); // We add an edge from node 1 to node 3, and we get the second diagram above.

g.shortestPath(0, 3); // return 6. The shortest path from 0 to 3 now is 0 -> 1 -> 3 with a total cost of 2 + 4 = 6.

**Constraints:**

*   `1 <= n <= 100`
*   `0 <= edges.length <= n * (n - 1)`
*   `edges[i].length == edge.length == 3`
*   `0 <= fromi, toi, from, to, node1, node2 <= n - 1`
*   `1 <= edgeCosti, edgeCost <= 106`
*   There are no repeated edges and no self-loops in the graph at any point.
*   At most `100` calls will be made for `addEdge`.
*   At most `100` calls will be made for `shortestPath`.

---
给你一个有 `n` 个节点的 **有向带权** 图，节点编号为 `0` 到 `n - 1` 。图中的初始边用数组 `edges` 表示，其中 `edges[i] = [fromi, toi, edgeCosti]` 表示从 `fromi` 到 `toi` 有一条代价为 `edgeCosti` 的边。

请你实现一个 `Graph` 类：

*   `Graph(int n, int[][] edges)` 初始化图有 `n` 个节点，并输入初始边。
*   `addEdge(int[] edge)` 向边集中添加一条边，其中 `edge = [from, to, edgeCost]` 。数据保证添加这条边之前对应的两个节点之间没有有向边。
*   `int shortestPath(int node1, int node2)` 返回从节点 `node1` 到 `node2` 的路径 **最小** 代价。如果路径不存在，返回 `-1` 。一条路径的代价是路径中所有边代价之和。

**示例 1：**

![](https://assets.leetcode.com/uploads/2023/01/11/graph3drawio-2.png)

**输入：**

\["Graph", "shortestPath", "shortestPath", "addEdge", "shortestPath"\]
\[\[4, \[\[0, 2, 5\], \[0, 1, 2\], \[1, 2, 1\], \[3, 0, 3\]\]\], \[3, 2\], \[0, 3\], \[\[1, 3, 4\]\], \[0, 3\]\]

**输出：**

\[null, 6, -1, null, 6\]

**解释：**

Graph g = new Graph(4, \[\[0, 2, 5\], \[0, 1, 2\], \[1, 2, 1\], \[3, 0, 3\]\]);
g.shortestPath(3, 2); // 返回 6 。从 3 到 2 的最短路径如第一幅图所示：3 -> 0 -> 1 -> 2 ，总代价为 3 + 2 + 1 = 6 。
g.shortestPath(0, 3); // 返回 -1 。没有从 0 到 3 的路径。
g.addEdge(\[1, 3, 4\]); // 添加一条节点 1 到节点 3 的边，得到第二幅图。
g.shortestPath(0, 3); // 返回 6 。从 0 到 3 的最短路径为 0 -> 1 -> 3 ，总代价为 2 + 4 = 6 。

**提示：**

*   `1 <= n <= 100`
*   `0 <= edges.length <= n * (n - 1)`
*   `edges[i].length == edge.length == 3`
*   `0 <= fromi, toi, from, to, node1, node2 <= n - 1`
*   `1 <= edgeCosti, edgeCost <= 106`
*   图中任何时候都不会有重边和自环。
*   调用 `addEdge` 至多 `100` 次。
*   调用 `shortestPath` 至多 `100` 次。
