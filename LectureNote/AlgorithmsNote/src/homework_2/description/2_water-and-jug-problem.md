# Water and Jug Problem（水壶问题）
You are given two jugs with capacities `x` liters and `y` liters. You have an infinite water supply. Return whether the total amount of water in both jugs may reach `target` using the following operations:

*   Fill either jug completely with water.
*   Completely empty either jug.
*   Pour water from one jug into another until the receiving jug is full, or the transferring jug is empty.

**Example 1:**

**Input:** x = 3, y = 5, target = 4

**Output:** true

**Explanation:**

Follow these steps to reach a total of 4 liters:

1.  Fill the 5-liter jug (0, 5).
2.  Pour from the 5-liter jug into the 3-liter jug, leaving 2 liters (3, 2).
3.  Empty the 3-liter jug (0, 2).
4.  Transfer the 2 liters from the 5-liter jug to the 3-liter jug (2, 0).
5.  Fill the 5-liter jug again (2, 5).
6.  Pour from the 5-liter jug into the 3-liter jug until the 3-liter jug is full. This leaves 4 liters in the 5-liter jug (3, 4).
7.  Empty the 3-liter jug. Now, you have exactly 4 liters in the 5-liter jug (0, 4).

Reference: The [Die Hard](https://www.youtube.com/watch?v=BVtQNK_ZUJg&ab_channel=notnek01) example.

**Example 2:**

**Input:** x = 2, y = 6, target = 5

**Output:** false

**Example 3:**

**Input:** x = 1, y = 2, target = 3

**Output:** true

**Explanation:** Fill both jugs. The total amount of water in both jugs is equal to 3 now.

**Constraints:**

*   `1 <= x, y, target <= 103`

---
有两个水壶，容量分别为 `x` 和 `y` 升。水的供应是无限的。确定是否有可能使用这两个壶准确得到 `target` 升。

你可以：

*   装满任意一个水壶
*   清空任意一个水壶
*   将水从一个水壶倒入另一个水壶，直到接水壶已满，或倒水壶已空。

**示例 1:** 

**输入:** x = 3,y = 5,target = 4
**输出:** true
**解释：**
按照以下步骤操作，以达到总共 4 升水：
1. 装满 5 升的水壶(0, 5)。
2. 把 5 升的水壶倒进 3 升的水壶，留下 2 升(3, 2)。
3. 倒空 3 升的水壶(0, 2)。
4. 把 2 升水从 5 升的水壶转移到 3 升的水壶(2, 0)。
5. 再次加满 5 升的水壶(2, 5)。
6. 从 5 升的水壶向 3 升的水壶倒水直到 3 升的水壶倒满。5 升的水壶里留下了 4 升水(3, 4)。
7. 倒空 3 升的水壶。现在，5 升的水壶里正好有 4 升水(0, 4)。

**示例 2:**

**输入:** x = 2, y = 6, target = 5
**输出:** false

**示例 3:**

**输入:** x = 1, y = 2, target = 3
**输出:** true
**解释：**同时倒满两个水壶。现在两个水壶中水的总量等于 3。

**提示:**

*   `1 <= x, y, target <= 103`
