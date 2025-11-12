#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "LeetCode Memo",
  author: "Xiyuan Yang",
  abstract: [Only programmers who do a LeetCode Hard problem every day deserve to be called programmers :)],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

Recordings for my LeetCode Journey.

== Table of Contents

#figure(
  table(
    columns: (2.5cm, 14cm),
    [2025/09/17], [重新开始 LeetCode 之旅，配置好在 Vscode Ubuntu 中刷 LeetCode 的环境],
    [2025/09/17], [开始算法第一个章节：动态规划刷题]
  ),
)

= Dynamic Programming


== T72 Edit Distance

=== Problems

- Url: https://leetcode.com/problems/edit-distance/

#problem("T72")[
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

- Insert a character
- Delete a character
- Replace a character
 

- Example 1:
  ```
  Input: word1 = "horse", word2 = "ros"
  Output: 3
  Explanation: 
  horse -> rorse (replace 'h' with 'r')
  rorse -> rose (remove 'r')
  rose -> ros (remove 'e')
  ```

- Example 2:

  ```
  Input: word1 = "intention", word2 = "execution"
  Output: 5
  Explanation: 
  intention -> inention (remove 't')
  inention -> enention (replace 'i' with 'e')
  enention -> exention (replace 'n' with 'x')
  exention -> exection (replace 'n' with 'c')
  exection -> execution (insert 'u')
  ```
 

- Constraints:
  - 0 <= word1.length, word2.length <= 500
  - word1 and word2 consist of lowercase English letters.
]

==== 前缀数组的考虑

#recordings("前缀数组")[
  - 动态规划的核心在于找到合适的子问题结构并且构建不同参数的子问题之下的动态联系（从图论的角度就是构建一个有向无环图）
  - 对于最常见的情况，问题可以被建模为一个数组，尤其是本身的数组问题或者字符串问题，可以考虑前缀数组或者后缀数组作为结构。
    - 因为这样的结构本身包含父子的包含关系，在一些约束中具有良好的结构，可以轻松的写出状态转移方程。
]

考虑构建前缀和数组，很显然，这里至少需要构造一个二维数组。下面的核心就是 define 好子结构的具体意义是什么？即 `dp[i][j]` 是什么意思。

这里考虑 `dp[i][j]` 代表着 对于字符串 word1 的前 i 个字符的 slice 和字符串 word2 的前 j 个字符的 slice，最少的操作次数是多少。因此创建二维数组的时候长度会比原先字符串的长度多 1 个，可以巧妙的避免空字符串的特判情况。

定义好子结构后，接下来就需要解决两个问题：

- 状态转移方程？
- 初始化？

=== 1-based 的情况下初始化的好处

在 1-based 的情况下，初始化很简单，填满 i = 0 和 j = 0 的两条边，因为这也对应着空字符串的情况

=== 状态转移方程

考虑计算 `dp[i][j]`，我们首先考虑从 `dp[i-1][j-1]` 的状态转移。

- 如果新加入的两个字符是相同的，那不需要做任何操作，并且反证法可以证明这是最优解。

- 如果新加入的两个字符不同，此时就需要状态转移，为了保证最优，我们的操作都对于新加入的字符串操作：
  - 这样简化问题保证正确性，因为这实际上是一个迭代的过程，最优的任何步骤都会在对应的步长时发生在末尾。
  - 在末尾插入：需要考虑 `dp[i-1][j]`
  - 在末尾删除：需要考虑 `dp[i][j-1]`
  - 在末尾替换：需要考虑 `dp[i-1][j-1]`
  - 最后去 min 加 1

=== Results

```cpp
#include <cmath>
#include <string>
#include <vector>
class Solution {
public:
    int minDistance(std::string word1, std::string word2) {
        int word_length_1 = word1.length();
        int word_length_2 = word2.length();

        std::vector<std::vector<int>> dp(word_length_1 + 1, std::vector<int>(word_length_2 + 1));

        // initialize
        for (int i = 0; i <= word_length_1; i++) {
            dp[i][0] = i;
        }

        for (int j = 0; j <= word_length_2; j++) {
            dp[0][j] = j;
        }

        if (word_length_1 == 0 && word_length_2 == 0) {
            return dp[word_length_1][word_length_2];
        }


        // start dp!
        for (int i = 1; i <= word_length_1; i++) {
            for (int j = 1; j <= word_length_2; j++) {

                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + std::min(dp[i - 1][j], std::min(dp[i][j - 1], dp[i - 1][j - 1]));
                }
            }
        }

        return dp[word_length_1][word_length_2];
    }
};
```

== T121 Best Time to Buy Stocks

#problem("T121")[
  You are given an array prices where prices`[i]` is the price of a given stock on the ith day.

  You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

  Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
]

=== Method1 Using Dynamic Programming

为了简化问题和状态转移方程，如果能给 dp 数组降维就需要降维。同样，对于离散的状态标记，也可以增加一个维度来存储，这不会提高时间复杂度。

考虑 `L[i][j]`:
- i 代表天数：`i = 0,1,2,...,n-1`
- j 代表状态，即当天是否持有股票（因为只允许交易一次股票，即一次买入和一次卖出）
- 这个值代表该天持有或者不持有股票可能会产生的最大利润。
- 最终状态，在最终，我们肯定是需要卖掉股票，这样才可以回本，因此我们需要计算 `L[n-1][0]`

状态转移方程：

- 如果在第 i 天持有：
  - `dp[i][1] = max(dp[i-1][1], -prices[i])`
- 如果在第 i 天不持有：
  - `dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])`


== T152 Max Product Sum SubArray

Given an integer `array` nums, find a subarray that has the largest product, and return the product.

=== Solution1 Simple Definition with $O(N^2)$.

定义如下的状态转移方程：

- `dp[i]` means the max product sum sub-arrays for `array[0:i+1]`
- All we want to solve is `dp[len(array)-1]`
- 对于最大最小值的最优解问题，可以分情况做分类讨论
  - 如果新加入的元素在对应的最大序列中，因为连续性可以保证一次遍历就可以找到当前情况的最大值
  - 如果不在，退化复用过去的子问题
- Time Complexity: $O(N^2)$

Aha, TLE.

```python
#
# @lc app=leetcode id=152 lang=python3
#
# [152] Maximum Product Subarray
#

# @lc code=start
from typing import List
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        length_nums = len(nums)
        if length_nums == 1:
            return nums[0]
        dp = [None] * (length_nums)
        dp[0] = nums[0]
        for i in range(1, length_nums):
            # find the current maximum
            current_maximum = nums[i]
            current_value = nums[i]
            for j in range(i-1,-1,-1):
                current_value *= nums[j]
                current_maximum = max(current_maximum, current_value)
            dp[i] = max(dp[i-1], current_maximum)
        return dp[length_nums-1]
        
# @lc code=end
```

=== Solution 2 Optimization for Linear Time Complexity

考虑简单的情况：如果数组元素全部是正整数，那最大的肯定是全部的数组，因为乘积会越乘越大。
- 考虑加入 0 的情况，则当前子数组乘积变为 0
- 考虑加入负数的情况，会导致正负翻转，当前的最小乘积可能在之后就变成了最大乘积

因此，我们需要在转移的过程中维护更多的信息，这样可以提高记忆化的效率，防止内部循环再来一次线性遍历。

#definition[
  - `max_ending[i]`: The maximum product for all the sub-arrays ending with index i.
  - `min_ending[i]`: The minimum product for all the sub-arrays ending with index i.

  在第一种算法中，加入新元素后采用粗暴的遍历方法，但是经过分析之后我们发现，我们可以 $O(1)$ 直接分析出最大值。这就是时间复杂度优化的关键。
]

```py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        length_nums = len(nums)
        if length_nums == 1:
            return nums[0]
        max_end = [None] * length_nums
        min_end = [None] * length_nums
        max_end[0] = nums[0]
        min_end[0] = nums[0]
        for i in range(1, length_nums):
            if nums[i] > 0:
                max_end[i] = max(max_end[i-1] * nums[i], nums[i])
                min_end[i] = min(min_end[i-1] * nums[i], nums[i])
            elif nums[i] == 0:
                max_end[i] = 0
                min_end[i] = 0
            else:
                max_end[i] = max(min_end[i-1] * nums[i], nums[i])
                min_end[i] = min(max_end[i-1] * nums[i], nums[i])
        return max(max_end)
```


= Divide and Conquer

#recordings("Divide and Conquer")[
  - 可以通过需要达到的时间复杂度倒退主定理递推式，再分析具体如何做递归
  - 有时候数组的切分是比较重要的，但是有时候数据的合并是比较重要的
  - 和动态规划类似，核心思想是在递归中不断的有效减小子问题的规模，直到达到基准问题，进而实现复杂度的优化。
]

== Median of two sorted arrays

#example("Median of two sorted arrays")[
  Given two sorted arrays, return the median number for the merged sorted arrays.
]

For the most simple intuition, we can merge the two arrays with time complexity $O(m+n)$, then find the median number in constant time. Total time complexity: $O(m+n)$.

We can optimize this problem using divide and conquer.

#recordings("Median of the two sorted arrays")[
  - 更加一般的子问题：寻找两个数组合并结果的第k小元素
    - 寻找中位数只是这种情况的特殊情况
  - 如何求解这个更加一般的子问题
    - 我们希望通过原先数组的有序性，来避免遍历查找，而是尽可能访问更少的数组元素。
]

因此，每个数组选择 $floor k/2 floor.r$ 的元素，通过比较最后一个数组元素，就可以排除掉 $floor k/2 floor.r$ 的数组元素。

时间复杂度分析：

$
  T(n) = T(n/2) + O(1)
$

$
  "Time Complexity": O(log(m+n))
$

```python
#
# @lc app=leetcode id=4 lang=python3
#
# [4] Median of Two Sorted Arrays
#

# @lc code=start
from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        m = len(nums1)
        n = len(nums2)
        total_len = m + n

        if total_len % 2 == 1:
            k = total_len // 2 + 1
            return self._findKth(nums1, 0, nums2, 0, k)
        else:
            k1 = total_len // 2
            k2 = total_len // 2 + 1
            median1 = self._findKth(nums1, 0, nums2, 0, k1)
            median2 = self._findKth(nums1, 0, nums2, 0, k2)
            return (median1 + median2) / 2.0

    def _findKth(self, nums1: list[int], i: int, nums2: list[int], j: int, k: int) -> int:
        # i and j means the starting index
        # len1 and len2 are the length of the sub arrays
        len1 = len(nums1) - i
        len2 = len(nums2) - j

        if len1 == 0:
            return nums2[j + k - 1]
        if len2 == 0:
            return nums1[i + k - 1]

        if k == 1:
            return min(nums1[i], nums2[j])

        half_k = k // 2
        
        # 确定要比较的元素索引
        idx1 = i + min(len1, half_k) - 1
        idx2 = j + min(len2, half_k) - 1
        val1 = nums1[idx1]
        val2 = nums2[idx2]

        if val1 <= val2:
            # 排除 nums1 中的 [i, idx1] 部分
            new_k = k - (idx1 - i + 1)
            return self._findKth(nums1, idx1 + 1, nums2, j, new_k)
        else: # val2 < val1
            new_k = k - (idx2 - j + 1)
            return self._findKth(nums1, i, nums2, idx2 + 1, new_k)

# @lc code=end

```

== K 路归并问题

#recordings[
  K 路归并的堆优化操作是非常经典的算法题目！
]

基本思路：

- 在数据量非常多的情况下，我们希望能够实现归并是尽可能少的比较元素就找到全局最大元素
  - 不考虑极端情况（考虑 K 路比较均衡的情况），K 路归并的优势是：
    - 可以实现流式输入（在线算法、实时更新）
    - 可以达到更快的时间复杂度
- 基本思想：*维护堆结构来找到局部最大和全局最大*
  - 对每一个分路维护堆结构，找到局部最大的元素
  - 局部最大元素组成一个堆，从而可以实现在 $O(log K)$ 的时间复杂度下找到全局最大值。

=== Basic 链表格式的 K 路归并

```python
#
# @lc app=leetcode id=23 lang=python3
#
# [23] Merge k Sorted Lists
#

import heapq
from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        min_heap = []
        counter = 0 
        
        for head in lists:
            if head:
                heapq.heappush(min_heap, (head.val, counter, head))
                counter += 1
        

        dummy = ListNode(0)
        current = dummy
        
        while min_heap:
            val, _, node = heapq.heappop(min_heap)
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(min_heap, (node.next.val, counter, node.next))
                counter += 1
                
        return dummy.next

# @lc code=end

```

=== Advanced 去重版的集合 K 路归并

#figure(
  image("images/kmerge.png")
)

```python
import heapq

def merge_k_unsorted_lists_top_n(lists, n):
    if not lists or n <= 0:
        return []
    
    negated_lists = []
    # 建队 O（N）
    for list_index, list in enumerate(lists):
        if list:
            negated_list = [(-b,a) for a,b in list]
            heapq.heapify(negated_list)
            negated_lists.append(negated_list)
        else:
            negated_lists.append([])

    # 初始化全局最小堆 O（N log N）
    # * 最坏情况 K 路分布不均匀，因此内部循环+弹出的时间复杂度最坏可能达到 N log N
    global_dict = {}
    global_heap = []
    heapq.heapify(global_heap)
    for list_index in range(len(lists)):
        while negated_lists[list_index]:
            new_element, new_id = heapq.heappop(negated_lists[list_index])
            if new_id not in global_dict or (-new_element) > global_dict[new_id]:
                global_dict[new_id] = -new_element
                heapq.heappush(global_heap, (new_element, new_id, list_index))
                break

    result = []
    # 寻找新的最大元素 O（n log N）
    while global_heap and len(result) < n:
        global_max, global_max_item_id, list_index = heapq.heappop(global_heap)
        result.append((global_max_item_id,-global_max))

        # inserting new elements into it
        if negated_lists[list_index]:
            new_element, new_id = heapq.heappop(negated_lists[list_index])
            if new_id not in global_dict or (-new_element) > global_dict[new_id]:
                global_dict[new_id] = -new_element
                heapq.heappush(global_heap, (new_element, new_id, list_index))
   
    return result

    # 总时间复杂度：O（N log N）
```

= Conclusion
