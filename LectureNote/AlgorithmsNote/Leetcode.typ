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

== Classical Problems

=== T72 Edit Distance

==== Problems

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

==== 1-based 的情况下初始化的好处

在 1-based 的情况下，初始化很简单，填满 i = 0 和 j = 0 的两条边，因为这也对应着空字符串的情况

==== 状态转移方程

考虑计算 `dp[i][j]`，我们首先考虑从 `dp[i-1][j-1]` 的状态转移。

- 如果新加入的两个字符是相同的，那不需要做任何操作，并且反证法可以证明这是最优解。

- 如果新加入的两个字符不同，此时就需要状态转移，为了保证最优，我们的操作都对于新加入的字符串操作：
  - 这样简化问题保证正确性，因为这实际上是一个迭代的过程，最优的任何步骤都会在对应的步长时发生在末尾。
  - 在末尾插入：需要考虑 `dp[i-1][j]`
  - 在末尾删除：需要考虑 `dp[i][j-1]`
  - 在末尾替换：需要考虑 `dp[i-1][j-1]`
  - 最后去 min 加 1

==== Results

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

=== T121 Best Time to Buy Stocks

#problem("T121")[
  You are given an array prices where prices`[i]` is the price of a given stock on the ith day.

  You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

  Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
]

==== Method1 Using Dynamic Programming

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



==== Method2 Better Algorithms

= Conclusion
