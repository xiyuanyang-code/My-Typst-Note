"""
AI1804 算法设计与分析 - 第7次课上练习
问题7-1：铸币问题

请完成动态规划算法的实现
"""


def coin_crafting(n, prices, melts):
    """
    返回:
        (max_revenue, chosen_items)

    时间复杂度目标: O(n^2)
    """
    dp = [[0] * (n + 1) for _ in range(n)]
    take = [[False] * (n + 1) for _ in range(n)]
    for k in range(n):
        for i in range(1, n + 1):
            dp[k][i] = max(dp[k][i], dp[k - 1][i])
            if i - melts[k] >= 0:
                dp[k][i] = max(dp[k][i], dp[k - 1][i - melts[k]] + prices[k])
                take[k][i] = True

    res_items = []
    i = n
    for k in range(n-1, 0, -1):
        if take[k][i]:
            res_items.append(k+1)  # 物品编号
            i -= melts[k]

    return dp[n - 1][n], res_items


if __name__ == "__main__":
    print("=" * 60)
    print("问题7-1：铸币问题")
    print("=" * 60)

    n = 5
    prices = [6, 3, 5, 4, 6]
    melts = [4, 2, 3, 1, 4]
    result = coin_crafting(n, prices, melts)
    print(f"n={n}")
    print(f"prices={prices}")
    print(f"melts={melts}")
    print(f"result={result}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
