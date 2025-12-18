"""
AI1804 算法设计与分析 - 第7次课上练习
问题7-2：招聘会最优化（Career Fair Optimization）

请完成动态规划算法的实现
"""


def career_fair_optimization(coolness, weights, times, b, h, k):
    """
    返回:
        max_coolness: 最大总酷炫值（整数）

    时间复杂度目标: O(n*b*k)
    """
    n = len(coolness)
    dp = [[0] * (b + 1) for _ in range(k + 1)]
    for t in range(k + 1):
        for c in range(1, b + 1):
            for i in range(n):
                if c - weights[i] >= 0 and t > times[i]:
                    dp[t][c] = max(
                        dp[t][c], dp[t - times[i] - 1][c - weights[i]] + coolness[i]
                    )
                if t > h + 1 + times[i] and b >= weights[i]:
                    dp[t][c] = max(
                        dp[t][c],
                        dp[t - h - 1 - times[i] - 1][b - weights[i]] + coolness[i],
                    )
    return dp[k][b]


if __name__ == "__main__":
    print("=" * 60)
    print("问题7-2：招聘会最优化（Career Fair Optimization）")
    print("=" * 60)

    cool = [10, 7]
    w = [3, 2]
    t = [2, 1]
    b = 4
    h = 3
    k = 8
    result = career_fair_optimization(cool, w, t, b, h, k)
    print(f"coolness={cool}, weights={w}, times={t}, b={b}, h={h}, k={k}")
    print(f"max_coolness={result}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
