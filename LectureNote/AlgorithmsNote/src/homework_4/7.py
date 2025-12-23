from typing import List


def get_min_height(n: int, h: List[int], k: int):
    dp = [[[float('inf')] * (n + 1) for _ in range(n + 1)] for _ in range(k + 1)]
    
    for egg in range(1, k + 1):
        for i in range(1, n + 1):
            dp[egg][i][i] = h[i-1]

            for j in range(i):
                dp[egg][i][j] = 0

    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i-1] + h[i-1]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            dp[1][i][j] = prefix_sum[j] - prefix_sum[i-1]

    for e in range(2, k + 1):
        for length in range(2, n + 1):
            for i in range(1, n - length + 2):
                j = i + length - 1

                for x in range(i, j + 1):
                    broken = dp[e-1][i][x-1] if x > i else 0
                    not_broken = dp[e][x+1][j] if x < j else 0
                    
                    current_worst = h[x-1] + max(broken, not_broken)
                    
                    dp[e][i][j] = min(dp[e][i][j], current_worst)

    return dp[k][1][n]

if __name__ == "__main__":
    n = int(input())
    h = [int(x.strip()) for x in input().strip().split(",") if x.strip()]
    k = int(input())
    print(get_min_height(n,h,k))