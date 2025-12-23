def career_fair_optimization(coolness, weights, times, b, h, k):
    n = len(coolness)
    dp = [[0] * (b + 1) for _ in range(k + 1)]

    for i in range(1, k + 1):
        for j in range(b + 1):
            for m in range(n):
                cost_t = times[m] + 1
                cost_w = weights[m]
                if i >= cost_t and j >= cost_w:
                    dp[i][j] = max(dp[i][j], coolness[m] + dp[i - cost_t][j - cost_w])
            
            cost_h = h + 1
            if i >= cost_h:
                dp[i][j] = max(dp[i][j], dp[i - cost_h][b])
                
            dp[i][j] = max(dp[i][j], dp[i-1][j])

    return dp[k][b]


if __name__ == "__main__":
    n = int(input())
    cool = [int(x.strip()) for x in input().split(",") if x.strip()]
    w = [int(x.strip()) for x in input().split(",") if x.strip()]
    t = [int(x.strip()) for x in input().split(",") if x.strip()]
    b = int(input())
    h = int(input())
    k = int(input())
    result = career_fair_optimization(cool, w, t, b, h, k)
    print(result)