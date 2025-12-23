from typing import List, Tuple

def coin_crafting(n: int, prices: List[int], melts: List[int]) -> Tuple[int, List[int]]:
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    for k in range(1, n + 1):
        price = prices[k-1]
        melt = melts[k-1]
        for i in range(1, n + 1):
            dp[k][i] = dp[k-1][i]
            if i >= melt:
                if dp[k-1][i-melt] + price > dp[k-1][i]:
                    dp[k][i] = dp[k-1][i-melt] + price


    # back tracking
    res_items = []
    current_capacity = n
    for k in range(n, 0, -1):
        if dp[k][current_capacity] != dp[k-1][current_capacity]:
            res_items.append(k)
            current_capacity -= melts[k-1]

    return dp[n][n], res_items[::]

if __name__ == "__main__":
    n = int(input().strip())
    prices = [int(x.strip()) for x in input().split(",") if x.strip()]
    melts = [int(x.strip()) for x in input().split(",") if x.strip()]
    
    max_revenue, chosen_items = coin_crafting(n, prices, melts)
    print(max_revenue)
    print(",".join(map(str, chosen_items)))