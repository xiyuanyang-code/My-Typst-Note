from typing import List


def get_max_value(total: str, k: int, values: List[str]):
    length = len(total)
    dp = [0] * (length + 1)
    for i in range(1, length + 1):
        # case1:
        if total[i - 1] in values:
            dp[i] = max(dp[i - 1] + 1, dp[i])
        else:
            dp[i] = max(dp[i - 1], dp[i])

        # case2:
        if i-2 >= 0:
            start = i-2
            end = max(i-k, 0)
            for j in range(start, end-1, -1):
                split_str = total[j:i]
                if split_str in values:
                    dp[i] = max(dp[i], dp[j]+1)
                else:
                    dp[i] = max(dp[i], dp[j])

    return dp[length]

if __name__ == "__main__":
    total = input()
    k = int(input())
    values = input().split(",")
    print(get_max_value(total=total, k=k, values=values))
