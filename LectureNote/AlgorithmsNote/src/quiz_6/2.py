from typing import List


def get_minimal_unswap_times(A: List[int], B: List[int]):
    n = len(A)
    m = len(B)
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidate = []
            candidate.append(dp[i - 1][j] + 1)
            candidate.append(dp[i][j - 1] + 1)
            if A[i - 1] == B[j - 1]:
                candidate.append(dp[i - 1][j - 1])

            if (
                i - 2 >= 0
                and j - 2 >= 0
                and A[i - 2] == B[j - 1]
                and A[i - 1] == B[j - 2]
            ):
                candidate.append(dp[i - 2][j - 2])

            dp[i][j] = min(candidate)

    return dp[n][m]


if __name__ == "__main__":
    A = [1, 4, 2, 3]
    B = [1, 2, 3, 4]
    print(get_minimal_unswap_times(A, B))
