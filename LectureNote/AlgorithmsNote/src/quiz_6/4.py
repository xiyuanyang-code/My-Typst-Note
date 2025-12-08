from typing import List


def count_paths(F: List[List[int]]):
    n = len(F)

    # K[i][j] 代表最大蘑菇数量
    # X[i][j] 代表拾取最大蘑菇数量的路径数量

    K = [[0] * n for _ in range(n)]
    X = [[0] * n for _ in range(n)]

    # ensure [0][0] and [n-1][n-1] does not have trees

    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                K[0][0] = 1 if F[0][0] == "m" else 0
                X[0][0] = 1
                continue
            # compute i,j
            if i - 1 >= 0 and F[i - 1][j] != "t":
                K[i][j] = max(K[i][j], K[i - 1][j])
            if j - 1 >= 0 and F[i][j - 1] != "t":
                K[i][j] = max(K[i][j], K[i][j - 1])

            # update X[i][j]
            if i - 1 >= 0 and K[i][j] == K[i - 1][j]:
                X[i][j] += X[i - 1][j]

            if j - 1 >= 0 and K[i][j] == K[i][j - 1]:
                X[i][j] += X[i][j - 1]

            if F[i][j] == "m":
                K[i][j] += 1

            if F[i][j] == "t":
                K[i][j] = 0
                F[i][j] = 0

    return X[n - 1][n - 1]


if __name__ == "__main__":
    F = [["x", "m", "x"], ["x", "x", "m"], ["x", "x", "x"]]
    assert count_paths(F) == 2
    print("All test passed")
